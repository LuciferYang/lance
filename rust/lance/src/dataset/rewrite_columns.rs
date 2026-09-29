// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Horizontal compaction: rewrite a few columns of every fragment into their
//! own data files, leaving the files that hold the other columns untouched.
//!
//! This is the horizontal pipeline of the pluggable compaction framework
//! ([`CompactionExecutor`] / [`CompactionCommitter`] in [`super::optimize`]).
//! Unlike vertical compaction (which merges whole fragments and re-encodes
//! every column), it re-reads only the named columns, writes them to one new
//! file per fragment, and tombstones them where they used to live. Rows,
//! fragment ids, row addresses and indices are unchanged, so it commits as an
//! `Operation::Update` in [`UpdateMode::RewriteColumns`] with an empty
//! `fields_modified` (values did not change: indices keep their coverage and
//! overlays keep shadowing).

use std::borrow::Cow;
use std::collections::HashSet;
use std::sync::Arc;

use lance_core::datatypes::Schema;
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_table::format::{Fragment, overlay::TOMBSTONE_FIELD_ID};

use super::fragment::FileFragment;
use super::optimize::{
    CompactionCommitter, CompactionExecutor, CompactionMetrics, CompactionOptions,
};
use super::transaction::{Operation, Transaction, UpdateMode};
use super::write::cleanup_data_fragments;
use super::{Dataset, versions};
use crate::dataset::optimize::remapping::IndexRemapperOptions;
use crate::{Error, Result};

/// The exact V2 version the new files get: `requested`, or the dataset's
/// default write version, which is never changed.
fn write_version(
    dataset: &Dataset,
    requested: Option<LanceFileVersion>,
) -> Result<ConcreteFileVersion> {
    let default_version = dataset.manifest.data_storage_format.lance_file_format();
    let version = requested
        .map(LanceFileVersion::resolve)
        .unwrap_or(default_version);
    versions::validate_write_version(default_version, version)?;
    if version == ConcreteFileVersion::V1 {
        return Err(Error::not_supported(
            "rewrite_columns requires the V2 file format: a V1 file cannot tombstone \
             single fields, so compact the dataset to V2 first",
        ));
    }
    Ok(version)
}
/// The schema of the new data file: `columns` in dataset order, all top-level.
fn rewrite_schema(dataset: &Dataset, columns: &[&str]) -> Result<Schema> {
    if columns.is_empty() {
        return Err(Error::invalid_input(
            "rewrite_columns needs at least one column",
        ));
    }
    let schema = dataset.schema();
    if let Some(column) = columns
        .iter()
        .find(|column| !schema.fields.iter().any(|field| &field.name == *column))
    {
        return Err(Error::invalid_input(format!(
            "Column \"{column}\" is not a top-level column of the dataset"
        )));
    }
    let ordered = schema
        .fields
        .iter()
        .map(|field| field.name.as_str())
        .filter(|name| columns.contains(name))
        .collect::<Vec<_>>();
    let schema = schema.project(&ordered)?;
    if let Some(blob) = schema.fields_pre_order().find(|field| field.is_blob()) {
        return Err(Error::not_supported(format!(
            "rewrite_columns cannot rewrite blob column \"{}\"",
            blob.name
        )));
    }
    Ok(schema)
}
impl FileFragment {
    /// Rewrite `columns` of this fragment into one new data file.
    ///
    /// The returned metadata has the new file appended and the rewritten fields
    /// tombstoned in the files they came from; a file left holding only
    /// tombstones is dropped. Nothing is committed. Returns `None` when the
    /// columns already sit alone in a file of the requested version, so a
    /// retried or resumed rewrite skips finished work.
    pub async fn rewrite_columns(
        &self,
        columns: &[&str],
        data_storage_version: Option<LanceFileVersion>,
    ) -> Result<Option<Fragment>> {
        let dataset = self.dataset();
        let write_schema = rewrite_schema(dataset, columns)?;
        let write_version = write_version(dataset, data_storage_version)?;
        let field_ids: HashSet<i32> = write_schema.field_ids().into_iter().collect();

        let already_split = self.metadata().files.iter().any(|file| {
            file.fields
                .iter()
                .copied()
                .filter(|id| *id != TOMBSTONE_FIELD_ID)
                .collect::<HashSet<_>>()
                == field_ids
                && file
                    .file_version()
                    .is_ok_and(|version| version == write_version)
        });
        if already_split {
            return Ok(None);
        }

        let mut updater = self
            .updater_with_version(
                Some(columns),
                Some((write_schema, dataset.schema().clone())),
                None,
                None,
                write_version,
            )
            .await?;
        let written: Result<Fragment> = async {
            while let Some(batch) = updater.next().await?.cloned() {
                updater.update(batch).await?;
            }
            updater.finish().await
        }
        .await;
        let mut fragment = match written {
            Ok(fragment) => fragment,
            Err(err) => {
                updater.cleanup_unfinished_writer().await;
                return Err(err);
            }
        };
        // No live rows, so no file was written and nothing to tombstone.
        if fragment.files.len() == self.metadata().files.len() {
            return Ok(None);
        }

        let new_file = fragment.files.len() - 1;
        for file in &mut fragment.files[..new_file] {
            file.fields = file
                .fields
                .iter()
                .map(|id| {
                    if field_ids.contains(id) {
                        TOMBSTONE_FIELD_ID
                    } else {
                        *id
                    }
                })
                .collect::<Vec<_>>()
                .into();
        }
        // A file holding only tombstones is unreachable to readers.
        fragment
            .files
            .retain(|file| file.fields.iter().any(|id| *id != TOMBSTONE_FIELD_ID));
        Ok(Some(fragment))
    }
}
/// Horizontal executor: rewrites one fragment's named columns into a new data
/// file (see [`FileFragment::rewrite_columns`]). The columns and target version
/// are the executor's configuration; the task is the fragment to rewrite.
#[derive(Debug, Clone)]
pub struct RewriteColumnsExecutor {
    columns: Vec<String>,
    data_storage_version: Option<LanceFileVersion>,
}

#[async_trait::async_trait]
impl CompactionExecutor for RewriteColumnsExecutor {
    type Task = FileFragment;
    /// `None` when the fragment already had the layout (nothing to commit).
    type TaskResult = Option<Fragment>;

    async fn execute(
        &self,
        _dataset: Cow<'_, Dataset>,
        task: FileFragment,
        _options: &CompactionOptions,
    ) -> Result<Option<Fragment>> {
        let columns: Vec<&str> = self.columns.iter().map(String::as_str).collect();
        task.rewrite_columns(&columns, self.data_storage_version)
            .await
    }
}
/// Commits horizontal-compaction results as one `Operation::Update` in
/// [`UpdateMode::RewriteColumns`] with an empty `fields_modified`: values did
/// not change, so index coverage is kept and overlays keep shadowing.
#[derive(Debug, Default, Clone, Copy)]
pub struct RewriteColumnsCommitter;

#[async_trait::async_trait]
impl CompactionCommitter for RewriteColumnsCommitter {
    type TaskResult = Option<Fragment>;

    async fn commit(
        &self,
        dataset: &mut Dataset,
        results: Vec<Option<Fragment>>,
        _remap_options: Arc<dyn IndexRemapperOptions>,
        _options: &CompactionOptions,
    ) -> Result<CompactionMetrics> {
        let updated_fragments: Vec<Fragment> = results.into_iter().flatten().collect();
        if updated_fragments.is_empty() {
            return Ok(CompactionMetrics::default());
        }
        // Row addresses and values are unchanged, so no fragment is added or
        // removed; only per-fragment files change. `files_added` is one new
        // file per rewritten fragment; `files_removed` is the fully-tombstoned
        // files that were dropped.
        let mut metrics = CompactionMetrics {
            files_added: updated_fragments.len(),
            ..CompactionMetrics::default()
        };
        for fragment in &updated_fragments {
            let original_files = dataset
                .get_fragment(fragment.id as usize)
                .map(|f| f.metadata().files.len())
                .unwrap_or(0);
            metrics.files_removed += (original_files + 1).saturating_sub(fragment.files.len());
        }
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::Update {
                removed_fragment_ids: Vec::new(),
                updated_fragments,
                new_fragments: Vec::new(),
                fields_modified: Vec::new(),
                compacted_sstables: Vec::new(),
                fields_for_preserving_frag_bitmap: Vec::new(),
                update_mode: Some(UpdateMode::RewriteColumns),
                inserted_rows_filter: None,
                updated_fragment_offsets: None,
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await?;
        Ok(metrics)
    }
}
impl Dataset {
    /// Rewrite `columns` of every fragment into one new data file per fragment,
    /// optionally in a different V2 version, without touching the files that
    /// hold the other columns. Fragments already in the requested layout are
    /// skipped, so an interrupted rewrite can be rerun.
    ///
    /// Runs through the horizontal compaction pipeline
    /// ([`RewriteColumnsExecutor`] + [`RewriteColumnsCommitter`]) and commits
    /// once. To spread the work over many machines, call
    /// [`FileFragment::rewrite_columns`] per fragment and commit the returned
    /// metadata with [`RewriteColumnsCommitter`].
    pub async fn rewrite_columns(
        &mut self,
        columns: &[&str],
        data_storage_version: Option<LanceFileVersion>,
    ) -> Result<CompactionMetrics> {
        let executor = RewriteColumnsExecutor {
            columns: columns.iter().map(|c| c.to_string()).collect(),
            data_storage_version,
        };
        let options = CompactionOptions::default();
        let fragments = self.get_fragments();
        let mut results = Vec::with_capacity(fragments.len());
        for fragment in fragments {
            match executor
                .execute(Cow::Borrowed(&*self), fragment, &options)
                .await
            {
                Ok(result) => results.push(result),
                Err(err) => {
                    self.cleanup_rewritten_files(&results).await;
                    return Err(err);
                }
            }
        }
        RewriteColumnsCommitter
            .commit(
                self,
                results,
                Arc::new(crate::dataset::index::DatasetIndexRemapperOptions::default()),
                &options,
            )
            .await
    }

    /// Delete the files a failed rewrite wrote: the last file of every
    /// rewritten fragment is the only one that did not exist before.
    async fn cleanup_rewritten_files(&self, results: &[Option<Fragment>]) {
        let new_files = results
            .iter()
            .flatten()
            .filter_map(|fragment| fragment.files.last())
            .map(|file| Fragment {
                files: vec![file.clone()],
                ..Fragment::new(0)
            })
            .collect::<Vec<_>>();
        cleanup_data_fragments(&self.object_store, &self.base, None, &new_files).await;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::WriteParams;
    use crate::index::DatasetIndexExt;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_index::IndexType;
    use lance_index::scalar::ScalarIndexParams;
    use std::sync::Arc;

    fn batch(start: i32, rows: i32) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
            Field::new("c", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(start..start + rows)),
                Arc::new(Int32Array::from_iter_values(
                    (start..start + rows).map(|v| v * 10),
                )),
                Arc::new(StringArray::from_iter_values(
                    (start..start + rows).map(|v| format!("row-{v}")),
                )),
            ],
        )
        .unwrap()
    }

    async fn write(uri: &str, version: LanceFileVersion) -> Dataset {
        let data = batch(0, 8);
        let reader = RecordBatchIterator::new([Ok(data.clone())], data.schema());
        Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                max_rows_per_file: 4,
                data_storage_version: Some(version),
                ..Default::default()
            }),
        )
        .await
        .unwrap()
    }

    fn file_versions(fragment: &Fragment) -> Vec<(Vec<i32>, ConcreteFileVersion)> {
        fragment
            .files
            .iter()
            .map(|file| (file.fields.to_vec(), file.file_version().unwrap()))
            .collect()
    }
    #[tokio::test]
    async fn rewrite_migrates_only_the_named_column() {
        let mut dataset = write("memory://", LanceFileVersion::V2_0).await;
        dataset.delete("a = 3").await.unwrap();
        let before = dataset.scan().try_into_batch().await.unwrap();
        let version = dataset.manifest.version;

        let metrics = dataset
            .rewrite_columns(&["c"], Some(LanceFileVersion::V2_2))
            .await
            .unwrap();
        assert_eq!(metrics.files_added, 2, "one new file per fragment");

        assert_eq!(dataset.manifest.version, version + 1);
        assert_eq!(dataset.manifest.fragments.len(), 2);
        for fragment in dataset.manifest.fragments.iter() {
            assert_eq!(
                file_versions(fragment),
                vec![
                    (vec![0, 1, TOMBSTONE_FIELD_ID], ConcreteFileVersion::V2_0),
                    (vec![2], ConcreteFileVersion::V2_2),
                ]
            );
        }
        // Values are byte-for-byte unchanged.
        assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
        // The dataset default write version is not changed.
        assert_eq!(
            dataset.manifest.data_storage_format.lance_file_format(),
            ConcreteFileVersion::V2_0
        );
        dataset.validate().await.unwrap();

        // Idempotent: already in the requested layout, nothing committed.
        dataset
            .rewrite_columns(&["c"], Some(LanceFileVersion::V2_2))
            .await
            .unwrap();
        assert_eq!(dataset.manifest.version, version + 1);
    }
    #[tokio::test]
    async fn rewrite_keeps_scalar_index() {
        let mut dataset = write("memory://", LanceFileVersion::V2_0).await;
        dataset
            .create_index(
                &["a"],
                IndexType::BTree,
                Some("a_idx".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let before = dataset.load_indices().await.unwrap()[0].clone();

        dataset.rewrite_columns(&["a"], None).await.unwrap();

        let after = dataset.load_indices().await.unwrap()[0].clone();
        assert_eq!(after.uuid, before.uuid);
        assert_eq!(after.fragment_bitmap, before.fragment_bitmap);
        let filtered = dataset
            .scan()
            .filter("a = 5")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(filtered.num_rows(), 1);
        dataset.validate().await.unwrap();
    }

    #[tokio::test]
    async fn rewrite_rejects_bad_input() {
        let mut dataset = write("memory://", LanceFileVersion::V2_0).await;
        let err = dataset
            .rewrite_columns(&["missing"], None)
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");

        let legacy_dir = TempStrDir::default();
        let mut legacy = write(&legacy_dir, LanceFileVersion::Legacy).await;
        let err = legacy.rewrite_columns(&["c"], None).await.unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err}");
    }

    #[tokio::test]
    async fn rewrite_collapses_backfilled_files() {
        use crate::dataset::NewColumnTransform;
        let mut dataset = write("memory://", LanceFileVersion::V2_0).await;
        // Two backfills, each appending one data file per fragment: the feature's
        // target — a fragment split across many per-column files.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("d".into(), "a + 1".into())]),
                None,
                None,
            )
            .await
            .unwrap();
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("e".into(), "a + 2".into())]),
                None,
                None,
            )
            .await
            .unwrap();
        let before = dataset.scan().try_into_batch().await.unwrap();
        assert!(
            dataset
                .column_layout_stats()
                .iter()
                .all(|s| s.live_file_count == 3),
            "each fragment carries three data files after two backfills: {:?}",
            dataset.column_layout_stats()
        );

        // Repack every column into a single file per fragment.
        dataset
            .rewrite_columns(&["a", "b", "c", "d", "e"], None)
            .await
            .unwrap();

        assert!(
            dataset
                .column_layout_stats()
                .iter()
                .all(|s| s.live_file_count == 1),
            "rewrite collapsed each fragment to one data file: {:?}",
            dataset.column_layout_stats()
        );
        assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
        dataset.validate().await.unwrap();
    }
}
