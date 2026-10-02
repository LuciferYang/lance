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
    run_compaction_pipeline,
};
use super::transaction::{Operation, Transaction, UpdateMode};
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
    /// tombstoned in the files they came from; a file left holding no column
    /// of the schema is dropped, unless it carries the fragment's spilled row
    /// lineage. Nothing is committed: commit the result with
    /// [`RewriteColumnsCommitter`]. Returns `None` when the columns already sit
    /// alone in a file of the requested version, so a retried or resumed
    /// rewrite skips finished work.
    pub async fn rewrite_columns(
        &self,
        columns: &[&str],
        data_storage_version: Option<LanceFileVersion>,
    ) -> Result<Option<RewriteColumnsResult>> {
        let dataset = self.dataset();
        let write_schema = rewrite_schema(dataset, columns)?;
        let resolved_version = write_version(dataset, data_storage_version)?;
        let field_ids: HashSet<i32> = write_schema.field_ids().into_iter().collect();
        // Ids of columns `drop_columns` removed stay in their files, and a
        // spilled row lineage sequence sits in a data file under a reserved id,
        // so a file's ids are only compared and kept by what the schema reads.
        let schema_ids: HashSet<i32> = dataset
            .schema()
            .fields_pre_order()
            .map(|field| field.id)
            .collect();

        let already_split = self.metadata().files.iter().any(|file| {
            file.fields
                .iter()
                .copied()
                .filter(|id| schema_ids.contains(id))
                .collect::<HashSet<_>>()
                == field_ids
                && file
                    .file_version()
                    .is_ok_and(|version| version == resolved_version)
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
                resolved_version,
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
        // Drop a file left with nothing the schema reads, the rule
        // `drop_columns` applies, but keep one carrying the fragment's spilled
        // row lineage: it is the only copy.
        let spilled = fragment.spilled_row_lineage_field_ids();
        fragment.files.retain(|file| {
            file.fields
                .iter()
                .any(|id| schema_ids.contains(id) || spilled.contains(id))
        });
        // The new file is the last one and is always kept.
        let files_removed = self.metadata().files.len() + 1 - fragment.files.len();
        Ok(Some(RewriteColumnsResult {
            fragment,
            read_version: dataset.manifest.version,
            files_removed,
        }))
    }
}
/// One fragment's horizontal-rewrite output, bound to the manifest version it
/// was read at. The committer commits against that version so the conflict
/// check catches any change made to the fragment between the read and the
/// commit (mirrors vertical compaction's `RewriteResult::read_version`).
#[derive(Debug, Clone)]
pub struct RewriteColumnsResult {
    /// The rewritten fragment metadata to publish.
    pub fragment: Fragment,
    /// The manifest version the fragment was read at.
    pub read_version: u64,
    /// How many of the fragment's data files the rewrite dropped.
    pub files_removed: usize,
}

/// Horizontal executor: rewrites one fragment's named columns into a new data
/// file (see [`FileFragment::rewrite_columns`]). The columns and target version
/// are the executor's configuration; the task is the fragment to rewrite.
#[derive(Debug, Clone)]
pub struct RewriteColumnsExecutor {
    columns: Vec<String>,
    data_storage_version: Option<LanceFileVersion>,
}

impl RewriteColumnsExecutor {
    /// Build an executor that repacks `columns` into one new file per fragment,
    /// optionally in a different V2 version.
    pub fn new(columns: &[&str], data_storage_version: Option<LanceFileVersion>) -> Self {
        Self {
            columns: columns.iter().map(|c| c.to_string()).collect(),
            data_storage_version,
        }
    }
}

#[async_trait::async_trait]
impl CompactionExecutor for RewriteColumnsExecutor {
    type Task = FileFragment;
    /// `None` when the fragment already had the layout (nothing to commit).
    type TaskResult = Option<RewriteColumnsResult>;

    /// The task is a [`FileFragment`] already bound to its read version, so the
    /// `dataset` argument the pipeline threads in is unused; the result carries
    /// the fragment's read version for the committer.
    async fn execute(
        &self,
        _dataset: Cow<'_, Dataset>,
        task: FileFragment,
        _options: &CompactionOptions,
    ) -> Result<Option<RewriteColumnsResult>> {
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
    type TaskResult = Option<RewriteColumnsResult>;

    async fn commit(
        &self,
        dataset: &mut Dataset,
        results: Vec<Option<RewriteColumnsResult>>,
        _remap_options: Arc<dyn IndexRemapperOptions>,
        _options: &CompactionOptions,
    ) -> Result<CompactionMetrics> {
        let results: Vec<RewriteColumnsResult> = results.into_iter().flatten().collect();
        if results.is_empty() {
            return Ok(CompactionMetrics::default());
        }
        // Commit against the earliest version any rewritten fragment was read
        // at, not the current one, so the conflict check catches a fragment
        // changed between the read and this commit (as vertical compaction does).
        let read_version = results
            .iter()
            .map(|result| result.read_version)
            .min()
            .expect("results is non-empty");
        // Row addresses and values are unchanged, so no fragment is added or
        // removed; only per-fragment files change: one new file per rewritten
        // fragment, and the files each rewrite dropped. A fragment changed or
        // removed since its read is left to the commit's conflict check.
        let metrics = CompactionMetrics {
            files_added: results.len(),
            files_removed: results.iter().map(|result| result.files_removed).sum(),
            ..CompactionMetrics::default()
        };
        let updated_fragments: Vec<Fragment> =
            results.into_iter().map(|result| result.fragment).collect();
        let transaction = Transaction::new(
            read_version,
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
    /// results with [`RewriteColumnsCommitter`].
    ///
    /// The new files carry only the rewritten columns. On a table with stable
    /// row ids whose row lineage was spilled into a data file, that file is
    /// kept for the lineage even after every column has moved out of it, until
    /// a vertical compaction rewrites the fragment.
    pub async fn rewrite_columns(
        &mut self,
        columns: &[&str],
        data_storage_version: Option<LanceFileVersion>,
    ) -> Result<CompactionMetrics> {
        let executor = RewriteColumnsExecutor::new(columns, data_storage_version);
        let options = CompactionOptions::default();
        let tasks = self.get_fragments();
        let concurrency = self.object_store.io_parallelism();
        // Like vertical compaction, a failed run leaves its new files
        // uncommitted and unreferenced; dataset cleanup reclaims them.
        run_compaction_pipeline(
            self,
            executor,
            RewriteColumnsCommitter,
            tasks,
            Arc::new(crate::dataset::index::DatasetIndexRemapperOptions::default()),
            &options,
            concurrency,
        )
        .await
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

    #[tokio::test]
    async fn rewrite_after_fragment_fully_deleted() {
        let mut dataset = write("memory://", LanceFileVersion::V2_0).await;
        // Deleting every row of a fragment removes it from the manifest, so the
        // rewrite runs over the surviving fragment(s); it must still succeed and
        // leave the live rows byte-for-byte unchanged.
        dataset.delete("a < 4").await.unwrap();
        let before = dataset.scan().try_into_batch().await.unwrap();

        dataset.rewrite_columns(&["c"], None).await.unwrap();

        assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
        dataset.validate().await.unwrap();
    }

    /// A delete that lands on the fragment between the rewrite's read and its
    /// commit, whether it removes one row or the whole fragment, must surface
    /// as a retryable conflict.
    #[rstest::rstest]
    #[case::one_row("a = 1", 7)]
    #[case::whole_fragment("a < 4", 4)]
    #[tokio::test]
    async fn rewrite_columns_conflicts_with_intervening_delete(
        #[case] predicate: &str,
        #[case] rows_left: usize,
    ) {
        use crate::dataset::index::DatasetIndexRemapperOptions;
        use crate::dataset::optimize::{
            CompactionCommitter, CompactionExecutor, CompactionOptions,
        };
        use std::borrow::Cow;

        let mut dataset = write("memory://", LanceFileVersion::V2_0).await;

        // Rewrite column "c" of fragment 0, reading at the current version.
        let executor = RewriteColumnsExecutor::new(&["c"], None);
        let fragment = dataset.get_fragment(0).unwrap();
        let stale = executor
            .execute(
                Cow::Borrowed(&dataset),
                fragment,
                &CompactionOptions::default(),
            )
            .await
            .unwrap()
            .expect("fragment 0 has columns to repack");

        // A delete lands on that same fragment before the rewrite is committed.
        dataset.delete(predicate).await.unwrap();
        let after_delete = dataset.scan().try_into_batch().await.unwrap();
        assert_eq!(after_delete.num_rows(), rows_left);

        // The rewrite was read before the delete, so committing it must conflict
        // rather than publish the stale layout and resurrect the deleted row.
        let err = RewriteColumnsCommitter
            .commit(
                &mut dataset,
                vec![Some(stale)],
                Arc::new(DatasetIndexRemapperOptions::default()),
                &CompactionOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "a rewrite read before the delete must not commit over it: {err}"
        );

        // The delete stands.
        assert_eq!(dataset.scan().try_into_batch().await.unwrap(), after_delete);
        assert_eq!(
            dataset
                .scan()
                .filter(predicate)
                .unwrap()
                .try_into_batch()
                .await
                .unwrap()
                .num_rows(),
            0
        );
        dataset.validate().await.unwrap();
    }

    /// `drop_columns` leaves a dropped column's id in the file that held it.
    /// Rewriting the rest of that file's columns leaves it with nothing the
    /// schema reads, so the rewrite must drop it, as `drop_columns` would.
    #[tokio::test]
    async fn rewrite_drops_file_left_with_only_dropped_columns() {
        use crate::dataset::NewColumnTransform;
        let mut dataset = write("memory://", LanceFileVersion::V2_0).await;
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("d".into(), "a + 1".into())]),
                None,
                None,
            )
            .await
            .unwrap();
        dataset.drop_columns(&["c"]).await.unwrap();
        let before = dataset.scan().try_into_batch().await.unwrap();

        dataset
            .rewrite_columns(&["a", "b", "d"], None)
            .await
            .unwrap();

        for fragment in dataset.get_fragments() {
            assert_eq!(
                fragment.metadata().files.len(),
                1,
                "only the new file is left: {:?}",
                fragment.metadata().files
            );
        }
        assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
        dataset.validate().await.unwrap();
    }

    /// A file whose only other column was dropped already holds exactly the
    /// requested columns, so there is nothing to rewrite.
    #[tokio::test]
    async fn rewrite_skips_file_whose_other_columns_were_dropped() {
        let mut dataset = write("memory://", LanceFileVersion::V2_0).await;
        dataset.drop_columns(&["c"]).await.unwrap();
        let version = dataset.manifest.version;

        dataset.rewrite_columns(&["a", "b"], None).await.unwrap();

        assert_eq!(dataset.manifest.version, version, "nothing was committed");
        dataset.validate().await.unwrap();
    }

    /// A `drop_columns` that commits between a rewrite's read and its commit
    /// would leave a file holding only the dropped column if the rewrite went
    /// through, whether that is the rewritten column's new file or a file the
    /// drop removed, so the commit must conflict. When the rewritten column
    /// survived the drop, rerunning the rewrite on the new version succeeds.
    #[rstest::rstest]
    #[case::rewritten_column("c")]
    #[case::other_column("d")]
    #[tokio::test]
    async fn rewrite_columns_conflicts_with_concurrent_drop(#[case] dropped: &str) {
        use crate::dataset::NewColumnTransform;
        use crate::dataset::index::DatasetIndexRemapperOptions;
        use crate::dataset::optimize::{
            CompactionCommitter, CompactionExecutor, CompactionOptions,
        };
        use std::borrow::Cow;

        let mut dataset = write("memory://", LanceFileVersion::V2_0).await;
        // `d` gets a file of its own in every fragment.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("d".into(), "a + 1".into())]),
                None,
                None,
            )
            .await
            .unwrap();

        let executor = RewriteColumnsExecutor::new(&["c"], None);
        let mut stale = Vec::new();
        for fragment in dataset.get_fragments() {
            stale.push(
                executor
                    .execute(
                        Cow::Borrowed(&dataset),
                        fragment,
                        &CompactionOptions::default(),
                    )
                    .await
                    .unwrap(),
            );
        }

        dataset.drop_columns(&[dropped]).await.unwrap();
        let version = dataset.manifest.version;
        let expected = dataset.scan().try_into_batch().await.unwrap();

        let err = RewriteColumnsCommitter
            .commit(
                &mut dataset,
                stale,
                Arc::new(DatasetIndexRemapperOptions::default()),
                &CompactionOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "{err}"
        );
        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.manifest.version, version, "nothing was committed");
        dataset.validate().await.unwrap();

        if dropped != "c" {
            dataset.rewrite_columns(&["c"], None).await.unwrap();
            assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
            dataset.validate().await.unwrap();
        }
    }

    /// The concurrent-drop check must not trip on a file holding only
    /// tombstones, which an in-place merge_insert leaves in its post-image for
    /// the commit to drop: a rename racing such a commit has to go through.
    #[tokio::test]
    async fn rewrite_columns_commit_ignores_tombstoned_file_on_rename() {
        use crate::dataset::index::DatasetIndexRemapperOptions;
        use crate::dataset::optimize::{
            CompactionCommitter, CompactionExecutor, CompactionOptions,
        };
        use crate::dataset::{ColumnAlteration, NewColumnTransform};
        use std::borrow::Cow;

        let mut dataset = write("memory://", LanceFileVersion::V2_0).await;
        // `d` gets a file of its own in every fragment.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("d".into(), "a + 1".into())]),
                None,
                None,
            )
            .await
            .unwrap();

        // `d` already sits alone, so repack it together with `c`.
        let executor = RewriteColumnsExecutor::new(&["c", "d"], None);
        let mut results = Vec::new();
        for fragment in dataset.get_fragments() {
            // The file `d` lived in, fully tombstoned, as merge_insert leaves it.
            let mut tombstoned = fragment.metadata().files[1].clone();
            tombstoned.fields = vec![TOMBSTONE_FIELD_ID].into();
            let mut result = executor
                .execute(
                    Cow::Borrowed(&dataset),
                    fragment,
                    &CompactionOptions::default(),
                )
                .await
                .unwrap()
                .expect("c and d have files to repack");
            result.fragment.files.insert(1, tombstoned);
            results.push(Some(result));
        }

        dataset
            .alter_columns(&[ColumnAlteration::new("b".into()).rename("b2".into())])
            .await
            .unwrap();
        let expected = dataset.scan().try_into_batch().await.unwrap();

        RewriteColumnsCommitter
            .commit(
                &mut dataset,
                results,
                Arc::new(DatasetIndexRemapperOptions::default()),
                &CompactionOptions::default(),
            )
            .await
            .unwrap();

        for fragment in dataset.get_fragments() {
            let layout: Vec<Vec<i32>> = fragment
                .metadata()
                .files
                .iter()
                .map(|file| file.fields.to_vec())
                .collect();
            assert_eq!(
                layout,
                vec![vec![0, 1, TOMBSTONE_FIELD_ID], vec![2, 3]],
                "the rewrite landed and the commit dropped the tombstoned file"
            );
        }
        assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
        dataset.validate().await.unwrap();
    }
}
