// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Column repacking, the compaction task that rewrites one fragment's columns
//! into fewer data files without moving rows.
//!
//! Every `add_columns` backfill gives each fragment one more data file, and a
//! large fragment with few deletions never qualifies for a rewrite, so its
//! files pile up. A [`CompactionTaskKind::RepackColumns`] task reads some of
//! the fragment's columns, writes them to new files, and commits the files as
//! an `Operation::DataReplacement` with `data_change: false`: the fragment id,
//! row addresses, overlays and index coverage are left as they are.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use lance_core::datatypes::{Field as LanceField, Schema};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::{DataFile, Fragment};

use super::{
    CompactionMetrics, CompactionOptions, CompactionTaskKind, RepackedFiles, RewriteResult,
    TaskData,
};
use crate::dataset::fragment::FileFragment;
use crate::dataset::transaction::{DataReplacementGroup, Operation, TransactionBuilder};
use crate::dataset::{Dataset, cleanup_data_fragments, versions};
use crate::{Error, Result};

/// The field ids of `field` and everything beneath it.
fn subtree_ids(field: &LanceField, ids: &mut HashSet<i32>) {
    ids.insert(field.id);
    for child in &field.children {
        subtree_ids(child, ids);
    }
}

fn holds_blob(field: &LanceField) -> bool {
    field.is_blob() || field.children.iter().any(holds_blob)
}

/// Every field id a data file answers for, the way a `DataReplacement`
/// commit counts it.
fn file_coverage(file: &DataFile, schema: &Schema) -> HashSet<i32> {
    file.schema(schema)
        .field_ids()
        .into_iter()
        .chain(file.fields.iter().copied())
        .filter(|id| *id >= 0)
        .collect()
}

/// The new files a repack of `fragment` should write, each the top-level
/// field ids it holds, or `None` when no repack is due.
///
/// `groups` partitions the top-level columns into the files wanted, as
/// `column_group_schemas` orders them. Without groups, the columns of one
/// live file stay together (the file holding the fragment's spilled row
/// lineage, else the one holding the most bytes) and every other column goes
/// to one new file; when `max_files` is 1, every column goes to one file. A
/// group is written unless one live file holds exactly its columns. A column
/// the fragment has no data for (added as all nulls) is left out of its
/// group, and a group holding a blob column, or a column whose fields are
/// only partly in the fragment's files, is left as it is.
///
/// The new files carry no row lineage, so a repack that would move every
/// column out of the file holding the fragment's spilled lineage is not
/// planned: that file would stay for the lineage alone, and a later rewrite
/// of the fragment is what reclaims it.
///
/// A group out of place is a reason to repack; so is `live_files` (the
/// fragment's [`FragmentColumnLayoutStats::live_file_count`]) above
/// `max_files`.
///
/// [`FragmentColumnLayoutStats::live_file_count`]: crate::dataset::compaction_stats::FragmentColumnLayoutStats::live_file_count
pub(super) fn plan_fragment_repack(
    schema: &Schema,
    fragment: &Fragment,
    live_files: usize,
    groups: Option<&[Vec<i32>]>,
    max_files: Option<usize>,
) -> Option<Vec<Vec<i32>>> {
    let over_file_limit = max_files.is_some_and(|max| live_files > max);
    if groups.is_none() && !over_file_limit {
        return None;
    }
    // A V1 file cannot tombstone single fields.
    if fragment.files.iter().any(|file| {
        !file
            .file_version()
            .is_ok_and(|version| version != ConcreteFileVersion::V1)
    }) {
        return None;
    }
    let schema_ids: HashSet<i32> = schema.fields_pre_order().map(|field| field.id).collect();
    let spilled = fragment.spilled_row_lineage_field_ids();
    let live: Vec<(&DataFile, HashSet<i32>)> = fragment
        .files
        .iter()
        .filter(|file| file.fields.iter().any(|id| schema_ids.contains(id)))
        .map(|file| (file, file_coverage(file, schema)))
        .collect();
    let covered: HashSet<i32> = live.iter().flat_map(|(_, ids)| ids).copied().collect();
    let column_ids: Vec<(i32, HashSet<i32>)> = schema
        .fields
        .iter()
        .map(|field| {
            let mut ids = HashSet::new();
            subtree_ids(field, &mut ids);
            (field.id, ids)
        })
        .collect();
    let present: BTreeSet<i32> = column_ids
        .iter()
        .filter(|(_, ids)| !ids.is_disjoint(&covered))
        .map(|(column, _)| *column)
        .collect();
    let movable: HashSet<i32> = schema
        .fields
        .iter()
        .zip(&column_ids)
        .filter(|(field, (_, ids))| !holds_blob(field) && ids.is_subset(&covered))
        .map(|(field, _)| field.id)
        .collect();
    // The top-level columns each live file holds any field of.
    let file_columns: Vec<BTreeSet<i32>> = live
        .iter()
        .map(|(_, coverage)| {
            column_ids
                .iter()
                .filter(|(_, ids)| !ids.is_disjoint(coverage))
                .map(|(column, _)| *column)
                .collect()
        })
        .collect();

    let wanted: Vec<BTreeSet<i32>> = match groups {
        Some(groups) => groups
            .iter()
            .map(|group| group.iter().copied().collect())
            .collect(),
        None if max_files == Some(1) => vec![present.clone()],
        None => {
            let kept = live
                .iter()
                .zip(&file_columns)
                .max_by_key(|((file, _), columns)| {
                    (
                        file.fields.iter().any(|id| spilled.contains(id)),
                        file.file_size_bytes.get().map(|size| size.get()),
                        columns.len(),
                    )
                })
                .map(|(_, columns)| columns.clone())
                .unwrap_or_default();
            let rest = present
                .iter()
                .copied()
                .filter(|column| !kept.contains(column) && movable.contains(column))
                .collect();
            vec![kept, rest]
        }
    };

    let out_of_place: Vec<Vec<i32>> = wanted
        .into_iter()
        .filter_map(|group| {
            let group: BTreeSet<i32> = group.intersection(&present).copied().collect();
            if group.is_empty() || !group.iter().all(|column| movable.contains(column)) {
                return None;
            }
            let holders: Vec<&BTreeSet<i32>> = file_columns
                .iter()
                .filter(|columns| !columns.is_disjoint(&group))
                .collect();
            let in_place = holders.len() == 1 && *holders[0] == group;
            (!in_place).then(|| group.into_iter().collect())
        })
        .collect();
    let moved: HashSet<i32> = out_of_place.iter().flatten().copied().collect();
    let strands_lineage = live.iter().zip(&file_columns).any(|((file, _), columns)| {
        file.fields.iter().any(|id| spilled.contains(id))
            && columns.iter().all(|column| moved.contains(column))
    });
    (!out_of_place.is_empty() && !strands_lineage).then_some(out_of_place)
}

/// How many of `fragment`'s data files a commit of `written` would drop: the
/// files left holding no field of the schema and none of the fragment's
/// spilled row lineage, the rule the `DataReplacement` commit applies.
fn files_dropped(fragment: &Fragment, written: &[DataFile], schema: &Schema) -> usize {
    let moved: HashSet<i32> = written
        .iter()
        .flat_map(|file| file_coverage(file, schema))
        .collect();
    let schema_ids: HashSet<i32> = schema.fields_pre_order().map(|field| field.id).collect();
    let spilled = fragment.spilled_row_lineage_field_ids();
    fragment
        .files
        .iter()
        .filter(|file| {
            !file
                .fields
                .iter()
                .any(|id| (schema_ids.contains(id) && !moved.contains(id)) || spilled.contains(id))
        })
        .count()
}

/// Write the new files of a [`CompactionTaskKind::RepackColumns`] task
/// against `dataset`, which must be at the plan's read version.
pub(super) async fn execute_repack(
    dataset: &Dataset,
    task: TaskData,
    options: &CompactionOptions,
) -> Result<RewriteResult> {
    let CompactionTaskKind::RepackColumns { files } = &task.kind else {
        return Err(Error::internal("execute_repack called with a rewrite task"));
    };
    let [fragment] = task.fragments.as_slice() else {
        return Err(Error::invalid_input(format!(
            "a RepackColumns task names exactly one fragment, got {}",
            task.fragments.len()
        )));
    };
    let write_version = options.write_version(dataset);
    versions::validate_write_version(
        dataset.manifest.data_storage_format.lance_file_format(),
        write_version,
    )?;
    if write_version == ConcreteFileVersion::V1 {
        return Err(Error::not_supported(
            "repacking columns needs the V2 file format: a V1 file cannot tombstone \
             single fields",
        ));
    }

    // Read the base values only: the overlays stay on the fragment and keep
    // shadowing them, so the new files hold exactly what the old ones did.
    let mut base = fragment.clone();
    base.overlays.clear();
    let source = FileFragment::new(Arc::new(dataset.clone()), base);
    let schema = dataset.schema();
    let batch_size = options.batch_size.map(|size| size as u32);

    let mut written: Vec<DataFile> = Vec::with_capacity(files.len());
    for column_ids in files {
        let names: Vec<&str> = schema
            .fields
            .iter()
            .filter(|field| column_ids.contains(&field.id))
            .map(|field| field.name.as_str())
            .collect();
        match write_one_file(&source, schema, &names, batch_size, write_version).await {
            Ok(Some(file)) => written.push(file),
            Ok(None) => {}
            Err(err) => {
                let partial = Fragment {
                    files: written,
                    ..Fragment::new(fragment.id)
                };
                cleanup_data_fragments(&dataset.object_store, &dataset.base, None, &[partial])
                    .await;
                return Err(err);
            }
        }
    }

    let metrics = CompactionMetrics {
        files_added: written.len(),
        files_removed: files_dropped(fragment, &written, schema),
        ..Default::default()
    };
    Ok(RewriteResult {
        metrics,
        new_fragments: Vec::new(),
        read_version: dataset.manifest.version,
        original_fragments: Vec::new(),
        row_addrs: None,
        repacked_files: Some(RepackedFiles {
            fragment_id: fragment.id,
            files: written,
        }),
    })
}

/// Write `columns` of `source` to one new data file. `None` when the
/// fragment has no rows to write.
async fn write_one_file(
    source: &FileFragment,
    schema: &Schema,
    columns: &[&str],
    batch_size: Option<u32>,
    write_version: ConcreteFileVersion,
) -> Result<Option<DataFile>> {
    let write_schema = schema.project(columns)?;
    let mut updater = source
        .updater_with_version(
            Some(columns),
            Some((write_schema, schema.clone())),
            batch_size,
            None,
            write_version,
        )
        .await?;
    let finished: Result<Fragment> = async {
        while let Some(batch) = updater.next().await?.cloned() {
            updater.update(batch).await?;
        }
        updater.finish().await
    }
    .await;
    let fragment = match finished {
        Ok(fragment) => fragment,
        Err(err) => {
            updater.cleanup_unfinished_writer().await;
            return Err(err);
        }
    };
    // The updater appends the file it wrote to the fragment's files.
    Ok((fragment.files.len() > source.metadata().files.len())
        .then(|| fragment.files.last().cloned())
        .flatten())
}

/// Commit the files of repack results as one `DataReplacement` that moves
/// values without changing them. Each result applies to its fragment as it
/// stands at commit, so a concurrent change to other columns of the fragment
/// is kept; the conflict check retries one that touches the moved columns.
pub(super) async fn commit_repacked_files(
    dataset: &mut Dataset,
    results: Vec<RewriteResult>,
    options: &CompactionOptions,
) -> Result<CompactionMetrics> {
    let mut metrics = CompactionMetrics::default();
    let mut read_version = u64::MAX;
    let mut replacements = Vec::new();
    for result in results {
        let Some(repacked) = result.repacked_files else {
            continue;
        };
        if repacked.files.is_empty() {
            continue;
        }
        metrics += result.metrics;
        read_version = read_version.min(result.read_version);
        replacements.extend(
            repacked
                .files
                .into_iter()
                .map(|file| DataReplacementGroup(repacked.fragment_id, file)),
        );
    }
    if replacements.is_empty() {
        return Ok(CompactionMetrics::default());
    }
    let transaction = TransactionBuilder::new(
        // The earliest version a repack read, so the conflict check covers
        // every write since (the same reason vertical compaction uses it).
        read_version,
        Operation::DataReplacement {
            replacements,
            data_change: false,
        },
    )
    .transaction_properties(options.transaction_properties.clone())
    .build();
    // Like a rewrite's, a repack result can be retried after an ambiguous
    // success, so its files are left for cleanup rather than deleted here.
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;
    Ok(metrics)
}
