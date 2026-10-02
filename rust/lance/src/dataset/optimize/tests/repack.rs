// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Column repacking through `compact_files`.

use super::*;
use crate::dataset::NewColumnTransform;
use arrow_array::StringArray;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;

fn abc_batch(start: i32, rows: i32) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
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

/// Two fragments of four rows, each `a, b, c` in one file.
async fn write(uri: &str, version: LanceFileVersion) -> Dataset {
    let data = abc_batch(0, 8);
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

/// `write`, then two backfills: every fragment holds its columns in three
/// files, `[a, b, c]`, `[d]` and `[e]`.
async fn write_backfilled() -> Dataset {
    let mut dataset = write("memory://", LanceFileVersion::V2_0).await;
    for (name, expr) in [("d", "a + 1"), ("e", "a + 2")] {
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![(name.into(), expr.into())]),
                None,
                None,
            )
            .await
            .unwrap();
    }
    dataset
}

fn repack_options(max_files: Option<usize>, groups: Vec<Vec<&str>>) -> CompactionOptions {
    CompactionOptions {
        max_data_files_per_fragment: max_files,
        column_groups: groups
            .into_iter()
            .map(|group| group.into_iter().map(String::from).collect())
            .collect(),
        scope: CompactionScope::RepackColumns,
        ..Default::default()
    }
}

fn layout(dataset: &Dataset) -> Vec<Vec<Vec<i32>>> {
    dataset
        .get_fragments()
        .iter()
        .map(|fragment| {
            fragment
                .metadata()
                .files
                .iter()
                .map(|file| file.fields.to_vec())
                .collect()
        })
        .collect()
}

fn live_file_counts(dataset: &Dataset) -> Vec<usize> {
    dataset
        .column_layout_stats()
        .iter()
        .map(|stats| stats.live_file_count)
        .collect()
}

#[tokio::test]
async fn repack_collapses_backfilled_files() {
    let mut dataset = write_backfilled().await;
    let before = dataset.scan().try_into_batch().await.unwrap();
    let version = dataset.manifest.version;
    assert_eq!(live_file_counts(&dataset), vec![3, 3]);

    let metrics = compact_files(&mut dataset, repack_options(Some(1), vec![]), None)
        .await
        .unwrap();

    assert_eq!(dataset.manifest.version, version + 1, "one commit");
    assert_eq!(live_file_counts(&dataset), vec![1, 1]);
    assert_eq!(layout(&dataset), vec![vec![vec![0, 1, 2, 3, 4]]; 2]);
    assert_eq!(metrics.files_added, 2);
    assert_eq!(metrics.files_removed, 6);
    assert_eq!(metrics.fragments_added, 0);
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();

    // Nothing is left to do, so nothing is committed.
    compact_files(&mut dataset, repack_options(Some(1), vec![]), None)
        .await
        .unwrap();
    assert_eq!(dataset.manifest.version, version + 1);
}

/// Above the limit, the columns of the biggest file stay where they are and
/// only the others are rewritten.
#[tokio::test]
async fn repack_keeps_the_biggest_file() {
    let mut dataset = write_backfilled().await;
    let before = dataset.scan().try_into_batch().await.unwrap();

    compact_files(&mut dataset, repack_options(Some(2), vec![]), None)
        .await
        .unwrap();

    for files in layout(&dataset) {
        assert_eq!(files.len(), 2, "{files:?}");
        assert_eq!(files[0], vec![0, 1, 2], "the base file is untouched");
        assert_eq!(files[1], vec![3, 4]);
    }
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn repack_follows_column_groups() {
    let mut dataset = write_backfilled().await;
    let before = dataset.scan().try_into_batch().await.unwrap();

    compact_files(
        &mut dataset,
        repack_options(None, vec![vec!["c", "e"]]),
        None,
    )
    .await
    .unwrap();

    for files in layout(&dataset) {
        let mut files = files
            .into_iter()
            .map(|fields| {
                fields
                    .into_iter()
                    .filter(|id| *id != TOMBSTONE_FIELD_ID)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        files.sort();
        // The columns no group claims share one file.
        assert_eq!(files, vec![vec![0, 1, 3], vec![2, 4]], "{files:?}");
    }
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();

    // The layout matches the groups now.
    let version = dataset.manifest.version;
    compact_files(
        &mut dataset,
        repack_options(None, vec![vec!["c", "e"]]),
        None,
    )
    .await
    .unwrap();
    assert_eq!(dataset.manifest.version, version);
}

/// Deleted rows stay deleted and the new files line up with the old ones.
#[tokio::test]
async fn repack_keeps_deletions() {
    let mut dataset = write_backfilled().await;
    dataset.delete("a = 1 OR a = 6").await.unwrap();
    let before = dataset.scan().try_into_batch().await.unwrap();

    let options = CompactionOptions {
        // Neither small nor deleted enough to be rewritten.
        target_rows_per_fragment: 4,
        materialize_deletions_threshold: 0.5,
        max_data_files_per_fragment: Some(1),
        ..Default::default()
    };
    compact_files(&mut dataset, options, None).await.unwrap();

    assert_eq!(live_file_counts(&dataset), vec![1, 1]);
    assert!(
        dataset
            .get_fragments()
            .iter()
            .all(|fragment| fragment.metadata().deletion_file.is_some())
    );
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn repack_keeps_scalar_index() {
    let mut dataset = write_backfilled().await;
    dataset
        .create_index(
            &["d"],
            IndexType::BTree,
            Some("d_idx".into()),
            &ScalarIndexParams::default(),
            false,
        )
        .await
        .unwrap();
    let before = dataset.load_indices().await.unwrap()[0].clone();

    compact_files(&mut dataset, repack_options(Some(1), vec![]), None)
        .await
        .unwrap();

    let after = dataset.load_indices().await.unwrap()[0].clone();
    assert_eq!(after.uuid, before.uuid, "the index is not rebuilt");
    assert_eq!(after.fragment_bitmap, before.fragment_bitmap);
    let filtered = dataset
        .scan()
        .filter("d = 6")
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(filtered.num_rows(), 1);
    dataset.validate().await.unwrap();
}

/// `drop_columns` leaves a dropped column's id in its file. A repack that
/// moves the other columns out drops that file, as `drop_columns` would.
#[tokio::test]
async fn repack_drops_file_left_with_only_dropped_columns() {
    let mut dataset = write_backfilled().await;
    dataset.drop_columns(&["c"]).await.unwrap();
    let before = dataset.scan().try_into_batch().await.unwrap();

    compact_files(&mut dataset, repack_options(Some(1), vec![]), None)
        .await
        .unwrap();

    assert_eq!(layout(&dataset), vec![vec![vec![0, 1, 3, 4]]; 2]);
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();
}

/// A run that rewrites some fragments and repacks others commits the
/// rewrite first and the repacks second, and never touches one fragment both
/// ways.
#[tokio::test]
async fn compaction_rewrites_and_repacks_in_one_run() {
    let mut dataset = write_backfilled().await;
    // Fragment 0 qualifies for a rewrite, fragment 1 only for a repack.
    dataset.delete("a = 0 OR a = 1").await.unwrap();
    let before = dataset.scan().try_into_batch().await.unwrap();
    let version = dataset.manifest.version;

    let options = CompactionOptions {
        target_rows_per_fragment: 3,
        max_data_files_per_fragment: Some(1),
        ..Default::default()
    };
    let plan = plan_compaction(&dataset, &options).await.unwrap();
    assert_eq!(plan.tasks.len(), 2, "{plan:?}");
    assert_eq!(plan.tasks[0].kind, CompactionTaskKind::RewriteFragments);
    assert_eq!(plan.tasks[0].fragments[0].id, 0);
    assert!(matches!(
        plan.tasks[1].kind,
        CompactionTaskKind::RepackColumns { .. }
    ));
    assert_eq!(plan.tasks[1].fragments[0].id, 1);

    let metrics = compact_files(&mut dataset, options, None).await.unwrap();
    let last = dataset.manifest.version;
    assert!(last >= version + 2);
    for (version, operation) in [(last - 1, "Rewrite"), (last, "DataReplacement")] {
        let transaction = dataset
            .checkout_version(version)
            .await
            .unwrap()
            .read_transaction()
            .await
            .unwrap()
            .unwrap();
        assert_eq!(transaction.operation.to_string(), operation);
    }
    assert_eq!(metrics.fragments_removed, 1);
    assert_eq!(metrics.fragments_added, 1);
    assert_eq!(live_file_counts(&dataset), vec![1, 1]);
    // The rewritten fragment gets a new id, so it now scans last; the repack
    // changes no value.
    let rewritten = dataset.checkout_version(last - 1).await.unwrap();
    let after_rewrite = rewritten.scan().try_into_batch().await.unwrap();
    assert_eq!(after_rewrite.num_rows(), before.num_rows());
    assert_eq!(
        dataset.scan().try_into_batch().await.unwrap(),
        after_rewrite
    );
    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn compaction_scope_selects_task_kinds() {
    let mut dataset = write_backfilled().await;
    dataset.delete("a = 0 OR a = 1").await.unwrap();
    let options = |scope| CompactionOptions {
        target_rows_per_fragment: 2,
        max_data_files_per_fragment: Some(1),
        scope,
        ..Default::default()
    };

    let rewrites = plan_compaction(&dataset, &options(CompactionScope::RewriteFragments))
        .await
        .unwrap();
    assert_eq!(rewrites.tasks.len(), 1);
    assert_eq!(rewrites.tasks[0].kind, CompactionTaskKind::RewriteFragments);

    let repacks = plan_compaction(&dataset, &options(CompactionScope::RepackColumns))
        .await
        .unwrap();
    assert_eq!(repacks.tasks.len(), 2, "both fragments are repacked");
    assert!(
        repacks
            .tasks
            .iter()
            .all(|task| matches!(task.kind, CompactionTaskKind::RepackColumns { .. }))
    );
}

/// Without `max_data_files_per_fragment` or `column_groups`, nothing is
/// repacked.
#[tokio::test]
async fn compaction_plans_no_repack_by_default() {
    let dataset = write_backfilled().await;
    let plan = plan_compaction(&dataset, &repack_options(None, vec![]))
        .await
        .unwrap();
    assert!(plan.tasks.is_empty(), "{plan:?}");
}

/// Repack tasks run on another machine: they are serialized, executed on
/// their own, and their results committed together.
#[tokio::test]
async fn repack_tasks_run_distributed() {
    let mut dataset = write_backfilled().await;
    let before = dataset.scan().try_into_batch().await.unwrap();
    let plan = plan_compaction(&dataset, &repack_options(Some(1), vec![]))
        .await
        .unwrap();
    assert_eq!(plan.tasks.len(), 2);

    let mut results = Vec::new();
    for task in plan.compaction_tasks() {
        let task: CompactionTask =
            serde_json::from_str(&serde_json::to_string(&task).unwrap()).unwrap();
        let result = task.execute(&dataset).await.unwrap();
        let result: RewriteResult =
            serde_json::from_str(&serde_json::to_string(&result).unwrap()).unwrap();
        results.push(result);
    }
    commit_compaction(
        &mut dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        plan.options(),
    )
    .await
    .unwrap();

    assert_eq!(live_file_counts(&dataset), vec![1, 1]);
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), before);
    dataset.validate().await.unwrap();
}

/// A task serialized before `kind` existed rewrites its fragments.
#[test]
fn task_data_without_kind_rewrites_fragments() {
    let task: TaskData = serde_json::from_str(r#"{"fragments": []}"#).unwrap();
    assert_eq!(task.kind, CompactionTaskKind::RewriteFragments);
}

/// Run a repack of every fragment to the point of commit.
async fn stale_repack_results(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Vec<RewriteResult> {
    let plan = plan_compaction(dataset, options).await.unwrap();
    let mut results = Vec::new();
    for task in plan.compaction_tasks() {
        results.push(task.execute(dataset).await.unwrap());
    }
    results
}

async fn commit_results(
    dataset: &mut Dataset,
    results: Vec<RewriteResult>,
) -> Result<CompactionMetrics> {
    commit_compaction(
        dataset,
        results,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &CompactionOptions::default(),
    )
    .await
}

/// A delete that removes rows of a repacked fragment leaves the new files
/// aligned (a deletion only marks rows), so the repack still commits; one
/// that removes the whole fragment makes it retry.
#[rstest]
#[case::one_row("a = 1", true)]
#[case::whole_fragment("a < 4", false)]
#[tokio::test]
async fn repack_against_concurrent_delete(#[case] predicate: &str, #[case] commits: bool) {
    let mut dataset = write_backfilled().await;
    let stale = stale_repack_results(&dataset, &repack_options(Some(1), vec![])).await;

    dataset.delete(predicate).await.unwrap();
    let expected = dataset.scan().try_into_batch().await.unwrap();
    let version = dataset.manifest.version;

    let result = commit_results(&mut dataset, stale).await;
    if commits {
        result.unwrap();
        assert_eq!(live_file_counts(&dataset), vec![1, 1]);
    } else {
        let err = result.unwrap_err();
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "{err}"
        );
        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.manifest.version, version);
    }
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
    dataset.validate().await.unwrap();
}

/// A concurrent `drop_columns` of a column the repack moves makes it retry.
/// One that drops a column the repack leaves alone does not: the new file is
/// applied to the fragment as the drop left it.
#[rstest]
#[case::moved_column("d", false)]
#[case::other_column("b", true)]
#[tokio::test]
async fn repack_against_concurrent_drop(#[case] dropped: &str, #[case] commits: bool) {
    let mut dataset = write_backfilled().await;
    // Moves d and e into one file and leaves the base file alone.
    let stale = stale_repack_results(&dataset, &repack_options(Some(2), vec![])).await;

    dataset.drop_columns(&[dropped]).await.unwrap();
    let expected = dataset.scan().try_into_batch().await.unwrap();
    let version = dataset.manifest.version;

    let result = commit_results(&mut dataset, stale).await;
    if commits {
        result.unwrap();
        assert_eq!(live_file_counts(&dataset), vec![2, 2]);
    } else {
        let err = result.unwrap_err();
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "{err}"
        );
        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.manifest.version, version);
    }
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
    dataset.validate().await.unwrap();
}

/// A concurrent update that writes new values into a moved column makes the
/// repack retry rather than publish the old values over the new ones.
#[tokio::test]
async fn repack_against_concurrent_update_of_moved_column() {
    let mut dataset = write_backfilled().await;
    let stale = stale_repack_results(&dataset, &repack_options(Some(1), vec![])).await;

    crate::dataset::UpdateBuilder::new(Arc::new(dataset.clone()))
        .update_where("a = 2")
        .unwrap()
        .set("d", "100")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap();
    dataset.checkout_latest().await.unwrap();
    let expected = dataset.scan().try_into_batch().await.unwrap();

    let err = commit_results(&mut dataset, stale).await.unwrap_err();
    assert!(
        matches!(err, Error::RetryableCommitConflict { .. }),
        "{err}"
    );
    dataset.checkout_latest().await.unwrap();
    assert_eq!(dataset.scan().try_into_batch().await.unwrap(), expected);
    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn repack_skips_legacy_files() {
    let dir = TempStrDir::default();
    let dataset = write(&dir, LanceFileVersion::Legacy).await;
    let plan = plan_compaction(&dataset, &repack_options(None, vec![vec!["c"]]))
        .await
        .unwrap();
    assert!(plan.tasks.is_empty(), "{plan:?}");
}

#[test]
fn max_data_files_per_fragment_must_be_positive() {
    let mut options = CompactionOptions {
        max_data_files_per_fragment: Some(0),
        ..Default::default()
    };
    assert!(options.validate().is_err());
}

#[test]
fn repack_options_parse_from_config() {
    let config = HashMap::from([
        (
            "lance.compaction.max_data_files_per_fragment".to_string(),
            "4".to_string(),
        ),
        (
            "lance.compaction.scope".to_string(),
            "repack_columns".to_string(),
        ),
    ]);
    let options = CompactionOptions::from_dataset_config(&config).unwrap();
    assert_eq!(options.max_data_files_per_fragment, Some(4));
    assert_eq!(options.scope, CompactionScope::RepackColumns);
}

/// The file holding a fragment's spilled row lineage keeps its columns: the
/// new files carry no lineage, so emptying that file would strand it.
#[test]
fn repack_keeps_the_spilled_lineage_file() {
    use lance_table::format::{ROW_ID_FIELD_ID, RowIdMeta};
    let schema = lance_core::datatypes::Schema::try_from(&Schema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Int32, true),
        Field::new("c", DataType::Int32, true),
    ]))
    .unwrap();
    let mut fragment = Fragment::new(0);
    fragment.add_file(
        "lineage.lance",
        vec![0, ROW_ID_FIELD_ID],
        vec![0, 1],
        lance_file::version::ConcreteFileVersion::V2_0,
        None,
    );
    for (path, field) in [("b.lance", 1), ("c.lance", 2)] {
        fragment.add_file(
            path,
            vec![field],
            vec![0],
            lance_file::version::ConcreteFileVersion::V2_0,
            None,
        );
    }
    fragment.row_id_meta = Some(RowIdMeta::Column);

    // Above the limit, the lineage file's column stays and the others move.
    assert_eq!(
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            3,
            None,
            Some(2)
        ),
        Some(vec![vec![1, 2]])
    );
    // Moving every column would leave the lineage file behind, so no repack.
    assert_eq!(
        crate::dataset::optimize::repack::plan_fragment_repack(
            &schema,
            &fragment,
            3,
            None,
            Some(1)
        ),
        None
    );
}
