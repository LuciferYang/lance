// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-fragment column-layout statistics.
//!
//! A compaction planner keys on how many data files each fragment carries: a
//! fragment split across many small per-column files (typically from repeated
//! `add_columns` backfills) is the signal horizontal compaction targets.
//! Exposing these stats via [`Dataset::column_layout_stats`] keeps the planning
//! decision observable instead of buried in a black-box heuristic.

use std::collections::HashSet;

use super::Dataset;

/// Column-layout statistics for a single fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentColumnLayoutStats {
    /// The fragment these stats describe.
    pub fragment_id: u64,
    /// Number of data files holding at least one column of the dataset schema.
    /// A large value on a wide dataset is what horizontal compaction collapses
    /// back into fewer files. A file left holding only tombstones or dropped
    /// columns, or kept only for the fragment's spilled row lineage, holds no
    /// column a read touches and is not counted.
    pub live_file_count: usize,
    /// Number of overlay files attached to the fragment.
    pub overlay_count: usize,
}

impl Dataset {
    /// Per-fragment column-layout stats, in manifest fragment order.
    ///
    /// This is the planning input for horizontal compaction: it reads only
    /// fragment metadata (no data files), so it is cheap to call.
    pub fn column_layout_stats(&self) -> Vec<FragmentColumnLayoutStats> {
        let schema_ids: HashSet<i32> = self
            .schema()
            .fields_pre_order()
            .map(|field| field.id)
            .collect();
        self.manifest
            .fragments
            .iter()
            .map(|fragment| FragmentColumnLayoutStats {
                fragment_id: fragment.id,
                live_file_count: fragment
                    .files
                    .iter()
                    .filter(|file| file.fields.iter().any(|id| schema_ids.contains(id)))
                    .count(),
                overlay_count: fragment.overlays.len(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{NewColumnTransform, WriteParams};
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use std::sync::Arc;

    #[tokio::test]
    async fn column_layout_stats_counts_files_per_fragment() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..8))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        // Two fragments (max_rows_per_file = 4), one data file each to start.
        let mut dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let stats = dataset.column_layout_stats();
        assert_eq!(stats.len(), 2);
        assert!(
            stats
                .iter()
                .all(|s| s.live_file_count == 1 && s.overlay_count == 0)
        );

        // add_columns appends a second data file per fragment.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("b".into(), "a + 1".into())]),
                None,
                None,
            )
            .await
            .unwrap();

        let stats = dataset.column_layout_stats();
        assert_eq!(stats.len(), 2);
        assert!(
            stats.iter().all(|s| s.live_file_count == 2),
            "each fragment should now have 2 data files: {stats:?}"
        );
    }

    /// Only files holding a schema column count. The extra files are added to
    /// the manifest by hand: the commit paths that leave such files behind (an
    /// in-place column update after a drop, a full horizontal rewrite on a
    /// table whose lineage was spilled into a data file) are not what this
    /// test is about.
    #[tokio::test]
    async fn column_layout_stats_skips_files_without_schema_columns() {
        use lance_file::version::ConcreteFileVersion;
        use lance_table::format::{
            DataFile, ROW_ID_FIELD_ID, RowIdMeta, overlay::TOMBSTONE_FIELD_ID,
        };

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..4))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let dropped_id = dataset.schema().max_field_id().unwrap() + 1;
        let mut manifest = dataset.manifest.as_ref().clone();
        let mut fragments = manifest.fragments.as_ref().clone();
        let file = |fields: Vec<i32>| {
            let indices = (0..fields.len() as i32).collect();
            DataFile::new(
                "extra.lance",
                fields,
                indices,
                ConcreteFileVersion::V2_0,
                None,
                None,
            )
        };
        fragments[0].files.push(file(vec![TOMBSTONE_FIELD_ID]));
        fragments[0].files.push(file(vec![dropped_id]));
        fragments[0]
            .files
            .push(file(vec![TOMBSTONE_FIELD_ID, ROW_ID_FIELD_ID]));
        fragments[0].row_id_meta = Some(RowIdMeta::Column);
        manifest.fragments = Arc::new(fragments);
        dataset.manifest = Arc::new(manifest);

        let stats = dataset.column_layout_stats();
        assert_eq!(stats[0].live_file_count, 1, "{stats:?}");
    }
}
