// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-fragment column-layout statistics.
//!
//! A compaction planner keys on how many data files each fragment carries: a
//! fragment split across many small per-column files (typically from repeated
//! `add_columns` backfills) is the signal horizontal compaction targets.
//! Exposing these stats via [`Dataset::column_layout_stats`] keeps the planning
//! decision observable instead of buried in a black-box heuristic.

use lance_table::format::overlay::TOMBSTONE_FIELD_ID;

use super::Dataset;

/// Column-layout statistics for a single fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentColumnLayoutStats {
    /// The fragment these stats describe.
    pub fragment_id: u64,
    /// Number of data files still holding at least one live (non-tombstoned)
    /// field. A large value on a wide dataset is what horizontal compaction
    /// collapses back into fewer files.
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
        self.manifest
            .fragments
            .iter()
            .map(|fragment| FragmentColumnLayoutStats {
                fragment_id: fragment.id,
                live_file_count: fragment
                    .files
                    .iter()
                    .filter(|file| file.fields.iter().any(|id| *id != TOMBSTONE_FIELD_ID))
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
}
