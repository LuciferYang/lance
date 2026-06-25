// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use super::retry::{execute_with_retry, RetryConfig, RetryExecutor};
use super::{write_fragments_internal, CommitBuilder, WriteParams};
use crate::dataset::rowids::get_row_id_index;
use crate::dataset::transaction::UpdateMode::RewriteRows;
use crate::dataset::transaction::{Operation, Transaction};
use crate::dataset::utils::make_rowid_capture_stream;
use crate::{io::exec::Planner, Dataset};
use crate::{Error, Result};
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, DataType, Schema as ArrowSchema};
use datafusion::common::DFSchema;
use datafusion::error::{DataFusionError, Result as DFResult};
use datafusion::logical_expr::ExprSchemable;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::PhysicalExpr;
use datafusion::prelude::Expr;
use datafusion::scalar::ScalarValue;
use futures::StreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::error::{box_error, InvalidInputSnafu};
use lance_core::utils::mask::RowAddrTreeMap;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_datafusion::expr::safe_coerce_scalar;
use lance_table::format::{Fragment, RowIdMeta};
use roaring::RoaringTreemap;
use snafu::{location, ResultExt};

/// Build an update operation.
///
/// This operation is similar to SQL's UPDATE statement. It allows you to change
/// the values of all or a subset of columns with SQL expressions.
///
/// Use the [UpdateBuilder] to construct an update job. For example:
///
/// ```
/// # use lance::{Dataset, Result};
/// # use lance::dataset::UpdateBuilder;
/// # use std::sync::Arc;
/// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
/// let result = UpdateBuilder::new(dataset)
///     .update_where("region_id = 10")?
///     .set("region_name", "New York")?
///     .build()?
///     .execute()
///     .await?;
/// # Ok(())
/// # }
/// ```
///
#[derive(Debug, Clone)]
pub struct UpdateBuilder {
    /// The dataset snapshot to update.
    dataset: Arc<Dataset>,
    /// The condition to apply to find matching rows to update. If None, all rows are updated.
    condition: Option<Expr>,
    /// The updates to apply to matching rows.
    updates: HashMap<String, Expr>,
    /// Number of times to retry on commit conflicts.
    conflict_retries: u32,
    /// Total timeout for retries.
    retry_timeout: Duration,
}

impl UpdateBuilder {
    pub fn new(dataset: Arc<Dataset>) -> Self {
        Self {
            dataset,
            condition: None,
            updates: HashMap::new(),
            conflict_retries: 10,
            retry_timeout: Duration::from_secs(30),
        }
    }

    pub fn update_where(mut self, filter: &str) -> Result<Self> {
        let planner = Planner::new(Arc::new(self.dataset.schema().into()));
        let expr = planner
            .parse_filter(filter)
            .map_err(box_error)
            .context(InvalidInputSnafu {
                location: location!(),
            })?;
        self.condition = Some(planner.optimize_expr(expr).map_err(box_error).context(
            InvalidInputSnafu {
                location: location!(),
            },
        )?);
        Ok(self)
    }

    pub fn set(mut self, column: impl AsRef<str>, value: &str) -> Result<Self> {
        let field = self
            .dataset
            .schema()
            .field(column.as_ref())
            .ok_or_else(|| {
                Error::invalid_input(
                    format!(
                        "Column '{}' does not exist in dataset schema: {:?}",
                        column.as_ref(),
                        self.dataset.schema()
                    ),
                    location!(),
                )
            })?;

        // TODO: support nested column references. This is mostly blocked on the
        // ability to insert them into the RecordBatch properly.
        if column.as_ref().contains('.') {
            return Err(Error::NotSupported {
                source: format!(
                    "Nested column references are not yet supported. Referenced: {}",
                    column.as_ref(),
                )
                .into(),
                location: location!(),
            });
        }

        let schema: Arc<ArrowSchema> = Arc::new(self.dataset.schema().into());
        let planner = Planner::new(schema.clone());
        let mut expr = planner
            .parse_expr(value)
            .map_err(box_error)
            .context(InvalidInputSnafu {
                location: location!(),
            })?;

        // Cast expression to the column's data type if necessary.
        let dest_type = field.data_type();
        let df_schema = DFSchema::try_from(schema.as_ref().clone())?;
        let src_type = expr
            .get_type(&df_schema)
            .map_err(box_error)
            .context(InvalidInputSnafu {
                location: location!(),
            })?;
        if dest_type != src_type {
            expr = match expr {
                // TODO: remove this branch once DataFusion supports casting List to FSL
                // This should happen in Arrow 51.0.0
                Expr::Literal(value @ ScalarValue::List(_), metadata)
                    if matches!(dest_type, DataType::FixedSizeList(_, _)) =>
                {
                    Expr::Literal(
                        safe_coerce_scalar(&value, &dest_type).ok_or_else(|| {
                            ArrowError::CastError(format!(
                                "Failed to cast {} to {} during planning",
                                value.data_type(),
                                dest_type
                            ))
                        })?,
                        metadata,
                    )
                }
                _ => expr
                    .cast_to(&dest_type, &df_schema)
                    .map_err(box_error)
                    .context(InvalidInputSnafu {
                        location: location!(),
                    })?,
            };
        }

        // Optimize the expression. For example, this might apply the cast on
        // literals. (Expr.cast_to() only wraps the expression in a Cast node,
        // it doesn't actually apply the cast to the literals.)
        let expr = planner
            .optimize_expr(expr)
            .map_err(box_error)
            .context(InvalidInputSnafu {
                location: location!(),
            })?;

        self.updates.insert(column.as_ref().to_string(), expr);
        Ok(self)
    }

    /// Set the number of times to retry on commit conflicts.
    ///
    /// Default is 10.
    pub fn conflict_retries(mut self, retries: u32) -> Self {
        self.conflict_retries = retries;
        self
    }

    /// Set the total timeout for all retries.
    ///
    /// Default is 30 seconds.
    pub fn retry_timeout(mut self, timeout: Duration) -> Self {
        self.retry_timeout = timeout;
        self
    }

    // TODO: set write params
    // pub fn with_write_params(mut self, params: WriteParams) -> Self { ... }

    pub fn build(self) -> Result<UpdateJob> {
        let mut updates = HashMap::new();

        let planner = Planner::new(Arc::new(self.dataset.schema().into()));

        for (column, expr) in self.updates {
            let physical_expr = planner.create_physical_expr(&expr)?;
            updates.insert(column, physical_expr);
        }

        if updates.is_empty() {
            return Err(Error::invalid_input("No updates provided", location!()));
        }

        let updates = Arc::new(updates);

        Ok(UpdateJob {
            dataset: self.dataset,
            condition: self.condition,
            updates,
            conflict_retries: self.conflict_retries,
            retry_timeout: self.retry_timeout,
        })
    }
}

// TODO: support distributed operation.

#[derive(Debug, Clone)]
pub struct UpdateResult {
    pub new_dataset: Arc<Dataset>,
    pub rows_updated: u64,
}

#[derive(Debug)]
pub struct UpdateData {
    removed_fragment_ids: Vec<u64>,
    old_fragments: Vec<Fragment>,
    new_fragments: Vec<Fragment>,
    affected_rows: RowAddrTreeMap,
    num_updated_rows: u64,
}

#[derive(Debug, Clone)]
pub struct UpdateJob {
    dataset: Arc<Dataset>,
    condition: Option<Expr>,
    updates: Arc<HashMap<String, Arc<dyn PhysicalExpr>>>,
    conflict_retries: u32,
    retry_timeout: Duration,
}

impl UpdateJob {
    pub async fn execute(self) -> Result<UpdateResult> {
        let dataset = self.dataset.clone();
        let config = RetryConfig {
            max_retries: self.conflict_retries,
            retry_timeout: self.retry_timeout,
        };

        Box::pin(execute_with_retry(self, dataset, config)).await
    }

    async fn execute_impl(self) -> Result<UpdateData> {
        // Preserve structural absence for Model A columns (a mutable initial-default with no
        // write-default).  Such a column is read-backfilled at scan time by `DefaultReader`, so a
        // full-schema scan would materialise the *current* default into the rewritten data files,
        // freezing the value and breaking the mutable-default contract (a later `set_column_default`
        // would no longer affect the matched rows).  We therefore exclude any Model A column that is
        // physically absent from a fragment from that fragment's rewrite, so it stays structurally
        // absent and continues to be governed by the live read-time default.
        //
        // A column the update explicitly writes is never excluded (the update supplies new values).
        // A column merely *referenced* by an update SET expression or the condition is also never
        // excluded: those expressions are evaluated in `apply_updates` against the scanned batch, so
        // the referenced column must be projected into the scan (it is materialised at its live
        // default for these fragments, matching pre-feature behaviour) rather than being dropped,
        // which would make the expression fail with a column-not-found error.
        //
        // Because a single rewritten fragment cannot represent a column with *mixed* presence
        // (present in some source fragments, absent in others) without either freezing the absent
        // rows or dropping the explicit values, we group the matched fragments by their Model A
        // absent-set and rewrite each group independently — mirroring compaction's absent-set
        // binning (see `optimize::model_a_absent_field_ids`).  Within a group every Model A column
        // is uniformly present or absent, so a single write schema is valid.
        let mut referenced_columns: HashSet<String> = HashSet::new();
        for expr in self.updates.values() {
            for col in datafusion::physical_expr::utils::collect_columns(expr) {
                referenced_columns.insert(col.name().to_string());
            }
        }
        if let Some(condition) = &self.condition {
            for col in condition.column_refs() {
                referenced_columns.insert(col.name.clone());
            }
        }

        let model_a_fields: Vec<(i32, String)> = self
            .dataset
            .schema()
            .fields
            .iter()
            .filter(|f| {
                f.initial_default_raw().is_some()
                    && f.write_default_raw().is_none()
                    && !self.updates.contains_key(&f.name)
                    && !referenced_columns.contains(&f.name)
            })
            .map(|f| (f.id, f.name.clone()))
            .collect();

        // Bin the dataset fragments by the set of Model A columns physically absent from them.
        // The key is the sorted set of absent column names for that bin.
        let mut groups: HashMap<Vec<String>, Vec<Fragment>> = HashMap::new();
        for frag in self.dataset.get_fragments() {
            let present: HashSet<i32> = frag
                .metadata()
                .files
                .iter()
                .flat_map(|file| file.fields.iter().copied())
                .collect();
            let mut absent: Vec<String> = model_a_fields
                .iter()
                .filter(|(id, _)| !present.contains(id))
                .map(|(_, name)| name.clone())
                .collect();
            absent.sort();
            groups
                .entry(absent)
                .or_default()
                .push(frag.metadata().clone());
        }

        let version = self
            .dataset
            .manifest()
            .data_storage_format
            .lance_file_version()?;
        let stable_row_ids = self.dataset.manifest.uses_stable_row_ids();
        let row_id_index = get_row_id_index(&self.dataset).await?;

        // Accumulate the new fragments produced across all groups and the union of all removed
        // row addresses (used once at the end to apply deletions to the source fragments).
        let mut new_fragments: Vec<Fragment> = Vec::new();
        let mut all_removed_addrs = RoaringTreemap::new();

        for (absent_default_columns, group_fragments) in groups {
            let (group_new_fragments, removed_addrs) = self
                .rewrite_group(
                    group_fragments,
                    &absent_default_columns,
                    version,
                    stable_row_ids,
                    row_id_index.as_deref(),
                )
                .await?;
            new_fragments.extend(group_new_fragments);
            all_removed_addrs |= removed_addrs;
        }

        // Apply deletions for the union of all matched rows.
        let row_addrs = Arc::new(all_removed_addrs);
        let (old_fragments, removed_fragment_ids) = self.apply_deletions(&row_addrs).await?;
        let affected_rows = RowAddrTreeMap::from(row_addrs.as_ref().clone());

        let num_updated_rows = new_fragments
            .iter()
            .map(|f| f.physical_rows.unwrap() as u64)
            .sum::<u64>();

        Ok(UpdateData {
            removed_fragment_ids,
            old_fragments,
            new_fragments,
            affected_rows,
            num_updated_rows,
        })
    }

    /// Rewrite the matched rows of a single absent-set group into new fragments.
    ///
    /// `absent_default_columns` are the Model A columns (sorted by name) that are structurally
    /// absent from *every* fragment in this group; they are projected out of both the scan and the
    /// write schema so they remain structurally absent (and read-backfilled by the live default) in
    /// the rewritten fragments.  Returns the new fragments and the set of source row addresses that
    /// were rewritten (to be deleted from the source fragments).
    async fn rewrite_group(
        &self,
        group_fragments: Vec<Fragment>,
        absent_default_columns: &[String],
        version: lance_file::version::LanceFileVersion,
        stable_row_ids: bool,
        row_id_index: Option<&lance_table::rowids::RowIdIndex>,
    ) -> Result<(Vec<Fragment>, RoaringTreemap)> {
        let mut scanner = self.dataset.scan();
        scanner.with_row_id();
        scanner.with_fragments(group_fragments);

        if let Some(expr) = &self.condition {
            scanner.filter_expr(expr.clone());
        }

        let absent_set: HashSet<&str> = absent_default_columns.iter().map(|s| s.as_str()).collect();
        let write_schema = if absent_set.is_empty() {
            self.dataset.schema().clone()
        } else {
            let kept_columns: Vec<&str> = self
                .dataset
                .schema()
                .fields
                .iter()
                .map(|f| f.name.as_str())
                .filter(|name| !absent_set.contains(name))
                .collect();
            scanner.project(&kept_columns)?;
            self.dataset.schema().project(&kept_columns)?
        };

        let stream = scanner.try_into_stream().await?.into();

        // We keep track of seen row ids so we can delete them from the existing
        // fragments and then set the row id segments in the new fragments.
        let (stream, row_id_rx) = make_rowid_capture_stream(stream, stable_row_ids)?;

        let schema = stream.schema();

        let expected_schema = (&write_schema).into();
        if schema.as_ref() != &expected_schema {
            return Err(Error::Internal {
                message: format!("Expected schema {:?} but got {:?}", expected_schema, schema),
                location: location!(),
            });
        }

        let updates_ref = self.updates.clone();
        let stream = stream
            .map(move |batch| {
                let updates = updates_ref.clone();
                tokio::task::spawn_blocking(move || Self::apply_updates(batch?, updates))
            })
            .buffered(get_num_compute_intensive_cpus())
            .map(|res| match res {
                Ok(Ok(batch)) => Ok(batch),
                Ok(Err(err)) => Err(err),
                Err(e) => Err(DataFusionError::ExecutionJoin(Box::new(e))),
            });
        let stream = RecordBatchStreamAdapter::new(schema, stream);

        let (mut new_fragments, _) = write_fragments_internal(
            Some(&self.dataset),
            self.dataset.object_store.clone(),
            &self.dataset.base,
            write_schema,
            Box::pin(stream),
            WriteParams::with_storage_version(version),
            None, // TODO: support multiple bases for update
        )
        .await?;

        let removed_row_ids = row_id_rx.try_recv().map_err(|err| Error::Internal {
            message: format!("Failed to receive row ids: {}", err),
            location: location!(),
        })?;

        if let Some(row_id_sequence) = removed_row_ids.row_id_sequence() {
            let fragment_sizes = new_fragments
                .iter()
                .map(|f| f.physical_rows.unwrap() as u64);
            let sequences = lance_table::rowids::rechunk_sequences(
                [row_id_sequence.clone()],
                fragment_sizes,
                false,
            )
            .map_err(|e| Error::Internal {
                message: format!(
                    "Captured row ids not equal to number of rows written: {}",
                    e
                ),
                location: location!(),
            })?;
            for (fragment, sequence) in new_fragments.iter_mut().zip(sequences) {
                let serialized = lance_table::rowids::write_row_ids(&sequence);
                fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
            }
        }

        let removed_addrs = removed_row_ids.row_addrs(row_id_index).into_owned();
        Ok((new_fragments, removed_addrs))
    }

    async fn commit_impl(
        &self,
        dataset: Arc<Dataset>,
        update_data: UpdateData,
    ) -> Result<UpdateResult> {
        let mut fields_for_preserving_frag_bitmap = Vec::new();
        for column_name in self.updates.keys() {
            if let Ok(field_id) = dataset.schema().field_id(column_name) {
                fields_for_preserving_frag_bitmap.push(field_id as u32);
            }
        }

        // Commit updated and new fragments
        let operation = Operation::Update {
            removed_fragment_ids: update_data.removed_fragment_ids,
            updated_fragments: update_data.old_fragments,
            new_fragments: update_data.new_fragments,
            // In "rewrite rows" mode, the rows that are updated in the fragment
            // are moved(deleted and appended).
            // so we do not need to handle the frag bitmap of the index about it.
            fields_modified: vec![],
            merged_generations: Vec::new(),
            fields_for_preserving_frag_bitmap,
            update_mode: Some(RewriteRows),
            inserted_rows_filter: None,
        };

        let transaction = Transaction::new(dataset.manifest.version, operation, None);

        let new_dataset = CommitBuilder::new(dataset)
            .with_affected_rows(update_data.affected_rows)
            .execute(transaction)
            .await?;

        Ok(UpdateResult {
            new_dataset: Arc::new(new_dataset),
            rows_updated: update_data.num_updated_rows,
        })
    }

    fn apply_updates(
        mut batch: RecordBatch,
        updates: Arc<HashMap<String, Arc<dyn PhysicalExpr>>>,
    ) -> DFResult<RecordBatch> {
        for (column, expr) in updates.iter() {
            let new_values = expr.evaluate(&batch)?.into_array(batch.num_rows())?;
            batch = batch.replace_column_by_name(column.as_str(), new_values)?;
        }
        Ok(batch)
    }

    /// Use previous found rows ids to delete rows from existing fragments.
    ///
    /// Returns the set of modified fragments and removed fragments, if any.
    async fn apply_deletions(
        &self,
        removed_row_addrs: &RoaringTreemap,
    ) -> Result<(Vec<Fragment>, Vec<u64>)> {
        let bitmaps = Arc::new(removed_row_addrs.bitmaps().collect::<BTreeMap<_, _>>());

        enum FragmentChange {
            Unchanged,
            Modified(Box<Fragment>),
            Removed(u64),
        }

        let mut updated_fragments = Vec::new();
        let mut removed_fragments = Vec::new();

        let mut stream = futures::stream::iter(self.dataset.get_fragments())
            .map(move |fragment| {
                let bitmaps_ref = bitmaps.clone();
                async move {
                    let fragment_id = fragment.id();
                    if let Some(bitmap) = bitmaps_ref.get(&(fragment_id as u32)) {
                        match fragment.extend_deletions(*bitmap).await {
                            Ok(Some(new_fragment)) => {
                                Ok(FragmentChange::Modified(Box::new(new_fragment.metadata)))
                            }
                            Ok(None) => Ok(FragmentChange::Removed(fragment_id as u64)),
                            Err(e) => Err(e),
                        }
                    } else {
                        Ok(FragmentChange::Unchanged)
                    }
                }
            })
            .buffer_unordered(self.dataset.object_store.io_parallelism());

        while let Some(res) = stream.next().await.transpose()? {
            match res {
                FragmentChange::Unchanged => {}
                FragmentChange::Modified(fragment) => updated_fragments.push(*fragment),
                FragmentChange::Removed(fragment_id) => removed_fragments.push(fragment_id),
            }
        }

        Ok((updated_fragments, removed_fragments))
    }
}

impl RetryExecutor for UpdateJob {
    type Data = UpdateData;
    type Result = UpdateResult;

    async fn execute_impl(&self) -> Result<Self::Data> {
        self.clone().execute_impl().await
    }

    async fn commit(&self, dataset: Arc<Dataset>, data: Self::Data) -> Result<Self::Result> {
        self.commit_impl(dataset, data).await
    }

    fn update_dataset(&mut self, dataset: Arc<Dataset>) {
        self.dataset = dataset;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::{
        dataset::{builder::DatasetBuilder, InsertBuilder, ReadParams, WriteParams},
        session::Session,
        utils::test::ThrottledStoreWrapper,
    };

    use super::*;

    use crate::dataset::{WriteDestination, WriteMode};
    use crate::index::vector::VectorIndexParams;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow::{array::AsArray, datatypes::UInt32Type};
    use arrow_array::types::Float32Type;
    use arrow_array::{Int64Array, RecordBatchIterator, StringArray, UInt32Array, UInt64Array};
    use arrow_schema::{Field, Schema as ArrowSchema};
    use arrow_select::concat::concat_batches;
    use futures::{future::try_join_all, TryStreamExt};
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::ROW_ID;
    use lance_datagen::{Dimension, RowCount};
    use lance_file::version::LanceFileVersion;
    use lance_index::scalar::ScalarIndexParams;
    use lance_index::DatasetIndexExt;
    use lance_index::IndexType;
    use lance_io::object_store::ObjectStoreParams;
    use lance_linalg::distance::MetricType;
    use object_store::throttle::ThrottleConfig;
    use rstest::rstest;
    use tokio::sync::Barrier;

    /// Returns a dataset with 3 fragments, each with 10 rows.
    ///
    /// Also returns the TempDir, which should be kept alive as long as the
    /// dataset is being accessed. Once that is dropped, the temp directory is
    /// deleted.
    async fn make_test_dataset(
        version: LanceFileVersion,
        enable_stable_row_ids: bool,
    ) -> (Arc<Dataset>, TempStrDir) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..30)),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "foo", 30,
                ))),
            ],
        )
        .unwrap();

        let write_params = WriteParams {
            max_rows_per_file: 10,
            data_storage_version: Some(version),
            enable_stable_row_ids,
            ..Default::default()
        };

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let batches = RecordBatchIterator::new([Ok(batch)], schema.clone());
        let ds = Dataset::write(batches, test_uri, Some(write_params))
            .await
            .unwrap();

        (Arc::new(ds), test_dir)
    }

    #[tokio::test]
    async fn test_update_validation() {
        let (dataset, _test_dir) = make_test_dataset(LanceFileVersion::Legacy, false).await;

        let builder = UpdateBuilder::new(dataset);

        assert!(
            matches!(
                builder.clone().update_where("foo = 10"),
                Err(Error::InvalidInput { .. })
            ),
            "Should return error if condition references non-existent column"
        );

        assert!(
            matches!(
                builder.clone().set("foo", "1"),
                Err(Error::InvalidInput { .. })
            ),
            "Should return error if update key references non-existent column"
        );

        assert!(
            matches!(
                builder.clone().set("id", "id2 + 1"),
                Err(Error::InvalidInput { .. })
            ),
            "Should return error if update expression references non-existent column"
        );

        assert!(
            matches!(builder.build(), Err(Error::InvalidInput { .. })),
            "Should return error if no update expressions are provided"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn test_update_all(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let (dataset, _test_dir) = make_test_dataset(version, enable_stable_row_ids).await;

        let update_result = UpdateBuilder::new(dataset)
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = update_result.new_dataset;
        let actual_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual_batch = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();

        let expected = RecordBatch::try_new(
            Arc::new(dataset.schema().into()),
            vec![
                Arc::new(Int64Array::from_iter_values(0..30)),
                Arc::new(StringArray::from_iter_values(
                    (0..30).map(|i| format!("bar{}", i)),
                )),
            ],
        )
        .unwrap();

        assert_eq!(actual_batch, expected);

        assert_eq!(dataset.get_fragments().len(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_update_conditional(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::V2_0)] version: LanceFileVersion,
        #[values(false, true)] enable_stable_row_ids: bool,
    ) {
        let (dataset, _test_dir) = make_test_dataset(version, enable_stable_row_ids).await;

        let original_fragments = dataset.get_fragments();

        let update_result = UpdateBuilder::new(dataset)
            .update_where("id >= 15")
            .unwrap()
            .set("name", "'bar' || cast(id as string)")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();

        let dataset = update_result.new_dataset;
        let actual_batches = dataset
            .scan()
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let actual_batch = concat_batches(&actual_batches[0].schema(), &actual_batches).unwrap();

        let expected = RecordBatch::try_new(
            Arc::new(dataset.schema().into()),
            vec![
                Arc::new(Int64Array::from_iter_values(0..30)),
                Arc::new(StringArray::from_iter_values(
                    (0..15)
                        .map(|_| "foo".to_string())
                        .chain((15..30).map(|i| format!("bar{}", i))),
                )),
            ],
        )
        .unwrap();

        assert_eq!(actual_batch, expected);

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 3);

        // One fragment not touched (id = 0..10)
        assert_eq!(fragments[0].metadata.id, original_fragments[0].metadata.id);
        assert_eq!(
            fragments[0].metadata.files,
            original_fragments[0].metadata.files
        );
        assert_eq!(
            fragments[0].metadata.physical_rows,
            original_fragments[0].metadata.physical_rows
        );
        assert_eq!(
            fragments[0].metadata.row_id_meta,
            original_fragments[0].metadata.row_id_meta
        );
        // One fragment partially modified (id = 10..15)
        assert_eq!(
            fragments[1].metadata.files,
            original_fragments[1].metadata.files,
        );
        assert_eq!(
            fragments[1]
                .metadata
                .deletion_file
                .as_ref()
                .and_then(|f| f.num_deleted_rows),
            Some(5)
        );
        // One fragment fully modified
        assert_eq!(fragments[2].metadata.physical_rows, Some(15));
    }

    #[rstest]
    #[tokio::test]
    async fn test_update_concurrency(#[values(false, true)] enable_stable_row_ids: bool) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("value", DataType::UInt32, false),
        ]));
        let concurrency = 3;
        let initial_data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(0..concurrency)),
                Arc::new(UInt32Array::from_iter_values(std::iter::repeat_n(
                    0,
                    concurrency as usize,
                ))),
            ],
        )
        .unwrap();

        // Increase likelihood of contention by throttling the store
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_call: Duration::from_millis(1),
                wait_get_per_call: Duration::from_millis(1),
                ..Default::default()
            },
        });
        let session = Arc::new(Session::default());

        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(throttled.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                enable_stable_row_ids,
                ..Default::default()
            })
            .execute(vec![initial_data])
            .await
            .unwrap();

        let barrier = Arc::new(Barrier::new(concurrency as usize));
        let mut handles = Vec::new();
        for i in 0..concurrency {
            let session_ref = session.clone();
            let barrier_ref = barrier.clone();
            let throttled_ref = throttled.clone();
            let handle = tokio::task::spawn(async move {
                let dataset = DatasetBuilder::from_uri("memory://")
                    .with_read_params(ReadParams {
                        store_options: Some(ObjectStoreParams {
                            object_store_wrapper: Some(throttled_ref.clone()),
                            ..Default::default()
                        }),
                        session: Some(session_ref.clone()),
                        ..Default::default()
                    })
                    .load()
                    .await
                    .unwrap();

                let job = UpdateBuilder::new(Arc::new(dataset))
                    .update_where(&format!("id = {}", i))
                    .unwrap()
                    .set("value", "1")
                    .unwrap()
                    .build()
                    .unwrap();
                barrier_ref.wait().await;

                job.execute().await.unwrap();
            });
            handles.push(handle);
        }

        try_join_all(handles).await.unwrap();

        dataset.checkout_latest().await.unwrap();

        let data = dataset.scan().try_into_batch().await.unwrap();

        let mut ids = data["id"]
            .as_primitive::<UInt32Type>()
            .values()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        ids.sort();
        assert_eq!(ids, vec![0, 1, 2],);
        let values = data["value"].as_primitive::<UInt32Type>().values();
        assert!(values.iter().all(|&value| value == 1));
    }

    #[rstest]
    #[tokio::test]
    async fn test_update_same_row_concurrency(#[values(false, true)] enable_stable_row_ids: bool) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::UInt32, false),
            Field::new("value", DataType::UInt32, false),
        ]));
        let concurrency = 3;
        // Create dataset with just one row that all workers will update
        let initial_data = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt32Array::from(vec![0])),
                Arc::new(UInt32Array::from(vec![10])),
            ],
        )
        .unwrap();

        // Increase likelihood of contention by throttling the store
        let throttled = Arc::new(ThrottledStoreWrapper {
            config: ThrottleConfig {
                wait_list_per_call: Duration::from_millis(10),
                wait_get_per_call: Duration::from_millis(10),
                ..Default::default()
            },
        });
        let session = Arc::new(Session::default());

        let mut dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                store_params: Some(ObjectStoreParams {
                    object_store_wrapper: Some(throttled.clone()),
                    ..Default::default()
                }),
                session: Some(session.clone()),
                enable_stable_row_ids,
                ..Default::default()
            })
            .execute(vec![initial_data])
            .await
            .unwrap();

        let barrier = Arc::new(Barrier::new(concurrency as usize));
        let mut handles = Vec::new();
        for _i in 0..concurrency {
            let session_ref = session.clone();
            let barrier_ref = barrier.clone();
            let throttled_ref = throttled.clone();
            let handle = tokio::task::spawn(async move {
                let dataset = DatasetBuilder::from_uri("memory://")
                    .with_read_params(ReadParams {
                        store_options: Some(ObjectStoreParams {
                            object_store_wrapper: Some(throttled_ref.clone()),
                            ..Default::default()
                        }),
                        session: Some(session_ref.clone()),
                        ..Default::default()
                    })
                    .load()
                    .await
                    .unwrap();

                let job = UpdateBuilder::new(Arc::new(dataset))
                    .update_where("id = 0")
                    .unwrap()
                    .set("value", "99")
                    .unwrap()
                    .build()
                    .unwrap();
                barrier_ref.wait().await;

                job.execute().await.unwrap();
            });
            handles.push(handle);
        }

        try_join_all(handles).await.unwrap();

        dataset.checkout_latest().await.unwrap();

        let data = dataset.scan().try_into_batch().await.unwrap();

        // With retry-based conflict resolution, all concurrent updates should succeed
        // Even though they all target the same row, they should not fail with commit conflicts
        // The final result should be exactly one row (not duplicated) because the retries
        // should work from the latest dataset state, preventing duplicate row creation
        let ids = data["id"].as_primitive::<UInt32Type>().values();
        assert_eq!(ids, &[0]);

        let values = data["value"].as_primitive::<UInt32Type>().values();
        assert_eq!(values, &[99]);
    }

    #[tokio::test]
    async fn test_row_ids_stable_after_update() {
        let (dataset, _test_dir) = make_test_dataset(LanceFileVersion::V2_0, true).await;

        let orig_batch = dataset.scan().with_row_id().try_into_batch().await.unwrap();
        let orig_row_ids = orig_batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let orig_ids = orig_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        let updated_batch = UpdateBuilder::new(dataset)
            .update_where("id >= 15")
            .unwrap()
            .set("name", "'updated'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset
            .scan()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        let updated_row_ids = updated_batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let updated_ids = updated_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        assert_eq!(orig_row_ids, updated_row_ids);
        assert_eq!(orig_ids, updated_ids);
    }

    #[tokio::test]
    async fn test_row_ids_stable_after_update_odd_id() {
        use std::collections::HashSet;

        let (dataset, _test_dir) = make_test_dataset(LanceFileVersion::V2_0, true).await;

        let orig_batch = dataset.scan().with_row_id().try_into_batch().await.unwrap();
        let orig_row_ids = orig_batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let orig_ids = orig_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let orig_names = orig_batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        let updated_batch = UpdateBuilder::new(dataset)
            .update_where("id % 2 = 1")
            .unwrap()
            .set("name", "'updated'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset
            .scan()
            .with_row_id()
            .try_into_batch()
            .await
            .unwrap();

        let updated_row_ids = updated_batch
            .column_by_name(ROW_ID)
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        let updated_ids = updated_batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let updated_names = updated_batch
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!(
            orig_row_ids
                .values()
                .iter()
                .cloned()
                .collect::<HashSet<_>>(),
            updated_row_ids
                .values()
                .iter()
                .cloned()
                .collect::<HashSet<_>>()
        );
        assert_eq!(
            orig_ids.values().iter().cloned().collect::<HashSet<_>>(),
            updated_ids.values().iter().cloned().collect::<HashSet<_>>()
        );

        for i in 0..orig_row_ids.len() {
            let row_id = orig_row_ids.value(i);
            let updated_idx = updated_row_ids
                .iter()
                .position(|rid| rid == Some(row_id))
                .unwrap();
            let id = orig_ids.value(i);
            let updated_name = updated_names.value(updated_idx);
            if id % 2 == 1 {
                assert_eq!(updated_name, "updated");
            } else {
                assert_eq!(updated_name, orig_names.value(i));
            }
        }
    }

    #[tokio::test]
    async fn test_update_affects_index_fragment_bitmap() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "str",
                lance_datagen::array::cycle_utf8_literals(&["a", "b", "c", "d", "e", "f"]),
            )
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(4)),
            )
            .into_ram_dataset_with_params(
                FragmentCount::from(2),
                FragmentRowCount::from(3),
                Some(WriteParams {
                    max_rows_per_file: 3,
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        let scalar_params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["str"],
                IndexType::Scalar,
                Some("str_idx".to_string()),
                &scalar_params,
                true,
            )
            .await
            .unwrap();

        let vector_params = VectorIndexParams::ivf_flat(1, MetricType::L2);
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("vec_idx".to_string()),
                &vector_params,
                true,
            )
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        let str_index = indices.iter().find(|idx| idx.name == "str_idx").unwrap();
        let vec_index = indices.iter().find(|idx| idx.name == "vec_idx").unwrap();

        assert_eq!(
            str_index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            vec_index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        let updated_dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("str = 'e'")
            .unwrap()
            .set("vec", "array[25.0, 26.0, 27.0, 28.0]")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;

        let updated_indices = updated_dataset.load_indices().await.unwrap();
        let updated_str_index = updated_indices
            .iter()
            .find(|idx| idx.name == "str_idx")
            .unwrap();
        let updated_vec_index = updated_indices
            .iter()
            .find(|idx| idx.name == "vec_idx")
            .unwrap();

        let str_bitmap = updated_str_index.fragment_bitmap.as_ref().unwrap();
        assert_eq!(str_bitmap.len(), 3);
        assert_eq!(str_bitmap.iter().collect::<Vec<_>>(), vec![0, 1, 2]);

        let vec_bitmap = updated_vec_index.fragment_bitmap.as_ref().unwrap();
        assert_eq!(vec_bitmap.len(), 2);
        assert_eq!(vec_bitmap.iter().collect::<Vec<_>>(), vec![0, 1]);

        let fragments = updated_dataset.get_fragments();
        assert!(fragments.len() > 2);

        let second_fragment = &fragments[1];
        assert!(second_fragment
            .get_deletion_vector()
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn test_update_mixed_indexed_unindexed_fragments() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "str",
                lance_datagen::array::cycle_utf8_literals(&["a", "b", "c", "d", "e", "f"]),
            )
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(4)),
            )
            .into_ram_dataset_with_params(
                FragmentCount::from(2),
                FragmentRowCount::from(3),
                Some(WriteParams {
                    max_rows_per_file: 3,
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();

        dataset
            .create_index(
                &["str"],
                IndexType::Scalar,
                Some("str_idx".to_string()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("vec_idx".to_string()),
                &VectorIndexParams::ivf_flat(1, MetricType::L2),
                true,
            )
            .await
            .unwrap();

        let initial_indices = dataset.load_indices().await.unwrap();
        let str_index = initial_indices
            .iter()
            .find(|idx| idx.name == "str_idx")
            .unwrap();
        let vec_index = initial_indices
            .iter()
            .find(|idx| idx.name == "vec_idx")
            .unwrap();

        assert_eq!(
            str_index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            vec_index
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        // insert data to create the third frag
        let new_batch = lance_datagen::gen_batch()
            .col(
                "str",
                lance_datagen::array::cycle_utf8_literals(&["g", "h", "i"]),
            )
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(4)),
            )
            .into_batch_rows(RowCount::from(3))
            .unwrap();

        dataset = InsertBuilder::new(WriteDestination::Dataset(Arc::new(dataset)))
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                enable_stable_row_ids: true,
                ..Default::default()
            })
            .execute(vec![new_batch])
            .await
            .unwrap();

        assert_eq!(dataset.get_fragments().len(), 3);

        let indices_after_insert = dataset.load_indices().await.unwrap();
        let str_index_after_insert = indices_after_insert
            .iter()
            .find(|idx| idx.name == "str_idx")
            .unwrap();
        let vec_index_after_insert = indices_after_insert
            .iter()
            .find(|idx| idx.name == "vec_idx")
            .unwrap();

        assert_eq!(
            str_index_after_insert
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert!(!str_index_after_insert
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .contains(2));
        assert_eq!(
            vec_index_after_insert
                .fragment_bitmap
                .as_ref()
                .unwrap()
                .len(),
            2
        );
        assert!(!vec_index_after_insert
            .fragment_bitmap
            .as_ref()
            .unwrap()
            .contains(2));

        let updated_dataset = UpdateBuilder::new(Arc::new(dataset))
            // 'a' in fragment 0，'g' in fragment 2, and frag 2 not in frag bitmap
            .update_where("str = 'a' OR str = 'g'")
            .unwrap()
            .set("vec", "array[99.0, 99.0, 99.0, 99.0]")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;

        // reload indices
        let updated_indices = updated_dataset.load_indices().await.unwrap();
        let updated_str_index = updated_indices
            .iter()
            .find(|idx| idx.name == "str_idx")
            .unwrap();
        let updated_vec_index = updated_indices
            .iter()
            .find(|idx| idx.name == "vec_idx")
            .unwrap();

        let str_bitmap = updated_str_index.fragment_bitmap.as_ref().unwrap();
        let vec_bitmap = updated_vec_index.fragment_bitmap.as_ref().unwrap();

        assert!(updated_dataset.get_fragments().len() > 3);
        assert_eq!(str_bitmap.len(), 2);
        assert_eq!(vec_bitmap.len(), 2);

        // frag 3 not in the index's frag bitmap
        for &fragment_id in str_bitmap.iter().collect::<Vec<_>>().iter() {
            assert!(fragment_id < 2,
                    "str index bitmap should not contain fragments with unindexed data, found fragment {}",
                    fragment_id);
        }

        // frag 3 not in the index's frag bitmap
        for &fragment_id in vec_bitmap.iter().collect::<Vec<_>>().iter() {
            assert!(fragment_id < 2,
                    "vec index bitmap should not contain fragments with unindexed data, found fragment {}",
                    fragment_id);
        }
    }

    /// Regression: updating one column must NOT freeze the live initial-default of a *different*
    /// Model A column (a mutable initial-default with no write-default) for the rewritten rows.
    /// Such a column is structurally absent and read-backfilled at scan time; the update rewrite
    /// path must keep it structurally absent so a later `set_column_default` still updates the
    /// matched rows retroactively.
    #[tokio::test]
    async fn test_update_preserves_mutable_initial_default_for_absent_column() {
        use crate::dataset::schema_evolution::NewColumnTransform;
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int32Type;
        use arrow_array::Array;
        use lance_core::datatypes::LANCE_INITIAL_DEFAULT_META_KEY;

        // Single fragment with an updatable column `id` only, v2 storage.
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..4))],
        )
        .unwrap();
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let batches = RecordBatchIterator::new([Ok(batch)], arrow_schema.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Add Model A column `c` (initial-default only, no write-default), absent in all rows.
        let c_field =
            Field::new("c", DataType::Int32, true).with_metadata(std::collections::HashMap::from(
                [(LANCE_INITIAL_DEFAULT_META_KEY.to_string(), "42".to_string())],
            ));
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![c_field]))),
                None,
                None,
            )
            .await
            .unwrap();
        dataset.checkout_latest().await.unwrap();

        // Update a DIFFERENT column (`id`) on a subset of rows, leaving `c` untouched.
        let result = UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id < 2")
            .unwrap()
            .set("id", "id + 100")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = Arc::try_unwrap(result.new_dataset).unwrap_or_else(|ds| (*ds).clone());

        // Changing the default must retroactively affect ALL rows, including the rewritten ones,
        // because `c` stayed structurally absent.
        dataset.set_column_default("c", "77").await.unwrap();
        dataset.checkout_latest().await.unwrap();

        let batch = dataset
            .scan()
            .project(&["id", "c"])
            .unwrap()
            .scan_in_order(true)
            .try_into_batch()
            .await
            .unwrap();
        let c = batch
            .column_by_name("c")
            .unwrap()
            .as_primitive::<Int32Type>();
        assert_eq!(c.null_count(), 0, "update must not introduce nulls in `c`");
        assert!(
            (0..c.len()).all(|i| c.value(i) == 77),
            "all rows (including rewritten ones) must reflect the new default 77, got {c:?}"
        );
    }

    /// Regression: an Update whose SET expression *references* a structurally-absent Model A column
    /// (added later with an initial-default only) must succeed.  The referenced column is projected
    /// out of the scan only when it is neither a target nor a reference; here it is a reference, so
    /// it must be kept in the scan projection and materialised at its live default so the expression
    /// can evaluate.  Before the fix this failed with a column-not-found error.
    #[tokio::test]
    async fn test_update_set_expr_references_absent_model_a_column() {
        use crate::dataset::schema_evolution::NewColumnTransform;
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int64Type;
        use lance_core::datatypes::LANCE_INITIAL_DEFAULT_META_KEY;

        // Single fragment with `id` and `total`, v2 storage.
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("total", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..4)),
                Arc::new(Int64Array::from(vec![0_i64; 4])),
            ],
        )
        .unwrap();
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;
        let batches = RecordBatchIterator::new([Ok(batch)], arrow_schema.clone());
        let mut dataset = Dataset::write(
            batches,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Add Model A column `base` (initial-default 10, no write-default), structurally absent.
        let base_field = Field::new("base", DataType::Int64, true).with_metadata(
            std::collections::HashMap::from([(
                LANCE_INITIAL_DEFAULT_META_KEY.to_string(),
                "10".to_string(),
            )]),
        );
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![base_field]))),
                None,
                None,
            )
            .await
            .unwrap();
        dataset.checkout_latest().await.unwrap();

        // UPDATE SET total = base + 1, where `base` is the absent Model A column. This must not
        // error: `base` is read-backfilled at its live default (10) for the matched fragment.
        let result = UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("id < 2")
            .unwrap()
            .set("total", "base + 1")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = Arc::try_unwrap(result.new_dataset).unwrap_or_else(|ds| (*ds).clone());

        let batch = dataset
            .scan()
            .project(&["id", "total"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let id = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int64Type>();
        let total = batch
            .column_by_name("total")
            .unwrap()
            .as_primitive::<Int64Type>();
        // Build an id -> total map so the assertion is independent of physical row order
        // (matched rows are rewritten and appended).
        let mut by_id: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
        for i in 0..batch.num_rows() {
            by_id.insert(id.value(i), total.value(i));
        }
        // Matched rows (id 0,1) become base(10) + 1 = 11; untouched rows (id 2,3) stay 0.
        assert_eq!(by_id.get(&0), Some(&11), "row id=0 must be base(10)+1=11");
        assert_eq!(by_id.get(&1), Some(&11), "row id=1 must be base(10)+1=11");
        assert_eq!(by_id.get(&2), Some(&0), "row id=2 untouched");
        assert_eq!(by_id.get(&3), Some(&0), "row id=3 untouched");
    }

    /// Regression: an Update that rewrites rows must NOT freeze the live initial-default of a
    /// Model A column with MIXED physical presence across fragments — absent in some source
    /// fragments (added via `add_columns(AllNulls)`) and present in others (an append that
    /// explicitly carried the column).  The originally-absent rows that get rewritten must stay
    /// structurally absent so a later `set_column_default` still updates them, while
    /// explicitly-written values are preserved.  Mirrors
    /// `optimize::test_compaction_preserves_mutable_initial_default_mixed_presence`.
    #[tokio::test]
    async fn test_update_preserves_mutable_initial_default_mixed_presence() {
        use crate::dataset::schema_evolution::NewColumnTransform;
        use arrow_array::cast::AsArray;
        use arrow_array::types::Int32Type;
        use arrow_array::{Array, Int32Array};
        use lance_core::datatypes::LANCE_INITIAL_DEFAULT_META_KEY;

        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Old fragment of column `a` only (will be absent in `c`).
        let schema_a = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            true,
        )]));
        let batch_a = RecordBatch::try_new(
            schema_a.clone(),
            vec![Arc::new(Int32Array::from(vec![1_i32, 2]))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch_a)], schema_a.clone()),
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 1024,
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Add Model A column `c` (initial-default only, no write-default).  Absent in the old
        // fragment (metadata-only AllNulls).
        let c_field =
            Field::new("c", DataType::Int32, true).with_metadata(std::collections::HashMap::from(
                [(LANCE_INITIAL_DEFAULT_META_KEY.to_string(), "42".to_string())],
            ));
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![c_field]))),
                None,
                None,
            )
            .await
            .unwrap();

        // Append a batch that explicitly includes `c` → present in the new fragment, absent in the
        // old one (mixed presence).
        let schema_ac = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("c", DataType::Int32, true),
        ]));
        let batch_ac = RecordBatch::try_new(
            schema_ac.clone(),
            vec![
                Arc::new(Int32Array::from(vec![3_i32, 4])),
                Arc::new(Int32Array::from(vec![100_i32, 200])),
            ],
        )
        .unwrap();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch_ac)], schema_ac),
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 1024,
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.checkout_latest().await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);

        // Update `a` for rows spanning BOTH fragments (a=1 from the absent fragment, a=3 from the
        // present fragment), leaving `c` untouched.  This rewrites matched rows from a
        // mixed-presence set; the absent-fragment row must remain structurally absent in `c`.
        let result = UpdateBuilder::new(Arc::new(dataset.clone()))
            .update_where("a = 1 OR a = 3")
            .unwrap()
            .set("a", "a + 1000")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        dataset = Arc::try_unwrap(result.new_dataset).unwrap_or_else(|ds| (*ds).clone());

        // Change the default → originally-absent rows (a was 1 and 2) must reflect it; explicitly
        // written `c` values (rows where a was 3 and 4) must be preserved, not frozen-to-default.
        dataset.set_column_default("c", "77").await.unwrap();
        dataset.checkout_latest().await.unwrap();

        let batch = dataset
            .scan()
            .project(&["a", "c"])
            .unwrap()
            .scan_in_order(true)
            .try_into_batch()
            .await
            .unwrap();
        let a = batch
            .column_by_name("a")
            .unwrap()
            .as_primitive::<Int32Type>();
        let c = batch
            .column_by_name("c")
            .unwrap()
            .as_primitive::<Int32Type>();
        assert_eq!(c.null_count(), 0, "update must not introduce nulls in `c`");
        for i in 0..a.len() {
            let expected = match a.value(i) {
                1001 | 2 => 77, // originally absent (a was 1, 2) → follows the new live default
                1003 => 100,    // explicitly written → preserved
                4 => 200,       // explicitly written → preserved
                other => panic!("unexpected a value {other}"),
            };
            assert_eq!(
                c.value(i),
                expected,
                "row a={} must read c={expected}, got {}",
                a.value(i),
                c.value(i)
            );
        }
    }
}
