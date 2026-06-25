// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchIterator};
use datafusion::execution::SendableRecordBatchStream;
use humantime::format_duration;
use lance_core::datatypes::{NullabilityComparison, Schema, SchemaCompareOptions};
use lance_core::utils::tracing::{DATASET_WRITING_EVENT, TRACE_DATASET_EVENTS};
use lance_core::{ROW_ADDR, ROW_ID, ROW_OFFSET};
use lance_datafusion::utils::StreamingWriteSource;
use lance_file::version::LanceFileVersion;
use lance_io::object_store::ObjectStore;
use lance_table::feature_flags::can_write_dataset;
use lance_table::format::Fragment;
use lance_table::io::commit::CommitHandler;
use object_store::path::Path;
use snafu::location;

use crate::dataset::builder::DatasetBuilder;
use crate::dataset::transaction::{Operation, Transaction, TransactionBuilder};
use crate::dataset::write::{validate_and_resolve_target_bases, write_fragments_internal};
use crate::dataset::ReadParams;
use crate::Dataset;
use crate::{Error, Result};
use tracing::info;

use super::commit::CommitBuilder;
use super::resolve_commit_handler;
use super::WriteDestination;
use super::WriteMode;
use super::WriteParams;
/// Insert or create a new dataset.
///
/// There are different variants of `execute()` methods. Those with the `_stream`
/// suffix take an iterator of data so that larger than memory data can be written
/// out. However, this eliminates optimizations that can be made when the full
/// data is known up-front.
///
/// Those with the `_uncommitted` suffix write the data files but do not commit
/// the transactions. These changes to the dataset will not be visible until
/// they are passed to the [`CommitBuilder`].
#[derive(Debug, Clone)]
pub struct InsertBuilder<'a> {
    dest: WriteDestination<'a>,
    // TODO: make these parameters a part of the builder, and add specific methods.
    params: Option<&'a WriteParams>,
}

impl<'a> InsertBuilder<'a> {
    pub fn new(dest: impl Into<WriteDestination<'a>>) -> Self {
        Self {
            dest: dest.into(),
            params: None,
        }
    }

    pub fn with_params(mut self, params: &'a WriteParams) -> Self {
        self.params = Some(params);
        self
    }

    /// Execute the insert operation with the given data.
    ///
    /// This writes the data fragments and commits them into the dataset.
    pub async fn execute(&self, data: Vec<RecordBatch>) -> Result<Dataset> {
        let (transaction, context) = self.write_uncommitted_impl(data).await?;
        Self::do_commit(&context, transaction).await
    }

    /// Execute the insert operation with the given stream.
    ///
    /// This writes the data fragments and commits them into the dataset.
    pub async fn execute_stream(&self, source: impl StreamingWriteSource) -> Result<Dataset> {
        let (stream, schema) = source.into_stream_and_schema().await?;
        self.execute_stream_impl(stream, schema).await
    }

    async fn execute_stream_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
    ) -> Result<Dataset> {
        let (transaction, context) = self.write_uncommitted_stream_impl(stream, schema).await?;
        Self::do_commit(&context, transaction).await
    }

    /// Write data files, but don't commit the transaction yet.
    ///
    /// Use [`CommitBuilder`] to commit the transaction.
    ///
    /// # Example: Append data to a dataset
    ///
    /// ```rust
    /// use lance::dataset::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
    ///
    /// # use std::sync::Arc;
    /// # use arrow_array::RecordBatch;
    /// # use lance::Result;
    /// # use lance::dataset::Dataset;
    /// # async fn example(dataset: Arc<Dataset>, data: Vec<RecordBatch>) -> Result<()> {
    /// let transaction = InsertBuilder::new(dataset.clone())
    ///     .with_params(&WriteParams { mode: WriteMode::Append, ..Default::default() })
    ///     .execute_uncommitted(data)
    ///     .await?;
    /// CommitBuilder::new(dataset)
    ///     .execute(transaction)
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn execute_uncommitted(&self, data: Vec<RecordBatch>) -> Result<Transaction> {
        self.write_uncommitted_impl(data).await.map(|(t, _)| t)
    }

    async fn do_commit(context: &WriteContext<'_>, transaction: Transaction) -> Result<Dataset> {
        let mut commit_builder = CommitBuilder::new(context.dest.clone())
            .use_stable_row_ids(context.params.enable_stable_row_ids)
            .with_storage_format(context.storage_version)
            .enable_v2_manifest_paths(context.params.enable_v2_manifest_paths)
            .with_commit_handler(context.commit_handler.clone())
            .with_object_store(context.object_store.clone())
            .with_skip_auto_cleanup(context.params.skip_auto_cleanup);

        if let Some(params) = context.params.store_params.as_ref() {
            commit_builder = commit_builder.with_store_params(params.clone());
        }

        if let Some(session) = context.params.session.as_ref() {
            commit_builder = commit_builder.with_session(session.clone());
        }

        commit_builder.execute(transaction).await
    }

    async fn write_uncommitted_impl(
        &self,
        data: Vec<RecordBatch>,
    ) -> Result<(Transaction, WriteContext<'_>)> {
        // TODO: This should be able to split the data up based on max_rows_per_file
        // and write in parallel. https://github.com/lance-format/lance/issues/1980
        if data.is_empty() {
            return Err(Error::InvalidInput {
                source: "No data to write".into(),
                location: location!(),
            });
        }
        let schema = data[0].schema();
        for batch in data.iter().skip(1) {
            if batch.schema() != schema {
                return Err(Error::InvalidInput {
                    source: "All record batches must have the same schema".into(),
                    location: location!(),
                });
            }
        }
        let reader = RecordBatchIterator::new(data.into_iter().map(Ok), schema);
        let (stream, schema) = reader.into_stream_and_schema().await?;
        self.write_uncommitted_stream_impl(stream, schema).await
    }

    /// Write data files, but don't commit the transaction yet.
    ///
    /// Use [`CommitBuilder`] to commit the transaction.
    pub async fn execute_uncommitted_stream(
        &self,
        source: impl StreamingWriteSource,
    ) -> Result<Transaction> {
        let (stream, schema) = source.into_stream_and_schema().await?;
        let (transaction, _) = self.write_uncommitted_stream_impl(stream, schema).await?;
        Ok(transaction)
    }

    async fn write_uncommitted_stream_impl(
        &self,
        stream: SendableRecordBatchStream,
        schema: Schema,
    ) -> Result<(Transaction, WriteContext<'_>)> {
        let mut context = self.resolve_context().await?;

        info!(
            target: TRACE_DATASET_EVENTS,
            event=DATASET_WRITING_EVENT,
            uri=context.dest.uri(),
            mode=?context.params.mode
        );

        self.validate_write(&mut context, &schema)?;

        let existing_base_paths = context.dest.dataset().map(|ds| &ds.manifest.base_paths);
        let target_base_info =
            validate_and_resolve_target_bases(&mut context.params, existing_base_paths).await?;

        // Genuine insert/append path: materialise write-time defaults for omitted columns.
        context.params.inject_write_defaults = true;

        let (written_fragments, written_schema) = write_fragments_internal(
            context.dest.dataset(),
            context.object_store.clone(),
            &context.base_path,
            schema.clone(),
            stream,
            context.params.clone(),
            target_base_info,
        )
        .await?;

        // For Overwrite, the committed manifest schema must include any columns that were
        // injected as write-defaults: they were omitted from the batch but physically
        // materialised into the data files by `augment_with_write_defaults`.
        // `write_fragments_internal` returns the augmented schema — use it so the manifest
        // correctly declares those columns (preventing the "silent vanish" hazard).
        //
        // For Create the schema comes from the batch and no augmentation was applied (there is no
        // existing dataset to read defaults from), so the original `schema` is still correct.
        // For Append the committed schema is the prior manifest schema (constructed inside
        // `build_transaction` from the existing dataset), so `written_schema` is not used there.
        let commit_schema = match context.params.mode {
            WriteMode::Overwrite => written_schema,
            WriteMode::Create | WriteMode::Append => schema,
        };

        let transaction = Self::build_transaction(commit_schema, written_fragments, &context)?;

        Ok((transaction, context))
    }

    fn build_transaction(
        schema: Schema,
        fragments: Vec<Fragment>,
        context: &WriteContext<'_>,
    ) -> Result<Transaction> {
        let operation = match context.params.mode {
            WriteMode::Create => {
                let config_upsert_values =
                    if let Some(auto_cleanup_params) = context.params.auto_cleanup.as_ref() {
                        let mut upsert_values = HashMap::new();
                        upsert_values.insert(
                            String::from("lance.auto_cleanup.interval"),
                            auto_cleanup_params.interval.to_string(),
                        );

                        let duration = auto_cleanup_params.older_than.to_std().map_err(|e| {
                            Error::InvalidInput {
                                source: e.into(),
                                location: location!(),
                            }
                        })?;
                        upsert_values.insert(
                            String::from("lance.auto_cleanup.older_than"),
                            format_duration(duration).to_string(),
                        );
                        Some(upsert_values)
                    } else {
                        None
                    };
                Operation::Overwrite {
                    // Use the full schema, not the written schema
                    schema,
                    fragments,
                    config_upsert_values,
                    initial_bases: context.params.initial_bases.clone(),
                }
            }
            WriteMode::Overwrite => Operation::Overwrite {
                schema,
                fragments,
                config_upsert_values: None,
                initial_bases: context.params.initial_bases.clone(),
            },
            WriteMode::Append => Operation::Append { fragments },
        };

        let transaction = TransactionBuilder::new(
            context
                .dest
                .dataset()
                .map(|ds| ds.manifest.version)
                .unwrap_or(0),
            operation,
        )
        .transaction_properties(context.params.transaction_properties.clone())
        .build();

        Ok(transaction)
    }

    fn validate_write(&self, context: &mut WriteContext, data_schema: &Schema) -> Result<()> {
        // Write mode
        match (&context.params.mode, &context.dest) {
            (WriteMode::Create, WriteDestination::Dataset(ds)) => {
                return Err(Error::DatasetAlreadyExists {
                    uri: ds.uri.clone(),
                    location: location!(),
                });
            }
            (WriteMode::Append | WriteMode::Overwrite, WriteDestination::Uri(uri)) => {
                log::warn!("No existing dataset at {uri}, it will be created");
                context.params.mode = WriteMode::Create;
            }
            _ => {}
        }

        // Validate schema
        if matches!(context.params.mode, WriteMode::Append) {
            if let WriteDestination::Dataset(dataset) = &context.dest {
                // If the dataset is already using (or not using) stable row ids, we need to match
                // and ignore whatever the user provided as input
                if context.params.enable_stable_row_ids != dataset.manifest.uses_stable_row_ids() {
                    log::info!(
                        "Ignoring user provided stable row ids setting of {}, dataset already has it set to {}",
                        context.params.enable_stable_row_ids,
                        dataset.manifest.uses_stable_row_ids()
                    );
                    context.params.enable_stable_row_ids = dataset.manifest.uses_stable_row_ids();
                }

                let schema_cmp_opts = SchemaCompareOptions {
                    compare_dictionary: dataset.manifest.should_use_legacy_format(),
                    compare_nullability: NullabilityComparison::Ignore,
                    allow_missing_if_nullable: true,
                    ignore_field_order: true,
                    ..Default::default()
                };

                // A column carrying a default may be omitted from the incoming batch even when it
                // is non-nullable.  Two cases:
                //   * write-default: `augment_with_write_defaults` materialises it before the data
                //     is written.
                //   * initial-default only (Model A): the column is left structurally absent and the
                //     read path (`DefaultReader`) backfills the live initial-default.
                // `allow_missing_if_nullable` only tolerates missing *nullable* fields, so exclude
                // such omitted defaulted columns from the expected schema for this pre-write check to
                // avoid a spurious SchemaMismatch. The post-augmentation `check_compatible` in
                // `write_fragments_internal` still validates the remaining schema.
                let incoming_names: std::collections::HashSet<&str> =
                    data_schema.fields.iter().map(|f| f.name.as_str()).collect();
                let expected_schema = Schema {
                    fields: dataset
                        .schema()
                        .fields
                        .iter()
                        .filter(|f| {
                            incoming_names.contains(f.name.as_str())
                                || (f.write_default_raw().is_none()
                                    && f.initial_default_raw().is_none())
                        })
                        .cloned()
                        .collect(),
                    metadata: dataset.schema().metadata.clone(),
                };

                data_schema.check_compatible(&expected_schema, &schema_cmp_opts)?;
            }
        }

        // Make sure we aren't using any reserved column names
        for field in data_schema.fields.iter() {
            if field.name == ROW_ID || field.name == ROW_ADDR || field.name == ROW_OFFSET {
                return Err(Error::InvalidInput {
                    source: format!(
                        "The column {} is a reserved name and cannot be used in a Lance dataset",
                        field.name
                    )
                    .into(),
                    location: location!(),
                });
            }
        }

        // Feature flags
        if let WriteDestination::Dataset(dataset) = &context.dest {
            if !can_write_dataset(dataset.manifest.writer_feature_flags) {
                let message = format!(
                    "This dataset cannot be written by this version of Lance. \
                Please upgrade Lance to write to this dataset.\n Flags: {}",
                    dataset.manifest.writer_feature_flags
                );
                return Err(Error::NotSupported {
                    source: message.into(),
                    location: location!(),
                });
            }
        }

        Ok(())
    }

    async fn resolve_context(&self) -> Result<WriteContext<'a>> {
        let params = self.params.cloned().unwrap_or_default();
        let (object_store, base_path, commit_handler) = match &self.dest {
            WriteDestination::Dataset(dataset) => (
                dataset.object_store.clone(),
                dataset.base.clone(),
                dataset.commit_handler.clone(),
            ),
            WriteDestination::Uri(uri) => {
                let registry = params
                    .session
                    .as_ref()
                    .map(|s| s.store_registry())
                    .unwrap_or_else(|| Arc::new(Default::default()));
                let (object_store, base_path) = ObjectStore::from_uri_and_params(
                    registry,
                    uri,
                    &params.store_params.clone().unwrap_or_default(),
                )
                .await?;
                let commit_handler = resolve_commit_handler(
                    uri,
                    params.commit_handler.clone(),
                    &params.store_params,
                )
                .await?;
                (object_store, base_path, commit_handler)
            }
        };
        let dest = match &self.dest {
            WriteDestination::Dataset(dataset) => WriteDestination::Dataset(dataset.clone()),
            WriteDestination::Uri(uri) => {
                // Check if it already exists.
                let builder = DatasetBuilder::from_uri(uri).with_read_params(ReadParams {
                    store_options: params.store_params.clone(),
                    commit_handler: params.commit_handler.clone(),
                    session: params.session.clone(),
                    ..Default::default()
                });

                match builder.load().await {
                    Ok(dataset) => WriteDestination::Dataset(Arc::new(dataset)),
                    Err(Error::DatasetNotFound { .. } | Error::NotFound { .. }) => {
                        WriteDestination::Uri(uri)
                    }
                    Err(e) => return Err(e),
                }
            }
        };

        let storage_version = match (&params.mode, &dest) {
            (WriteMode::Overwrite, WriteDestination::Dataset(dataset)) => {
                // If overwriting an existing dataset, allow the user to specify but use
                // the existing version if they don't
                params.data_storage_version.map(Ok).unwrap_or_else(|| {
                    let m = dataset.manifest.as_ref();
                    m.data_storage_format.lance_file_version()
                })?
            }
            (_, WriteDestination::Dataset(dataset)) => {
                // If appending to an existing dataset, always use the dataset version
                let m = dataset.manifest.as_ref();
                m.data_storage_format.lance_file_version()?
            }
            // Otherwise (no existing dataset) fallback to the default if the user didn't specify
            (_, WriteDestination::Uri(_)) => params.storage_version_or_default(),
        };

        Ok(WriteContext {
            params,
            dest,
            object_store,
            base_path,
            commit_handler,
            storage_version,
        })
    }
}

#[derive(Debug)]
struct WriteContext<'a> {
    params: WriteParams,
    dest: WriteDestination<'a>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    commit_handler: Arc<dyn CommitHandler>,
    storage_version: LanceFileVersion,
}

#[cfg(test)]
mod test {
    use arrow_array::{Int32Array, RecordBatchReader, StructArray};
    use arrow_schema::{ArrowError, DataType, Field, Schema};

    use crate::session::Session;

    use super::*;

    #[tokio::test]
    async fn test_pass_session() {
        let session = Arc::new(Session::new(0, 0, Default::default()));
        let dataset = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                session: Some(session.clone()),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![],
                Arc::new(Schema::new(vec![Field::new("col", DataType::Int32, false)])),
            ))
            .await
            .unwrap();

        assert_eq!(Arc::as_ptr(&dataset.session()), Arc::as_ptr(&session));
    }

    #[tokio::test]
    async fn test_write_empty_struct() {
        // Regresses a 2.1 issue where empty structs did not get assigned any columns
        // in the file because we only look at leaf columns.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "empties",
            DataType::Struct(Vec::<Field>::new().into()),
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StructArray::new_empty_fields(1, None))],
        )
        .unwrap();
        let dataset = InsertBuilder::new("memory://")
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
            .await
            .unwrap();

        assert_eq!(
            dataset
                .count_rows(Some("empties IS NOT NULL".to_string()))
                .await
                .unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn prevent_blob_version_upgrade_on_overwrite() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
            .unwrap();

        let dataset = InsertBuilder::new("memory://blob-version-guard")
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(batch.clone())],
                schema.clone(),
            ))
            .await
            .unwrap();

        let dataset = Arc::new(dataset);
        let params = WriteParams {
            mode: WriteMode::Overwrite,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };

        let result = InsertBuilder::new(dataset.clone())
            .with_params(&params)
            .execute_stream(RecordBatchIterator::new(vec![Ok(batch)], schema.clone()))
            .await;

        assert!(matches!(result, Err(Error::InvalidInput { .. })));
    }

    mod external_error {
        use super::*;
        use std::fmt;

        #[derive(Debug)]
        struct MyTestError {
            code: i32,
            details: String,
        }

        impl fmt::Display for MyTestError {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "MyTestError({}): {}", self.code, self.details)
            }
        }

        impl std::error::Error for MyTestError {}

        fn create_failing_iterator(
            schema: Arc<Schema>,
            fail_at_batch: usize,
            error_code: i32,
        ) -> impl Iterator<Item = std::result::Result<RecordBatch, ArrowError>> {
            let mut batch_count = 0;
            std::iter::from_fn(move || {
                if batch_count >= 5 {
                    return None;
                }
                batch_count += 1;
                if batch_count == fail_at_batch {
                    Some(Err(ArrowError::ExternalError(Box::new(MyTestError {
                        code: error_code,
                        details: format!("Failed at batch {}", batch_count),
                    }))))
                } else {
                    let batch = RecordBatch::try_new(
                        schema.clone(),
                        vec![Arc::new(Int32Array::from(vec![batch_count as i32; 10]))],
                    )
                    .unwrap();
                    Some(Ok(batch))
                }
            })
        }

        #[tokio::test]
        async fn test_insert_builder_preserves_external_error() {
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

            let error_code = 42;
            let iter = create_failing_iterator(schema.clone(), 3, error_code);
            let reader = RecordBatchIterator::new(iter, schema);

            let result = InsertBuilder::new("memory://test_external_error")
                .execute_stream(Box::new(reader) as Box<dyn RecordBatchReader + Send>)
                .await;

            match result {
                Err(Error::External { source }) => {
                    let original = source
                        .downcast_ref::<MyTestError>()
                        .expect("Should be able to downcast to MyTestError");
                    assert_eq!(original.code, error_code);
                    assert!(original.details.contains("batch 3"));
                }
                Err(other) => panic!("Expected Error::External variant, got: {:?}", other),
                Ok(_) => panic!("Expected error, got success"),
            }
        }

        #[tokio::test]
        async fn test_insert_builder_first_batch_error() {
            let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

            let error_code = 999;
            let iter = std::iter::once(Err(ArrowError::ExternalError(Box::new(MyTestError {
                code: error_code,
                details: "immediate failure".to_string(),
            }))));
            let reader = RecordBatchIterator::new(iter, schema);

            let result = InsertBuilder::new("memory://test_first_batch_error")
                .execute_stream(Box::new(reader) as Box<dyn RecordBatchReader + Send>)
                .await;

            match result {
                Err(Error::External { source }) => {
                    let original = source.downcast_ref::<MyTestError>().unwrap();
                    assert_eq!(original.code, error_code);
                }
                Err(other) => panic!("Expected External, got: {:?}", other),
                Ok(_) => panic!("Expected error"),
            }
        }
    }
}

#[cfg(test)]
mod write_default_tests {
    use std::sync::Arc;

    use arrow_array::{cast::AsArray, Array, Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::{LANCE_INITIAL_DEFAULT_META_KEY, LANCE_WRITE_DEFAULT_META_KEY};
    use lance_file::version::LanceFileVersion;

    use crate::dataset::write::insert::InsertBuilder;
    use crate::dataset::{Dataset, WriteMode, WriteParams};

    async fn make_dataset_with_write_default() -> Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("c", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1_i32, 2])),
                Arc::new(Int32Array::from(vec![10_i32, 20])),
            ],
        )
        .unwrap();
        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://write_default_mat",
            None,
        )
        .await
        .unwrap();

        ds.update_field_metadata()
            .update(
                "c",
                [
                    (LANCE_INITIAL_DEFAULT_META_KEY, Some("1")),
                    (LANCE_WRITE_DEFAULT_META_KEY, Some("2")),
                ],
            )
            .unwrap()
            .await
            .unwrap();

        ds
    }

    #[tokio::test]
    async fn test_write_default_materialized_on_append() {
        let ds = make_dataset_with_write_default().await;
        let ds = Arc::new(ds);

        let append_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let append_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![3_i32, 4]))],
        )
        .unwrap();

        let ds = InsertBuilder::new(ds)
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(append_batch)],
                append_schema,
            ))
            .await
            .unwrap();

        let batches = ds
            .scan()
            .project(&["a", "c"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let c_col = batches.column_by_name("c").unwrap();
        let c_vals = c_col.as_primitive::<arrow_array::types::Int32Type>();

        assert_eq!(c_vals.len(), 4);
        // Rows 0,1 — original values
        assert_eq!(c_vals.value(0), 10);
        assert_eq!(c_vals.value(1), 20);
        // Rows 2,3 — write-default materialized (== 2, physically present, NOT NULL)
        assert!(
            !c_vals.is_null(2),
            "row 2 c should be non-null (materialized)"
        );
        assert_eq!(c_vals.value(2), 2);
        assert!(
            !c_vals.is_null(3),
            "row 3 c should be non-null (materialized)"
        );
        assert_eq!(c_vals.value(3), 2);
    }

    #[tokio::test]
    async fn test_write_default_materialized_on_append_non_nullable() {
        // A NON-NULLABLE column carrying a write-default that is omitted from the append batch
        // must be materialised, not rejected by the pre-write schema-compatibility check.
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("c", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1_i32, 2])),
                Arc::new(Int32Array::from(vec![10_i32, 20])),
            ],
        )
        .unwrap();
        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://write_default_mat_non_nullable",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        ds.update_field_metadata()
            .update("c", [(LANCE_WRITE_DEFAULT_META_KEY, Some("2"))])
            .unwrap()
            .await
            .unwrap();

        let ds = Arc::new(ds);

        // Append omitting the non-nullable write-default column `c`.
        let append_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let append_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![3_i32, 4]))],
        )
        .unwrap();

        let ds = InsertBuilder::new(ds)
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(append_batch)],
                append_schema,
            ))
            .await
            .expect("append omitting a non-nullable write-default column must succeed");

        let batches = ds
            .scan()
            .project(&["a", "c"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let c_col = batches.column_by_name("c").unwrap();
        let c_vals = c_col.as_primitive::<arrow_array::types::Int32Type>();

        assert_eq!(c_vals.len(), 4);
        assert_eq!(c_vals.value(0), 10);
        assert_eq!(c_vals.value(1), 20);
        // Rows 2,3 — write-default materialized (== 2, physically present, NOT NULL)
        assert!(
            !c_vals.is_null(2),
            "row 2 c should be non-null (materialized)"
        );
        assert_eq!(c_vals.value(2), 2);
        assert!(
            !c_vals.is_null(3),
            "row 3 c should be non-null (materialized)"
        );
        assert_eq!(c_vals.value(3), 2);
    }

    #[tokio::test]
    async fn test_no_write_default_left_absent() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("c", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1_i32])),
                Arc::new(Int32Array::from(vec![99_i32])),
            ],
        )
        .unwrap();
        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://write_default_absent",
            None,
        )
        .await
        .unwrap();

        // Only initial-default, no write-default
        ds.update_field_metadata()
            .update("c", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("1"))])
            .unwrap()
            .await
            .unwrap();

        let ds = Arc::new(ds);
        let append_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let append_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![2_i32]))],
        )
        .unwrap();

        let ds = InsertBuilder::new(ds)
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(append_batch)],
                append_schema,
            ))
            .await
            .unwrap();

        let batches = ds
            .scan()
            .project(&["a", "c"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let c_col = batches.column_by_name("c").unwrap();
        let c_vals = c_col.as_primitive::<arrow_array::types::Int32Type>();

        assert_eq!(c_vals.len(), 2);
        assert_eq!(c_vals.value(0), 99);
        // Row 1: absent fragment — read-backfill applies initial-default = 1
        assert_eq!(c_vals.value(1), 1);
    }

    /// Regression: a NON-nullable column carrying only an initial-default (no write-default)
    /// may be omitted from an Append.  The column is left structurally absent and the read path
    /// backfills the initial-default.  Previously this was rejected with a SchemaMismatch
    /// because the omitted non-nullable field was kept in the expected schema and
    /// `allow_missing_if_nullable` only tolerates nullable fields.
    #[tokio::test]
    async fn test_append_omitting_non_nullable_initial_default_column() {
        use crate::dataset::schema_evolution::NewColumnTransform;

        // Dataset with a single column `a`.
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1_i32, 2, 3]))],
        )
        .unwrap();
        let mut ds = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://append_omit_non_nullable_default",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Add a NON-nullable column `c` with only an initial-default (Model A).
        let c_field = ArrowField::new("c", DataType::Int32, false).with_metadata(
            std::collections::HashMap::from([(
                LANCE_INITIAL_DEFAULT_META_KEY.to_string(),
                "42".to_string(),
            )]),
        );
        ds.add_columns(
            NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![c_field]))),
            None,
            None,
        )
        .await
        .unwrap();

        // Append a batch that omits `c` entirely — must succeed (not SchemaMismatch).
        let ds = Arc::new(ds);
        let append_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let append_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![4_i32, 5]))],
        )
        .unwrap();

        let ds = InsertBuilder::new(ds)
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(append_batch)],
                append_schema,
            ))
            .await
            .expect("append omitting a non-nullable initial-default column must succeed");

        // All rows (original absent + newly appended absent) read the initial-default 42.
        let batches = ds
            .scan()
            .project(&["a", "c"])
            .unwrap()
            .scan_in_order(true)
            .try_into_batch()
            .await
            .unwrap();
        let c_vals = batches
            .column_by_name("c")
            .unwrap()
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(c_vals.len(), 5);
        assert_eq!(
            c_vals.null_count(),
            0,
            "non-null backfill must produce no nulls"
        );
        assert!((0..5).all(|i| c_vals.value(i) == 42));
    }

    #[tokio::test]
    async fn test_explicit_null_preserved() {
        let ds = make_dataset_with_write_default().await;
        let ds = Arc::new(ds);

        // Append a batch that provides c=NULL explicitly
        let append_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("c", DataType::Int32, true),
        ]));
        let c_null: Option<i32> = None;
        let append_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![5_i32])),
                Arc::new(Int32Array::from(vec![c_null])),
            ],
        )
        .unwrap();

        let ds = InsertBuilder::new(ds)
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(append_batch)],
                append_schema,
            ))
            .await
            .unwrap();

        let batches = ds
            .scan()
            .project(&["a", "c"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let c_col = batches.column_by_name("c").unwrap();
        let c_vals = c_col.as_primitive::<arrow_array::types::Int32Type>();

        // Row 2 (index 2): explicitly written NULL must survive
        assert!(
            c_vals.is_null(2),
            "explicitly written NULL must be preserved"
        );
    }

    #[tokio::test]
    async fn test_rewrite_callers_dont_inject() {
        // WriteParams::default() must have inject_write_defaults = false
        let params = WriteParams::default();
        assert!(
            !params.inject_write_defaults,
            "inject_write_defaults must default to false"
        );
    }

    // -----------------------------------------------------------------------
    // Model B Overwrite tests (committed-schema correctness)
    // -----------------------------------------------------------------------

    /// Overwrite an existing Model-B dataset omitting a write-default column `c`.
    ///
    /// After the overwrite:
    /// * The **committed manifest schema** must still declare `c` (with its write-default
    ///   metadata) — this is the key correctness invariant; a test that only checks data
    ///   values would miss the committed-schema bug.
    /// * Rows in the new version must carry the materialised write-default value for `c`.
    #[tokio::test]
    async fn test_overwrite_write_default_committed_schema_retains_column() {
        // Create an initial dataset with columns `a` and `c`, then add write-defaults.
        let ds = make_dataset_with_write_default().await;
        let ds = Arc::new(ds);

        // Overwrite using a batch that includes only `a` — `c` is absent.
        let overwrite_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let overwrite_batch = RecordBatch::try_new(
            overwrite_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![10_i32, 20]))],
        )
        .unwrap();

        let ds_after = InsertBuilder::new(ds)
            .with_params(&WriteParams {
                mode: WriteMode::Overwrite,
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(overwrite_batch)],
                overwrite_schema,
            ))
            .await
            .unwrap();

        // --- Committed schema assertion ---
        // The manifest schema must still include `c` with its write-default metadata.
        let committed_schema = ds_after.schema();
        let c_field = committed_schema.field("c").unwrap_or_else(|| {
            panic!("committed manifest schema must retain column 'c' after overwrite")
        });
        assert!(
            c_field.write_default_raw().is_some(),
            "committed field 'c' must preserve write-default metadata; got: {:?}",
            c_field.metadata
        );

        // --- Data assertion ---
        // Both rows must carry the materialised write-default value (== 2).
        let batches = ds_after
            .scan()
            .project(&["a", "c"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();

        let c_col = batches.column_by_name("c").unwrap();
        let c_vals = c_col.as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(c_vals.len(), 2, "expected 2 rows after overwrite");
        assert!(
            !c_vals.is_null(0),
            "row 0 c must be non-null (materialised)"
        );
        assert_eq!(c_vals.value(0), 2, "row 0 c must equal the write-default");
        assert!(
            !c_vals.is_null(1),
            "row 1 c must be non-null (materialised)"
        );
        assert_eq!(c_vals.value(1), 2, "row 1 c must equal the write-default");
    }

    /// Regression: a plain Append after the fix still commits the prior manifest schema
    /// (not the augmented written schema), preserving all metadata and field ids.
    #[tokio::test]
    async fn test_append_after_overwrite_fix_regression() {
        let ds = make_dataset_with_write_default().await;
        let prior_schema = ds.schema().clone();
        let ds = Arc::new(ds);

        let append_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let append_batch = RecordBatch::try_new(
            append_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![5_i32]))],
        )
        .unwrap();

        let ds_after = InsertBuilder::new(ds)
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(append_batch)],
                append_schema,
            ))
            .await
            .unwrap();

        // Committed schema after Append must equal the prior manifest schema.
        let committed_schema = ds_after.schema();
        assert_eq!(
            committed_schema.fields.len(),
            prior_schema.fields.len(),
            "Append must not alter the number of committed schema fields"
        );
        for (before, after) in prior_schema
            .fields
            .iter()
            .zip(committed_schema.fields.iter())
        {
            assert_eq!(
                before.name, after.name,
                "Append must preserve field names in the committed schema"
            );
            assert_eq!(
                before.metadata, after.metadata,
                "Append must preserve field metadata (incl. write-default) in the committed schema"
            );
        }
    }

    /// A normal Overwrite that supplies all columns (no omission) must remain unaffected:
    /// the committed schema must include all columns that were written.
    #[tokio::test]
    async fn test_overwrite_all_columns_no_regression() {
        let ds = make_dataset_with_write_default().await;
        let ds = Arc::new(ds);

        // Overwrite providing both `a` and `c` — no column is omitted.
        let overwrite_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("c", DataType::Int32, true),
        ]));
        let overwrite_batch = RecordBatch::try_new(
            overwrite_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![7_i32])),
                Arc::new(Int32Array::from(vec![99_i32])),
            ],
        )
        .unwrap();

        let ds_after = InsertBuilder::new(ds)
            .with_params(&WriteParams {
                mode: WriteMode::Overwrite,
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(overwrite_batch)],
                overwrite_schema,
            ))
            .await
            .unwrap();

        let committed_schema = ds_after.schema();
        assert!(
            committed_schema.field("a").is_some(),
            "column 'a' must be in committed schema after full overwrite"
        );
        assert!(
            committed_schema.field("c").is_some(),
            "column 'c' must be in committed schema after full overwrite"
        );

        let batches = ds_after
            .scan()
            .project(&["a", "c"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let c_col = batches.column_by_name("c").unwrap();
        let c_vals = c_col.as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(
            c_vals.value(0),
            99,
            "explicitly written value must be preserved"
        );
    }

    /// Regression for the duplicate-field-id corruption on Overwrite: when the
    /// overwrite batch omits a write-default column AND introduces a brand-new
    /// column, the new column receives a fresh id that previously collided with
    /// the injected write-default field's stale id (cloned from the prior dataset
    /// schema).  The collision was not caught at commit, but the resulting
    /// dataset was unreadable ("Duplicate field name"/"Duplicate field id").
    /// The injected column must get a fresh, non-colliding id and the dataset
    /// must be scannable.
    #[tokio::test]
    async fn test_overwrite_write_default_new_column_no_duplicate_field_id() {
        // Initial dataset: a(id=0), c(id=1, write-default=2, initial-default=1).
        let ds = make_dataset_with_write_default().await;
        let ds = Arc::new(ds);

        // Overwrite with a batch [a, b] that omits `c` and adds a new column `b`.
        // The incoming `a`/`b` get fresh ids 0/1; injected `c` must NOT keep id 1.
        let overwrite_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, false),
            ArrowField::new("b", DataType::Int32, true),
        ]));
        let overwrite_batch = RecordBatch::try_new(
            overwrite_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![10_i32, 20])),
                Arc::new(Int32Array::from(vec![100_i32, 200])),
            ],
        )
        .unwrap();

        let ds_after = InsertBuilder::new(ds)
            .with_params(&WriteParams {
                mode: WriteMode::Overwrite,
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            })
            .execute_stream(RecordBatchIterator::new(
                vec![Ok(overwrite_batch)],
                overwrite_schema,
            ))
            .await
            .unwrap();

        // Committed schema must have unique field ids (validate would have caught
        // a duplicate at commit time after the fix).
        ds_after
            .schema()
            .validate()
            .expect("committed schema must have unique field ids/names");

        // `c` must be retained with its write-default preserved.
        let committed_schema = ds_after.schema();
        let c_field = committed_schema
            .field("c")
            .expect("committed schema must retain column 'c'");
        assert!(
            c_field.write_default_raw().is_some(),
            "committed field 'c' must preserve write-default metadata"
        );

        // The dataset must be readable: prior to the fix this scan failed with a
        // duplicate-field error.
        let batches = ds_after
            .scan()
            .project(&["a", "b", "c"])
            .unwrap()
            .try_into_batch()
            .await
            .expect("scan of overwritten dataset must succeed");

        assert_eq!(batches.num_rows(), 2);
        let c_col = batches.column_by_name("c").unwrap();
        let c_vals = c_col.as_primitive::<arrow_array::types::Int32Type>();
        assert!(
            !c_vals.is_null(0),
            "row 0 c must be materialised (non-null)"
        );
        assert_eq!(c_vals.value(0), 2, "row 0 c must equal the write-default");
        assert_eq!(c_vals.value(1), 2, "row 1 c must equal the write-default");
    }
}
