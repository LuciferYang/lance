// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Typed setters for column default values on an existing dataset.
//!
//! This module exposes two public methods on [`Dataset`]:
//!
//! * [`Dataset::set_column_default`] – persist or update the
//!   `lance-schema:initial-default` metadata key for a top-level column.
//! * [`Dataset::remove_column_default`] – delete that key from a top-level
//!   column.
//!
//! Both methods apply the §1.4a scalar-index safety guard: if the target
//! column is covered by a scalar index (BTree, Bitmap, Inverted, …) the
//! operation is rejected with an actionable error message.  The caller must
//! drop the index, set/remove the default, then recreate the index.
//!
//! ## Model-selection flag (Task 3a.1)
//!
//! The constant [`COLUMN_DEFAULTS_MODEL_CONFIG_KEY`] and helper
//! [`column_defaults_model_b_enabled`] implement the per-dataset manifest flag
//! that selects between **Model A** (single mutable default; default) and
//! **Model B** (two-value defaults: initial-default + fill-default).
//!
//! Set the flag via the existing config-update API:
//!
//! ```ignore
//! dataset.update_config([(COLUMN_DEFAULTS_MODEL_CONFIG_KEY, "B")]).await?;
//! ```
//!
//! **Phase 3b** will consume [`column_defaults_model_b_enabled`] inside
//! `Transaction::build_manifest` to gate the immutability guard on this flag.

use lance_core::datatypes::{validate_default_assignable, LANCE_INITIAL_DEFAULT_META_KEY};
use lance_index::DatasetIndexExt;
use lance_table::format::Manifest;
use snafu::location;

use super::metadata::UpdateFieldMetadataBuilder;
use super::Dataset;
use crate::{Error, Result};

// ── Model-selection flag ──────────────────────────────────────────────────────

/// Manifest table-config key that selects the column-defaults model.
///
/// * **Absent** (or any value other than `"B"`) → **Model A**: single mutable
///   default stored as `lance-schema:initial-default` in field metadata.
/// * **`"B"`** → **Model B**: two-value defaults (initial-default +
///   fill-default).  The Phase-3b immutability guard in
///   `Transaction::build_manifest` is gated on this flag.
///
/// # Setting the flag
///
/// Use the existing config-update API — no new manifest field is required:
///
/// ```ignore
/// dataset.update_config([(COLUMN_DEFAULTS_MODEL_CONFIG_KEY, "B")]).await?;
/// ```
pub const COLUMN_DEFAULTS_MODEL_CONFIG_KEY: &str = "lance.column-defaults.model";

/// The value stored in the manifest config to activate Model B.
const COLUMN_DEFAULTS_MODEL_B_VALUE: &str = "B";

/// Return `true` if the manifest's table config has the column-defaults model
/// key set to `"B"` (Model B active).
///
/// Absent or any other value → `false` (Model A, the default).
///
/// When `true`, `Transaction::build_manifest` enforces that the
/// `lance-schema:initial-default` metadata key on any existing field id is
/// immutable — it cannot be changed or dropped once set.
pub fn column_defaults_model_b_enabled(manifest: &Manifest) -> bool {
    manifest
        .config
        .get(COLUMN_DEFAULTS_MODEL_CONFIG_KEY)
        .map(|v| v == COLUMN_DEFAULTS_MODEL_B_VALUE)
        .unwrap_or(false)
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Resolve a *bare, top-level* column name to its field id.
///
/// Rejects dotted paths (nested columns) and non-existent names.
fn resolve_top_level_field_id(dataset: &Dataset, column: &str) -> Result<i32> {
    // Reject dotted paths up-front so the user gets a clear message.
    if column.contains('.') {
        return Err(Error::invalid_input(
            format!(
                "column '{}': dotted paths are not supported — \
                 column defaults can only be set on top-level columns",
                column
            ),
            location!(),
        ));
    }

    // schema().field() does a DFS lookup; for bare names it returns the
    // first top-level field that matches.
    let schema = dataset.schema();
    let field = schema.field(column).ok_or_else(|| {
        Error::invalid_input(
            format!("column '{}' not found in dataset schema", column),
            location!(),
        )
    })?;

    // Double-check that the field is genuinely top-level, i.e. it appears
    // directly in schema.fields (not only as a child of a struct).
    if !schema.fields.iter().any(|f| f.id == field.id) {
        return Err(Error::invalid_input(
            format!(
                "column '{}': only top-level columns are supported — \
                 nested fields cannot carry a column default",
                column
            ),
            location!(),
        ));
    }

    Ok(field.id)
}

/// Return `Some(index_name)` if any **scalar** index currently covers
/// `field_id`, otherwise `None`.
///
/// A scalar index is identified by its `index_details` type URL **not**
/// ending with `"VectorIndexDetails"`.  Indices with `index_details == None`
/// (legacy format) are conservatively treated as scalar indices to err on
/// the side of safety.
async fn covering_scalar_index_name(dataset: &Dataset, field_id: i32) -> Result<Option<String>> {
    let indices = dataset.load_indices().await?;
    for idx in indices.iter() {
        if !idx.fields.contains(&field_id) {
            continue;
        }
        let is_vector = idx
            .index_details
            .as_ref()
            .map(|d| d.type_url.ends_with("VectorIndexDetails"))
            .unwrap_or(false); // None → legacy scalar → treat as scalar
        if !is_vector {
            return Ok(Some(idx.name.clone()));
        }
    }
    Ok(None)
}

/// Apply the §1.4a scalar-index guard.  Returns `Err` if a covering scalar
/// index exists.
///
/// TODO(model-flag): When the model-selection flag (Phase 3) lands, gate
/// this guard on `model == ModelA`.  For now it runs unconditionally because
/// Model A is the only option.
async fn check_no_covering_scalar_index(
    dataset: &Dataset,
    column: &str,
    field_id: i32,
) -> Result<()> {
    if let Some(idx_name) = covering_scalar_index_name(dataset, field_id).await? {
        return Err(Error::invalid_input(
            format!(
                "cannot set/change a default on column '{}' because it has a \
                 scalar index ('{}'); drop the index first, set the default, \
                 then recreate the index",
                column, idx_name
            ),
            location!(),
        ));
    }
    Ok(())
}

/// Return `true` if `field_id` is physically absent (not materialized in any
/// data file) in at least one fragment of the dataset.
///
/// A field is physically present in a fragment when its id appears in any of
/// that fragment's data files' `fields` lists.  A dataset with no fragments has
/// no rows requiring backfill, so this returns `false`.
fn field_absent_in_any_fragment(dataset: &Dataset, field_id: i32) -> bool {
    dataset.manifest.fragments.iter().any(|frag| {
        !frag
            .files
            .iter()
            .any(|file| file.fields.contains(&field_id))
    })
}

// ── public API ────────────────────────────────────────────────────────────────

impl Dataset {
    /// Persist `default_literal` as the `lance-schema:initial-default` for
    /// an existing top-level column.
    ///
    /// Stores the JSON-serialized literal in the field's Arrow metadata under
    /// the key `lance-schema:initial-default`.  Pre-existing rows that were
    /// written before the column was added will read back this value instead of
    /// `null` when scanned.
    ///
    /// # Errors
    ///
    /// * Column name is dotted / nested.
    /// * Column does not exist.
    /// * `default_literal` fails [`validate_default_assignable`]
    ///   (non-nullable, nested type, Dictionary/REE, PK column, or
    ///   value that cannot be cast to the column's data type).
    /// * A scalar index covers this column (§1.4a guard): drop the index,
    ///   set the default, then recreate the index.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::{Dataset, Result};
    /// # use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    /// # use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    /// # async fn example() -> Result<()> {
    /// let schema = Arc::new(ArrowSchema::new(vec![
    ///     ArrowField::new("id",    DataType::Int32, true),
    ///     ArrowField::new("score", DataType::Int32, true),
    /// ]));
    /// let batch = RecordBatch::try_new(
    ///     schema.clone(),
    ///     vec![
    ///         Arc::new(Int32Array::from(vec![1_i32, 2, 3])),
    ///         Arc::new(Int32Array::from(vec![None, None, None])),
    ///     ],
    /// ).unwrap();
    /// let mut dataset = Dataset::write(
    ///     RecordBatchIterator::new(vec![Ok(batch)], schema),
    ///     "memory://",
    ///     None,
    /// ).await?;
    ///
    /// // Store "0" as the initial default for the nullable `score` column.
    /// dataset.set_column_default("score", "0").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn set_column_default(&mut self, column: &str, default_literal: &str) -> Result<()> {
        let field_id = resolve_top_level_field_id(self, column)?;

        // Validate the literal against the field's type and constraints.
        let field = self.schema().field_by_id(field_id).ok_or_else(|| {
            Error::invalid_input(
                format!("column '{column}': field id {field_id} not found in schema"),
                location!(),
            )
        })?;
        validate_default_assignable(field, default_literal).map_err(|e| {
            Error::invalid_input(
                format!("column '{}': invalid initial default value: {}", column, e),
                location!(),
            )
        })?;

        // §1.4a scalar-index guard (unconditional in Model A).
        check_no_covering_scalar_index(self, column, field_id).await?;

        // Key-scoped merge: replace=false, single key.
        UpdateFieldMetadataBuilder::new(self)
            .update(
                field_id,
                [(LANCE_INITIAL_DEFAULT_META_KEY, default_literal)],
            )?
            .await?;

        Ok(())
    }

    /// Remove the `lance-schema:initial-default` key from an existing
    /// top-level column.
    ///
    /// Deletes the `lance-schema:initial-default` metadata key from *column*.
    /// For a **nullable** column, pre-existing rows that were written before the
    /// column was added will read back `null` instead of the previously stored
    /// default after removal.
    ///
    /// No-ops gracefully if the key was not set.
    ///
    /// # Errors
    ///
    /// * Column name is dotted / nested.
    /// * Column does not exist.
    /// * A scalar index covers this column (§1.4a guard): drop the index,
    ///   remove the default, then recreate the index.
    /// * The column is **non-nullable** and is physically absent in at least one
    ///   fragment.  The initial-default is the only backfill source for those
    ///   rows, so removing it would leave them unreadable (the read path cannot
    ///   produce a non-null value).  Mirrors the add-time guard in
    ///   `schema_evolution.rs`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::{Dataset, Result};
    /// # use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    /// # use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    /// # use lance_core::datatypes::LANCE_INITIAL_DEFAULT_META_KEY;
    /// # async fn example() -> Result<()> {
    /// let schema = Arc::new(ArrowSchema::new(vec![
    ///     ArrowField::new("id",    DataType::Int32, true),
    ///     ArrowField::new("score", DataType::Int32, true),
    /// ]));
    /// let batch = RecordBatch::try_new(
    ///     schema.clone(),
    ///     vec![
    ///         Arc::new(Int32Array::from(vec![1_i32, 2, 3])),
    ///         Arc::new(Int32Array::from(vec![None, None, None])),
    ///     ],
    /// ).unwrap();
    /// let mut dataset = Dataset::write(
    ///     RecordBatchIterator::new(vec![Ok(batch)], schema),
    ///     "memory://",
    ///     None,
    /// ).await?;
    ///
    /// // First set a default, then remove it.
    /// dataset.set_column_default("score", "0").await?;
    /// dataset.remove_column_default("score").await?;
    ///
    /// // The key is now absent.
    /// let field = dataset.schema().field("score").unwrap();
    /// assert!(!field.metadata.contains_key(LANCE_INITIAL_DEFAULT_META_KEY));
    /// # Ok(())
    /// # }
    /// ```
    pub async fn remove_column_default(&mut self, column: &str) -> Result<()> {
        let field_id = resolve_top_level_field_id(self, column)?;

        // Non-nullable + physically-absent guard.  This is the inverse of the
        // add-time guard in `schema_evolution.rs`: a non-nullable column that is
        // absent in some fragment can only be read by backfilling its
        // initial-default.  Removing the default would leave those rows with no
        // valid value, making the column (and any projection including it)
        // unreadable.  Only allow removal when the field is nullable or it is
        // materialized in every fragment.
        let field = self.schema().field_by_id(field_id).ok_or_else(|| {
            Error::invalid_input(
                format!("column '{column}': field id {field_id} not found in schema"),
                location!(),
            )
        })?;
        if !field.nullable && field_absent_in_any_fragment(self, field_id) {
            return Err(Error::invalid_input(
                format!(
                    "cannot remove the initial-default from non-nullable column '{}' \
                     because it is physically absent in at least one fragment; the \
                     initial-default is the only backfill source for those rows. \
                     Make the column nullable or materialize it in every fragment first.",
                    column
                ),
                location!(),
            ));
        }

        // §1.4a scalar-index guard (unconditional in Model A).
        check_no_covering_scalar_index(self, column, field_id).await?;

        // Key-scoped delete: value = None → key removed.
        UpdateFieldMetadataBuilder::new(self)
            .update(
                field_id,
                [(LANCE_INITIAL_DEFAULT_META_KEY, None as Option<&str>)],
            )?
            .await?;

        Ok(())
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::LANCE_INITIAL_DEFAULT_META_KEY;
    use lance_index::{scalar::ScalarIndexParams, DatasetIndexExt, IndexType};

    use super::super::Dataset;
    use crate::{Error, Result};

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Create a simple dataset with two nullable top-level columns:
    /// `id` (Int32, nullable) and `tag` (Utf8, nullable).
    async fn make_dataset() -> Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int32, true),
            ArrowField::new("tag", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1_i32, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            None,
        )
        .await
        .unwrap()
    }

    // ── 1. round-trip: set → persist → remove → gone ─────────────────────────

    /// Setting a valid default persists the `initial-default` key; a subsequent
    /// `remove_column_default` deletes exactly that key without disturbing other
    /// metadata on the field.
    #[tokio::test]
    async fn test_set_and_remove_column_default_round_trip() -> Result<()> {
        let mut dataset = make_dataset().await;

        // Pre-condition: no default key yet.
        let field = dataset.schema().field("id").unwrap();
        assert!(!field.metadata.contains_key(LANCE_INITIAL_DEFAULT_META_KEY));

        // First, place some unrelated metadata on the field so we can verify
        // the key-scoped merge does NOT disturb it.
        let version_before = dataset.version().version;
        dataset
            .update_field_metadata()
            .update("id", [("my-own-key", "my-value")])?
            .await?;
        assert_eq!(dataset.version().version, version_before + 1);

        // Set a valid default.
        dataset.set_column_default("id", "42").await?;
        let field = dataset.schema().field("id").unwrap();
        assert_eq!(
            field
                .metadata
                .get(LANCE_INITIAL_DEFAULT_META_KEY)
                .map(|s| s.as_str()),
            Some("42"),
            "initial-default key must be persisted after set_column_default"
        );
        // Unrelated key must survive.
        assert_eq!(
            field.metadata.get("my-own-key").map(|s| s.as_str()),
            Some("my-value"),
            "key-scoped merge must not disturb co-located metadata"
        );

        // Update the default to a different value.
        dataset.set_column_default("id", "99").await?;
        let field = dataset.schema().field("id").unwrap();
        assert_eq!(
            field
                .metadata
                .get(LANCE_INITIAL_DEFAULT_META_KEY)
                .map(|s| s.as_str()),
            Some("99"),
            "initial-default key must be updated"
        );

        // Remove the default.
        dataset.remove_column_default("id").await?;
        let field = dataset.schema().field("id").unwrap();
        assert!(
            !field.metadata.contains_key(LANCE_INITIAL_DEFAULT_META_KEY),
            "initial-default key must be absent after remove_column_default"
        );
        // Unrelated key must still survive.
        assert_eq!(
            field.metadata.get("my-own-key").map(|s| s.as_str()),
            Some("my-value"),
            "remove_column_default must not disturb co-located metadata"
        );

        Ok(())
    }

    // ── 2. validation: uncastable literal → Err, no version advance ───────────

    /// An uncastable literal (e.g., `"not-a-number"` for an Int32 column)
    /// must be rejected before committing; the dataset version must not
    /// advance.
    #[tokio::test]
    async fn test_set_column_default_uncastable_literal_rejected() -> Result<()> {
        let mut dataset = make_dataset().await;
        let version_before = dataset.version().version;

        let result = dataset.set_column_default("id", "not-a-number").await;
        assert!(
            result.is_err(),
            "expected Err for uncastable literal, got Ok"
        );
        assert!(
            matches!(result.unwrap_err(), Error::InvalidInput { .. }),
            "expected InvalidInput error variant"
        );
        // Version must not have advanced.
        assert_eq!(
            dataset.version().version,
            version_before,
            "version must not advance on validation failure"
        );

        Ok(())
    }

    // ── 3. §1.4a scalar-index guard ───────────────────────────────────────────

    /// Building a scalar (BTree) index on a column and then calling
    /// `set_column_default` on that column must be rejected with an
    /// actionable error.  Likewise for `remove_column_default`.
    #[tokio::test]
    async fn test_scalar_index_guard_blocks_set_and_remove() -> Result<()> {
        // Temporary directory needed for index files.
        let tmp = lance_core::utils::tempfile::TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "score",
            DataType::Int32,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![10_i32, 20, 30]))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            tmp.as_ref(),
            None,
        )
        .await?;

        // Build a BTree scalar index on "score".
        let params = ScalarIndexParams::default();
        dataset
            .create_index(
                &["score"],
                IndexType::BTree,
                Some("score_btree".to_string()),
                &params,
                true,
            )
            .await?;

        // set_column_default must be rejected.
        let version_before = dataset.version().version;
        let set_result = dataset.set_column_default("score", "0").await;
        assert!(
            set_result.is_err(),
            "set_column_default must be rejected when a scalar index exists"
        );
        let err_msg = set_result.unwrap_err().to_string();
        assert!(
            err_msg.contains("scalar index"),
            "error message must mention 'scalar index', got: {err_msg}"
        );
        assert!(
            err_msg.contains("score_btree"),
            "error message must mention the index name, got: {err_msg}"
        );
        assert!(
            err_msg.contains("drop the index"),
            "error message must be actionable ('drop the index'), got: {err_msg}"
        );
        assert_eq!(
            dataset.version().version,
            version_before,
            "version must not advance when guard fires"
        );

        // remove_column_default must also be rejected.
        let remove_result = dataset.remove_column_default("score").await;
        assert!(
            remove_result.is_err(),
            "remove_column_default must be rejected when a scalar index exists"
        );
        let err_msg = remove_result.unwrap_err().to_string();
        assert!(
            err_msg.contains("scalar index"),
            "remove error message must mention 'scalar index', got: {err_msg}"
        );

        Ok(())
    }

    // ── 4. invalid column names ───────────────────────────────────────────────

    /// A dotted path must produce a clear error, distinct from a
    /// non-existent column error.
    #[tokio::test]
    async fn test_dotted_path_rejected() -> Result<()> {
        let mut dataset = make_dataset().await;

        let result = dataset.set_column_default("id.sub", "1").await;
        assert!(result.is_err(), "expected Err for dotted path");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("dotted paths are not supported"),
            "error message must explain dotted-path rejection, got: {err_msg}"
        );

        Ok(())
    }

    /// A non-existent column must produce a clear "not found" error.
    #[tokio::test]
    async fn test_nonexistent_column_rejected() -> Result<()> {
        let mut dataset = make_dataset().await;

        let result = dataset.set_column_default("no_such_column", "1").await;
        assert!(result.is_err(), "expected Err for non-existent column");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not found"),
            "error message must say 'not found', got: {err_msg}"
        );

        Ok(())
    }

    // ── 6. remove_column_default guard: non-nullable + physically absent ───────

    /// Removing the initial-default from a NON-nullable column that is
    /// physically absent in pre-existing fragments must be rejected: the
    /// default is the only backfill source for those rows, so removing it
    /// would leave the column (and any projection of it) unreadable.
    #[tokio::test]
    async fn test_remove_default_rejected_for_nonnullable_absent_column() -> Result<()> {
        use crate::dataset::schema_evolution::NewColumnTransform;

        let mut dataset = make_dataset().await;

        // Add a NON-nullable column `c` with an initial-default via AllNulls.
        // The column is physically absent in the existing fragment (metadata-only).
        let c_field = ArrowField::new("c", DataType::Int32, false).with_metadata(
            std::collections::HashMap::from([(
                LANCE_INITIAL_DEFAULT_META_KEY.to_string(),
                "42".to_string(),
            )]),
        );
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![c_field]))),
                None,
                None,
            )
            .await?;

        // Sanity: the column reads back the default before removal.
        let batch = dataset.scan().project(&["c"])?.try_into_batch().await?;
        assert_eq!(batch.column(0).null_count(), 0);

        // Removing the default must be rejected.
        let version_before = dataset.version().version;
        let result = dataset.remove_column_default("c").await;
        assert!(
            result.is_err(),
            "removing initial-default from a non-nullable absent column must be rejected"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("non-nullable") && err_msg.contains("absent"),
            "error must explain the non-nullable/absent reason, got: {err_msg}"
        );
        assert_eq!(
            dataset.version().version,
            version_before,
            "version must not advance when the guard fires"
        );

        // The column must still be readable (default intact).
        let batch = dataset.scan().project(&["c"])?.try_into_batch().await?;
        assert_eq!(batch.column(0).null_count(), 0);

        Ok(())
    }

    /// Removing the initial-default from a NULLABLE absent column is allowed
    /// (the rows simply read back null afterwards).
    #[tokio::test]
    async fn test_remove_default_allowed_for_nullable_absent_column() -> Result<()> {
        use crate::dataset::schema_evolution::NewColumnTransform;

        let mut dataset = make_dataset().await;

        let c_field = ArrowField::new("c", DataType::Int32, true).with_metadata(
            std::collections::HashMap::from([(
                LANCE_INITIAL_DEFAULT_META_KEY.to_string(),
                "42".to_string(),
            )]),
        );
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![c_field]))),
                None,
                None,
            )
            .await?;

        dataset.remove_column_default("c").await?;
        let field = dataset.schema().field("c").unwrap();
        assert!(!field.metadata.contains_key(LANCE_INITIAL_DEFAULT_META_KEY));

        // Absent nullable rows now read back null.
        let batch = dataset.scan().project(&["c"])?.try_into_batch().await?;
        assert_eq!(batch.column(0).null_count(), batch.num_rows());

        Ok(())
    }

    // ── 5. Model-selection flag (Task 3a.1) ──────────────────────────────────

    /// A fresh dataset must default to Model A:
    /// `column_defaults_model_b_enabled` returns `false` when the config key
    /// is absent.
    #[tokio::test]
    async fn test_model_flag_absent_means_model_a() -> Result<()> {
        let dataset = make_dataset().await;

        assert!(
            !super::column_defaults_model_b_enabled(&dataset.manifest),
            "fresh dataset must report Model A (flag absent)"
        );
        // Confirm the key is genuinely not in the config map.
        assert!(
            !dataset
                .config()
                .contains_key(super::COLUMN_DEFAULTS_MODEL_CONFIG_KEY),
            "fresh dataset must have no column-defaults model config key"
        );

        Ok(())
    }

    /// Setting the config key to `"B"` via `Dataset::update_config` persists
    /// it in the manifest; after re-reading the manifest the helper returns
    /// `true` (Model B active).
    #[tokio::test]
    async fn test_model_flag_set_to_b_persists_and_reader_returns_true() -> Result<()> {
        let mut dataset = make_dataset().await;

        // Pre-condition: Model A.
        assert!(
            !super::column_defaults_model_b_enabled(&dataset.manifest),
            "should start as Model A"
        );

        // Set the flag via the existing config-update API.
        dataset
            .update_config([(super::COLUMN_DEFAULTS_MODEL_CONFIG_KEY, "B")])
            .await?;

        // Reader helper must now return true.
        assert!(
            super::column_defaults_model_b_enabled(&dataset.manifest),
            "reader must return true after setting model key to 'B'"
        );
        // The config map must contain the key with value "B".
        assert_eq!(
            dataset
                .config()
                .get(super::COLUMN_DEFAULTS_MODEL_CONFIG_KEY)
                .map(|s| s.as_str()),
            Some("B"),
            "config map must contain 'B' for the model key"
        );

        Ok(())
    }

    /// Re-opening the dataset confirms the flag survives a round-trip through
    /// the manifest (written to disk, then read back from a fresh `Dataset::open`).
    #[tokio::test]
    async fn test_model_flag_survives_reopen() -> Result<()> {
        // Use a temporary directory so the manifest is written to real files
        // and can be re-opened independently.
        let tmp = lance_core::utils::tempfile::TempStrDir::default();
        let uri = tmp.as_ref();

        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            DataType::Int32,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1_i32]))],
        )
        .unwrap();
        let mut dataset =
            Dataset::write(RecordBatchIterator::new(vec![Ok(batch)], schema), uri, None).await?;

        // Set Model B flag.
        dataset
            .update_config([(super::COLUMN_DEFAULTS_MODEL_CONFIG_KEY, "B")])
            .await?;

        // Re-open the dataset from the same path (independent read).
        let reopened = Dataset::open(uri).await?;

        assert!(
            super::column_defaults_model_b_enabled(&reopened.manifest),
            "Model B flag must survive reopen (persisted in manifest)"
        );

        Ok(())
    }

    /// Setting the flag to a value other than `"B"` must keep Model A.
    #[tokio::test]
    async fn test_model_flag_non_b_value_means_model_a() -> Result<()> {
        let mut dataset = make_dataset().await;

        // Set the key to something that is not "B".
        dataset
            .update_config([(super::COLUMN_DEFAULTS_MODEL_CONFIG_KEY, "A")])
            .await?;

        assert!(
            !super::column_defaults_model_b_enabled(&dataset.manifest),
            "value 'A' must be treated as Model A (only 'B' activates Model B)"
        );

        Ok(())
    }
}

// ── Enforcement tests (§3 initial-default immutability + PK co-presence) ─────

#[cfg(test)]
mod enforcement_tests {
    //! Tests for the two invariants enforced in `Transaction::build_manifest`:
    //!
    //! 1. `initial-default` immutability — gated on Model B.
    //! 2. PK + default co-presence rejection — always-on, both models.

    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::{LANCE_INITIAL_DEFAULT_META_KEY, LANCE_WRITE_DEFAULT_META_KEY};

    use super::super::Dataset;
    use crate::dataset::schema_evolution::NewColumnTransform;
    use crate::Result;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build a minimal dataset (`x` Int32) at `memory://`.
    async fn make_dataset() -> Dataset {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            DataType::Int32,
            true,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1_i32, 2, 3]))],
        )
        .unwrap();
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            None,
        )
        .await
        .unwrap()
    }

    /// Enable Model B on a dataset.
    async fn enable_model_b(dataset: &mut Dataset) -> Result<()> {
        dataset
            .update_config([(super::COLUMN_DEFAULTS_MODEL_CONFIG_KEY, "B")])
            .await
            .map(|_| ())
    }

    // ── (a) update_field_metadata changing an existing initial-default → REJECTED

    #[tokio::test]
    async fn test_immutability_update_changes_initial_default_rejected() -> Result<()> {
        let mut dataset = make_dataset().await;

        // Set an initial-default on field `x` (Model A — allowed).
        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("10"))])?
            .await?;

        // Activate Model B.
        enable_model_b(&mut dataset).await?;

        // Now try to change the initial-default via update_field_metadata → must fail.
        let result = dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("99"))])?
            .await;

        assert!(
            result.is_err(),
            "changing initial-default under Model B must be rejected"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("initial-default") || msg.contains("immutable"),
            "error must mention immutability, got: {msg}"
        );

        Ok(())
    }

    // ── (b) replace=true dropping initial-default → REJECTED ─────────────────

    #[tokio::test]
    async fn test_immutability_replace_drops_initial_default_rejected() -> Result<()> {
        let mut dataset = make_dataset().await;

        // Set initial-default in Model A.
        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("42"))])?
            .await?;

        enable_model_b(&mut dataset).await?;

        // replace=true write that does NOT include the initial-default key
        // → effectively drops it → must be rejected.
        let result = dataset
            .update_field_metadata()
            .replace("x", [("some-other-key", Some("value"))])?
            .await;

        assert!(
            result.is_err(),
            "replace that drops initial-default under Model B must be rejected"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("initial-default") || msg.contains("immutable"),
            "error must mention immutability, got: {msg}"
        );

        // Confirm field is unchanged after the rejected write.
        assert_eq!(
            dataset
                .schema()
                .field("x")
                .unwrap()
                .metadata
                .get(LANCE_INITIAL_DEFAULT_META_KEY)
                .map(|s| s.as_str()),
            Some("42"),
            "initial-default must be unchanged after rejected write"
        );

        // Also test via Dataset::replace_field_metadata.
        let field_id = dataset.schema().field("x").unwrap().id;
        let mut new_meta = HashMap::new();
        new_meta.insert("other-key".to_string(), "val".to_string());
        let result = dataset
            .replace_field_metadata([(field_id as u32, new_meta)])
            .await;

        assert!(
            result.is_err(),
            "replace_field_metadata dropping initial-default must be rejected"
        );

        Ok(())
    }

    // ── (c) add_columns setting initial-default on new field id → ALLOWED ─────

    #[tokio::test]
    async fn test_immutability_new_field_may_set_initial_default() -> Result<()> {
        let mut dataset = make_dataset().await;
        enable_model_b(&mut dataset).await?;

        // Add a new column with an initial-default in its metadata.
        let new_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "y",
            DataType::Int32,
            true,
        )
        .with_metadata(HashMap::from([(
            LANCE_INITIAL_DEFAULT_META_KEY.to_string(),
            "7".to_string(),
        )]))]));
        dataset
            .add_columns(NewColumnTransform::AllNulls(new_schema), None, None)
            .await?;

        let field = dataset.schema().field("y").unwrap();
        assert_eq!(
            field
                .metadata
                .get(LANCE_INITIAL_DEFAULT_META_KEY)
                .map(|s| s.as_str()),
            Some("7"),
            "new field must carry its initial-default"
        );

        Ok(())
    }

    // ── (d) A→B switch: Model A allows mutation, Model B rejects ─────────────

    #[tokio::test]
    async fn test_immutability_model_a_allows_then_b_rejects() -> Result<()> {
        let mut dataset = make_dataset().await;

        // Model A: set initial-default.
        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("1"))])?
            .await?;

        // Model A: change initial-default → allowed.
        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("2"))])?
            .await
            .expect("Model A must allow changing initial-default");

        // Switch to Model B.
        enable_model_b(&mut dataset).await?;

        // Model B: try to change initial-default again → rejected.
        let result = dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("3"))])?
            .await;

        assert!(
            result.is_err(),
            "Model B must reject changing existing initial-default"
        );

        Ok(())
    }

    // ── (e) replace path changing initial-default → REJECTED (schema-carrying) ─

    #[tokio::test]
    async fn test_immutability_replace_changes_initial_default_rejected() -> Result<()> {
        let mut dataset = make_dataset().await;

        // Attach an initial-default to `x` in Model A.
        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("5"))])?
            .await?;

        enable_model_b(&mut dataset).await?;

        // replace=true that includes a DIFFERENT initial-default → rejected.
        let result = dataset
            .update_field_metadata()
            .replace("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("99"))])?
            .await;

        assert!(
            result.is_err(),
            "replace changing initial-default under Model B must be rejected"
        );

        Ok(())
    }

    // ── (f) decoded-scalar: equivalent JSON → allowed; different → rejected ───

    #[tokio::test]
    async fn test_immutability_same_json_string_is_allowed() -> Result<()> {
        let mut dataset = make_dataset().await;

        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("42"))])?
            .await?;

        enable_model_b(&mut dataset).await?;

        // Re-commit the exact same string → must NOT be rejected.
        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("42"))])?
            .await
            .expect("identical JSON string must be allowed under Model B");

        Ok(())
    }

    #[tokio::test]
    async fn test_immutability_different_scalar_rejected() -> Result<()> {
        let mut dataset = make_dataset().await;

        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("10"))])?
            .await?;

        enable_model_b(&mut dataset).await?;

        let result = dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("20"))])?
            .await;

        assert!(
            result.is_err(),
            "different scalar must be rejected under Model B"
        );

        Ok(())
    }

    // ── PK co-presence — NOT gated (both models) ──────────────────────────────

    /// initial-default field that then receives a truthy unenforced-primary-key → rejected.
    #[tokio::test]
    async fn test_pk_coprohibition_initial_default_then_pk_key_truthy() -> Result<()> {
        let mut dataset = make_dataset().await;

        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("1"))])?
            .await?;

        let result = dataset
            .update_field_metadata()
            .update("x", [("lance-schema:unenforced-primary-key", Some("true"))])?
            .await;

        assert!(
            result.is_err(),
            "adding unenforced-primary-key to a defaulted field must be rejected"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("primary-key") || msg.contains("unenforced"),
            "error must mention primary key, got: {msg}"
        );

        Ok(())
    }

    /// initial-default + :position PK key → rejected.
    #[tokio::test]
    async fn test_pk_coprohibition_initial_default_then_pk_position_key() -> Result<()> {
        let mut dataset = make_dataset().await;

        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("5"))])?
            .await?;

        let result = dataset
            .update_field_metadata()
            .update(
                "x",
                [("lance-schema:unenforced-primary-key:position", Some("1"))],
            )?
            .await;

        assert!(
            result.is_err(),
            "adding :position PK key to a defaulted field must be rejected"
        );

        Ok(())
    }

    /// Reverse order: PK field that then gets initial-default → rejected.
    #[tokio::test]
    async fn test_pk_coprohibition_pk_field_then_initial_default_rejected() -> Result<()> {
        let mut dataset = make_dataset().await;

        dataset
            .update_field_metadata()
            .update("x", [("lance-schema:unenforced-primary-key", Some("true"))])?
            .await?;

        let result = dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("0"))])?
            .await;

        assert!(
            result.is_err(),
            "adding initial-default to a PK field must be rejected"
        );

        Ok(())
    }

    /// write-default on a PK column → rejected.
    #[tokio::test]
    async fn test_pk_coprohibition_write_default_on_pk_rejected() -> Result<()> {
        let mut dataset = make_dataset().await;

        dataset
            .update_field_metadata()
            .update("x", [("lance-schema:unenforced-primary-key", Some("true"))])?
            .await?;

        let result = dataset
            .update_field_metadata()
            .update("x", [(LANCE_WRITE_DEFAULT_META_KEY, Some("0"))])?
            .await;

        assert!(
            result.is_err(),
            "adding write-default to a PK field must be rejected"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("primary-key") || msg.contains("unenforced"),
            "error must mention primary key, got: {msg}"
        );

        Ok(())
    }

    /// PK guard fires even in Model A (not gated on model flag).
    #[tokio::test]
    async fn test_pk_coprohibition_fires_in_model_a() -> Result<()> {
        let mut dataset = make_dataset().await;

        // Confirm Model A.
        assert!(
            !super::column_defaults_model_b_enabled(&dataset.manifest),
            "should be Model A"
        );

        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("1"))])?
            .await?;

        // PK annotation → must be rejected even in Model A.
        let result = dataset
            .update_field_metadata()
            .update("x", [("lance-schema:unenforced-primary-key", Some("true"))])?
            .await;

        assert!(
            result.is_err(),
            "PK co-presence guard must fire in Model A too (not gated)"
        );

        Ok(())
    }

    // ── Overwrite exemption tests ─────────────────────────────────────────────

    /// **HIGH BUG regression** (must FAIL before the fix, PASS after):
    /// A Model B dataset with an `initial-default` on a column can be
    /// overwritten with a new schema.  `Overwrite` is a full schema
    /// replacement — field-id coincidences carry no semantic meaning — so
    /// the immutability guard must NOT fire.
    #[tokio::test]
    async fn test_overwrite_exempt_from_immutability_guard() -> Result<()> {
        use crate::dataset::write::{WriteMode, WriteParams};
        use std::sync::Arc;

        let mut dataset = make_dataset().await;

        // Set initial-default on `x` in Model A, then enable Model B.
        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("10"))])?
            .await?;
        enable_model_b(&mut dataset).await?;

        // Overwrite the entire dataset with a fresh schema (same column name,
        // field ids will be reassigned from 0, but the default is not carried
        // over). This must SUCCEED — Overwrite is a full schema replacement.
        let new_schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "x",
            arrow_schema::DataType::Int32,
            true,
        )]));
        let batch = arrow_array::RecordBatch::try_new(
            new_schema.clone(),
            vec![Arc::new(arrow_array::Int32Array::from(vec![99_i32]))],
        )
        .unwrap();
        let result = Dataset::write(
            arrow_array::RecordBatchIterator::new(vec![Ok(batch)], new_schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await;

        assert!(
            result.is_ok(),
            "Overwrite on a Model B dataset must succeed (immutability does not \
             apply across a full schema replacement); got: {:?}",
            result.err()
        );

        Ok(())
    }

    /// `Overwrite` is exempt from the immutability guard, but the
    /// **PK + default co-presence** check still applies: if the new schema
    /// has a field with both `initial-default` and `unenforced-primary-key`
    /// metadata, the commit must be rejected.
    #[tokio::test]
    async fn test_overwrite_still_pk_guarded() -> Result<()> {
        use crate::dataset::write::{WriteMode, WriteParams};
        use std::sync::Arc;

        let dataset = make_dataset().await;

        // Build a schema where the column carries BOTH an initial-default and a
        // primary-key annotation — this is always invalid regardless of operation.
        // Note: PK columns must be non-nullable (nullable=false) to pass the
        // schema-level nullable check before reaching the PK+default guard.
        let field_meta: std::collections::HashMap<String, String> = [
            (LANCE_INITIAL_DEFAULT_META_KEY.to_string(), "5".to_string()),
            (
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            ),
        ]
        .into_iter()
        .collect();
        let new_schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "x",
            arrow_schema::DataType::Int32,
            false,
        )
        .with_metadata(field_meta)]));
        let batch = arrow_array::RecordBatch::try_new(
            new_schema.clone(),
            vec![Arc::new(arrow_array::Int32Array::from(vec![1_i32]))],
        )
        .unwrap();
        let result = Dataset::write(
            arrow_array::RecordBatchIterator::new(vec![Ok(batch)], new_schema),
            Arc::new(dataset),
            Some(WriteParams {
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await;

        assert!(
            result.is_err(),
            "Overwrite whose new schema has PK+default on the same field must be rejected"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("primary-key") || msg.contains("unenforced"),
            "error must mention primary key, got: {msg}"
        );

        Ok(())
    }

    /// Regression: the always-on PK + default co-presence guard must also fire on
    /// a brand-new dataset `Create` (no prior manifest).  Previously the guard
    /// only ran when a prior manifest existed, so a Create whose field carried
    /// BOTH a primary-key annotation and a column default committed successfully,
    /// breaking the documented invariant.
    #[tokio::test]
    async fn test_pk_coprohibition_fires_on_create() -> Result<()> {
        use std::sync::Arc;

        let field_meta: std::collections::HashMap<String, String> = [
            (LANCE_INITIAL_DEFAULT_META_KEY.to_string(), "7".to_string()),
            (
                "lance-schema:unenforced-primary-key".to_string(),
                "true".to_string(),
            ),
        ]
        .into_iter()
        .collect();
        let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "k",
            arrow_schema::DataType::Int32,
            false,
        )
        .with_metadata(field_meta)]));
        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(arrow_array::Int32Array::from(vec![1_i32]))],
        )
        .unwrap();

        // Brand-new dataset (Create): no prior manifest exists.
        let result = Dataset::write(
            arrow_array::RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            None,
        )
        .await;

        assert!(
            result.is_err(),
            "Create whose schema has PK+default on the same field must be rejected"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("primary-key") || msg.contains("unenforced"),
            "error must mention primary key, got: {msg}"
        );

        Ok(())
    }

    // ── Equivalent-JSON same scalar (§5 #10f) ────────────────────────────────

    /// Under Model B, re-committing an `initial-default` that decodes to the
    /// **same scalar** as the existing one — but via a different JSON encoding
    /// — must be ALLOWED.  Different JSON encoding → REJECTED.
    ///
    /// We use a `Float64` column: `"42"` and `"42.0"` decode to the same
    /// IEEE-754 value; `"43"` is genuinely different.
    #[tokio::test]
    async fn test_immutability_equivalent_json_same_scalar_allowed_different_rejected() -> Result<()>
    {
        use std::sync::Arc;

        // Create a dataset with a Float64 column.
        let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "v",
            arrow_schema::DataType::Float64,
            true,
        )]));
        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(arrow_array::Float64Array::from(vec![1.0_f64]))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            arrow_array::RecordBatchIterator::new(vec![Ok(batch)], schema),
            "memory://",
            None,
        )
        .await
        .unwrap();

        // Set initial-default as "42" (integer encoding).
        dataset
            .update_field_metadata()
            .update("v", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("42"))])?
            .await?;

        // Enable Model B.
        enable_model_b(&mut dataset).await?;

        // Re-commit with "42.0" — same scalar, different JSON string → ALLOWED.
        dataset
            .update_field_metadata()
            .update("v", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("42.0"))])?
            .await
            .expect(
                "equivalent JSON encoding (\"42\" == \"42.0\" as Float64) must be \
                 allowed under Model B",
            );

        // Now try a genuinely different value → REJECTED.
        let result = dataset
            .update_field_metadata()
            .update("v", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("43.0"))])?
            .await;

        assert!(
            result.is_err(),
            "different scalar (43.0 ≠ 42) must be rejected under Model B"
        );

        Ok(())
    }

    // ── Plain Append regression ───────────────────────────────────────────────

    /// A plain `append` of rows to a Model B dataset that has a defaulted
    /// column must SUCCEED — `Append` is not a schema-evolution operation and
    /// must not be blocked by the immutability guard.
    #[tokio::test]
    async fn test_plain_append_not_blocked_by_immutability_guard() -> Result<()> {
        use std::sync::Arc;

        let mut dataset = make_dataset().await;

        // Set initial-default on `x` and enable Model B.
        dataset
            .update_field_metadata()
            .update("x", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("7"))])?
            .await?;
        enable_model_b(&mut dataset).await?;

        // Append more rows without touching the schema.
        let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
            "x",
            arrow_schema::DataType::Int32,
            true,
        )]));
        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(arrow_array::Int32Array::from(vec![4_i32, 5, 6]))],
        )
        .unwrap();
        let result = dataset
            .append(
                arrow_array::RecordBatchIterator::new(vec![Ok(batch)], schema),
                None,
            )
            .await;

        assert!(
            result.is_ok(),
            "plain append must not be blocked by the immutability guard; got: {:?}",
            result.err()
        );

        Ok(())
    }
}

// ── Model B end-to-end behavior table tests (§3) ─────────────────────────────

#[cfg(test)]
mod model_b_e2e_tests {
    //! Integration tests that validate every row in the Model B two-value-default
    //! behavior table end-to-end, exercising read-backfill, append materialization,
    //! write-default mutability, initial-default immutability, and write-default
    //! deletion.

    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{cast::AsArray, Array, Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use lance_core::datatypes::{LANCE_INITIAL_DEFAULT_META_KEY, LANCE_WRITE_DEFAULT_META_KEY};
    use lance_file::version::LanceFileVersion;

    use super::super::schema_evolution::NewColumnTransform;
    use super::super::Dataset;
    use super::super::WriteParams;
    use crate::Result;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Enable Model B on the dataset via `update_config`.
    async fn enable_model_b(dataset: &mut Dataset) -> Result<()> {
        dataset
            .update_config([(super::COLUMN_DEFAULTS_MODEL_CONFIG_KEY, "B")])
            .await
            .map(|_| ())
    }

    /// Run a full scan projecting `["a", "c"]` in insertion order, collect into a
    /// single RecordBatch, then return the `c` column as an `Int32Array`.
    async fn scan_c(dataset: &Dataset) -> Int32Array {
        let batch = dataset
            .scan()
            .project(&["a", "c"])
            .unwrap()
            .scan_in_order(true)
            .try_into_batch()
            .await
            .unwrap();
        let col = batch.column_by_name("c").unwrap();
        col.as_primitive::<arrow_array::types::Int32Type>().clone()
    }

    // ── The eight-step Model B behavior table ────────────────────────────────

    /// Complete Model B §3 behavior table validated end-to-end.
    ///
    /// Steps (matching the spec):
    ///  1. Add column `c` with BOTH `initial-default = "1"` and `write-default = "1"`;
    ///     enable Model B; verify pre-existing rows read 1 (initial-default backfill).
    ///  2. Append omitting `c` → write-default materialised → new rows read 1.
    ///  3. Append with explicit `c = [10, NULL, 30]` → those values are preserved.
    ///  4. Update `write-default` to `"2"` (mutable) → succeeds, not blocked.
    ///  5. Pre-existing absent-`c` rows STILL read 1 (initial-default unchanged).
    ///  6. Append omitting `c` after update → new rows materialise write-default 2.
    ///  7. Attempt to change `initial-default` → REJECTED under Model B.
    ///  8. Remove `write-default`; append omitting `c` → falls back to initial-default 1.
    #[tokio::test]
    async fn test_model_b_full_behavior_table() -> Result<()> {
        // ── Setup: initial dataset (column `a` only) ────────────────────────
        // Pre-existing rows: a = [100, 200] (no `c` yet)
        let schema_a = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            true,
        )]));
        let batch_a = RecordBatch::try_new(
            schema_a.clone(),
            vec![Arc::new(Int32Array::from(vec![100_i32, 200]))],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch_a)], schema_a),
            "memory://model_b_e2e",
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_1),
                ..Default::default()
            }),
        )
        .await?;

        // ── Step 1: Add column `c` with BOTH defaults; enable Model B ────────
        // The Arrow field carries both metadata keys so add_columns (AllNulls)
        // preserves them in the committed schema.
        let c_field = ArrowField::new("c", DataType::Int32, true).with_metadata(HashMap::from([
            (LANCE_INITIAL_DEFAULT_META_KEY.to_string(), "1".to_string()),
            (LANCE_WRITE_DEFAULT_META_KEY.to_string(), "1".to_string()),
        ]));
        dataset
            .add_columns(
                NewColumnTransform::AllNulls(Arc::new(ArrowSchema::new(vec![c_field]))),
                None,
                None,
            )
            .await?;

        // Enable Model B AFTER setting the defaults so the add_columns above is
        // not subject to the immutability guard (it sets a new field id).
        enable_model_b(&mut dataset).await?;

        // Verify the metadata survived add_columns.
        {
            let field = dataset.schema().field("c").unwrap();
            assert_eq!(
                field
                    .metadata
                    .get(LANCE_INITIAL_DEFAULT_META_KEY)
                    .map(|s| s.as_str()),
                Some("1"),
                "step 1: initial-default must be '1' after add_columns"
            );
            assert_eq!(
                field
                    .metadata
                    .get(LANCE_WRITE_DEFAULT_META_KEY)
                    .map(|s| s.as_str()),
                Some("1"),
                "step 1: write-default must be '1' after add_columns"
            );
        }

        // Pre-existing rows (fragment physically lacks `c`) → read-backfill via initial-default.
        let c = scan_c(&dataset).await;
        assert_eq!(c.len(), 2, "step 1: expected 2 pre-existing rows");
        assert_eq!(c.value(0), 1, "step 1: pre-existing row 0 must read 1");
        assert_eq!(c.value(1), 1, "step 1: pre-existing row 1 must read 1");
        assert_eq!(c.null_count(), 0, "step 1: no nulls expected from backfill");

        // ── Step 2: Append omitting `c` → write-default materialised ─────────
        // New rows: a = [300, 400]; `c` is absent from the batch.
        let schema_a_only = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "a",
            DataType::Int32,
            true,
        )]));
        let batch_step2 = RecordBatch::try_new(
            schema_a_only.clone(),
            vec![Arc::new(Int32Array::from(vec![300_i32, 400]))],
        )
        .unwrap();

        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch_step2)], schema_a_only.clone()),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_1),
                    ..Default::default()
                }),
            )
            .await?;

        // Total rows: [100,200] (backfill) + [300,400] (materialised write-default)
        let c = scan_c(&dataset).await;
        assert_eq!(c.len(), 4, "step 2: expected 4 rows total");
        assert_eq!(c.value(0), 1, "step 2: row 0 still reads initial-default 1");
        assert_eq!(c.value(1), 1, "step 2: row 1 still reads initial-default 1");
        assert_eq!(
            c.value(2),
            1,
            "step 2: new row 0 materialised as write-default 1"
        );
        assert_eq!(
            c.value(3),
            1,
            "step 2: new row 1 materialised as write-default 1"
        );
        assert_eq!(c.null_count(), 0, "step 2: no nulls expected");

        // ── Step 3: Append with explicit c = [10, NULL, 30] ──────────────────
        let schema_ac = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", DataType::Int32, true),
            ArrowField::new("c", DataType::Int32, true),
        ]));
        let batch_step3 = RecordBatch::try_new(
            schema_ac.clone(),
            vec![
                Arc::new(Int32Array::from(vec![500_i32, 600, 700])),
                Arc::new(Int32Array::from(vec![Some(10_i32), None, Some(30)])),
            ],
        )
        .unwrap();

        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch_step3)], schema_ac),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_1),
                    ..Default::default()
                }),
            )
            .await?;

        let c = scan_c(&dataset).await;
        assert_eq!(c.len(), 7, "step 3: expected 7 rows total");
        assert_eq!(c.value(4), 10, "step 3: explicit 10 must be preserved");
        assert!(c.is_null(5), "step 3: explicit NULL must be preserved");
        assert_eq!(c.value(6), 30, "step 3: explicit 30 must be preserved");

        // ── Step 4: Update write-default to "2" (mutable — must succeed) ─────
        let version_before_step4 = dataset.version().version;
        dataset
            .update_field_metadata()
            .update("c", [(LANCE_WRITE_DEFAULT_META_KEY, Some("2"))])?
            .await?;

        assert!(
            dataset.version().version > version_before_step4,
            "step 4: version must advance after write-default update"
        );
        assert_eq!(
            dataset
                .schema()
                .field("c")
                .unwrap()
                .metadata
                .get(LANCE_WRITE_DEFAULT_META_KEY)
                .map(|s| s.as_str()),
            Some("2"),
            "step 4: write-default must now be '2'"
        );
        // initial-default must be unchanged.
        assert_eq!(
            dataset
                .schema()
                .field("c")
                .unwrap()
                .metadata
                .get(LANCE_INITIAL_DEFAULT_META_KEY)
                .map(|s| s.as_str()),
            Some("1"),
            "step 4: initial-default must still be '1' after write-default update"
        );

        // ── Step 5: Pre-existing absent-`c` rows still read initial-default 1 ─
        let c = scan_c(&dataset).await;
        assert_eq!(c.value(0), 1, "step 5: row 0 still reads initial-default 1");
        assert_eq!(c.value(1), 1, "step 5: row 1 still reads initial-default 1");

        // ── Step 6: Append omitting `c` after write-default update ───────────
        let batch_step6 = RecordBatch::try_new(
            schema_a_only.clone(),
            vec![Arc::new(Int32Array::from(vec![800_i32, 900]))],
        )
        .unwrap();

        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch_step6)], schema_a_only.clone()),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_1),
                    ..Default::default()
                }),
            )
            .await?;

        let c = scan_c(&dataset).await;
        assert_eq!(c.len(), 9, "step 6: expected 9 rows total");
        assert_eq!(
            c.value(7),
            2,
            "step 6: new row 0 must materialise write-default 2"
        );
        assert_eq!(
            c.value(8),
            2,
            "step 6: new row 1 must materialise write-default 2"
        );

        // ── Step 7: Attempt to change initial-default → must be REJECTED ──────
        let version_before_step7 = dataset.version().version;
        let result = dataset
            .update_field_metadata()
            .update("c", [(LANCE_INITIAL_DEFAULT_META_KEY, Some("9"))])?
            .await;

        assert!(
            result.is_err(),
            "step 7: changing initial-default under Model B must be rejected"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("initial-default") || err_msg.contains("immutable"),
            "step 7: error must mention immutability, got: {err_msg}"
        );
        assert_eq!(
            dataset.version().version,
            version_before_step7,
            "step 7: version must not advance after rejected initial-default change"
        );
        assert_eq!(
            dataset
                .schema()
                .field("c")
                .unwrap()
                .metadata
                .get(LANCE_INITIAL_DEFAULT_META_KEY)
                .map(|s| s.as_str()),
            Some("1"),
            "step 7: initial-default must remain '1' after rejected update"
        );

        // ── Step 8: Remove write-default; append omitting `c` → backfill ─────
        dataset
            .update_field_metadata()
            .update("c", [(LANCE_WRITE_DEFAULT_META_KEY, None as Option<&str>)])?
            .await?;

        assert_eq!(
            dataset
                .schema()
                .field("c")
                .unwrap()
                .metadata
                .get(LANCE_WRITE_DEFAULT_META_KEY),
            None,
            "step 8: write-default must be absent after removal"
        );

        // Append omitting `c`; no write-default → fragment absent → read-backfill (== 1).
        let batch_step8 = RecordBatch::try_new(
            schema_a_only.clone(),
            vec![Arc::new(Int32Array::from(vec![1000_i32]))],
        )
        .unwrap();

        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch_step8)], schema_a_only),
                Some(WriteParams {
                    data_storage_version: Some(LanceFileVersion::V2_1),
                    ..Default::default()
                }),
            )
            .await?;

        let c = scan_c(&dataset).await;
        assert_eq!(c.len(), 10, "step 8: expected 10 rows total");
        assert_eq!(
            c.value(9),
            1,
            "step 8: new row without write-default must fall back to initial-default 1"
        );
        // Previously materialised rows (indices 7-8) must still read 2.
        assert_eq!(
            c.value(7),
            2,
            "step 8: row 7 (stored write-default 2) must still read 2"
        );
        assert_eq!(
            c.value(8),
            2,
            "step 8: row 8 (stored write-default 2) must still read 2"
        );

        Ok(())
    }
}
