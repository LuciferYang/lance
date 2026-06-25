// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Column-default invariants checked while a transaction builds its manifest.
//!
//! The checks live here rather than in [`super::validate`] because they compare
//! the *post-commit* schema against the manifest being replaced, and the
//! post-commit schema only exists once [`super::Transaction::build_manifest`]
//! has applied every schema mutation the operation carries.

use crate::format::Manifest;
use lance_core::datatypes::{
    LANCE_INITIAL_DEFAULT_META_KEY, LANCE_WRITE_DEFAULT_META_KEY, Schema, decode_default,
};
use lance_core::{Error, Result};

/// Manifest table-config key that selects the column-defaults model.
///
/// * **Absent** (or any value other than `"B"`) → **Model A**: single mutable
///   default stored as `lance-schema:initial-default` in field metadata.
/// * **`"B"`** → **Model B**: two-value defaults (initial-default +
///   write-default). The immutability guard in
///   [`enforce_default_constraints`] is gated on this flag.
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
/// When `true`, manifest building enforces that the
/// `lance-schema:initial-default` metadata key on any existing field id is
/// immutable — it cannot be changed or dropped once set.
pub fn column_defaults_model_b_enabled(manifest: &Manifest) -> bool {
    manifest
        .config
        .get(COLUMN_DEFAULTS_MODEL_CONFIG_KEY)
        .map(|v| v == COLUMN_DEFAULTS_MODEL_B_VALUE)
        .unwrap_or(false)
}

/// Enforce column-default constraints on the post-commit schema.
///
/// Two invariants are checked (at the single point where both the prior
/// manifest and the post-commit schema are in scope):
///
/// 1. **`initial-default` immutability** (gated on Model B + `apply_immutability`):
///    If [`column_defaults_model_b_enabled`] is true for `prior_manifest` AND
///    `apply_immutability` is true, any field id that existed in the prior
///    schema and carried an `initial-default` must carry the *same decoded
///    scalar* in the post-commit schema.  Dropping or changing the value is
///    rejected.  Newly added field ids may set `initial-default` freely.
///
///    Pass `apply_immutability = false` for operations that replace the schema
///    wholesale (`Overwrite`, `Restore`) rather than evolving existing column
///    identities in place.  `Overwrite` assigns fresh field ids from 0, so a
///    coincident id is not semantically the same column; the immutability
///    guarantee does not cross an `Overwrite`/`Restore` boundary.
///
/// 2. **PK + default co-presence** (always-on, both models):
///    Any field in the post-commit schema that simultaneously carries a
///    default key (`initial-default` or `write-default`) AND is an
///    unenforced primary key (`is_unenforced_pk_raw()`) is rejected,
///    regardless of `apply_immutability`.
pub(super) fn enforce_default_constraints(
    prior_manifest: &Manifest,
    post_schema: &Schema,
    apply_immutability: bool,
) -> Result<()> {
    let model_b = column_defaults_model_b_enabled(prior_manifest);
    let prior_schema = &prior_manifest.schema;

    // ── (1) initial-default immutability — gated on Model B ───────────────
    // Only applies to in-place schema-evolution operations (UpdateConfig,
    // Merge, Project) where column identity (field id) is preserved across
    // the commit.  Overwrite and Restore are full replacements and are
    // exempt (`apply_immutability == false`).
    if model_b && apply_immutability {
        for post_field in post_schema.fields_pre_order() {
            let Some(prior_field) = prior_schema.field_by_id(post_field.id) else {
                // Newly added field — allowed to set initial-default freely.
                continue;
            };
            let prior_initial = prior_field
                .metadata
                .get(LANCE_INITIAL_DEFAULT_META_KEY)
                .cloned();
            let post_initial = post_field
                .metadata
                .get(LANCE_INITIAL_DEFAULT_META_KEY)
                .cloned();

            match (prior_initial, post_initial) {
                // Prior had no initial-default → no constraint on the post state.
                (None, _) => {}
                // Prior had initial-default, post drops it → rejected.
                (Some(prior_json), None) => {
                    return Err(Error::invalid_input(format!(
                        "field '{}' (id={}) had an initial-default ({prior_json:?}) but \
                         it was dropped in this transaction; \
                         initial-default is immutable once set (Model B active)",
                        post_field.name, post_field.id,
                    )));
                }
                // Both present — compare decoded scalars; forbid a different value.
                (Some(prior_json), Some(post_json)) => {
                    // Same string → definitely the same value (fast path).
                    if prior_json == post_json {
                        continue;
                    }
                    // Different strings — decode and compare scalars to allow
                    // equivalent re-encodings (e.g. "42" vs "42.0").
                    let dt = post_field.data_type();
                    let prior_arr = decode_default(&prior_json, &dt).map_err(|e| {
                        Error::invalid_input(format!(
                            "field '{}' (id={}): could not decode prior initial-default \
                             {prior_json:?}: {e}",
                            post_field.name, post_field.id,
                        ))
                    })?;
                    let post_arr = decode_default(&post_json, &dt).map_err(|e| {
                        Error::invalid_input(format!(
                            "field '{}' (id={}): could not decode new initial-default \
                             {post_json:?}: {e}",
                            post_field.name, post_field.id,
                        ))
                    })?;
                    if prior_arr.as_ref() != post_arr.as_ref() {
                        return Err(Error::invalid_input(format!(
                            "field '{}' (id={}) initial-default changed from \
                             {prior_json:?} to {post_json:?}; \
                             initial-default is immutable once set (Model B active)",
                            post_field.name, post_field.id,
                        )));
                    }
                }
            }
        }
    }

    // ── (2) PK + default co-presence — always-on ──────────────────────────
    enforce_pk_default_co_presence(post_schema)?;

    Ok(())
}

/// Reject any field that is simultaneously an unenforced primary key and
/// carries a column default (`initial-default` or `write-default`).
///
/// This invariant is always-on and independent of any prior manifest, so it
/// must run for brand-new dataset `Create` as well as in-place evolution.
pub(super) fn enforce_pk_default_co_presence(post_schema: &Schema) -> Result<()> {
    for post_field in post_schema.fields_pre_order() {
        let has_initial = post_field
            .metadata
            .contains_key(LANCE_INITIAL_DEFAULT_META_KEY);
        let has_write = post_field
            .metadata
            .contains_key(LANCE_WRITE_DEFAULT_META_KEY);
        if (has_initial || has_write) && post_field.is_unenforced_pk_raw() {
            let which = if has_initial && has_write {
                format!(
                    "both '{LANCE_INITIAL_DEFAULT_META_KEY}' and '{LANCE_WRITE_DEFAULT_META_KEY}'"
                )
            } else if has_initial {
                format!("'{LANCE_INITIAL_DEFAULT_META_KEY}'")
            } else {
                format!("'{LANCE_WRITE_DEFAULT_META_KEY}'")
            };
            return Err(Error::invalid_input(format!(
                "field '{}' (id={}) is an unenforced primary-key column and cannot \
                 also carry a column default ({}); \
                 remove the default or the primary-key designation before committing",
                post_field.name, post_field.id, which,
            )));
        }
    }

    Ok(())
}
