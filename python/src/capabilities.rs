// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compiled-in capability probes for lance.capabilities.
//!
//! These booleans are compiled into the native binary at build time, so their
//! truthfulness is guaranteed by the binary itself.  An older pylance binary
//! that does not include this module will not have these attributes, and callers
//! (e.g., lance-ray) should treat their absence as `False`.

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// `True` iff the DefaultReader read-backfill (Phase 1) is compiled into this
/// binary.  When `True`, reading a dataset whose schema has column default values
/// will backfill those values for rows that pre-date the column's addition.
#[pyfunction]
pub fn has_initial_default_backfill() -> bool {
    true
}

/// `True` iff the write-default materialization (Phase 4) is compiled into this
/// binary.  When `True`, appending or overwriting a dataset that has column
/// default values will inject those defaults into the written data automatically.
#[pyfunction]
pub fn has_write_default_materialization() -> bool {
    true
}

/// Register the `capabilities` submodule on the top-level `lance` module.
pub fn register_capabilities(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let capabilities = PyModule::new(py, "capabilities")?;
    capabilities.add_wrapped(wrap_pyfunction!(has_initial_default_backfill))?;
    capabilities.add_wrapped(wrap_pyfunction!(has_write_default_materialization))?;
    m.add_submodule(&capabilities)?;
    Ok(())
}
