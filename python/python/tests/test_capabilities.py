# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for lance.capabilities capability probes (Task 5.1)."""

import lance


def test_capabilities_module_importable():
    """lance.capabilities can be imported from the top-level lance namespace."""
    import lance.capabilities  # noqa: F401 (import for side-effect / smoke test)


def test_top_level_import():
    """Both probes are importable directly from lance.capabilities."""
    from lance.capabilities import (
        has_initial_default_backfill,
        has_write_default_materialization,
    )
    # Basic sanity: names exist in the module namespace
    assert has_initial_default_backfill is not None
    assert has_write_default_materialization is not None


def test_probes_are_bool():
    """Both probes are plain Python bool values."""
    from lance.capabilities import (
        has_initial_default_backfill,
        has_write_default_materialization,
    )
    assert isinstance(has_initial_default_backfill, bool)
    assert isinstance(has_write_default_materialization, bool)


def test_probes_are_true_in_this_build():
    """Both capabilities are compiled into this binary, so both probes are True."""
    from lance.capabilities import (
        has_initial_default_backfill,
        has_write_default_materialization,
    )
    assert has_initial_default_backfill is True
    assert has_write_default_materialization is True


def test_native_submodule_accessible_via_lance_dot_capabilities():
    """lance.capabilities is also accessible as an attribute on the lance package."""
    # Importing lance.capabilities makes it available as lance.capabilities
    import lance.capabilities  # noqa: F401
    assert hasattr(lance, "capabilities")


def test_native_submodule_attributes():
    """The underlying native submodule also exposes the probe functions."""
    # The Rust capabilities submodule is accessible from the native extension.
    from lance.lance import capabilities as _native

    assert hasattr(_native, "has_initial_default_backfill")
    assert hasattr(_native, "has_write_default_materialization")
    assert _native.has_initial_default_backfill() is True
    assert _native.has_write_default_materialization() is True
