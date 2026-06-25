# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Compiled-in capability probes for the Lance binary.

These module-level booleans are resolved at import time from the native
extension module compiled alongside this Python package.  Each probe is
``True`` iff the corresponding feature is compiled into the running binary.

Callers (e.g. lance-ray, Phase 6) that need to adapt behaviour based on
whether a given Lance version supports a feature should guard their code
like this::

    try:
        from lance.capabilities import has_initial_default_backfill
    except ImportError:
        has_initial_default_backfill = False

Absence of this module (``ImportError``) means the binary pre-dates the
column-default-values feature and both probes should be treated as ``False``.

Probes
------
has_initial_default_backfill : bool
    ``True`` iff the DefaultReader read-backfill (Phase 1) is compiled in.
    When ``True``, reading a dataset whose schema has column default values
    will backfill those values for rows written before the column was added.

has_write_default_materialization : bool
    ``True`` iff write-default injection (Phase 4) is compiled in.
    When ``True``, appending or overwriting a dataset that declares column
    default values will materialise those defaults into the written batches
    automatically.
"""

from .lance import capabilities as _native_capabilities

has_initial_default_backfill: bool = _native_capabilities.has_initial_default_backfill()
has_write_default_materialization: bool = (
    _native_capabilities.has_write_default_materialization()
)

__all__ = [
    "has_initial_default_backfill",
    "has_write_default_materialization",
]
