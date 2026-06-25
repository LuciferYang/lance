# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Forward-compatibility test for the column-default-values feature.

This test loads a small Lance dataset checked into the repository under
``historical_datasets/column_defaults_v2.0.0/`` and asserts that:

* The ``initial_only`` column (added via ``add_column_with_default(default=42)``)
  reads back ``42`` for every pre-existing row — proving that the
  ``lance-schema:initial-default`` metadata key is stable and readable by this
  build.

* The ``both_defaults`` column (added with both ``lance-schema:initial-default=7``
  and ``lance-schema:write-default=9`` in field metadata) also reads back ``7``
  for every pre-existing row — the initial-default is used for back-fill.

Design note:  Old readers (without the DefaultReader read-path) would return
NULL for both columns instead of the default.  This is expected — those readers
simply ignore the unknown metadata keys.  This test verifies that **this** build
reads them correctly.

The dataset was generated with:
    python python/tests/historical_datasets/column_defaults_v2.0.0/datagen.py
(lance 2.0.0, branch feat/column-default-values, local feature build).
"""

from pathlib import Path

import lance
import pyarrow as pa

# Path to the checked-in dataset (relative to this test file).
_DATASET_DIR = (
    Path(__file__).parent
    / "historical_datasets"
    / "column_defaults_v2.0.0"
    / "dataset.lance"
)

_KEY_INITIAL = b"lance-schema:initial-default"
_KEY_WRITE = b"lance-schema:write-default"


def test_compat_dataset_exists():
    """The checked-in dataset directory must be present."""
    assert _DATASET_DIR.exists(), (
        f"Checked-in dataset not found at {_DATASET_DIR}.\n"
        "Re-run python/python/tests/historical_datasets/column_defaults_v2.0.0/datagen.py"
        " to regenerate it."
    )


def test_compat_initial_default_reads_back_42():
    """Pre-existing rows for ``initial_only`` column must read back 42.

    The column was added via ``add_column_with_default(pa.field('initial_only',
    pa.int32()), default=42)`` which stores only ``lance-schema:initial-default``
    in field metadata.  The DefaultReader back-fill returns 42 for rows written
    before the column existed.
    """
    ds = lance.dataset(_DATASET_DIR)
    result = ds.to_table(columns=["initial_only"]).column("initial_only").to_pylist()
    assert result == [42, 42, 42], (
        f"Expected initial_only to read [42, 42, 42] from checked-in dataset, got {result}"
    )


def test_compat_initial_default_metadata_key_present():
    """The ``lance-schema:initial-default`` metadata key must be present on ``initial_only``."""
    ds = lance.dataset(_DATASET_DIR)
    meta = ds.schema.field("initial_only").metadata or {}
    assert meta.get(_KEY_INITIAL) == b"42", (
        f"Expected initial-default metadata '42', got {meta.get(_KEY_INITIAL)!r}"
    )


def test_compat_both_defaults_reads_back_initial():
    """Pre-existing rows for ``both_defaults`` column must read back 7 (initial-default).

    The column carries both ``lance-schema:initial-default = 7`` and
    ``lance-schema:write-default = 9``.  Pre-existing rows (written before the
    column existed) are back-filled using the *initial* default only.
    """
    ds = lance.dataset(_DATASET_DIR)
    result = ds.to_table(columns=["both_defaults"]).column("both_defaults").to_pylist()
    assert result == [7, 7, 7], (
        f"Expected both_defaults to read [7, 7, 7] (initial-default backfill), got {result}"
    )


def test_compat_both_defaults_metadata_keys():
    """Both metadata keys must be present and correct on ``both_defaults``."""
    ds = lance.dataset(_DATASET_DIR)
    meta = ds.schema.field("both_defaults").metadata or {}
    assert meta.get(_KEY_INITIAL) == b"7", (
        f"Expected initial-default '7', got {meta.get(_KEY_INITIAL)!r}"
    )
    assert meta.get(_KEY_WRITE) == b"9", (
        f"Expected write-default '9', got {meta.get(_KEY_WRITE)!r}"
    )


def test_compat_row_count_and_schema():
    """The dataset must have 3 rows and the expected schema."""
    ds = lance.dataset(_DATASET_DIR)
    assert ds.count_rows() == 3

    schema = ds.schema
    assert schema.field("id").type == pa.int32()
    assert schema.field("initial_only").type == pa.int32()
    assert schema.field("both_defaults").type == pa.int32()
