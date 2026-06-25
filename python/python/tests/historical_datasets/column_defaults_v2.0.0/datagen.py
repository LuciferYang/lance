#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Forward-compatibility dataset for the column-default-values feature.

Run this script once (with the local feature build) to (re-)generate the
checked-in Lance dataset that ``test_column_default_compat.py`` loads.

Usage::

    cd python/
    .venv/bin/python python/tests/historical_datasets/column_defaults_v2.0.0/datagen.py

The dataset is intentionally small (3 rows).  It exercises two scenarios:

* **Column ``initial_only``** – added via ``add_column_with_default(default=42)``.
  Only ``lance-schema:initial-default`` is stored; pre-existing rows that were
  written before the column existed must back-fill to ``42``.

* **Column ``both_defaults``** – added manually with *both*
  ``lance-schema:initial-default = "7"`` *and*
  ``lance-schema:write-default = "9"`` embedded in the Arrow field metadata.
  Pre-existing rows (absent column) back-fill via the initial-default ``7``;
  future appends that omit the column would materialise the write-default
  ``9`` (not exercised here since we only need a stable on-disk fixture).

This file was generated with lance.__version__ == 2.0.0 (local feature build,
branch feat/column-default-values).  The feature was not yet released on PyPI
at generation time.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import lance
import pyarrow as pa

# ── Version assertion ──────────────────────────────────────────────────────────
GENERATED_WITH_VERSION = "2.0.0"
if lance.__version__ != GENERATED_WITH_VERSION:
    import warnings

    warnings.warn(
        f"This datagen was originally run with lance {GENERATED_WITH_VERSION!r}; "
        f"you are running with {lance.__version__!r}.  The generated dataset may "
        "differ from the checked-in fixture.",
        stacklevel=2,
    )

# ── Output path ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATASET_PATH = HERE / "dataset.lance"

# ── Remove any existing dataset first ─────────────────────────────────────────
if DATASET_PATH.exists():
    shutil.rmtree(DATASET_PATH)

# ── Step 1: write the base table (id column only) ─────────────────────────────
base_table = pa.table({"id": pa.array([1, 2, 3], type=pa.int32())})
ds = lance.write_dataset(base_table, str(DATASET_PATH))
assert ds.count_rows() == 3

# ── Step 2: add ``initial_only`` with initial-default = 42 ────────────────────
#
# add_column_with_default stores the value as ``lance-schema:initial-default``
# in the Arrow field metadata and writes physical NULLs for all existing rows.
# The DefaultReader read-path back-fills them to 42 when the column is scanned.
ds.add_column_with_default(pa.field("initial_only", pa.int32()), default=42)

# Verify the default reads back correctly for all existing rows.
result = ds.to_table().column("initial_only").to_pylist()
assert result == [42, 42, 42], f"Expected [42, 42, 42], got {result}"

# Verify the metadata key is present.
_KEY = b"lance-schema:initial-default"
field_meta = ds.schema.field("initial_only").metadata or {}
assert field_meta.get(_KEY) == b"42", (
    f"Expected initial-default metadata '42', got {field_meta.get(_KEY)!r}"
)

# ── Step 3: add ``both_defaults`` with initial=7 and write=9 ─────────────────
#
# We embed both metadata keys directly into the Arrow field so that
# add_columns (AllNulls path) carries them into the committed schema.
both_field = pa.field(
    "both_defaults",
    pa.int32(),
    metadata={
        "lance-schema:initial-default": "7",
        "lance-schema:write-default": "9",
    },
)
ds.add_columns(both_field)

# Verify that pre-existing rows back-fill to 7 (the initial-default).
result_both = ds.to_table().column("both_defaults").to_pylist()
assert result_both == [7, 7, 7], (
    f"Expected initial-default back-fill [7, 7, 7], got {result_both}"
)

# Verify both metadata keys are present.
both_meta = ds.schema.field("both_defaults").metadata or {}
assert both_meta.get(b"lance-schema:initial-default") == b"7"
assert both_meta.get(b"lance-schema:write-default") == b"9"

print(f"Dataset written to {DATASET_PATH}")
print(f"  lance version : {lance.__version__}")
print(f"  rows          : {ds.count_rows()}")
print(f"  schema        : {ds.schema}")
print("Done.")
