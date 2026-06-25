# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Model A end-to-end integration tests (Task 2.5).

Model A defines a single mutable ``initial-default`` that is applied on READ
to rows where the column is physically absent (i.e. the fragment was written
before the column existed, or was written without the column).  Rows where the
column IS physically present — including stored NULLs — are read verbatim; the
default is not substituted.

Tests
-----
1. add-with-default backfills old (absent) rows.
2. Appending a batch that includes the column preserves values and NULLs.
3. set_column_default retroactively updates the value seen for absent rows.
4. remove_column_default causes absent rows to read NULL again.
5. Appending a batch that OMITS the defaulted column (Phase-4 boundary probe).
"""

from pathlib import Path

import lance
import pyarrow as pa

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_v2_simple(path: str) -> lance.LanceDataset:
    """Write a small two-column dataset (no 'c' yet) and return it."""
    table = pa.table({"id": pa.array([1, 2, 3], type=pa.int64())})
    return lance.write_dataset(table, path)


# ---------------------------------------------------------------------------
# Test 1 — add-with-default backfills absent rows
# ---------------------------------------------------------------------------


def test_add_with_default_backfills_old_rows(tmp_path: Path):
    """Existing rows read the initial-default after add_columns(default=...).

    After calling add_columns(pa.field('c', pa.int32()), default=42):
    - Every pre-existing row's 'c' reads as 42.
    - schema.field('c').initial_default()  == 42
    - schema.field('c').write_default()    is None   (Model A — no write-default)
    - schema.field('c').effective_default() == 42
    """
    path = str(tmp_path / "ds")
    ds = _write_v2_simple(path)

    ds.add_columns(pa.field("c", pa.int32()), default=42)

    tbl = ds.to_table()
    assert tbl.column("c").to_pylist() == [42, 42, 42], (
        "Pre-existing rows must read 42 after add_columns(default=42)"
    )

    lf = ds.lance_schema.field("c")
    initial = lf.initial_default()
    assert initial is not None, "initial_default() must not be None"
    assert initial.as_py() == 42, f"Expected 42, got {initial.as_py()!r}"

    assert lf.write_default() is None, (
        "write_default() must be None in Model A (only initial-default is stored)"
    )

    effective = lf.effective_default()
    assert effective is not None, "effective_default() must not be None"
    assert effective.as_py() == 42, (
        f"Expected effective_default()==42, got {effective.as_py()!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — append WITH column c: values and NULLs are preserved
# ---------------------------------------------------------------------------


def test_append_with_column_preserves_values_and_nulls(tmp_path: Path):
    """Appending a batch that includes 'c' stores those values verbatim.

    After adding 'c' with default=42 and then appending rows where
    c = [10, None, 30]:
    - The original rows (absent fragment) still read 42.
    - The appended rows read 10, NULL, 30 exactly — the stored NULL is NOT
      replaced by the default.
    """
    path = str(tmp_path / "ds")
    ds = _write_v2_simple(path)
    ds.add_columns(pa.field("c", pa.int32()), default=42)

    # Append new rows that physically include 'c', including a stored NULL.
    new_rows = pa.table(
        {
            "id": pa.array([4, 5, 6], type=pa.int64()),
            "c": pa.array([10, None, 30], type=pa.int32()),
        }
    )
    lance.write_dataset(new_rows, path, mode="append")

    ds = lance.dataset(path)
    tbl = ds.to_table()

    ids = tbl.column("id").to_pylist()
    c_vals = tbl.column("c").to_pylist()

    # Determine ordering (Lance may reorder fragments but guarantees id/c align).
    id_to_c = dict(zip(ids, c_vals))

    # Original rows: column physically absent — default applies.
    for row_id in (1, 2, 3):
        assert id_to_c[row_id] == 42, (
            f"Row id={row_id} (absent fragment) should read 42, got {id_to_c[row_id]!r}"
        )

    # Appended rows: column physically present — read verbatim.
    assert id_to_c[4] == 10, f"Row id=4 should read 10, got {id_to_c[4]!r}"
    assert id_to_c[5] is None, (
        f"Row id=5 has stored NULL — must read NULL (not default), got {id_to_c[5]!r}"
    )
    assert id_to_c[6] == 30, f"Row id=6 should read 30, got {id_to_c[6]!r}"


# ---------------------------------------------------------------------------
# Test 3 — set_column_default is retroactive for absent rows only
# ---------------------------------------------------------------------------


def test_set_column_default_retroactive_for_absent_rows_only(tmp_path: Path):
    """set_column_default(c, 99) retroactively changes the read value for absent rows.

    After the setup in test 2 (original rows absent, appended rows present):
    - set_column_default('c', 99)
    - Original rows (absent fragment) now read 99.
    - Appended rows (present fragment: 10, NULL, 30) are unchanged.
    - schema.field('c').initial_default() == 99.
    """
    path = str(tmp_path / "ds")
    ds = _write_v2_simple(path)
    ds.add_columns(pa.field("c", pa.int32()), default=42)

    new_rows = pa.table(
        {
            "id": pa.array([4, 5, 6], type=pa.int64()),
            "c": pa.array([10, None, 30], type=pa.int32()),
        }
    )
    lance.write_dataset(new_rows, path, mode="append")

    ds = lance.dataset(path)
    ds.set_column_default("c", 99)

    tbl = ds.to_table()
    ids = tbl.column("id").to_pylist()
    c_vals = tbl.column("c").to_pylist()
    id_to_c = dict(zip(ids, c_vals))

    # Absent rows now read the updated default.
    for row_id in (1, 2, 3):
        assert id_to_c[row_id] == 99, (
            f"Row id={row_id} (absent) should read 99 after set_column_default, "
            f"got {id_to_c[row_id]!r}"
        )

    # Present rows are unaffected by the default change.
    assert id_to_c[4] == 10, (
        f"Row id=4 (present) should still read 10, got {id_to_c[4]!r}"
    )
    assert id_to_c[5] is None, (
        f"Row id=5 (stored NULL) should still read NULL, got {id_to_c[5]!r}"
    )
    assert id_to_c[6] == 30, (
        f"Row id=6 (present) should still read 30, got {id_to_c[6]!r}"
    )

    # Accessor reflects the updated default.
    lf = ds.lance_schema.field("c")
    initial = lf.initial_default()
    assert initial is not None
    assert initial.as_py() == 99, (
        f"initial_default() should be 99, got {initial.as_py()!r}"
    )


# ---------------------------------------------------------------------------
# Test 4 — remove_column_default: absent rows read NULL again
# ---------------------------------------------------------------------------


def test_remove_column_default_absent_rows_read_null(tmp_path: Path):
    """remove_column_default causes absent rows to read NULL; present rows unchanged.

    After the setup in tests 2–3 (set to 99), calling remove_column_default:
    - Original rows (absent fragment) revert to NULL.
    - Appended rows (present fragment: 10, NULL, 30) are unaffected.
    - schema.field('c').initial_default() is None.
    """
    path = str(tmp_path / "ds")
    ds = _write_v2_simple(path)
    ds.add_columns(pa.field("c", pa.int32()), default=42)

    new_rows = pa.table(
        {
            "id": pa.array([4, 5, 6], type=pa.int64()),
            "c": pa.array([10, None, 30], type=pa.int32()),
        }
    )
    lance.write_dataset(new_rows, path, mode="append")

    ds = lance.dataset(path)
    ds.set_column_default("c", 99)
    ds.remove_column_default("c")

    tbl = ds.to_table()
    ids = tbl.column("id").to_pylist()
    c_vals = tbl.column("c").to_pylist()
    id_to_c = dict(zip(ids, c_vals))

    # Absent rows revert to NULL once the default is removed.
    for row_id in (1, 2, 3):
        assert id_to_c[row_id] is None, (
            f"Row id={row_id} (absent, no default) should read NULL, "
            f"got {id_to_c[row_id]!r}"
        )

    # Present rows are unaffected.
    assert id_to_c[4] == 10, (
        f"Row id=4 (present) should still read 10, got {id_to_c[4]!r}"
    )
    assert id_to_c[5] is None, (
        f"Row id=5 (stored NULL) should still read NULL, got {id_to_c[5]!r}"
    )
    assert id_to_c[6] == 30, (
        f"Row id=6 (present) should still read 30, got {id_to_c[6]!r}"
    )

    lf = ds.lance_schema.field("c")
    assert lf.initial_default() is None, (
        "initial_default() must be None after remove_column_default"
    )


# ---------------------------------------------------------------------------
# Test 5 — "append omitting the column" Phase-4 boundary probe
# ---------------------------------------------------------------------------


def test_append_omitting_defaulted_column_is_phase4(tmp_path: Path):
    """Document the current behavior when a batch is appended WITHOUT column 'c'.

    NOTE: Empirically, as of this task, the write path already handles the case
    where a batch omits a column that has an initial-default: the append
    SUCCEEDS and the omitted rows read the current initial-default on subsequent
    reads.  This is Model A behavior working correctly — the new fragment is
    treated as an 'absent' fragment, so the default applies to all its rows.

    If this behavior changes (e.g. the writer is tightened to require all
    columns), update this test and the comment above accordingly.

    Historical note: The original task spec anticipated that this path might
    raise (deferring to a Phase-4 write path).  This test documents/locks the
    ACTUAL observed behavior.  If Phase 4 later changes the semantics
    (e.g. write-default materialization), update the assertions here.
    """
    path = str(tmp_path / "ds")
    ds = _write_v2_simple(path)
    ds.add_columns(pa.field("c", pa.int32()), default=42)

    # Append rows that do NOT include 'c' at all.
    new_rows = pa.table({"id": pa.array([4, 5], type=pa.int64())})

    # CURRENT BEHAVIOR: the append succeeds.
    # The omitted fragment is treated as "absent" — rows 4 and 5 read the
    # current initial-default (42) on subsequent reads.
    lance.write_dataset(new_rows, path, mode="append")

    ds2 = lance.dataset(path)
    tbl = ds2.to_table()
    ids = tbl.column("id").to_pylist()
    c_vals = tbl.column("c").to_pylist()
    id_to_c = dict(zip(ids, c_vals))

    # Original absent rows still read 42.
    for row_id in (1, 2, 3):
        assert id_to_c[row_id] == 42, (
            f"Row id={row_id} (original absent) should read 42, got {id_to_c[row_id]!r}"
        )

    # Newly-appended rows (also absent) also read 42.
    for row_id in (4, 5):
        assert id_to_c[row_id] == 42, (
            f"Row id={row_id} (appended, column omitted) should read 42, "
            f"got {id_to_c[row_id]!r}"
        )

    # Verify that changing the default retroactively affects the omitted rows too,
    # confirming they are stored as "absent" (not as NULLs).
    ds2.set_column_default("c", 77)
    tbl2 = ds2.to_table()
    ids2 = tbl2.column("id").to_pylist()
    c_vals2 = tbl2.column("c").to_pylist()
    id_to_c2 = dict(zip(ids2, c_vals2))

    for row_id in (1, 2, 3, 4, 5):
        assert id_to_c2[row_id] == 77, (
            f"Row id={row_id} (absent) should read 77 after set_column_default(77), "
            f"got {id_to_c2[row_id]!r}"
        )


def test_append_omitting_non_nullable_defaulted_column(tmp_path: Path):
    """A NON-nullable column carrying only an initial-default may be omitted on Append.

    Mirrors ``test_append_omitting_defaulted_column_is_phase4`` but with
    ``nullable=False``.  Previously the append was rejected with a SchemaMismatch
    even though the read path backfills the non-null initial-default for the
    structurally-absent column.  Per Model A ("a column with no write-default is
    left structurally absent, read-backfills initial") and "a required column
    with a default IS supported", the append must succeed.
    """
    path = str(tmp_path / "ds")
    ds = _write_v2_simple(path)
    ds.add_columns(pa.field("c", pa.int32(), nullable=False), default=42)

    # Append rows that do NOT include 'c' at all — must succeed.
    new_rows = pa.table({"id": pa.array([4, 5], type=pa.int64())})
    lance.write_dataset(new_rows, path, mode="append")

    ds2 = lance.dataset(path)
    tbl = ds2.to_table()
    id_to_c = dict(zip(tbl.column("id").to_pylist(), tbl.column("c").to_pylist()))

    # All rows (original absent + newly appended absent) read the initial-default 42.
    for row_id in (1, 2, 3, 4, 5):
        assert id_to_c[row_id] == 42, (
            f"Row id={row_id} (absent) should read 42, got {id_to_c[row_id]!r}"
        )

    # The default is still mutable for the absent rows.
    ds2.set_column_default("c", 77)
    tbl2 = ds2.to_table()
    id_to_c2 = dict(zip(tbl2.column("id").to_pylist(), tbl2.column("c").to_pylist()))
    for row_id in (1, 2, 3, 4, 5):
        assert id_to_c2[row_id] == 77, (
            f"Row id={row_id} (absent) should read 77 after set_column_default(77), "
            f"got {id_to_c2[row_id]!r}"
        )
