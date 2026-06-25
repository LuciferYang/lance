# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for the add_columns(default=) and add_column_with_default APIs (Task 2.3)."""

from pathlib import Path

import lance
import pyarrow as pa
import pytest

_LANCE_INITIAL_DEFAULT_META_KEY = "lance-schema:initial-default"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_simple(tmp_path: Path) -> lance.LanceDataset:
    """Write a single-column int64 dataset to *tmp_path*."""
    table = pa.table({"id": pa.array([1, 2, 3], type=pa.int64())})
    return lance.write_dataset(table, tmp_path)


# ---------------------------------------------------------------------------
# Core happy-path tests
# ---------------------------------------------------------------------------


def test_add_columns_with_default_int_reads_back(tmp_path: Path):
    """add_columns(pa.field(...), default=7) causes pre-existing rows to read 7."""
    ds = _write_simple(tmp_path)

    ds.add_columns(pa.field("score", pa.int32()), default=7)

    tbl = ds.to_table()
    assert tbl.schema.field("score").type == pa.int32()
    score_col = tbl.column("score")
    # Every pre-existing row must read back the default value.
    assert score_col.to_pylist() == [7, 7, 7]


def test_add_columns_with_default_stores_metadata(tmp_path: Path):
    """The lance-schema:initial-default metadata key is persisted on the field."""
    ds = _write_simple(tmp_path)

    ds.add_columns(pa.field("score", pa.int32()), default=7)

    field = ds.schema.field("score")
    meta = field.metadata or {}
    key = _LANCE_INITIAL_DEFAULT_META_KEY.encode()
    assert key in meta, f"Expected metadata key {_LANCE_INITIAL_DEFAULT_META_KEY!r}"
    assert meta[key] == b"7"


def test_add_column_with_default_convenience_reads_back(tmp_path: Path):
    """add_column_with_default is a correct alias for the metadata-only path."""
    ds = _write_simple(tmp_path)

    ds.add_column_with_default(pa.field("label", pa.utf8()), default="unknown")

    tbl = ds.to_table()
    assert tbl.column("label").to_pylist() == ["unknown", "unknown", "unknown"]


def test_add_columns_field_list_with_default(tmp_path: Path):
    """add_columns with a list of pa.Field plus default= works correctly."""
    ds = _write_simple(tmp_path)

    # Passing a single-element list — the schema path.
    ds.add_columns([pa.field("flag", pa.bool_())], default=True)

    tbl = ds.to_table()
    assert tbl.column("flag").to_pylist() == [True, True, True]


def test_add_columns_schema_with_default_single_field(tmp_path: Path):
    """add_columns with a pa.Schema and a single field encodes the default."""
    ds = _write_simple(tmp_path)

    ds.add_columns(pa.schema([pa.field("val", pa.float32())]), default=3.14)

    tbl = ds.to_table()
    result = tbl.column("val").to_pylist()
    assert all(abs(v - 3.14) < 1e-5 for v in result)


def test_add_columns_no_default_still_works(tmp_path: Path):
    """add_columns without default= still works (null backfill)."""
    ds = _write_simple(tmp_path)

    ds.add_columns(pa.field("extra", pa.int32()))

    tbl = ds.to_table()
    # Without a default the column backfills as null.
    assert tbl.column("extra").to_pylist() == [None, None, None]


# ---------------------------------------------------------------------------
# Rejection tests
# ---------------------------------------------------------------------------


def test_add_columns_default_on_dict_raises(tmp_path: Path):
    """default= on the dict/SQL transform route raises ValueError."""
    ds = _write_simple(tmp_path)

    with pytest.raises(ValueError, match="default="):
        ds.add_columns({"computed": "id * 2"}, default=0)


def test_add_columns_default_on_udf_raises(tmp_path: Path):
    """default= on a BatchUDF raises ValueError."""
    ds = _write_simple(tmp_path)

    @lance.batch_udf()
    def make_col(batch):
        import pandas as pd

        return pd.DataFrame({"doubled": batch["id"].to_pylist()})

    with pytest.raises(ValueError, match="default="):
        ds.add_columns(make_col, default=0)


def test_add_columns_invalid_default_type_raises(tmp_path: Path):
    """Encoding an incompatible default value raises ValueError."""
    ds = _write_simple(tmp_path)

    # "not_an_int" cannot be cast to int32.
    with pytest.raises((ValueError, pa.lib.ArrowInvalid)):
        ds.add_columns(pa.field("bad", pa.int32()), default="not_an_int")


def test_add_columns_unsupported_type_raises(tmp_path: Path):
    """Encoding a default for an unsupported type (struct) raises ValueError."""
    ds = _write_simple(tmp_path)

    struct_field = pa.field(
        "nested", pa.struct([pa.field("a", pa.int32())]), nullable=True
    )
    with pytest.raises((ValueError, pa.lib.ArrowInvalid, Exception)):
        ds.add_columns(struct_field, default={"a": 1})


# ---------------------------------------------------------------------------
# Encode helper round-trip (pure Python)
# ---------------------------------------------------------------------------


def test_encode_default_value_int():
    """_encode_default_value produces correct JSON for an int32 scalar."""
    from lance.lance import _encode_default_value

    arr = pa.array([42], type=pa.int32())
    assert _encode_default_value(arr) == "42"


def test_encode_default_value_string():
    """_encode_default_value produces correct JSON for a utf8 scalar."""
    from lance.lance import _encode_default_value

    arr = pa.array(["hello"], type=pa.utf8())
    assert _encode_default_value(arr) == '"hello"'


def test_encode_default_value_float():
    """_encode_default_value produces correct JSON for a float64 scalar."""
    from lance.lance import _encode_default_value

    arr = pa.array([1.5], type=pa.float64())
    assert _encode_default_value(arr) == "1.5"


def test_encode_default_value_bool():
    """_encode_default_value produces correct JSON for a boolean scalar."""
    from lance.lance import _encode_default_value

    arr = pa.array([True], type=pa.bool_())
    assert _encode_default_value(arr) == "true"


def test_encode_default_value_empty_raises():
    """_encode_default_value raises ValueError for an empty array."""
    from lance.lance import _encode_default_value

    arr = pa.array([], type=pa.int32())
    with pytest.raises(ValueError):
        _encode_default_value(arr)


# ---------------------------------------------------------------------------
# Required (non-nullable) column tests
# ---------------------------------------------------------------------------


def test_add_required_column_with_default_succeeds(tmp_path: Path):
    """add_column_with_default(nullable=False, default=42) succeeds.

    - Existing rows read back 42.
    - null_count is 0.
    - The schema field is non-nullable.
    """
    ds = _write_simple(tmp_path)

    ds.add_column_with_default(pa.field("c", pa.int32(), nullable=False), default=42)

    tbl = ds.to_table()
    c_col = tbl.column("c")
    assert c_col.to_pylist() == [42, 42, 42], (
        f"Expected all 42, got {c_col.to_pylist()}"
    )
    assert c_col.null_count == 0, f"null_count={c_col.null_count}"
    schema_field = tbl.schema.field("c")
    assert not schema_field.nullable, "Schema field 'c' must be non-nullable"


def test_add_required_column_no_default_raises(tmp_path: Path):
    """add_columns(nullable=False) without a default raises a clear error."""
    ds = _write_simple(tmp_path)

    with pytest.raises(Exception, match="non-nullable|initial-default|required"):
        # No default supplied — the Rust core must reject this.
        ds.add_columns(pa.field("c", pa.int32(), nullable=False))


def test_append_providing_required_column_works(tmp_path: Path):
    """Appending a batch that explicitly provides the non-nullable column succeeds."""
    ds = _write_simple(tmp_path)
    ds.add_column_with_default(pa.field("c", pa.int32(), nullable=False), default=42)

    # Append a new batch that explicitly includes the 'c' column.
    new_batch = pa.table(
        {
            "id": pa.array([4, 5], type=pa.int64()),
            "c": pa.array([10, 20], type=pa.int32()),
        }
    )
    lance.write_dataset(new_batch, tmp_path, mode="append")

    # Re-open the dataset to pick up the new version.
    ds2 = lance.dataset(tmp_path)
    tbl = ds2.to_table()
    c_col = tbl.column("c")
    # Original 3 rows → 42; new 2 rows → 10, 20
    assert c_col.to_pylist() == [42, 42, 42, 10, 20]
    assert c_col.null_count == 0
