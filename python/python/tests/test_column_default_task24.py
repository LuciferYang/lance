# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for Task 2.4: typed set/remove column default + two-value read accessors.

Part A: LanceDataset.set_column_default / remove_column_default
Part B: LanceField.initial_default / write_default / effective_default
        plus _decode_default_value helper.
"""

from pathlib import Path

import lance
import pyarrow as pa
import pytest

_LANCE_INITIAL_DEFAULT_META_KEY = "lance-schema:initial-default"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_simple(tmp_path: Path) -> lance.LanceDataset:
    """Write a simple int32+utf8 dataset to *tmp_path* and return it."""
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int32()),
            "tag": pa.array(["a", "b", "c"], type=pa.utf8()),
        }
    )
    return lance.write_dataset(table, tmp_path)


def _lance_field(ds: lance.LanceDataset, name: str):
    """Return the LanceField for *name* from the lance schema of *ds*."""
    from lance.lance import LanceSchema

    return ds.lance_schema.field(name)


# ---------------------------------------------------------------------------
# Part A — set_column_default
# ---------------------------------------------------------------------------


class TestSetColumnDefault:
    def test_set_int_default_reads_back(self, tmp_path: Path):
        """set_column_default(col, 5) causes absent rows to read 5."""
        ds = _write_simple(tmp_path)
        # Add a new column without a default first (reads NULL).
        ds.add_columns(pa.field("score", pa.int32()))
        assert ds.to_table().column("score").to_pylist() == [None, None, None]

        # Now set a default.
        ds.set_column_default("score", 5)

        result = ds.to_table().column("score").to_pylist()
        assert result == [5, 5, 5], f"Expected [5, 5, 5], got {result}"

    def test_set_default_metadata_key_present(self, tmp_path: Path):
        """After set_column_default the metadata key is present in field.metadata."""
        ds = _write_simple(tmp_path)
        ds.add_columns(pa.field("score", pa.int32()))
        ds.set_column_default("score", 5)

        # Check via PyArrow schema (the pa.Schema field has bytes metadata keys).
        arrow_field = ds.schema.field("score")
        meta = arrow_field.metadata or {}
        key = _LANCE_INITIAL_DEFAULT_META_KEY.encode()
        assert key in meta, "initial-default key must be present in Arrow field metadata"
        assert meta[key] == b"5"

    def test_set_then_remove_default(self, tmp_path: Path):
        """set then remove_column_default: key gone, column reads NULL again."""
        ds = _write_simple(tmp_path)
        ds.add_columns(pa.field("score", pa.int32()))
        ds.set_column_default("score", 7)

        # Verify set worked.
        arrow_field = ds.schema.field("score")
        key = _LANCE_INITIAL_DEFAULT_META_KEY.encode()
        assert (arrow_field.metadata or {}).get(key) == b"7"

        # Remove the default.
        ds.remove_column_default("score")

        # Key must be absent.
        arrow_field_after = ds.schema.field("score")
        assert key not in (arrow_field_after.metadata or {}), (
            "initial-default key must be absent after remove_column_default"
        )

        # Column reads NULL again (prior rows have no stored value).
        result = ds.to_table().column("score").to_pylist()
        assert result == [None, None, None]

    def test_set_default_string_column(self, tmp_path: Path):
        """set_column_default works on a utf8 column."""
        ds = _write_simple(tmp_path)
        ds.add_columns(pa.field("label", pa.utf8()))
        ds.set_column_default("label", "unknown")

        result = ds.to_table().column("label").to_pylist()
        assert result == ["unknown", "unknown", "unknown"]

    def test_set_default_float_column(self, tmp_path: Path):
        """set_column_default works on a float64 column."""
        ds = _write_simple(tmp_path)
        ds.add_columns(pa.field("score_f", pa.float64()))
        ds.set_column_default("score_f", 3.14)

        result = ds.to_table().column("score_f").to_pylist()
        assert all(abs(v - 3.14) < 1e-9 for v in result), f"Unexpected result: {result}"

    def test_set_invalid_value_raises(self, tmp_path: Path):
        """set_column_default with a value incompatible with the column type raises."""
        ds = _write_simple(tmp_path)
        ds.add_columns(pa.field("score", pa.int32()))

        with pytest.raises((OSError, ValueError, pa.lib.ArrowInvalid, Exception)):
            ds.set_column_default("score", "not_an_int")

    def test_set_default_on_scalar_indexed_column_raises(self, tmp_path: Path):
        """set_column_default on a scalar-indexed column is rejected (§1.4a guard)."""
        import tempfile

        # Need a real file path (not memory://) for index creation.
        real_path = str(tmp_path / "indexed_ds")
        table = pa.table({"score": pa.array([10, 20, 30], type=pa.int32())})
        ds = lance.write_dataset(table, real_path)
        ds.create_scalar_index("score", "BTREE")

        with pytest.raises(OSError, match="scalar index"):
            ds.set_column_default("score", 0)

    def test_remove_default_on_scalar_indexed_column_raises(self, tmp_path: Path):
        """remove_column_default on a scalar-indexed column is rejected."""
        real_path = str(tmp_path / "indexed_ds")
        table = pa.table({"score": pa.array([10, 20, 30], type=pa.int32())})
        ds = lance.write_dataset(table, real_path)
        ds.create_scalar_index("score", "BTREE")

        with pytest.raises(OSError, match="scalar index"):
            ds.remove_column_default("score")

    def test_set_default_nonexistent_column_raises(self, tmp_path: Path):
        """set_column_default on a non-existent column raises."""
        ds = _write_simple(tmp_path)

        with pytest.raises((OSError, KeyError, Exception)):
            ds.set_column_default("no_such_column", 99)

    def test_set_default_does_not_disturb_other_metadata(self, tmp_path: Path):
        """set_column_default only writes the initial-default key; other keys survive."""
        ds = _write_simple(tmp_path)
        ds.add_columns(pa.field("score", pa.int32()))

        # Place some other metadata first.
        ds.update_field_metadata({"score": {"my-custom-key": "my-value"}})

        # Now set the default.
        ds.set_column_default("score", 42)

        arrow_field = ds.schema.field("score")
        meta = arrow_field.metadata or {}
        assert meta.get(b"my-custom-key") == b"my-value", (
            "set_column_default must not disturb co-located metadata"
        )
        assert meta.get(_LANCE_INITIAL_DEFAULT_META_KEY.encode()) == b"42"


# ---------------------------------------------------------------------------
# Part B — read accessors: initial_default / write_default / effective_default
# ---------------------------------------------------------------------------


class TestReadAccessors:
    def _set_and_get_field(
        self, tmp_path: Path, value, col_type=pa.int32()
    ):
        """Helper: write dataset, add+set default on 'score', return LanceField."""
        ds = _write_simple(tmp_path)
        ds.add_columns(pa.field("score", col_type))
        ds.set_column_default("score", value)
        return _lance_field(ds, "score")

    def test_initial_default_returns_scalar(self, tmp_path: Path):
        """initial_default() returns a pyarrow scalar equal to 7."""
        field = self._set_and_get_field(tmp_path, 7)
        val = field.initial_default()
        assert val is not None, "initial_default() must not be None"
        # Compare as Python int.
        assert val.as_py() == 7, f"Expected 7, got {val.as_py()!r}"

    def test_initial_default_string(self, tmp_path: Path):
        """initial_default() works for utf8 columns."""
        field = self._set_and_get_field(tmp_path, "hello", col_type=pa.utf8())
        val = field.initial_default()
        assert val is not None
        assert val.as_py() == "hello"

    def test_initial_default_none_when_absent(self, tmp_path: Path):
        """initial_default() returns None for a field with no default."""
        ds = _write_simple(tmp_path)
        field = _lance_field(ds, "id")
        assert field.initial_default() is None

    def test_write_default_always_none_model_a(self, tmp_path: Path):
        """write_default() returns None in Model A (no write-default key)."""
        field = self._set_and_get_field(tmp_path, 7)
        assert field.write_default() is None, (
            "write_default() must be None in Model A"
        )

    def test_write_default_none_when_no_default(self, tmp_path: Path):
        """write_default() returns None for a field with no defaults at all."""
        ds = _write_simple(tmp_path)
        field = _lance_field(ds, "id")
        assert field.write_default() is None

    def test_effective_default_returns_initial_in_model_a(self, tmp_path: Path):
        """effective_default() falls back to initial_default() in Model A."""
        field = self._set_and_get_field(tmp_path, 7)
        val = field.effective_default()
        assert val is not None
        assert val.as_py() == 7

    def test_effective_default_none_when_no_defaults(self, tmp_path: Path):
        """effective_default() returns None when neither key is set."""
        ds = _write_simple(tmp_path)
        field = _lance_field(ds, "id")
        assert field.effective_default() is None

    def test_all_three_after_remove(self, tmp_path: Path):
        """After remove_column_default, all three accessors return None."""
        ds = _write_simple(tmp_path)
        ds.add_columns(pa.field("score", pa.int32()))
        ds.set_column_default("score", 7)
        ds.remove_column_default("score")

        field = _lance_field(ds, "score")
        assert field.initial_default() is None
        assert field.write_default() is None
        assert field.effective_default() is None

    def test_bool_default(self, tmp_path: Path):
        """initial_default() works for boolean columns."""
        field = self._set_and_get_field(tmp_path, True, col_type=pa.bool_())
        val = field.initial_default()
        assert val is not None
        assert val.as_py() is True

    def test_float_default(self, tmp_path: Path):
        """initial_default() works for float32 columns."""
        field = self._set_and_get_field(tmp_path, 1.5, col_type=pa.float32())
        val = field.initial_default()
        assert val is not None
        assert abs(val.as_py() - 1.5) < 1e-5


# ---------------------------------------------------------------------------
# Part B — _decode_default_value helper
# ---------------------------------------------------------------------------


class TestDecodeDefaultValue:
    def test_decode_int(self):
        """_decode_default_value decodes an integer JSON literal."""
        from lance.lance import _decode_default_value

        arr = _decode_default_value("42", pa.int32())
        assert len(arr) == 1
        assert arr[0].as_py() == 42

    def test_decode_string(self):
        """_decode_default_value decodes a JSON string literal."""
        from lance.lance import _decode_default_value

        arr = _decode_default_value('"hello"', pa.utf8())
        assert len(arr) == 1
        assert arr[0].as_py() == "hello"

    def test_decode_float(self):
        """_decode_default_value decodes a float JSON literal."""
        from lance.lance import _decode_default_value

        arr = _decode_default_value("1.5", pa.float64())
        assert len(arr) == 1
        assert abs(arr[0].as_py() - 1.5) < 1e-9

    def test_decode_bool(self):
        """_decode_default_value decodes a boolean JSON literal."""
        from lance.lance import _decode_default_value

        arr = _decode_default_value("true", pa.bool_())
        assert len(arr) == 1
        assert arr[0].as_py() is True

    def test_decode_invalid_raises(self):
        """_decode_default_value raises ValueError for an incompatible literal."""
        from lance.lance import _decode_default_value

        with pytest.raises(ValueError):
            _decode_default_value('"hello"', pa.int32())

    def test_decode_not_json_raises(self):
        """_decode_default_value raises ValueError for non-JSON input."""
        from lance.lance import _decode_default_value

        with pytest.raises(ValueError):
            _decode_default_value("not-json", pa.int32())

    def test_roundtrip_encode_decode(self):
        """Encoding then decoding a value produces the original value."""
        from lance.lance import _decode_default_value, _encode_default_value

        original = pa.array([99], type=pa.int32())
        encoded = _encode_default_value(original)
        decoded = _decode_default_value(encoded, pa.int32())
        assert decoded[0].as_py() == 99
