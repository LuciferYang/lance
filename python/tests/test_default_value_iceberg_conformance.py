# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Iceberg conformance tests for Lance's column-default-value JSON codec.

Lance's per-column default mechanism uses a JSON single-value codec
(``rust/lance-core/src/datatypes/default_value.rs``) that is intentionally
aligned with Apache Iceberg's ``SingleValueParser``, with a few *deliberate*
divergences (scalars only, decimal restrictions, lenient timestamp offsets).

The reference vectors below come from Iceberg's own test:
``core/src/test/java/org/apache/iceberg/TestSingleValueParser.java``
(method ``testValidDefaults``).

This file proves the PyO3 binding (``_encode_default_value`` /
``_decode_default_value``) preserves the Iceberg-aligned encoding for the
scalar subset, and that the deliberate divergences raise. A separate,
self-skipping section runs a *live* cross-check against PyIceberg when it
happens to be installed (it is NOT a test dependency).
"""

import pyarrow as pa
import pytest

# `_encode_default_value` / `_decode_default_value` are PyO3 helpers exposed by
# the native extension. `dataset.py` imports `_encode_default_value` from
# `lance.lance`; `_decode_default_value` lives in the same module.
from lance.lance import _decode_default_value, _encode_default_value


def _roundtrip(json: str, dtype: pa.DataType) -> str:
    """decode(json) -> Array, then encode(Array) -> canonical JSON string."""
    arr = _decode_default_value(json, dtype)
    assert len(arr) == 1
    return _encode_default_value(arr)


# ── PARITY vectors ─────────────────────────────────────────────────────────────
#
# Each tuple is (iceberg_json, arrow_type, expected_lance_canonical_json).
# `expected` is Lance's ACTUAL canonical output (determined by running the
# codec). Where it equals the Iceberg vector byte-for-byte it is "parity";
# where it differs textually but is semantically equal it is annotated as a
# divergence in the comment next to the case.
#
# Source: TestSingleValueParser.java::testValidDefaults (Iceberg).

PARITY_CASES = [
    # parity with Iceberg "true"
    pytest.param("true", pa.bool_(), "true", id="boolean_true"),
    # parity with Iceberg "1"
    pytest.param("1", pa.int32(), "1", id="int32"),
    # parity with Iceberg "9999999"
    pytest.param("9999999", pa.int64(), "9999999", id="int64"),
    # parity with Iceberg "1.23"
    pytest.param("1.23", pa.float32(), "1.23", id="float32"),
    # parity with Iceberg "123.456"
    pytest.param("123.456", pa.float64(), "123.456", id="float64"),
    # parity with Iceberg "2007-12-03"
    pytest.param('"2007-12-03"', pa.date32(), '"2007-12-03"', id="date32"),
    # divergence from Iceberg "2007-12-03T10:15:30": Lance always emits the full
    # sub-second field for the column unit, so a microsecond column pads to
    # ".000000" (same instant).
    pytest.param(
        '"2007-12-03T10:15:30"',
        pa.timestamp("us"),
        '"2007-12-03T10:15:30.000000"',
        id="ts_micro_naive",
    ),
    # divergence from Iceberg "2007-12-03T10:15:30+00:00": Lance pads the micros
    # field to ".000000" (same instant, same +00:00 offset).
    pytest.param(
        '"2007-12-03T10:15:30+00:00"',
        pa.timestamp("us", tz="UTC"),
        '"2007-12-03T10:15:30.000000+00:00"',
        id="ts_micro_utc",
    ),
    # parity with Iceberg "2007-12-03T10:15:30.123456789"
    pytest.param(
        '"2007-12-03T10:15:30.123456789"',
        pa.timestamp("ns"),
        '"2007-12-03T10:15:30.123456789"',
        id="ts_nano_naive",
    ),
    # parity with Iceberg "2007-12-03T10:15:30.123456789+00:00"
    pytest.param(
        '"2007-12-03T10:15:30.123456789+00:00"',
        pa.timestamp("ns", tz="UTC"),
        '"2007-12-03T10:15:30.123456789+00:00"',
        id="ts_nano_utc",
    ),
    # parity with Iceberg "foo"
    pytest.param('"foo"', pa.string(), '"foo"', id="string"),
    # divergence from Iceberg "111f" (lowercase hex): Lance normalises hex output
    # to UPPERCASE, so the canonical form is "111F" (same bytes 0x11 0x1f).
    pytest.param('"111f"', pa.binary(2), '"111F"', id="fixed2_uppercase_hex"),
    # divergence from Iceberg "0000ff" (lowercase hex): Lance emits uppercase
    # "0000FF" (same bytes 0x00 0x00 0xff).
    pytest.param('"0000ff"', pa.binary(), '"0000FF"', id="binary_uppercase_hex"),
    # parity with Iceberg "123.4500"
    pytest.param(
        '"123.4500"',
        pa.decimal128(9, 4),
        '"123.4500"',
        id="decimal_9_4",
    ),
    # parity with Iceberg "2"
    pytest.param('"2"', pa.decimal128(9, 0), '"2"', id="decimal_9_0"),
]


@pytest.mark.parametrize("iceberg_json,dtype,expected", PARITY_CASES)
def test_iceberg_scalar_roundtrip(iceberg_json, dtype, expected):
    """Each Iceberg scalar vector decodes and re-encodes to Lance's canonical form."""
    assert _roundtrip(iceberg_json, dtype) == expected


def test_fixed2_value_bytes():
    """Iceberg Fixed(2) "111f" decodes to the exact two bytes 0x11 0x1f."""
    arr = _decode_default_value('"111f"', pa.binary(2))
    assert arr[0].as_py() == bytes([0x11, 0x1F])


def test_binary_value_bytes():
    """Iceberg Binary "0000ff" decodes to bytes 0x00 0x00 0xff."""
    arr = _decode_default_value('"0000ff"', pa.binary())
    assert arr[0].as_py() == bytes([0x00, 0x00, 0xFF])


# ── DELIBERATE DIVERGENCES ─────────────────────────────────────────────────────


def test_decimal_negative_scale_rejected():
    """Iceberg accepts Decimal(9,-20) default "2E+20"; Lance REJECTS it.

    Two reasons combine: (1) negative-scale decimals cannot round-trip through
    Lance's parse/encode pair, and (2) scientific notation is rejected. pyarrow
    cannot even construct decimal128(9, -20), so we exercise the scientific
    notation rejection on a representable type instead, plus assert the
    negative-scale path is rejected where pyarrow allows it.

    Source: TestSingleValueParser.java vector {DecimalType.of(9, -20), "2E+20"}.
    """
    # Scientific notation rejected even on a valid positive-scale column.
    with pytest.raises(ValueError):
        _decode_default_value('"2E+20"', pa.decimal128(20, 0))
    with pytest.raises(ValueError):
        _decode_default_value('"1.8e-2"', pa.decimal128(10, 2))


def test_decimal_negative_scale_type_rejected():
    """Negative-scale decimal columns reject defaults (when pyarrow builds them)."""
    try:
        neg_scale = pa.decimal128(10, -2)
    except Exception:
        pytest.skip("pyarrow cannot construct a negative-scale decimal type")
    with pytest.raises(ValueError):
        _decode_default_value('"100"', neg_scale)


def test_struct_nested_default_rejected():
    """Iceberg accepts struct defaults; Lance REJECTS nested types (scalars only).

    Source: TestSingleValueParser.java struct vector
    {StructType.of(req(4,f1,int), opt(5,f2,string)), {"4":1,"5":"bar"}}.
    """
    struct_t = pa.struct([("f1", pa.int32()), ("f2", pa.string())])
    with pytest.raises(ValueError):
        _decode_default_value('{"f1": 1, "f2": "bar"}', struct_t)


def test_list_nested_default_rejected():
    """Iceberg accepts list defaults; Lance REJECTS them.

    Source: TestSingleValueParser.java list vector
    {ListType.ofOptional(1, int), [1, 2, 3]}.
    """
    with pytest.raises(ValueError):
        _decode_default_value("[1, 2, 3]", pa.list_(pa.int32()))


def test_timestamp_any_offset_normalised():
    """Iceberg requires UTC offset; Lance accepts ANY offset and normalises.

    Iceberg's testInvalidTimestamptz REJECTS "+01:00"; Lance accepts e.g.
    "+05:00" for a UTC column and normalises the stored instant. The canonical
    re-encoding is the equivalent UTC wall-clock with "+00:00".

    Source: contrast with TestSingleValueParser.java::testInvalidTimestamptz.
    """
    # 2007-12-03T10:15:30+05:00 == 2007-12-03T05:15:30Z (micros padded to .000000)
    out = _roundtrip('"2007-12-03T10:15:30+05:00"', pa.timestamp("us", tz="UTC"))
    assert out == '"2007-12-03T05:15:30.000000+00:00"'


def test_json_null_default_rejected_for_typed_column():
    """Iceberg lists Boolean default `null`; Lance's codec REJECTS JSON null.

    Lance's `decode_default` requires a concrete boolean for a Boolean column;
    a bare `null` literal is not a valid boolean and is rejected. (Nullability
    of a column is a schema concern, not a default-literal value.)

    Source: TestSingleValueParser.java vector {BooleanType.get(), "null"}.
    """
    with pytest.raises(ValueError):
        _decode_default_value("null", pa.bool_())


# ── Live PyIceberg cross-check (self-skipping; pyiceberg is NOT a dependency) ───
#
# This section is dormant by default: it self-skips when pyiceberg is not
# installed (pyiceberg is NOT a test dependency and must not be installed for
# the suite to pass). When pyiceberg IS present, it parses each Iceberg JSON
# default for the matching Iceberg type via PyIceberg's own machinery and
# asserts Lance decodes the same JSON to an equivalent scalar value.


def _pyiceberg_default_parser():
    """Return a callable (iceberg_type, json_str) -> python value, or None.

    PyIceberg's entry point for parsing a single-value JSON default has moved
    across versions, so probe for a working one. Returns None if none is found.
    """
    import json as _json

    candidates = []
    try:
        from pyiceberg.conversions import from_json  # type: ignore

        candidates.append(lambda t, s: from_json(t, _json.loads(s)))
    except Exception:
        pass
    try:
        # Some versions expose literal parsing that accepts the JSON scalar.
        from pyiceberg.expressions.literals import literal  # type: ignore

        candidates.append(lambda t, s: literal(_json.loads(s)).to(t).value)
    except Exception:
        pass
    return candidates[0] if candidates else None


def test_live_pyiceberg_scalar_values_match():
    """Cross-check Lance's decode against a live PyIceberg parse, when available.

    Skips cleanly when pyiceberg is not installed (the default state of this
    repo's test env). When present, only the scalar subset PyIceberg can parse
    is checked; if PyIceberg lacks a usable JSON-default API the check narrows
    to whatever it supports and documents the limitation via a skip.
    """
    pytest.importorskip("pyiceberg")
    from pyiceberg import types as it  # type: ignore

    parse_default = _pyiceberg_default_parser()
    if parse_default is None:
        pytest.skip("No usable PyIceberg JSON-default parsing API found")

    # (iceberg_type, iceberg_json, arrow_type)
    live_cases = [
        (it.BooleanType(), "true", pa.bool_()),
        (it.IntegerType(), "1", pa.int32()),
        (it.LongType(), "9999999", pa.int64()),
        (it.FloatType(), "1.23", pa.float32()),
        (it.DoubleType(), "123.456", pa.float64()),
        (it.StringType(), '"foo"', pa.string()),
    ]

    checked = 0
    for ice_t, json_str, arrow_t in live_cases:
        try:
            ice_val = parse_default(ice_t, json_str)
        except Exception:
            # This particular type isn't parseable by this PyIceberg API version.
            continue
        lance_val = _decode_default_value(json_str, arrow_t)[0].as_py()
        if isinstance(ice_val, float):
            assert lance_val == pytest.approx(ice_val)
        else:
            assert lance_val == ice_val
        checked += 1

    if checked == 0:
        pytest.skip("PyIceberg API present but parsed no scalar cases")
