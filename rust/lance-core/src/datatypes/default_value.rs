// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! JSON ↔ Arrow single-value codec for column default values.
//!
//! This module provides [`encode_default`] and [`decode_default`] for the
//! primitive Arrow data types, following Iceberg's JSON single-value encoding:
//!
//! - Signed/unsigned integers → JSON number (e.g. `42`)
//! - `Float32`/`Float64` → JSON number (finite values only)
//! - `Boolean` → `true` / `false`
//! - `Utf8` / `LargeUtf8` → JSON string (e.g. `"unknown"`)
//! - `Decimal128(p,s)` / `Decimal256(p,s)` → fixed-point JSON string (e.g. `"0.00"`)
//! - `Timestamp(unit, tz)` → ISO-8601 string (offset required iff tz-aware)
//! - `Date32` → date string `"YYYY-MM-DD"`
//! - `Binary` / `LargeBinary` / `FixedSizeBinary(n)` → UPPERCASE hex JSON string (e.g. `"0A0B0C"`)
//!
//! Unsupported data types return [`DefaultValueError::UnsupportedType`]; this
//! function never panics on bad input.
//!
//! # Validation entry point
//!
//! [`validate_default_assignable`] is the set-time validation helper that checks
//! whether a JSON literal may be assigned as a default value to a resolved
//! [`Field`].  It enforces nullability, type eligibility (nested / physically-encoded
//! types are rejected), primary-key restrictions, and literal castability via
//! [`decode_default`].  Top-level existence and addressing checks (e.g. whether
//! the column path refers to a real column) are the **caller's** responsibility.
//!
//! # Architecture Invariant
//!
//! **INVARIANT: This module depends only on `arrow` + `serde_json` (+ `chrono` via Arrow).**
//! **It MUST NOT use DataFusion or `ScalarValue`.**
//!
//! The persisted default value format is a stable, library-independent JSON form
//! (by design, per the column-default-values feature decision). Any conversion to/from
//! `ScalarValue` belongs in the `lance` or `lance-datafusion` crates, not here. This
//! isolation ensures that the codec is portable and can be used by any consumer
//! of `lance-core` without pulling in DataFusion as a transitive dependency.

use std::sync::Arc;

use arrow_array::{
    types::{Decimal128Type, Decimal256Type},
    ArrayRef, BinaryArray, BooleanArray, Date32Array, Decimal128Array, Decimal256Array,
    FixedSizeBinaryArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    Int8Array, LargeBinaryArray, LargeStringArray, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt16Array,
    UInt32Array, UInt64Array, UInt8Array,
};
use arrow_cast::parse::parse_decimal;
use arrow_schema::{DataType, TimeUnit};
use chrono::{DateTime, NaiveDate, NaiveDateTime};
use snafu::Snafu;

use super::field::Field;

/// Errors returned by the JSON ↔ Arrow default-value codec.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum DefaultValueError {
    /// The data type is not yet supported by this codec.
    ///
    /// Support for additional types (Decimal, Timestamp, Date, Binary, …) will
    /// be added in future tasks.
    #[snafu(display("Unsupported data type for default value codec: {data_type:?}"))]
    UnsupportedType { data_type: DataType },

    /// The JSON text could not be parsed as the expected Arrow type.
    #[snafu(display(
        "Invalid default value literal '{literal}' for type {data_type:?}: {reason}"
    ))]
    InvalidLiteral {
        literal: String,
        data_type: DataType,
        reason: String,
    },

    /// The Arrow array has a different data type than expected.
    #[snafu(display("Type mismatch: expected {expected:?}, got {actual:?}"))]
    TypeMismatch {
        expected: DataType,
        actual: DataType,
    },

    /// Nested types (Struct, List, LargeList, FixedSizeList, Map, Union) are not supported.
    #[snafu(display(
        "Column '{name}' has a nested data type ({data_type:?}); \
         nested-type defaults are not supported"
    ))]
    NestedNotSupported { name: String, data_type: DataType },

    /// Physically-encoded scalars (Dictionary, RunEndEncoded) are not supported.
    #[snafu(display(
        "Column '{name}' uses a physically-encoded type ({data_type:?}); \
         Dictionary and run-end-encoded columns are not supported"
    ))]
    PhysicalEncodingNotSupported { name: String, data_type: DataType },

    /// Primary-key columns may not have a default value.
    #[snafu(display(
        "Column '{name}' is part of the primary key; \
         primary-key columns may not have a default value"
    ))]
    PrimaryKeyNotAllowed { name: String },
}

/// Result type for this module.
pub type DefaultValueResult<T> = std::result::Result<T, DefaultValueError>;

/// Validate that a JSON literal may be assigned as a default value to a resolved [`Field`].
///
/// This is the set-time validation helper for the column-default-values feature.  It checks
/// the *intrinsic eligibility* of the field and the castability of the literal:
///
/// 1. **No nested types** — `Struct`, `List`, `LargeList`, `FixedSizeList`, `Map`, and
///    `Union` are rejected; only scalar leaf types are supported.
/// 2. **No physically-encoded types** — `Dictionary(_,_)` and `RunEndEncoded(_,_)` are
///    rejected because the codec has no stable JSON representation for them.
/// 3. **No primary-key columns** — columns marked as unenforced primary keys (either via
///    the legacy boolean metadata key or the position-only key) are rejected.
/// 4. **Literal must cast** — [`decode_default`] is called to verify the JSON literal
///    parses and fits the column's Arrow type; any error is propagated directly.
///
/// # Non-nullable columns
///
/// A non-nullable column WITH a valid literal is accepted — the default value is what
/// satisfies the non-null contract for pre-existing (absent) rows on read.  The higher-level
/// AllNulls create-time guard separately rejects non-nullable columns that have NO
/// initial-default metadata.
///
/// # Scope
///
/// This function validates a **resolved** field's intrinsic eligibility (type, nullability,
/// PK, and literal).  Top-level existence checks (does the column exist?) and dotted-path
/// addressing (rejecting `"parent.child"` names) are the **caller's** responsibility and
/// are enforced at the higher set/create API layer via `project_by_schema`.
///
/// # Errors
///
/// Returns a recoverable [`DefaultValueError`] variant on the first failing check.
pub fn validate_default_assignable(field: &Field, json: &str) -> DefaultValueResult<()> {
    // 1. Reject nested types.
    let dt = field.data_type();
    let is_nested = matches!(
        dt,
        DataType::Struct(_)
            | DataType::List(_)
            | DataType::LargeList(_)
            | DataType::FixedSizeList(_, _)
            | DataType::Map(_, _)
            | DataType::Union(_, _)
    );
    if is_nested {
        return Err(DefaultValueError::NestedNotSupported {
            name: field.name.clone(),
            data_type: dt,
        });
    }

    // 2. Reject physically-encoded types (Dictionary, RunEndEncoded).
    let is_physical = matches!(
        dt,
        DataType::Dictionary(_, _) | DataType::RunEndEncoded(_, _)
    );
    if is_physical {
        return Err(DefaultValueError::PhysicalEncodingNotSupported {
            name: field.name.clone(),
            data_type: dt,
        });
    }

    // 3. Reject primary-key columns.
    if field.is_unenforced_pk_raw() {
        return Err(DefaultValueError::PrimaryKeyNotAllowed {
            name: field.name.clone(),
        });
    }

    // 4. Literal must decode/cast to the column type.
    decode_default(json, &field.data_type())?;

    Ok(())
}

/// Decode a JSON single-value string into a length-1 Arrow array of type `dt`.
///
/// # Supported types
///
/// | `DataType`                            | JSON form                          |
/// |---------------------------------------|------------------------------------|
/// | `Int8`, `Int16`, `Int32`, `Int64`     | JSON number                        |
/// | `UInt8`, `UInt16`, `UInt32`, `UInt64` | JSON number                        |
/// | `Float32`, `Float64`                  | JSON number                        |
/// | `Boolean`                             | `true` / `false`                   |
/// | `Utf8`, `LargeUtf8`                   | JSON string                        |
/// | `Decimal128(p,s)`, `Decimal256(p,s)`  | fixed-point string                 |
/// | `Timestamp(unit, Some(tz))`           | ISO-8601 string with UTC offset    |
/// | `Timestamp(unit, None)`               | ISO-8601 string without offset     |
/// | `Date32`                              | date string `"YYYY-MM-DD"`         |
/// | `Binary`, `LargeBinary`               | uppercase hex JSON string          |
/// | `FixedSizeBinary(n)`                  | uppercase hex JSON string (n bytes)|
///
/// # Errors
///
/// Returns [`DefaultValueError::UnsupportedType`] for data types not listed
/// above, or [`DefaultValueError::InvalidLiteral`] when the JSON cannot be
/// parsed as the requested type.
pub fn decode_default(json: &str, dt: &DataType) -> DefaultValueResult<ArrayRef> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| DefaultValueError::InvalidLiteral {
            literal: json.to_string(),
            data_type: dt.clone(),
            reason: e.to_string(),
        })?;

    match dt {
        DataType::Boolean => {
            let v = as_bool(&value, json, dt)?;
            Ok(Arc::new(BooleanArray::from(vec![v])))
        }
        DataType::Int8 => {
            let wide = as_integer::<i64>(&value, json, dt)?;
            let v = i8::try_from(wide).map_err(|_| DefaultValueError::InvalidLiteral {
                literal: json.to_string(),
                data_type: dt.clone(),
                reason: format!("value {wide} is out of range for Int8"),
            })?;
            Ok(Arc::new(Int8Array::from(vec![v])))
        }
        DataType::Int16 => {
            let wide = as_integer::<i64>(&value, json, dt)?;
            let v = i16::try_from(wide).map_err(|_| DefaultValueError::InvalidLiteral {
                literal: json.to_string(),
                data_type: dt.clone(),
                reason: format!("value {wide} is out of range for Int16"),
            })?;
            Ok(Arc::new(Int16Array::from(vec![v])))
        }
        DataType::Int32 => {
            let wide = as_integer::<i64>(&value, json, dt)?;
            let v = i32::try_from(wide).map_err(|_| DefaultValueError::InvalidLiteral {
                literal: json.to_string(),
                data_type: dt.clone(),
                reason: format!("value {wide} is out of range for Int32"),
            })?;
            Ok(Arc::new(Int32Array::from(vec![v])))
        }
        DataType::Int64 => {
            let v = as_integer::<i64>(&value, json, dt)?;
            Ok(Arc::new(Int64Array::from(vec![v])))
        }
        DataType::UInt8 => {
            let wide = as_integer::<u64>(&value, json, dt)?;
            let v = u8::try_from(wide).map_err(|_| DefaultValueError::InvalidLiteral {
                literal: json.to_string(),
                data_type: dt.clone(),
                reason: format!("value {wide} is out of range for UInt8"),
            })?;
            Ok(Arc::new(UInt8Array::from(vec![v])))
        }
        DataType::UInt16 => {
            let wide = as_integer::<u64>(&value, json, dt)?;
            let v = u16::try_from(wide).map_err(|_| DefaultValueError::InvalidLiteral {
                literal: json.to_string(),
                data_type: dt.clone(),
                reason: format!("value {wide} is out of range for UInt16"),
            })?;
            Ok(Arc::new(UInt16Array::from(vec![v])))
        }
        DataType::UInt32 => {
            let wide = as_integer::<u64>(&value, json, dt)?;
            let v = u32::try_from(wide).map_err(|_| DefaultValueError::InvalidLiteral {
                literal: json.to_string(),
                data_type: dt.clone(),
                reason: format!("value {wide} is out of range for UInt32"),
            })?;
            Ok(Arc::new(UInt32Array::from(vec![v])))
        }
        DataType::UInt64 => {
            let v = as_integer::<u64>(&value, json, dt)?;
            Ok(Arc::new(UInt64Array::from(vec![v])))
        }
        DataType::Float32 => {
            let v = as_float(&value, json, dt)?;
            let v32 = v as f32;
            if v.is_finite() && !v32.is_finite() {
                return Err(DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: "value out of range for Float32".into(),
                });
            }
            Ok(Arc::new(Float32Array::from(vec![v32])))
        }
        DataType::Float64 => {
            let v = as_float(&value, json, dt)?;
            Ok(Arc::new(Float64Array::from(vec![v])))
        }
        DataType::Utf8 => {
            let v = as_string(&value, json, dt)?;
            Ok(Arc::new(StringArray::from(vec![v])))
        }
        DataType::LargeUtf8 => {
            let v = as_string(&value, json, dt)?;
            Ok(Arc::new(LargeStringArray::from(vec![v])))
        }
        DataType::Decimal128(precision, scale) => {
            let s = as_decimal_string(&value, json, dt)?;
            reject_negative_decimal_scale(json, dt, *scale)?;
            validate_decimal_frac_digits(s, json, dt, *scale)?;
            let native = parse_decimal::<Decimal128Type>(s, *precision, *scale).map_err(|e| {
                DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: e.to_string(),
                }
            })?;
            let arr = Decimal128Array::from_iter_values([native])
                .with_precision_and_scale(*precision, *scale)
                .map_err(|e| DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: e.to_string(),
                })?;
            Ok(Arc::new(arr))
        }
        DataType::Decimal256(precision, scale) => {
            let s = as_decimal_string(&value, json, dt)?;
            reject_negative_decimal_scale(json, dt, *scale)?;
            validate_decimal_frac_digits(s, json, dt, *scale)?;
            let native = parse_decimal::<Decimal256Type>(s, *precision, *scale).map_err(|e| {
                DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: e.to_string(),
                }
            })?;
            let arr = Decimal256Array::from_iter_values([native])
                .with_precision_and_scale(*precision, *scale)
                .map_err(|e| DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: e.to_string(),
                })?;
            Ok(Arc::new(arr))
        }
        DataType::Timestamp(unit, tz_opt) => {
            let s = as_string(&value, json, dt)?;
            decode_timestamp(s, json, dt, unit, tz_opt.as_deref())
        }
        DataType::Date32 => {
            let s = as_string(&value, json, dt)?;
            let nd = NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|e| {
                DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: format!("expected YYYY-MM-DD date string: {e}"),
                }
            })?;
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch is valid");
            let days = nd.signed_duration_since(epoch).num_days();
            let days_i32 = i32::try_from(days).map_err(|_| DefaultValueError::InvalidLiteral {
                literal: json.to_string(),
                data_type: dt.clone(),
                reason: format!(
                    "date {nd} is out of range for Date32 (days since epoch overflows i32)"
                ),
            })?;
            Ok(Arc::new(Date32Array::from(vec![days_i32])))
        }
        DataType::Binary => {
            let s = as_string(&value, json, dt)?;
            let bytes = parse_hex_bytes(s, json, dt)?;
            Ok(Arc::new(BinaryArray::from_iter_values([bytes])))
        }
        DataType::LargeBinary => {
            let s = as_string(&value, json, dt)?;
            let bytes = parse_hex_bytes(s, json, dt)?;
            Ok(Arc::new(LargeBinaryArray::from_iter_values([bytes])))
        }
        DataType::FixedSizeBinary(width) => {
            let s = as_string(&value, json, dt)?;
            let bytes = parse_hex_bytes(s, json, dt)?;
            let expected = *width as usize;
            if bytes.len() != expected {
                return Err(DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: format!(
                        "decoded {} byte(s) but FixedSizeBinary width is {expected}",
                        bytes.len()
                    ),
                });
            }
            let arr = FixedSizeBinaryArray::try_from_iter([bytes.as_slice()].into_iter()).map_err(
                |e| DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: e.to_string(),
                },
            )?;
            Ok(Arc::new(arr))
        }
        other => Err(DefaultValueError::UnsupportedType {
            data_type: other.clone(),
        }),
    }
}

/// Encode element 0 of an Arrow array as a JSON single-value string.
///
/// # Errors
///
/// Returns [`DefaultValueError::UnsupportedType`] for data types not supported
/// by this codec, [`DefaultValueError::TypeMismatch`] if the array's concrete
/// type does not match the reported `data_type()` (a defensive guard against
/// malformed Arrow arrays — not an ordinary-usage error),
/// [`DefaultValueError::InvalidLiteral`] for non-finite float values (NaN /
/// Infinity) which have no JSON representation, when the array is empty
/// (no element 0 to encode), or when element 0 is null. Never panics on bad
/// input.
pub fn encode_default(arr: &ArrayRef) -> DefaultValueResult<String> {
    if arr.is_empty() {
        return Err(DefaultValueError::InvalidLiteral {
            literal: "<empty array>".into(),
            data_type: arr.data_type().clone(),
            reason: "cannot encode a default from an empty array".into(),
        });
    }
    if arr.is_null(0) {
        return Err(DefaultValueError::InvalidLiteral {
            literal: "<null value>".into(),
            data_type: arr.data_type().clone(),
            reason: "cannot encode a default from a null value".into(),
        });
    }
    match arr.data_type() {
        DataType::Boolean => {
            let arr = arr.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::Boolean,
                    actual: arr.data_type().clone(),
                }
            })?;
            let v = arr.value(0);
            Ok(serde_json::to_string(&v).expect("bool always serializable"))
        }
        DataType::Int8 => {
            let arr = arr.as_any().downcast_ref::<Int8Array>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::Int8,
                    actual: arr.data_type().clone(),
                }
            })?;
            Ok(serde_json::to_string(&arr.value(0)).expect("int always serializable"))
        }
        DataType::Int16 => {
            let arr = arr.as_any().downcast_ref::<Int16Array>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::Int16,
                    actual: arr.data_type().clone(),
                }
            })?;
            Ok(serde_json::to_string(&arr.value(0)).expect("int always serializable"))
        }
        DataType::Int32 => {
            let arr = arr.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::Int32,
                    actual: arr.data_type().clone(),
                }
            })?;
            Ok(serde_json::to_string(&arr.value(0)).expect("int always serializable"))
        }
        DataType::Int64 => {
            let arr = arr.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::Int64,
                    actual: arr.data_type().clone(),
                }
            })?;
            Ok(serde_json::to_string(&arr.value(0)).expect("int always serializable"))
        }
        DataType::UInt8 => {
            let arr = arr.as_any().downcast_ref::<UInt8Array>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::UInt8,
                    actual: arr.data_type().clone(),
                }
            })?;
            Ok(serde_json::to_string(&arr.value(0)).expect("uint always serializable"))
        }
        DataType::UInt16 => {
            let arr = arr.as_any().downcast_ref::<UInt16Array>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::UInt16,
                    actual: arr.data_type().clone(),
                }
            })?;
            Ok(serde_json::to_string(&arr.value(0)).expect("uint always serializable"))
        }
        DataType::UInt32 => {
            let arr = arr.as_any().downcast_ref::<UInt32Array>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::UInt32,
                    actual: arr.data_type().clone(),
                }
            })?;
            Ok(serde_json::to_string(&arr.value(0)).expect("uint always serializable"))
        }
        DataType::UInt64 => {
            let arr = arr.as_any().downcast_ref::<UInt64Array>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::UInt64,
                    actual: arr.data_type().clone(),
                }
            })?;
            Ok(serde_json::to_string(&arr.value(0)).expect("uint always serializable"))
        }
        DataType::Float32 => {
            let arr = arr.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::Float32,
                    actual: arr.data_type().clone(),
                }
            })?;
            let v = arr.value(0);
            if !v.is_finite() {
                return Err(DefaultValueError::InvalidLiteral {
                    literal: v.to_string(),
                    data_type: DataType::Float32,
                    reason: "non-finite floats (NaN/Infinity) are not representable in JSON".into(),
                });
            }
            Ok(serde_json::to_string(&v).expect("finite f32 always serializable"))
        }
        DataType::Float64 => {
            let arr = arr.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::Float64,
                    actual: arr.data_type().clone(),
                }
            })?;
            let v = arr.value(0);
            if !v.is_finite() {
                return Err(DefaultValueError::InvalidLiteral {
                    literal: v.to_string(),
                    data_type: DataType::Float64,
                    reason: "non-finite floats (NaN/Infinity) are not representable in JSON".into(),
                });
            }
            Ok(serde_json::to_string(&v).expect("finite f64 always serializable"))
        }
        DataType::Utf8 => {
            let arr = arr.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::Utf8,
                    actual: arr.data_type().clone(),
                }
            })?;
            Ok(serde_json::to_string(arr.value(0)).expect("string always serializable"))
        }
        DataType::LargeUtf8 => {
            let arr = arr
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| DefaultValueError::TypeMismatch {
                    expected: DataType::LargeUtf8,
                    actual: arr.data_type().clone(),
                })?;
            Ok(serde_json::to_string(arr.value(0)).expect("string always serializable"))
        }
        dt @ DataType::Decimal128(_, _) => {
            let typed = arr
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| DefaultValueError::TypeMismatch {
                    expected: dt.clone(),
                    actual: arr.data_type().clone(),
                })?;
            let fixed_point = typed.value_as_string(0);
            Ok(serde_json::to_string(&fixed_point).expect("string always serializable"))
        }
        dt @ DataType::Decimal256(_, _) => {
            let typed = arr
                .as_any()
                .downcast_ref::<Decimal256Array>()
                .ok_or_else(|| DefaultValueError::TypeMismatch {
                    expected: dt.clone(),
                    actual: arr.data_type().clone(),
                })?;
            let fixed_point = typed.value_as_string(0);
            Ok(serde_json::to_string(&fixed_point).expect("string always serializable"))
        }
        dt @ DataType::Timestamp(unit, tz_opt) => {
            encode_timestamp(arr, dt, unit, tz_opt.as_deref())
        }
        DataType::Date32 => {
            let typed = arr.as_any().downcast_ref::<Date32Array>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::Date32,
                    actual: arr.data_type().clone(),
                }
            })?;
            let days = typed.value(0) as i64;
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch is valid");
            let out_of_range = || DefaultValueError::InvalidLiteral {
                literal: days.to_string(),
                data_type: DataType::Date32,
                reason: "Date32 day count is out of the representable date range".into(),
            };
            let delta = chrono::Duration::try_days(days).ok_or_else(out_of_range)?;
            let nd = epoch.checked_add_signed(delta).ok_or_else(out_of_range)?;
            let s = nd.format("%Y-%m-%d").to_string();
            Ok(serde_json::to_string(&s).expect("string always serializable"))
        }
        DataType::Binary => {
            let typed = arr.as_any().downcast_ref::<BinaryArray>().ok_or_else(|| {
                DefaultValueError::TypeMismatch {
                    expected: DataType::Binary,
                    actual: arr.data_type().clone(),
                }
            })?;
            let hex = encode_hex_bytes(typed.value(0));
            Ok(serde_json::to_string(&hex).expect("string always serializable"))
        }
        DataType::LargeBinary => {
            let typed = arr
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| DefaultValueError::TypeMismatch {
                    expected: DataType::LargeBinary,
                    actual: arr.data_type().clone(),
                })?;
            let hex = encode_hex_bytes(typed.value(0));
            Ok(serde_json::to_string(&hex).expect("string always serializable"))
        }
        dt @ DataType::FixedSizeBinary(_) => {
            let typed = arr
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or_else(|| DefaultValueError::TypeMismatch {
                    expected: dt.clone(),
                    actual: arr.data_type().clone(),
                })?;
            let hex = encode_hex_bytes(typed.value(0));
            Ok(serde_json::to_string(&hex).expect("string always serializable"))
        }
        other => Err(DefaultValueError::UnsupportedType {
            data_type: other.clone(),
        }),
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn as_bool(value: &serde_json::Value, literal: &str, dt: &DataType) -> DefaultValueResult<bool> {
    value
        .as_bool()
        .ok_or_else(|| DefaultValueError::InvalidLiteral {
            literal: literal.to_string(),
            data_type: dt.clone(),
            reason: "expected JSON boolean".to_string(),
        })
}

/// Helper trait for extracting an integer of the target width from a `serde_json::Value`.
trait IntFromJson: Sized {
    fn from_json(v: &serde_json::Value) -> Option<Self>;
}

impl IntFromJson for i64 {
    fn from_json(v: &serde_json::Value) -> Option<Self> {
        v.as_i64()
    }
}

impl IntFromJson for u64 {
    fn from_json(v: &serde_json::Value) -> Option<Self> {
        v.as_u64()
    }
}

fn as_integer<T: IntFromJson>(
    value: &serde_json::Value,
    literal: &str,
    dt: &DataType,
) -> DefaultValueResult<T> {
    T::from_json(value).ok_or_else(|| DefaultValueError::InvalidLiteral {
        literal: literal.to_string(),
        data_type: dt.clone(),
        reason: "expected JSON integer".to_string(),
    })
}

fn as_float(value: &serde_json::Value, literal: &str, dt: &DataType) -> DefaultValueResult<f64> {
    value
        .as_f64()
        .ok_or_else(|| DefaultValueError::InvalidLiteral {
            literal: literal.to_string(),
            data_type: dt.clone(),
            reason: "expected JSON number".to_string(),
        })
}

fn as_string<'a>(
    value: &'a serde_json::Value,
    literal: &str,
    dt: &DataType,
) -> DefaultValueResult<&'a str> {
    value
        .as_str()
        .ok_or_else(|| DefaultValueError::InvalidLiteral {
            literal: literal.to_string(),
            data_type: dt.clone(),
            reason: "expected JSON string".to_string(),
        })
}

/// For decimal types, the JSON value must be a quoted string (not a bare number).
fn as_decimal_string<'a>(
    value: &'a serde_json::Value,
    literal: &str,
    dt: &DataType,
) -> DefaultValueResult<&'a str> {
    match value {
        serde_json::Value::String(s) => Ok(s.as_str()),
        _ => Err(DefaultValueError::InvalidLiteral {
            literal: literal.to_string(),
            data_type: dt.clone(),
            reason: "decimal default must be a quoted fixed-point string (e.g. \"1.23\"), \
                     not a bare JSON number"
                .to_string(),
        }),
    }
}

/// Encode a byte slice as an UPPERCASE hex string (Iceberg-aligned, no `0x` prefix).
///
/// Each byte is formatted as exactly two uppercase hex digits (e.g. `0x0a` → `"0A"`).
fn encode_hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02X}")).collect()
}

/// Parse an UPPERCASE (or lowercase) hex string into a byte vector.
///
/// Rejects strings with odd length or characters outside `[0-9A-Fa-f]`.
fn parse_hex_bytes(s: &str, literal: &str, dt: &DataType) -> DefaultValueResult<Vec<u8>> {
    if s.len() % 2 != 0 {
        return Err(DefaultValueError::InvalidLiteral {
            literal: literal.to_string(),
            data_type: dt.clone(),
            reason: format!(
                "hex string must have an even number of characters, got {} character(s)",
                s.len()
            ),
        });
    }
    let mut bytes = Vec::with_capacity(s.len() / 2);
    for chunk in s.as_bytes().chunks(2) {
        let hi = hex_char_to_nibble(chunk[0]).ok_or_else(|| DefaultValueError::InvalidLiteral {
            literal: literal.to_string(),
            data_type: dt.clone(),
            reason: format!("invalid hex character: {:?}", chunk[0] as char),
        })?;
        let lo = hex_char_to_nibble(chunk[1]).ok_or_else(|| DefaultValueError::InvalidLiteral {
            literal: literal.to_string(),
            data_type: dt.clone(),
            reason: format!("invalid hex character: {:?}", chunk[1] as char),
        })?;
        bytes.push((hi << 4) | lo);
    }
    Ok(bytes)
}

/// Convert a single hex ASCII byte to its nibble value, or `None` if invalid.
fn hex_char_to_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Reject default values for Decimal columns with a negative scale.
///
/// `arrow_cast::parse::parse_decimal` interprets the literal as the *unscaled* integer and
/// never applies a negative scale, while `value_as_string` (used by `encode_default`) *does*
/// apply it. As a result a negative-scale literal would not round-trip and would silently
/// store a value `10^|scale|` larger than the user wrote (e.g. `"100"` at scale `-2` stores an
/// unscaled `100`, read back as `"10000"`). Until correct round-tripping is implemented,
/// negative-scale Decimal defaults are rejected at decode-time (which is also the set-time
/// validation path), so such a value can never be persisted incorrectly.
fn reject_negative_decimal_scale(
    literal: &str,
    dt: &DataType,
    scale: i8,
) -> DefaultValueResult<()> {
    if scale < 0 {
        return Err(DefaultValueError::InvalidLiteral {
            literal: literal.to_string(),
            data_type: dt.clone(),
            reason: format!(
                "negative-scale Decimal columns (scale {scale}) do not support default values"
            ),
        });
    }
    Ok(())
}

/// Reject strings that have more fractional digits than the column's declared scale.
///
/// `arrow_cast::parse::parse_decimal` silently truncates extra fractional digits, so we
/// must check this ourselves before calling it.
fn validate_decimal_frac_digits(
    s: &str,
    literal: &str,
    dt: &DataType,
    scale: i8,
) -> DefaultValueResult<()> {
    // The documented decimal default format is a fixed-point string (e.g. "1.23").
    // Scientific/exponent notation is rejected: `parse_decimal` would silently apply
    // the exponent (shifting the decimal point) and then truncate to `scale`, which
    // would persist a value different from what the user wrote. Rejecting 'e'/'E'
    // here also closes the gap where the fractional-digit count below cannot see the
    // exponent (e.g. "1.8e-2" has 1 literal frac digit but an effective scale of 3).
    if s.bytes().any(|b| b == b'e' || b == b'E') {
        return Err(DefaultValueError::InvalidLiteral {
            literal: literal.to_string(),
            data_type: dt.clone(),
            reason: "decimal default must be a fixed-point string (e.g. \"1.23\"); \
                     scientific/exponent notation is not allowed"
                .to_string(),
        });
    }
    // Strip an optional leading sign so we don't confuse '-' with a digit.
    let digits_part = s.trim_start_matches(['+', '-']);
    if let Some(dot_pos) = digits_part.find('.') {
        let frac_str = &digits_part[dot_pos + 1..];
        let frac_len = frac_str.bytes().take_while(|b| b.is_ascii_digit()).count();
        // `scale` can be negative; clamp to 0 so any fractional digit is rejected then.
        let max_frac = scale.max(0) as usize;
        if frac_len > max_frac {
            return Err(DefaultValueError::InvalidLiteral {
                literal: literal.to_string(),
                data_type: dt.clone(),
                reason: format!(
                    "value has {frac_len} fractional digit(s) but column scale is {scale}"
                ),
            });
        }
    }
    Ok(())
}

// ── timestamp helpers ─────────────────────────────────────────────────────────

/// Maximum allowed fractional-second digits for each `TimeUnit`.
fn max_frac_digits(unit: &TimeUnit) -> usize {
    match unit {
        TimeUnit::Second => 0,
        TimeUnit::Millisecond => 3,
        TimeUnit::Microsecond => 6,
        TimeUnit::Nanosecond => 9,
    }
}

/// Count fractional-second digits in an ISO timestamp string.
///
/// Looks for a `.` after the seconds field and counts digits that follow it,
/// stopping at the first non-digit (e.g. `+`, `Z`, or end-of-string).
fn count_frac_digits(s: &str) -> usize {
    // The time portion looks like  HH:MM:SS[.ddd...][offset]
    // Find the last ':' (before seconds) then scan past the two digit seconds.
    // A simpler approach: scan for the first '.' after the 'T' separator.
    let after_t = if let Some(t_pos) = s.find('T').or_else(|| s.find(' ')) {
        &s[t_pos + 1..]
    } else {
        s
    };
    if let Some(dot_pos) = after_t.find('.') {
        after_t[dot_pos + 1..]
            .bytes()
            .take_while(|b| b.is_ascii_digit())
            .count()
    } else {
        0
    }
}

/// Validate that the literal's sub-second precision fits the column's `TimeUnit`.
/// Excess precision is rejected; fewer digits than the unit is fine.
fn validate_timestamp_frac(
    s: &str,
    literal: &str,
    dt: &DataType,
    unit: &TimeUnit,
) -> DefaultValueResult<()> {
    let actual = count_frac_digits(s);
    let allowed = max_frac_digits(unit);
    if actual > allowed {
        return Err(DefaultValueError::InvalidLiteral {
            literal: literal.to_string(),
            data_type: dt.clone(),
            reason: format!(
                "literal has {actual} sub-second digit(s) but column unit {unit:?} allows at most {allowed}"
            ),
        });
    }
    Ok(())
}

/// Decode a timestamp literal into a length-1 Arrow timestamp array.
fn decode_timestamp(
    s: &str,
    json: &str,
    dt: &DataType,
    unit: &TimeUnit,
    tz: Option<&str>,
) -> DefaultValueResult<ArrayRef> {
    // Precision check first (applies to both tz-aware and naive).
    validate_timestamp_frac(s, json, dt, unit)?;

    // Detect whether the literal carries an offset.
    // Strategy: try to parse as offset-aware; if that fails, as naive.
    // Then enforce the column's tz-aware/naive requirement.
    let has_offset = has_utc_offset(s);

    match tz {
        Some(col_tz) => {
            // tz-aware column: literal MUST carry an explicit offset.
            if !has_offset {
                return Err(DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: "timestamp literal for a tz-aware column must include a UTC offset \
                             (e.g. +00:00 or Z)"
                        .to_string(),
                });
            }
            // Parse the offset-bearing literal; any offset is accepted.
            let dt_parsed =
                DateTime::parse_from_rfc3339(s).map_err(|e| DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: format!("failed to parse timestamp: {e}"),
                })?;
            // Compute UTC-epoch value at the column's unit.
            let epoch_value = epoch_value_from_datetime(dt_parsed.into(), json, dt, unit)?;
            let col_tz_arc: Arc<str> = col_tz.into();
            let arr: ArrayRef = match unit {
                TimeUnit::Second => Arc::new(
                    TimestampSecondArray::from(vec![epoch_value]).with_timezone(col_tz_arc),
                ),
                TimeUnit::Millisecond => Arc::new(
                    TimestampMillisecondArray::from(vec![epoch_value]).with_timezone(col_tz_arc),
                ),
                TimeUnit::Microsecond => Arc::new(
                    TimestampMicrosecondArray::from(vec![epoch_value]).with_timezone(col_tz_arc),
                ),
                TimeUnit::Nanosecond => Arc::new(
                    TimestampNanosecondArray::from(vec![epoch_value]).with_timezone(col_tz_arc),
                ),
            };
            Ok(arr)
        }
        None => {
            // tz-naive column: literal MUST NOT carry an offset.
            if has_offset {
                return Err(DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: "timestamp literal for a tz-naive column must not include a UTC offset"
                        .to_string(),
                });
            }
            let ndt = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f")
                .or_else(|_| NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S%.f"))
                .map_err(|e| DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: dt.clone(),
                    reason: format!("failed to parse naive timestamp: {e}"),
                })?;
            let epoch_value = epoch_value_from_naive(ndt, json, dt, unit)?;
            let arr: ArrayRef = match unit {
                TimeUnit::Second => Arc::new(TimestampSecondArray::from(vec![epoch_value])),
                TimeUnit::Millisecond => {
                    Arc::new(TimestampMillisecondArray::from(vec![epoch_value]))
                }
                TimeUnit::Microsecond => {
                    Arc::new(TimestampMicrosecondArray::from(vec![epoch_value]))
                }
                TimeUnit::Nanosecond => Arc::new(TimestampNanosecondArray::from(vec![epoch_value])),
            };
            Ok(arr)
        }
    }
}

/// Encode element 0 of a timestamp array as a JSON string.
fn encode_timestamp(
    arr: &ArrayRef,
    dt: &DataType,
    unit: &TimeUnit,
    tz: Option<&str>,
) -> DefaultValueResult<String> {
    match tz {
        Some(col_tz) => {
            // Parse the timezone string to convert stored UTC epoch → local time.
            // The stored value is always the UTC epoch count at `unit`.
            let epoch_value: i64 = downcast_timestamp_value(arr, dt, unit)?;
            let utc_dt: chrono::DateTime<chrono::Utc> =
                epoch_to_utc_datetime(epoch_value, unit, dt)?;

            // Try to parse col_tz as a fixed-offset (+HH:MM / Z) or IANA name.
            // For round-trip we format in UTC (offset +00:00) when col_tz is "UTC",
            // or as fixed offset when it is parseable as such; otherwise fall back to UTC.
            let s = format_datetime_with_tz(utc_dt, col_tz, unit);
            Ok(serde_json::to_string(&s).expect("string always serializable"))
        }
        None => {
            let epoch_value: i64 = downcast_timestamp_value(arr, dt, unit)?;
            let ndt: NaiveDateTime = epoch_to_naive_datetime(epoch_value, unit, dt)?;
            let s = format_naive_datetime(ndt, unit);
            Ok(serde_json::to_string(&s).expect("string always serializable"))
        }
    }
}

/// Returns `true` if the string ends with a UTC offset (`Z`, `+HH:MM`, or `-HH:MM`).
fn has_utc_offset(s: &str) -> bool {
    if s.ends_with('Z') {
        return true;
    }
    // Look for ±HH:MM at the end (length 6: ±HHMM or ±HH:MM).
    let bytes = s.as_bytes();
    let len = bytes.len();
    // Pattern: last 6 chars are `±HH:MM`
    if len >= 6 {
        let sign_pos = len - 6;
        let candidate = &bytes[sign_pos..];
        if (candidate[0] == b'+' || candidate[0] == b'-')
            && candidate[1].is_ascii_digit()
            && candidate[2].is_ascii_digit()
            && candidate[3] == b':'
            && candidate[4].is_ascii_digit()
            && candidate[5].is_ascii_digit()
        {
            return true;
        }
    }
    false
}

/// Compute the UTC-epoch integer value (in `unit`) from an offset-aware `DateTime<Utc>`.
fn epoch_value_from_datetime(
    dt: chrono::DateTime<chrono::Utc>,
    json: &str,
    data_type: &DataType,
    unit: &TimeUnit,
) -> DefaultValueResult<i64> {
    let v = match unit {
        TimeUnit::Second => dt.timestamp(),
        TimeUnit::Millisecond => dt.timestamp_millis(),
        TimeUnit::Microsecond => dt.timestamp_micros(),
        TimeUnit::Nanosecond => {
            dt.timestamp_nanos_opt()
                .ok_or_else(|| DefaultValueError::InvalidLiteral {
                    literal: json.to_string(),
                    data_type: data_type.clone(),
                    reason: "timestamp is out of range for nanosecond precision".to_string(),
                })?
        }
    };
    Ok(v)
}

/// Compute the epoch integer value (in `unit`) from a `NaiveDateTime`.
///
/// The naive time is treated as a UTC epoch position directly (i.e. it is
/// interpreted as if it were already UTC, not as a local wall-clock time).
fn epoch_value_from_naive(
    ndt: NaiveDateTime,
    json: &str,
    data_type: &DataType,
    unit: &TimeUnit,
) -> DefaultValueResult<i64> {
    // For naive timestamps we treat the wall-clock value directly.
    epoch_value_from_datetime(ndt.and_utc(), json, data_type, unit)
}

/// Downcast an Arrow timestamp array to its i64 value at index 0.
fn downcast_timestamp_value(
    arr: &ArrayRef,
    dt: &DataType,
    unit: &TimeUnit,
) -> DefaultValueResult<i64> {
    Ok(match unit {
        TimeUnit::Second => arr
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .ok_or_else(|| DefaultValueError::TypeMismatch {
                expected: dt.clone(),
                actual: arr.data_type().clone(),
            })?
            .value(0),
        TimeUnit::Millisecond => arr
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .ok_or_else(|| DefaultValueError::TypeMismatch {
                expected: dt.clone(),
                actual: arr.data_type().clone(),
            })?
            .value(0),
        TimeUnit::Microsecond => arr
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| DefaultValueError::TypeMismatch {
                expected: dt.clone(),
                actual: arr.data_type().clone(),
            })?
            .value(0),
        TimeUnit::Nanosecond => arr
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .ok_or_else(|| DefaultValueError::TypeMismatch {
                expected: dt.clone(),
                actual: arr.data_type().clone(),
            })?
            .value(0),
    })
}

/// Convert a stored UTC epoch value (in `unit`) to a `DateTime<Utc>`.
///
/// Uses chrono's own unit constructors so that pre-epoch (negative) values
/// with a non-zero sub-second remainder are handled correctly — the manual
/// arithmetic `((v % unit) * scale) as u32` produces a negative intermediate
/// for those values, which wraps under `as u32` and then causes
/// `timestamp_opt` to return `None`.
fn epoch_to_utc_datetime(
    v: i64,
    unit: &TimeUnit,
    dt: &DataType,
) -> DefaultValueResult<chrono::DateTime<chrono::Utc>> {
    let utc_dt: Option<chrono::DateTime<chrono::Utc>> = match unit {
        TimeUnit::Second => chrono::DateTime::from_timestamp(v, 0),
        TimeUnit::Millisecond => chrono::DateTime::from_timestamp_millis(v),
        TimeUnit::Microsecond => chrono::DateTime::from_timestamp_micros(v),
        // from_timestamp_nanos is infallible in chrono 0.4.41+; wrap in Some
        // so all arms share the same Option<DateTime<Utc>> type.
        TimeUnit::Nanosecond => Some(chrono::DateTime::from_timestamp_nanos(v)),
    };
    utc_dt.ok_or_else(|| DefaultValueError::InvalidLiteral {
        literal: v.to_string(),
        data_type: dt.clone(),
        reason: "stored epoch value is out of range for datetime conversion".to_string(),
    })
}

/// Convert a stored naive epoch value (in `unit`) to a `NaiveDateTime`.
fn epoch_to_naive_datetime(
    v: i64,
    unit: &TimeUnit,
    dt: &DataType,
) -> DefaultValueResult<NaiveDateTime> {
    let utc = epoch_to_utc_datetime(v, unit, dt)?;
    Ok(utc.naive_utc())
}

/// Format a `DateTime<Utc>` for the given timezone string and unit.
///
/// The formatted string must re-decode to the **same UTC epoch value** that
/// was stored (value-stable round-trip).
///
/// - For a **fixed-offset** column tz (e.g. `+05:30`, `-07:00`): convert the
///   UTC instant to that local time and emit the local wall-clock with the
///   offset suffix.  Example: `2024-01-01T00:00:00Z` with `+05:30` encodes to
///   `"2024-01-01T05:30:00+05:30"`, which re-parses to the same UTC instant.
/// - For `"UTC"`, `"Z"`, or any non-fixed-offset / IANA-name string: emit the
///   UTC wall-clock with `+00:00`.  This is always value-stable even though
///   the column's IANA label is not preserved in the text (documented
///   limitation — no IANA DB is linked).
fn format_datetime_with_tz(
    utc_dt: chrono::DateTime<chrono::Utc>,
    col_tz: &str,
    unit: &TimeUnit,
) -> String {
    // Try to parse col_tz as a fixed-offset (e.g. "+05:30", "-07:00").
    // UTC / Z / IANA names are handled by the else-branch below.
    if !col_tz.eq_ignore_ascii_case("utc") && col_tz != "Z" {
        if let Ok(fo) = col_tz.parse::<chrono::FixedOffset>() {
            // Convert the UTC instant to local time in this fixed offset.
            let local_dt: chrono::DateTime<chrono::FixedOffset> = utc_dt.with_timezone(&fo);
            let date_part = local_dt.format("%Y-%m-%dT%H:%M:%S");
            let frac_part = match unit {
                TimeUnit::Second => String::new(),
                TimeUnit::Millisecond => {
                    format!(".{:03}", local_dt.timestamp_subsec_millis())
                }
                TimeUnit::Microsecond => {
                    format!(".{:06}", local_dt.timestamp_subsec_micros())
                }
                TimeUnit::Nanosecond => {
                    format!(".{:09}", local_dt.timestamp_subsec_nanos())
                }
            };
            let total_secs = fo.local_minus_utc();
            let sign = if total_secs >= 0 { '+' } else { '-' };
            let abs = total_secs.unsigned_abs();
            let offset_str = format!("{sign}{:02}:{:02}", abs / 3600, (abs % 3600) / 60);
            return format!("{date_part}{frac_part}{offset_str}");
        }
    }

    // UTC, Z, or unknown IANA name — emit UTC wall-clock with +00:00.
    let date_part = utc_dt.format("%Y-%m-%dT%H:%M:%S");
    let frac_part = match unit {
        TimeUnit::Second => String::new(),
        TimeUnit::Millisecond => format!(".{:03}", utc_dt.timestamp_subsec_millis()),
        TimeUnit::Microsecond => format!(".{:06}", utc_dt.timestamp_subsec_micros()),
        TimeUnit::Nanosecond => format!(".{:09}", utc_dt.timestamp_subsec_nanos()),
    };
    format!("{date_part}{frac_part}+00:00")
}

/// Format a `NaiveDateTime` for a tz-naive column (no offset suffix).
fn format_naive_datetime(ndt: NaiveDateTime, unit: &TimeUnit) -> String {
    let date_part = ndt.format("%Y-%m-%dT%H:%M:%S");
    let utc = ndt.and_utc();
    let frac_part = match unit {
        TimeUnit::Second => String::new(),
        TimeUnit::Millisecond => format!(".{:03}", utc.timestamp_subsec_millis()),
        TimeUnit::Microsecond => format!(".{:06}", utc.timestamp_subsec_micros()),
        TimeUnit::Nanosecond => format!(".{:09}", utc.timestamp_subsec_nanos()),
    };
    format!("{date_part}{frac_part}")
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{Field as ArrowField, Fields};
    use rstest::rstest;

    #[cfg(test)]
    use super::super::field::{
        LANCE_UNENFORCED_PRIMARY_KEY, LANCE_UNENFORCED_PRIMARY_KEY_POSITION,
    };

    // ── round-trip: primitives ────────────────────────────────────────────────

    #[rstest]
    #[case::int8(DataType::Int8, "-128")]
    #[case::int16(DataType::Int16, "-32768")]
    #[case::int32(DataType::Int32, "42")]
    #[case::int64(DataType::Int64, "9007199254740992")]
    #[case::uint8(DataType::UInt8, "255")]
    #[case::uint16(DataType::UInt16, "65535")]
    #[case::uint32(DataType::UInt32, "4294967295")]
    #[case::uint64(DataType::UInt64, "18446744073709551615")]
    #[case::float32(DataType::Float32, "1.5")]
    #[case::float64(DataType::Float64, "3.141592653589793")]
    #[case::bool_true(DataType::Boolean, "true")]
    #[case::bool_false(DataType::Boolean, "false")]
    #[case::utf8(DataType::Utf8, "\"unknown\"")]
    #[case::large_utf8(DataType::LargeUtf8, "\"hello world\"")]
    fn json_roundtrip_primitive(#[case] dt: DataType, #[case] json: &str) {
        let arr = decode_default(json, &dt).unwrap();
        assert_eq!(arr.len(), 1, "array must have exactly one element");
        let encoded = encode_default(&arr).unwrap();
        assert_eq!(encoded, json);
    }

    // ── decode: error cases ───────────────────────────────────────────────────

    #[test]
    fn decode_unsupported_type_returns_error() {
        // Date64 is not yet supported — use it as the sentinel for UnsupportedType.
        let result = decode_default("42", &DataType::Date64);
        assert!(matches!(
            result,
            Err(DefaultValueError::UnsupportedType { .. })
        ));
    }

    #[test]
    fn decode_invalid_json_returns_error() {
        let result = decode_default("not-json", &DataType::Int32);
        assert!(matches!(
            result,
            Err(DefaultValueError::InvalidLiteral { .. })
        ));
    }

    #[test]
    fn decode_type_mismatch_int_returns_error() {
        // JSON string for an integer type
        let result = decode_default("\"hello\"", &DataType::Int32);
        assert!(matches!(
            result,
            Err(DefaultValueError::InvalidLiteral { .. })
        ));
    }

    #[test]
    fn decode_type_mismatch_bool_returns_error() {
        let result = decode_default("42", &DataType::Boolean);
        assert!(matches!(
            result,
            Err(DefaultValueError::InvalidLiteral { .. })
        ));
    }

    // ── Issue 2: out-of-range integer rejection ───────────────────────────────

    #[test]
    fn decode_int8_out_of_range_returns_error() {
        // 200 is valid JSON integer but out of range for i8 (-128..=127)
        let result = decode_default("200", &DataType::Int8);
        assert!(
            matches!(result, Err(DefaultValueError::InvalidLiteral { .. })),
            "expected Err(InvalidLiteral) for 200 in Int8, got {result:?}"
        );
    }

    #[test]
    fn decode_int8_in_range_succeeds() {
        let arr = decode_default("-100", &DataType::Int8).unwrap();
        let arr = arr.as_any().downcast_ref::<Int8Array>().expect("Int8Array");
        assert_eq!(arr.value(0), -100i8);
    }

    #[test]
    fn decode_uint8_out_of_range_returns_error() {
        // 300 is out of range for u8 (0..=255)
        let result = decode_default("300", &DataType::UInt8);
        assert!(
            matches!(result, Err(DefaultValueError::InvalidLiteral { .. })),
            "expected Err(InvalidLiteral) for 300 in UInt8, got {result:?}"
        );
    }

    #[test]
    fn decode_uint8_in_range_succeeds() {
        let arr = decode_default("255", &DataType::UInt8).unwrap();
        let arr = arr
            .as_any()
            .downcast_ref::<UInt8Array>()
            .expect("UInt8Array");
        assert_eq!(arr.value(0), 255u8);
    }

    #[test]
    fn decode_int16_out_of_range_returns_error() {
        let result = decode_default("40000", &DataType::Int16);
        assert!(matches!(
            result,
            Err(DefaultValueError::InvalidLiteral { .. })
        ));
    }

    #[test]
    fn decode_uint16_out_of_range_returns_error() {
        let result = decode_default("70000", &DataType::UInt16);
        assert!(matches!(
            result,
            Err(DefaultValueError::InvalidLiteral { .. })
        ));
    }

    #[test]
    fn decode_int32_out_of_range_returns_error() {
        let result = decode_default("9999999999", &DataType::Int32);
        assert!(matches!(
            result,
            Err(DefaultValueError::InvalidLiteral { .. })
        ));
    }

    #[test]
    fn decode_uint32_out_of_range_returns_error() {
        let result = decode_default("9999999999", &DataType::UInt32);
        assert!(matches!(
            result,
            Err(DefaultValueError::InvalidLiteral { .. })
        ));
    }

    // ── Issue 1: non-finite float rejection ──────────────────────────────────

    #[test]
    fn encode_f64_nan_returns_error() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![f64::NAN]));
        let result = encode_default(&arr);
        assert!(
            matches!(result, Err(DefaultValueError::InvalidLiteral { .. })),
            "expected Err(InvalidLiteral) for NaN, got {result:?}"
        );
    }

    #[test]
    fn encode_f64_infinity_returns_error() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![f64::INFINITY]));
        let result = encode_default(&arr);
        assert!(
            matches!(result, Err(DefaultValueError::InvalidLiteral { .. })),
            "expected Err(InvalidLiteral) for Infinity, got {result:?}"
        );
    }

    #[test]
    fn encode_f64_neg_infinity_returns_error() {
        let arr: ArrayRef = Arc::new(Float64Array::from(vec![f64::NEG_INFINITY]));
        let result = encode_default(&arr);
        assert!(matches!(
            result,
            Err(DefaultValueError::InvalidLiteral { .. })
        ));
    }

    #[test]
    fn encode_f32_nan_returns_error() {
        let arr: ArrayRef = Arc::new(Float32Array::from(vec![f32::NAN]));
        let result = encode_default(&arr);
        assert!(
            matches!(result, Err(DefaultValueError::InvalidLiteral { .. })),
            "expected Err(InvalidLiteral) for f32::NAN, got {result:?}"
        );
    }

    #[test]
    fn encode_f32_infinity_returns_error() {
        let arr: ArrayRef = Arc::new(Float32Array::from(vec![f32::INFINITY]));
        let result = encode_default(&arr);
        assert!(matches!(
            result,
            Err(DefaultValueError::InvalidLiteral { .. })
        ));
    }

    // ── Important 1: empty array rejected by encode_default ──────────────────

    #[test]
    fn encode_empty_array_returns_error() {
        let arr: ArrayRef = Arc::new(Int32Array::from(Vec::<i32>::new()));
        let result = encode_default(&arr);
        assert!(
            matches!(result, Err(DefaultValueError::InvalidLiteral { .. })),
            "expected Err(InvalidLiteral) for empty array, got {result:?}"
        );
    }

    #[test]
    fn encode_null_element_returns_error() {
        // A length-1 array whose only element is null must not silently encode
        // the physical buffer value (0 for ints) as a real default.
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![None]));
        let result = encode_default(&arr);
        assert!(
            matches!(result, Err(DefaultValueError::InvalidLiteral { .. })),
            "expected Err(InvalidLiteral) for null element, got {result:?}"
        );
    }

    // ── Important 2: f32 overflow rejected by decode_default ─────────────────

    #[test]
    fn decode_f32_overflow_returns_error() {
        // 1e39 exceeds f32::MAX (~3.4e38) — must not silently become +Infinity
        let result = decode_default("1e39", &DataType::Float32);
        assert!(
            matches!(result, Err(DefaultValueError::InvalidLiteral { .. })),
            "expected Err(InvalidLiteral) for 1e39 in Float32, got {result:?}"
        );
    }

    #[test]
    fn decode_f32_in_range_succeeds() {
        // Use a value clearly within f32 range; just verify it is finite.
        let arr = decode_default("1.5", &DataType::Float32).unwrap();
        let arr = arr
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("Float32Array");
        let v = arr.value(0);
        assert!(v.is_finite(), "expected finite f32, got {v}");
        assert_eq!(v, 1.5f32);
    }

    // ── decimal round-trip ────────────────────────────────────────────────────

    #[test]
    fn decimal_is_fixed_point_string() {
        let dt = DataType::Decimal128(10, 2);
        assert_eq!(
            encode_default(&decode_default("\"0.00\"", &dt).unwrap()).unwrap(),
            "\"0.00\""
        );
        assert_eq!(
            encode_default(&decode_default("\"-1234.50\"", &dt).unwrap()).unwrap(),
            "\"-1234.50\""
        );
        // bare JSON number rejected (must be quoted string)
        assert!(decode_default("0.00", &dt).is_err());
        // scale overflow (3 frac digits > scale 2) rejected
        assert!(decode_default("\"1.234\"", &dt).is_err());
    }

    #[test]
    fn decimal256_round_trip() {
        let dt = DataType::Decimal256(20, 4);
        // Round-trip: "0.0000" → decode → encode → "0.0000"
        assert_eq!(
            encode_default(&decode_default("\"0.0000\"", &dt).unwrap()).unwrap(),
            "\"0.0000\""
        );
        // Round-trip: "-12345.6789" → decode → encode → "-12345.6789"
        assert_eq!(
            encode_default(&decode_default("\"-12345.6789\"", &dt).unwrap()).unwrap(),
            "\"-12345.6789\""
        );
        // Scale overflow (5 frac digits > scale 4) rejected
        assert!(decode_default("\"1.23456\"", &dt).is_err());
    }

    #[test]
    fn decimal_negative_scale_rejected() {
        // Negative-scale Decimal columns cannot round-trip through parse_decimal /
        // value_as_string (decode stores the unscaled integer, encode applies the
        // scale), so such defaults are rejected rather than silently stored wrong.
        for dt in [DataType::Decimal128(10, -2), DataType::Decimal256(20, -2)] {
            assert!(
                decode_default("\"100\"", &dt).is_err(),
                "expected negative-scale {dt:?} default to be rejected"
            );
            // The higher-level set-time validator must reject it too.
            let field = field_with(dt.clone(), true);
            assert!(
                validate_default_assignable(&field, "\"100\"").is_err(),
                "expected validate_default_assignable to reject negative-scale {dt:?}"
            );
        }
        // Positive/zero scale remains accepted and round-trips correctly.
        let dt = DataType::Decimal128(10, 2);
        assert_eq!(
            encode_default(&decode_default("\"1.23\"", &dt).unwrap()).unwrap(),
            "\"1.23\""
        );
    }

    #[test]
    fn decimal_rejects_scientific_notation() {
        // Scientific/exponent notation would otherwise bypass the fractional-digit
        // scale guard and be silently truncated by parse_decimal (e.g. "1.8e-2"
        // would persist as "0.01" instead of the intended 0.018).
        let dt = DataType::Decimal128(10, 2);
        for lit in [
            "\"1.8e-2\"",
            "\"9.99e-1\"",
            "\"12e-3\"",
            "\"1E2\"",
            "\"1e3\"",
        ] {
            assert!(
                decode_default(lit, &dt).is_err(),
                "expected {lit} to be rejected"
            );
        }
        // Plain fixed-point forms remain accepted / correctly range-checked.
        assert!(decode_default("\"0.01\"", &dt).is_ok());
        assert!(decode_default("\"0.018\"", &dt).is_err());
    }

    // ── encode: error cases ───────────────────────────────────────────────────

    #[test]
    fn encode_unsupported_type_returns_error() {
        // Date64 is not yet supported — use it as the sentinel for UnsupportedType.
        let arr: ArrayRef = Arc::new(arrow_array::Date64Array::from(vec![0i64]));
        let result = encode_default(&arr);
        assert!(matches!(
            result,
            Err(DefaultValueError::UnsupportedType { .. })
        ));
    }

    // ── specific value checks ─────────────────────────────────────────────────

    #[test]
    fn decode_int32_produces_correct_value() {
        let arr = decode_default("42", &DataType::Int32).unwrap();
        let arr = arr
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32Array");
        assert_eq!(arr.value(0), 42);
    }

    #[test]
    fn decode_bool_true_produces_correct_value() {
        let arr = decode_default("true", &DataType::Boolean).unwrap();
        let arr = arr
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("BooleanArray");
        assert!(arr.value(0));
    }

    #[test]
    fn decode_utf8_produces_correct_value() {
        let arr = decode_default("\"unknown\"", &DataType::Utf8).unwrap();
        let arr = arr
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("StringArray");
        assert_eq!(arr.value(0), "unknown");
    }

    // ── timestamp grammar tests (§1.2 rules) ─────────────────────────────────

    #[rstest]
    // tz-aware column with explicit UTC offset — OK
    #[case::tz_aware_ok(
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        "\"2024-01-01T00:00:00.000000+00:00\"",
        true
    )]
    // tz-aware column with literal that lacks an offset — REJECTED
    #[case::tz_aware_needs_offset(
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        "\"2024-01-01T00:00:00.000000\"",
        false
    )]
    // tz-naive column with offset-bearing literal — REJECTED
    #[case::tz_naive_rejects_offset(
        DataType::Timestamp(TimeUnit::Microsecond, None),
        "\"2024-01-01T00:00:00.000000+00:00\"",
        false
    )]
    // microsecond column with nanosecond-precision literal — REJECTED (excess precision)
    #[case::excess_precision(
        DataType::Timestamp(TimeUnit::Microsecond, None),
        "\"2024-01-01T00:00:00.000000001\"",
        false
    )]
    // tz-naive column with no offset — OK
    #[case::tz_naive_ok(
        DataType::Timestamp(TimeUnit::Microsecond, None),
        "\"2024-01-01T00:00:00.000000\"",
        true
    )]
    // tz-aware with non-UTC offset (+05:30) — OK (any offset accepted)
    #[case::tz_aware_non_utc_offset(
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        "\"2024-01-01T05:30:00.000000+05:30\"",
        true
    )]
    // millisecond column allows fewer digits (0 frac digits is fine)
    #[case::ms_zero_frac(
        DataType::Timestamp(TimeUnit::Millisecond, None),
        "\"2024-01-01T00:00:00\"",
        true
    )]
    // millisecond column rejects microsecond precision
    #[case::ms_excess_precision(
        DataType::Timestamp(TimeUnit::Millisecond, None),
        "\"2024-01-01T00:00:00.000001\"",
        false
    )]
    // second column rejects any fractional digits
    #[case::sec_excess_precision(
        DataType::Timestamp(TimeUnit::Second, None),
        "\"2024-01-01T00:00:00.1\"",
        false
    )]
    // nanosecond column accepts 9 digits
    #[case::ns_nine_digits(
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        "\"2024-01-01T00:00:00.123456789\"",
        true
    )]
    // tz-aware with Z suffix — OK
    #[case::tz_aware_z_suffix(
        DataType::Timestamp(TimeUnit::Second, Some("UTC".into())),
        "\"2024-01-01T00:00:00Z\"",
        true
    )]
    fn timestamp_grammar(#[case] dt: DataType, #[case] json: &str, #[case] ok: bool) {
        let result = decode_default(json, &dt);
        assert_eq!(
            result.is_ok(),
            ok,
            "decode_default({json:?}, {dt:?}) -> {result:?}, expected ok={ok}"
        );
    }

    // ── timestamp round-trips ────────────────────────────────────────────────

    #[rstest]
    // tz-aware microsecond round-trip
    #[case::tz_aware_us(
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        "\"2024-01-01T00:00:00.000000+00:00\""
    )]
    // tz-naive microsecond round-trip
    #[case::tz_naive_us(
        DataType::Timestamp(TimeUnit::Microsecond, None),
        "\"2024-01-01T12:34:56.123456\""
    )]
    // tz-naive second round-trip (no fractional part)
    #[case::tz_naive_s(DataType::Timestamp(TimeUnit::Second, None), "\"2024-06-15T08:00:00\"")]
    // tz-naive millisecond round-trip
    #[case::tz_naive_ms(
        DataType::Timestamp(TimeUnit::Millisecond, None),
        "\"2024-06-15T08:00:00.500\""
    )]
    // tz-aware second round-trip
    #[case::tz_aware_s(
        DataType::Timestamp(TimeUnit::Second, Some("UTC".into())),
        "\"2024-06-15T08:00:00+00:00\""
    )]
    fn timestamp_roundtrip(#[case] dt: DataType, #[case] json: &str) {
        let arr = decode_default(json, &dt).expect("decode must succeed");
        assert_eq!(arr.len(), 1, "array must have exactly one element");
        let encoded = encode_default(&arr).expect("encode must succeed");
        assert_eq!(encoded, json, "round-trip mismatch for {dt:?}");
    }

    // ── CRITICAL 1: fixed-offset column timezone round-trip (epoch i64 stable) ──

    #[rstest]
    #[case::plus_05_30(
        DataType::Timestamp(TimeUnit::Microsecond, Some("+05:30".into())),
        // 2024-01-01T00:00:00Z stored as UTC microseconds
        1_704_067_200_000_000i64
    )]
    #[case::minus_07_00(
        DataType::Timestamp(TimeUnit::Microsecond, Some("-07:00".into())),
        // 2024-06-15T12:00:00Z stored as UTC microseconds
        1_718_452_800_000_000i64
    )]
    fn timestamp_fixed_offset_roundtrip(#[case] dt: DataType, #[case] original_epoch: i64) {
        // Build an array directly from the stored epoch value.
        let arr: ArrayRef = match &dt {
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz)) => Arc::new(
                TimestampMicrosecondArray::from(vec![original_epoch]).with_timezone(tz.clone()),
            ),
            _ => panic!("test case misconfigured"),
        };
        // Encode → should produce a local-wall-clock string with the fixed offset.
        let encoded = encode_default(&arr).expect("encode must succeed");
        // Decode → must recover the same UTC epoch value.
        let arr2 = decode_default(&encoded, &dt).expect("decode must succeed");
        let epoch2 = arr2
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("TimestampMicrosecondArray")
            .value(0);
        assert_eq!(
            epoch2, original_epoch,
            "round-trip epoch mismatch for {dt:?}: encoded as {encoded}, \
             original={original_epoch}, recovered={epoch2}"
        );
    }

    // Test that an IANA-named tz column also round-trips by epoch value (UTC text, stable i64).
    #[test]
    fn timestamp_iana_tz_roundtrip_by_value() {
        let dt = DataType::Timestamp(TimeUnit::Microsecond, Some("America/New_York".into()));
        let original_epoch = 1_704_067_200_000_000i64; // 2024-01-01T00:00:00Z
        let arr: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![original_epoch]).with_timezone("America/New_York"),
        );
        let encoded = encode_default(&arr).expect("encode must succeed");
        let arr2 = decode_default(&encoded, &dt).expect("decode must succeed");
        let epoch2 = arr2
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("TimestampMicrosecondArray")
            .value(0);
        assert_eq!(
            epoch2, original_epoch,
            "IANA tz round-trip epoch mismatch: encoded as {encoded}, \
             original={original_epoch}, recovered={epoch2}"
        );
    }

    // ── CRITICAL 2: pre-epoch fractional timestamp encode round-trip ──────────

    #[rstest]
    // Pre-1970, half-second before midnight UTC on 1969-12-31
    // microseconds since epoch: -500_000 (−0.5 s = −500_000 µs)
    #[case::pre_epoch_micros(
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        "\"1969-12-31T23:59:59.500000+00:00\""
    )]
    // Same instant in milliseconds: -500 ms
    #[case::pre_epoch_millis(
        DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
        "\"1969-12-31T23:59:59.500+00:00\""
    )]
    fn timestamp_pre_epoch_fractional_roundtrip(#[case] dt: DataType, #[case] json: &str) {
        // First decode — should not error.
        let arr = decode_default(json, &dt).expect("decode of pre-epoch fractional must succeed");
        // The stored epoch value must be negative.
        let epoch: i64 = match &dt {
            DataType::Timestamp(TimeUnit::Microsecond, _) => arr
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("TimestampMicrosecondArray")
                .value(0),
            DataType::Timestamp(TimeUnit::Millisecond, _) => arr
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .expect("TimestampMillisecondArray")
                .value(0),
            _ => panic!("test case misconfigured"),
        };
        assert!(epoch < 0, "pre-epoch value must be negative, got {epoch}");
        // Encode must succeed.
        let encoded = encode_default(&arr).expect("encode of pre-epoch fractional must succeed");
        // Decode again — epoch must be identical (value-stable).
        let arr2 = decode_default(&encoded, &dt)
            .expect("second decode of pre-epoch fractional must succeed");
        let epoch2: i64 = match &dt {
            DataType::Timestamp(TimeUnit::Microsecond, _) => arr2
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("TimestampMicrosecondArray")
                .value(0),
            DataType::Timestamp(TimeUnit::Millisecond, _) => arr2
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .expect("TimestampMillisecondArray")
                .value(0),
            _ => panic!("test case misconfigured"),
        };
        assert_eq!(
            epoch2, epoch,
            "pre-epoch fractional round-trip epoch mismatch: original={epoch}, \
             encoded={encoded:?}, recovered={epoch2}"
        );
    }

    // ── timestamp: non-UTC offset normalised to UTC epoch ────────────────────

    #[test]
    fn timestamp_non_utc_offset_normalised() {
        // +05:30 literal: 2024-01-01T05:30:00+05:30 == 2024-01-01T00:00:00Z
        let dt = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        let arr = decode_default("\"2024-01-01T05:30:00.000000+05:30\"", &dt).expect("decode ok");
        let arr = arr
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .expect("TimestampMicrosecondArray");
        // Stored UTC epoch for 2024-01-01T00:00:00Z in microseconds
        let expected_epoch = 1_704_067_200_000_000i64;
        assert_eq!(arr.value(0), expected_epoch);
    }

    // ── date32 tests ─────────────────────────────────────────────────────────

    #[test]
    fn date32_round_trip() {
        let dt = DataType::Date32;
        let json = "\"2024-01-01\"";
        let arr = decode_default(json, &dt).expect("decode must succeed");
        assert_eq!(arr.len(), 1);
        let encoded = encode_default(&arr).expect("encode must succeed");
        assert_eq!(encoded, json);
    }

    #[test]
    fn date32_epoch() {
        let dt = DataType::Date32;
        // 1970-01-01 → day 0
        let arr = decode_default("\"1970-01-01\"", &dt).expect("decode ok");
        let arr = arr
            .as_any()
            .downcast_ref::<Date32Array>()
            .expect("Date32Array");
        assert_eq!(arr.value(0), 0);
    }

    #[test]
    fn date32_positive_days() {
        let dt = DataType::Date32;
        // 1970-01-02 → day 1
        let arr = decode_default("\"1970-01-02\"", &dt).expect("decode ok");
        let arr = arr
            .as_any()
            .downcast_ref::<Date32Array>()
            .expect("Date32Array");
        assert_eq!(arr.value(0), 1);
    }

    #[test]
    fn date32_negative_days() {
        let dt = DataType::Date32;
        // 1969-12-31 → day -1
        let arr = decode_default("\"1969-12-31\"", &dt).expect("decode ok");
        let arr = arr
            .as_any()
            .downcast_ref::<Date32Array>()
            .expect("Date32Array");
        assert_eq!(arr.value(0), -1);
    }

    #[test]
    fn date32_malformed_rejects() {
        let dt = DataType::Date32;
        assert!(decode_default("\"not-a-date\"", &dt).is_err());
        assert!(decode_default("\"2024/01/01\"", &dt).is_err());
        // bare JSON number rejected
        assert!(decode_default("20240101", &dt).is_err());
    }

    #[rstest]
    #[case::epoch("\"1970-01-01\"")]
    #[case::recent("\"2024-06-22\"")]
    #[case::before_epoch("\"1969-07-20\"")]
    fn date32_roundtrip_cases(#[case] json: &str) {
        let dt = DataType::Date32;
        let arr = decode_default(json, &dt).expect("decode must succeed");
        let encoded = encode_default(&arr).expect("encode must succeed");
        assert_eq!(encoded, json);
    }

    // ── binary / large-binary / fixed-size-binary hex round-trip ─────────────

    #[test]
    fn binary_hex_roundtrip() {
        let dt = DataType::Binary;
        assert_eq!(
            encode_default(&decode_default("\"0A0B0C0D\"", &dt).unwrap()).unwrap(),
            "\"0A0B0C0D\""
        );
    }

    #[test]
    fn large_binary_hex_roundtrip() {
        let dt = DataType::LargeBinary;
        assert_eq!(
            encode_default(&decode_default("\"DEADBEEF\"", &dt).unwrap()).unwrap(),
            "\"DEADBEEF\""
        );
    }

    #[test]
    fn binary_empty_roundtrip() {
        // Zero-length binary is valid — empty hex string
        let dt = DataType::Binary;
        assert_eq!(
            encode_default(&decode_default("\"\"", &dt).unwrap()).unwrap(),
            "\"\""
        );
    }

    #[test]
    fn fixed_size_binary_preserves_width() {
        let dt = DataType::FixedSizeBinary(4);
        assert!(decode_default("\"0A0B0C0D\"", &dt).is_ok()); // 4 bytes ok
        assert!(decode_default("\"0A0B\"", &dt).is_err()); // 2 bytes != width 4 -> error
    }

    #[test]
    fn fixed_size_binary_hex_roundtrip() {
        let dt = DataType::FixedSizeBinary(4);
        assert_eq!(
            encode_default(&decode_default("\"0A0B0C0D\"", &dt).unwrap()).unwrap(),
            "\"0A0B0C0D\""
        );
    }

    #[test]
    fn binary_lowercase_hex_accepted() {
        // Lowercase hex input is accepted; output is normalised to uppercase.
        let dt = DataType::Binary;
        assert_eq!(
            encode_default(&decode_default("\"0a0b0c0d\"", &dt).unwrap()).unwrap(),
            "\"0A0B0C0D\""
        );
    }

    #[test]
    fn binary_odd_length_hex_rejected() {
        // Odd-length hex string must be rejected.
        assert!(decode_default("\"0A0\"", &DataType::Binary).is_err());
    }

    #[test]
    fn binary_invalid_hex_char_rejected() {
        // Characters outside [0-9A-Fa-f] must be rejected.
        assert!(decode_default("\"ZZ\"", &DataType::Binary).is_err());
    }

    // ── non-finite float decode rejection ─────────────────────────────────────

    #[test]
    fn non_finite_float_decode_rejected() {
        // Bare NaN / Infinity / -Infinity are not valid JSON; serde_json::from_str
        // fails and the error is mapped to InvalidLiteral (not a panic).
        assert!(decode_default("NaN", &DataType::Float64).is_err());
        assert!(decode_default("Infinity", &DataType::Float64).is_err());
        assert!(decode_default("-Infinity", &DataType::Float64).is_err());
    }

    #[test]
    fn non_finite_float32_decode_rejected() {
        assert!(decode_default("NaN", &DataType::Float32).is_err());
        assert!(decode_default("Infinity", &DataType::Float32).is_err());
        assert!(decode_default("-Infinity", &DataType::Float32).is_err());
    }

    // ── validate_default_assignable ───────────────────────────────────────────

    /// Construct a `Field` from the given Arrow `DataType` and `nullable` flag.
    fn field_with(dt: DataType, nullable: bool) -> Field {
        Field::try_from(&ArrowField::new("c", dt, nullable))
            .expect("field construction must succeed")
    }

    #[test]
    fn validate_accepts_nullable_scalar() {
        let f = field_with(DataType::Int32, true);
        assert!(
            validate_default_assignable(&f, "42").is_ok(),
            "nullable Int32 with integer literal must be accepted"
        );
    }

    /// A non-nullable field with a valid literal must now be accepted — the default
    /// value is what satisfies the non-null contract for absent (pre-existing) rows.
    #[test]
    fn validate_accepts_non_nullable_with_default() {
        let f = field_with(DataType::Int32, false);
        assert!(
            validate_default_assignable(&f, "42").is_ok(),
            "non-nullable Int32 with valid literal must be accepted"
        );
    }

    #[test]
    fn validate_rejects_nested_types() {
        // Struct is nested → rejected
        let s = field_with(DataType::Struct(Fields::empty()), true);
        assert!(
            matches!(
                validate_default_assignable(&s, "null"),
                Err(DefaultValueError::NestedNotSupported { .. })
            ),
            "Struct must be rejected as nested"
        );

        // List is nested → rejected
        let l = field_with(
            DataType::List(Arc::new(ArrowField::new("i", DataType::Int32, true))),
            true,
        );
        assert!(
            matches!(
                validate_default_assignable(&l, "[]"),
                Err(DefaultValueError::NestedNotSupported { .. })
            ),
            "List must be rejected as nested"
        );
    }

    #[test]
    fn validate_rejects_dictionary_and_ree() {
        // Dictionary → rejected as physically-encoded.
        let d = field_with(
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        );
        assert!(
            matches!(
                validate_default_assignable(&d, "\"x\""),
                Err(DefaultValueError::PhysicalEncodingNotSupported { .. })
            ),
            "Dictionary must be rejected as physically-encoded"
        );

        // Note: RunEndEncoded is also rejected by the `PhysicalEncodingNotSupported` branch
        // in `validate_default_assignable`, but the Lance `Field` type itself cannot represent
        // a RunEndEncoded column (LogicalType conversion fails), so no separate test is needed
        // — the defensive check in the function guards against hypothetical future changes.
    }

    #[test]
    fn validate_rejects_unenforced_pk() {
        // Legacy truthy key → rejected
        let mut f = field_with(DataType::Int32, true);
        f.metadata
            .insert(LANCE_UNENFORCED_PRIMARY_KEY.to_string(), "true".to_string());
        assert!(
            matches!(
                validate_default_assignable(&f, "42"),
                Err(DefaultValueError::PrimaryKeyNotAllowed { .. })
            ),
            "unenforced PK (legacy bool) must be rejected"
        );

        // Position-only PK → rejected
        let mut f2 = field_with(DataType::Int32, true);
        f2.metadata.insert(
            LANCE_UNENFORCED_PRIMARY_KEY_POSITION.to_string(),
            "1".to_string(),
        );
        assert!(
            matches!(
                validate_default_assignable(&f2, "42"),
                Err(DefaultValueError::PrimaryKeyNotAllowed { .. })
            ),
            "unenforced PK (position-only) must be rejected"
        );
    }

    #[test]
    fn validate_rejects_uncastable_literal() {
        let f = field_with(DataType::Int32, true);
        let result = validate_default_assignable(&f, "\"not an int\"");
        assert!(
            result.is_err(),
            "string literal for Int32 field must be rejected, got {result:?}"
        );
    }

    // ── Iceberg conformance ───────────────────────────────────────────────────
    //
    // These cases lock Lance's behavior against Apache Iceberg's reference
    // single-value codec. The vectors come from Iceberg's own test:
    //   core/src/test/java/org/apache/iceberg/TestSingleValueParser.java
    //   (method `testValidDefaults`, the scalar subset).
    //
    // The codec is intentionally aligned with Iceberg's `SingleValueParser` with
    // a few DELIBERATE divergences (scalars only; uppercase hex; decimal
    // restrictions; lenient timestamp offsets). Each `expected` below is Lance's
    // ACTUAL canonical output (captured by running the codec), and the comment
    // records whether that is byte-identical to Iceberg (parity) or a documented
    // textual divergence.

    /// PARITY + textual-divergence round-trips: decode the Iceberg JSON vector,
    /// re-encode, and assert Lance's ACTUAL canonical string.
    #[rstest]
    // parity with Iceberg "true"  (Boolean → true)
    #[case::iceberg_boolean(DataType::Boolean, "true", "true")]
    // parity with Iceberg "1"  (Integer → Int32)
    #[case::iceberg_int32(DataType::Int32, "1", "1")]
    // parity with Iceberg "9999999"  (Long → Int64)
    #[case::iceberg_int64(DataType::Int64, "9999999", "9999999")]
    // parity with Iceberg "1.23"  (Float → Float32)
    #[case::iceberg_float32(DataType::Float32, "1.23", "1.23")]
    // parity with Iceberg "123.456"  (Double → Float64)
    #[case::iceberg_float64(DataType::Float64, "123.456", "123.456")]
    // parity with Iceberg "2007-12-03"  (Date → Date32)
    #[case::iceberg_date32(DataType::Date32, "\"2007-12-03\"", "\"2007-12-03\"")]
    // divergence from Iceberg "2007-12-03T10:15:30": Lance always emits the full
    // sub-second field for the column's unit, so a Microsecond column pads to
    // ".000000" (same instant).
    #[case::iceberg_ts_micro_naive(
        DataType::Timestamp(TimeUnit::Microsecond, None),
        "\"2007-12-03T10:15:30\"",
        "\"2007-12-03T10:15:30.000000\""
    )]
    // divergence from Iceberg "2007-12-03T10:15:30+00:00": Lance pads the
    // micros field to ".000000" (same instant, same +00:00 offset).
    #[case::iceberg_ts_micro_utc(
        DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
        "\"2007-12-03T10:15:30+00:00\"",
        "\"2007-12-03T10:15:30.000000+00:00\""
    )]
    // parity with Iceberg "2007-12-03T10:15:30.123456789"  (TimestampNano withoutZone)
    #[case::iceberg_ts_nano_naive(
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        "\"2007-12-03T10:15:30.123456789\"",
        "\"2007-12-03T10:15:30.123456789\""
    )]
    // parity with Iceberg "2007-12-03T10:15:30.123456789+00:00"  (TimestampNano withZone)
    #[case::iceberg_ts_nano_utc(
        DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
        "\"2007-12-03T10:15:30.123456789+00:00\"",
        "\"2007-12-03T10:15:30.123456789+00:00\""
    )]
    // parity with Iceberg "foo"  (String → Utf8)
    #[case::iceberg_string(DataType::Utf8, "\"foo\"", "\"foo\"")]
    // divergence from Iceberg "111f" (lowercase hex): Lance normalises hex output
    // to UPPERCASE → "111F" (same bytes 0x11 0x1f).
    #[case::iceberg_fixed2(DataType::FixedSizeBinary(2), "\"111f\"", "\"111F\"")]
    // divergence from Iceberg "0000ff" (lowercase hex): Lance emits "0000FF".
    #[case::iceberg_binary(DataType::Binary, "\"0000ff\"", "\"0000FF\"")]
    // parity with Iceberg "123.4500"  (Decimal(9,4))
    #[case::iceberg_decimal_9_4(DataType::Decimal128(9, 4), "\"123.4500\"", "\"123.4500\"")]
    // parity with Iceberg "2"  (Decimal(9,0))
    #[case::iceberg_decimal_9_0(DataType::Decimal128(9, 0), "\"2\"", "\"2\"")]
    fn iceberg_scalar_roundtrip(
        #[case] dt: DataType,
        #[case] iceberg_json: &str,
        #[case] expected: &str,
    ) {
        let arr = decode_default(iceberg_json, &dt).expect("decode of Iceberg vector must succeed");
        assert_eq!(arr.len(), 1, "array must have exactly one element");
        let encoded = encode_default(&arr).expect("encode must succeed");
        assert_eq!(
            encoded, expected,
            "Iceberg vector {iceberg_json:?} for {dt:?} re-encoded to {encoded:?}, \
             expected {expected:?}"
        );
    }

    /// Lance accepts ANY timestamp offset and normalises to the column tz, where
    /// Iceberg requires UTC (`testInvalidTimestamptz` rejects "+01:00"). A
    /// non-UTC offset Iceberg would reject is accepted and the instant is
    /// normalised; re-encoding yields the equivalent UTC wall-clock.
    ///
    /// Source: contrast with TestSingleValueParser.java::testInvalidTimestamptz.
    #[test]
    fn iceberg_divergence_timestamp_any_offset_normalised() {
        let dt = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
        // 2007-12-03T10:15:30+05:00 == 2007-12-03T05:15:30Z
        let arr = decode_default("\"2007-12-03T10:15:30+05:00\"", &dt)
            .expect("Lance accepts a non-UTC offset Iceberg would reject");
        let encoded = encode_default(&arr).expect("encode must succeed");
        assert_eq!(encoded, "\"2007-12-03T05:15:30.000000+00:00\"");
    }

    /// Iceberg accepts Decimal(9,-20) default "2E+20" (negative scale +
    /// scientific notation); Lance REJECTS both. `decode_default` and the
    /// set-time validator must both error.
    ///
    /// Source: TestSingleValueParser.java vector {DecimalType.of(9, -20), "2E+20"}.
    #[test]
    fn iceberg_divergence_decimal_negative_scale_and_scientific() {
        // Exact Iceberg vector: negative scale is rejected.
        let neg = DataType::Decimal128(9, -20);
        assert!(
            decode_default("\"2E+20\"", &neg).is_err(),
            "negative-scale Decimal default must be rejected"
        );
        // Scientific notation is rejected even on a representable positive-scale column.
        let pos = DataType::Decimal128(20, 0);
        assert!(
            decode_default("\"2E+20\"", &pos).is_err(),
            "scientific-notation Decimal default must be rejected"
        );
    }

    /// Iceberg accepts nested defaults (list/map/struct); Lance is scalars-only.
    /// `validate_default_assignable` rejects a struct field, and a struct field
    /// is the addressing target a dotted scalar leaf would resolve to (dotted
    /// addressing is rejected at the higher set/create layer).
    ///
    /// Source: TestSingleValueParser.java list/map/struct vectors.
    #[test]
    fn iceberg_divergence_nested_defaults_rejected() {
        // List default — Iceberg accepts [1,2,3]; Lance rejects the nested column.
        let list = field_with(
            DataType::List(Arc::new(ArrowField::new("e", DataType::Int32, true))),
            true,
        );
        assert!(
            matches!(
                validate_default_assignable(&list, "[1, 2, 3]"),
                Err(DefaultValueError::NestedNotSupported { .. })
            ),
            "List default must be rejected (scalars only)"
        );

        // Struct default — Iceberg accepts {"4":1,"5":"bar"}; Lance rejects it.
        let struct_field = field_with(
            DataType::Struct(Fields::from(vec![
                ArrowField::new("f1", DataType::Int32, false),
                ArrowField::new("f2", DataType::Utf8, true),
            ])),
            true,
        );
        assert!(
            matches!(
                validate_default_assignable(&struct_field, "{\"f1\": 1, \"f2\": \"bar\"}"),
                Err(DefaultValueError::NestedNotSupported { .. })
            ),
            "Struct default must be rejected (scalars only)"
        );
        // The same struct field rejects a scalar leaf addressed within it: the
        // resolved target type is still the Struct, which is nested.
        assert!(
            matches!(
                validate_default_assignable(&struct_field, "1"),
                Err(DefaultValueError::NestedNotSupported { .. })
            ),
            "a scalar leaf nested in a struct must still be rejected at the struct boundary"
        );
    }

    /// Iceberg lists a Boolean default of `null`; Lance's codec REJECTS a bare
    /// JSON `null` as a value (the literal must be a concrete value of the
    /// column type; column nullability is a schema concern, not a default value).
    ///
    /// Source: TestSingleValueParser.java vector {BooleanType.get(), "null"}.
    #[test]
    fn iceberg_divergence_json_null_default_rejected() {
        assert!(
            matches!(
                decode_default("null", &DataType::Boolean),
                Err(DefaultValueError::InvalidLiteral { .. })
            ),
            "a bare JSON null default must be rejected for a typed column"
        );
    }

    /// Iceberg has UUID and Time scalar types; Lance's codec supports neither
    /// (no native Arrow mapping wired into the codec). Verified here as a
    /// divergence/skip: Lance treats the closest Arrow analogues as unsupported.
    ///
    /// Source: TestSingleValueParser.java vectors {UUIDType, "..."} and
    /// {TimeType, "10:15:30"}.
    #[test]
    fn iceberg_divergence_uuid_and_time_unsupported() {
        // Time: Arrow Time64(Microsecond) is the closest analogue; the codec has
        // no arm for it → UnsupportedType.
        assert!(
            matches!(
                decode_default("\"10:15:30\"", &DataType::Time64(TimeUnit::Microsecond)),
                Err(DefaultValueError::UnsupportedType { .. })
            ),
            "Time defaults are not supported by Lance's codec"
        );
        // UUID: Iceberg stores UUID as a 16-byte value; FixedSizeBinary(16) is the
        // closest Arrow analogue, but the UUID *text* form Iceberg uses is not hex
        // and is rejected as an invalid hex literal (not silently accepted).
        assert!(
            decode_default(
                "\"eb26bdb1-a1d8-4aa6-990e-da940875492c\"",
                &DataType::FixedSizeBinary(16)
            )
            .is_err(),
            "Iceberg's UUID text form is not accepted by Lance's binary codec"
        );
    }
}
