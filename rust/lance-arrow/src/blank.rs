// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Minimum-size placeholder values.
//!
//! Some write paths have to materialize a row they will never read back: the
//! updater, for instance, rewrites a fragment column-by-column and must emit a
//! row for every physical row the fragment has, including the deleted ones. The
//! value does not matter, but its type and its nullability do, so a null is not
//! always allowed and copying a real row can be expensive.

use arrow_array::{ArrayRef, make_array, new_null_array};
use arrow_data::{ArrayData, ArrayDataBuilder};
use arrow_schema::{DataType, Field, TimeUnit};

/// Builds a one-row array holding the smallest value `field` can hold: a null
/// where the field allows one, and the smallest non-null value of its type
/// where it does not. A `Null` field yields a null either way, since that is the
/// only value the type has.
///
/// "Smallest non-null" means no payload beyond what the type forces: an empty
/// string or byte slice, an empty list, zero for a fixed-width type. A
/// `FixedSizeList` still costs its full width, since the type has no way to
/// express a shorter value.
///
/// Nullability is honored at every level, so a nullable child stays null inside
/// a non-nullable parent. That matters wherever a writer dispatches on a child
/// being present rather than on its contents: a blob descriptor with a non-null
/// empty `uri` is read as an external reference to `""`, while a null `uri` is
/// read as the inline blob it is.
///
/// Returns `None` for a type this has no minimal value for: `Union`,
/// `RunEndEncoded` and the view list types at any depth, and a type no valid
/// array can have, such as a dictionary keyed by something other than an
/// integer. [`is_supported`] is the whole list.
pub fn minimal_value(field: &Field) -> Option<ArrayRef> {
    if !is_supported(field.data_type()) {
        return None;
    }
    if field.data_type() == &DataType::Null {
        return Some(new_null_array(field.data_type(), 1));
    }
    let data = ArrayData::new_null(field.data_type(), 1);
    Some(make_array(minimal_from_null(data, field.is_nullable())?))
}

/// Builds a one-row array of `data_type` holding the smallest non-null value of
/// that type, for a caller that has a type rather than a field. `Null` is the
/// exception, since it has no non-null value.
fn minimal_non_null_array(data_type: &DataType) -> Option<ArrayRef> {
    minimal_value(&Field::new("", data_type.clone(), false))
}

/// Whether a placeholder can be built for `data_type` at all. Two kinds of
/// exclusion, and both have to hold for every node of the type tree:
///
/// - `Union`, `RunEndEncoded` and the view lists, which have no minimal value
///   worth defining here and which no Lance schema can hold. `new_null` also
///   unwraps a union's first variant and rejects a run-end type outside
///   `Int16`/`Int32`/`Int64` with `unreachable!`.
/// - Types no valid array can have but a `DataType` can still spell: a
///   dictionary keyed by a non-integer (`new_null` unwraps the key width, and a
///   `Utf8` or `Struct` key has none), a map
///   whose entries are not a two-field struct (`MapArray::from` panics), a
///   negative fixed width (`width as usize` becomes a huge allocation), and a
///   time unit arrow has no array for — `primitive_width` reports the same width
///   for every unit, so only `make_array` turns `Time32(Nanosecond)` away, and it
///   does so with `unimplemented!`.
fn is_supported(data_type: &DataType) -> bool {
    match data_type {
        DataType::Union(_, _)
        | DataType::RunEndEncoded(_, _)
        | DataType::ListView(_)
        | DataType::LargeListView(_) => false,
        DataType::Time32(unit) => matches!(unit, TimeUnit::Second | TimeUnit::Millisecond),
        DataType::Time64(unit) => matches!(unit, TimeUnit::Microsecond | TimeUnit::Nanosecond),
        DataType::Dictionary(key_type, value_type) => {
            key_type.is_dictionary_key_type() && is_supported(value_type)
        }
        DataType::Map(entries, _) => {
            matches!(entries.data_type(), DataType::Struct(fields) if fields.len() == 2)
                && is_supported(entries.data_type())
        }
        DataType::FixedSizeBinary(width) => *width >= 0,
        DataType::FixedSizeList(field, width) => *width >= 0 && is_supported(field.data_type()),
        DataType::List(field) | DataType::LargeList(field) => is_supported(field.data_type()),
        DataType::Struct(fields) => fields.iter().all(|field| is_supported(field.data_type())),
        _ => true,
    }
}

/// Turns `new_null`'s zeroed payload into the value described above: keeps the
/// validity mask where the field is nullable and drops it where it is not, and
/// gives a dictionary the single values entry its key 0 points at, which
/// `new_null` leaves empty. Returns `None` if the result would not be a valid
/// array.
fn minimal_from_null(data: ArrayData, nullable: bool) -> Option<ArrayData> {
    let children = match data.data_type() {
        DataType::Dictionary(_, value_type) => vec![minimal_non_null_array(value_type)?.to_data()],
        DataType::Struct(fields) => data
            .child_data()
            .iter()
            .zip(fields.iter())
            .map(|(child, field)| minimal_from_null(child.clone(), field.is_nullable()))
            .collect::<Option<Vec<_>>>()?,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => data
            .child_data()
            .iter()
            .map(|child| minimal_from_null(child.clone(), field.is_nullable()))
            .collect::<Option<Vec<_>>>()?,
        _ => data
            .child_data()
            .iter()
            .map(|child| minimal_from_null(child.clone(), false))
            .collect::<Option<Vec<_>>>()?,
    };
    let mut builder = ArrayDataBuilder::from(data).child_data(children);
    if !nullable {
        builder = builder.nulls(None);
    }
    builder.build().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, ArrayRef, StringArray, cast::AsArray};
    use arrow_schema::{Field, Fields, TimeUnit};
    use rstest::rstest;
    use std::sync::Arc;

    fn byte_size(array: &ArrayRef) -> usize {
        array.to_data().buffers().iter().map(|b| b.len()).sum()
    }

    fn dictionary_type() -> DataType {
        DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8))
    }

    fn map_type(value_type: DataType) -> DataType {
        DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Field::new("keys", DataType::Utf8, false),
                        Field::new("values", value_type, true),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        )
    }

    fn union_type() -> DataType {
        let fields =
            std::iter::once((0i8, Arc::new(Field::new("a", DataType::Int32, true)))).collect();
        DataType::Union(fields, arrow_schema::UnionMode::Sparse)
    }

    fn run_end_encoded(run_ends: DataType) -> DataType {
        DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", run_ends, false)),
            Arc::new(Field::new("values", DataType::Int32, true)),
        )
    }

    #[rstest]
    #[case::boolean(DataType::Boolean)]
    #[case::int32(DataType::Int32)]
    #[case::uint64(DataType::UInt64)]
    #[case::float32(DataType::Float32)]
    #[case::decimal128(DataType::Decimal128(20, 4))]
    #[case::date32(DataType::Date32)]
    #[case::time64(DataType::Time64(TimeUnit::Microsecond))]
    #[case::timestamp_tz(DataType::Timestamp(
        TimeUnit::Nanosecond,
        Some("America/New_York".into())
    ))]
    #[case::duration(DataType::Duration(TimeUnit::Second))]
    #[case::utf8(DataType::Utf8)]
    #[case::large_utf8(DataType::LargeUtf8)]
    #[case::binary(DataType::Binary)]
    #[case::large_binary(DataType::LargeBinary)]
    #[case::utf8_view(DataType::Utf8View)]
    #[case::binary_view(DataType::BinaryView)]
    #[case::fixed_size_binary(DataType::FixedSizeBinary(16))]
    #[case::list(DataType::List(Arc::new(Field::new("item", DataType::Int32, true))))]
    #[case::large_list(DataType::LargeList(Arc::new(Field::new("item", DataType::Int32, true))))]
    #[case::fixed_size_list(DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, false)),
        8
    ))]
    #[case::struct_non_nullable(DataType::Struct(
        vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]
        .into()
    ))]
    #[case::dictionary(DataType::Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8)))]
    #[case::struct_of_dictionary(DataType::Struct(
        vec![Field::new("d", dictionary_type(), false)].into()
    ))]
    #[case::fixed_size_list_of_dictionary(DataType::FixedSizeList(
        Arc::new(Field::new("item", dictionary_type(), false)),
        4
    ))]
    #[case::list_of_dictionary(DataType::List(Arc::new(Field::new(
        "item",
        dictionary_type(),
        false
    ))))]
    #[case::dictionary_of_null(DataType::Dictionary(
        Box::new(DataType::UInt8),
        Box::new(DataType::Null)
    ))]
    #[case::map(map_type(DataType::Int32))]
    #[case::map_of_dictionary(map_type(dictionary_type()))]
    fn minimal_value_is_one_row_with_no_null_of_its_own(#[case] data_type: DataType) {
        let array = minimal_non_null_array(&data_type)
            .unwrap_or_else(|| panic!("no minimal value for {data_type}"));
        assert_eq!(array.data_type(), &data_type);
        assert_eq!(array.len(), 1);
        assert_eq!(
            array.null_count(),
            0,
            "a placeholder must carry no null of its own"
        );
        array.to_data().validate_full().unwrap();
    }

    #[rstest]
    #[case::utf8(DataType::Utf8)]
    #[case::binary(DataType::Binary)]
    #[case::large_binary(DataType::LargeBinary)]
    #[case::large_utf8(DataType::LargeUtf8)]
    #[case::list(DataType::List(Arc::new(Field::new("item", DataType::Int32, true))))]
    #[case::large_list(DataType::LargeList(Arc::new(Field::new("item", DataType::Int32, true))))]
    fn variable_width_placeholder_carries_no_payload(#[case] data_type: DataType) {
        let array = minimal_non_null_array(&data_type).unwrap();
        // Offsets for one value plus the terminator, and nothing else. This is
        // the property that keeps a blank row from consuming offset space in a
        // 32-bit offset column.
        let offset_width = match &data_type {
            DataType::LargeBinary | DataType::LargeUtf8 | DataType::LargeList(_) => 8,
            _ => 4,
        };
        assert_eq!(byte_size(&array), 2 * offset_width);
    }

    #[test]
    fn fixed_size_list_placeholder_is_zeroed_and_valid() {
        let data_type =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 4);
        let array = minimal_non_null_array(&data_type).unwrap();
        let list = array.as_fixed_size_list();
        assert_eq!(list.value_length(), 4);
        let values = list.value(0);
        assert_eq!(
            values.null_count(),
            0,
            "a non-nullable child must stay valid"
        );
        assert_eq!(
            values
                .as_primitive::<arrow_array::types::Float32Type>()
                .values(),
            &[0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn dictionary_placeholder_points_at_a_real_entry() {
        let array = minimal_non_null_array(&dictionary_type()).unwrap();
        let dict = array.as_dictionary::<arrow_array::types::UInt8Type>();
        assert_eq!(dict.keys().value(0), 0);
        let values = dict.values().as_string::<i32>();
        assert_eq!(values.len(), 1, "key 0 needs an entry to point at");
        assert_eq!(values.value(0), "");
    }

    #[test]
    fn nested_dictionary_keeps_its_length_and_gets_an_entry() {
        // The values array has to be spliced into the dictionary node, not
        // substituted for it: under a `FixedSizeList` the node itself still
        // owes the parent `list_size` keys.
        let data_type =
            DataType::FixedSizeList(Arc::new(Field::new("item", dictionary_type(), false)), 4);
        let array = minimal_non_null_array(&data_type).unwrap();
        let dict = array
            .as_fixed_size_list()
            .values()
            .as_dictionary::<arrow_array::types::UInt8Type>();
        assert_eq!(dict.len(), 4, "one key per list slot");
        assert_eq!(dict.keys().values(), &[0, 0, 0, 0]);
        assert_eq!(dict.values().len(), 1, "key 0 needs an entry to point at");
    }

    #[test]
    fn struct_placeholder_strips_validity_recursively() {
        let data_type = DataType::Struct(
            vec![
                Field::new("a", DataType::Int32, false),
                Field::new("b", DataType::Utf8, false),
            ]
            .into(),
        );
        let array = minimal_non_null_array(&data_type).unwrap();
        let st = array.as_struct();
        assert_eq!(st.column(0).null_count(), 0);
        assert_eq!(st.column(1).null_count(), 0);
        assert_eq!(
            st.column(0)
                .as_primitive::<arrow_array::types::Int32Type>()
                .value(0),
            0
        );
        assert_eq!(st.column(1).as_string::<i32>().value(0), "");
    }

    #[test]
    fn null_type_yields_a_null() {
        // `Null` has no non-null value, so the field's nullability makes no
        // difference. A `NullArray` carries no validity buffer, so the null shows
        // up as a logical null rather than in `null_count`.
        let array = minimal_non_null_array(&DataType::Null).unwrap();
        assert_eq!(array.len(), 1);
        assert_eq!(array.null_count(), 0);
        assert_eq!(array.logical_null_count(), 1);
    }

    #[test]
    fn a_nullable_field_yields_a_null() {
        // The cheapest value a nullable field can hold, and the one a writer
        // reads as "absent" rather than as an empty value.
        let array = minimal_value(&Field::new("s", DataType::Utf8, true)).unwrap();
        assert_eq!(array.len(), 1);
        assert_eq!(array.null_count(), 1);
    }

    #[test]
    fn a_nullable_child_stays_null_inside_a_non_nullable_parent() {
        // The shape of a blob descriptor: a writer that dispatches on `uri`
        // being present must not see an empty string there, or it resolves the
        // blank row as an external reference to "".
        let data_type = DataType::Struct(
            vec![
                Field::new("data", DataType::LargeBinary, true),
                Field::new("uri", DataType::Utf8, true),
                Field::new("size", DataType::UInt64, false),
            ]
            .into(),
        );
        let array = minimal_value(&Field::new("blob", data_type, false)).unwrap();
        let blob = array.as_struct();
        assert_eq!(blob.null_count(), 0, "the parent field is not nullable");
        assert_eq!(blob.column(0).null_count(), 1, "data");
        assert_eq!(blob.column(1).null_count(), 1, "uri");
        assert_eq!(
            blob.column(2).null_count(),
            0,
            "a non-nullable child still gets a value"
        );
        array.to_data().validate_full().unwrap();
    }

    #[test]
    fn unsupported_types_are_reported_not_guessed() {
        // Only the types the doc comment names, and each one both bare and
        // nested.
        for data_type in [
            union_type(),
            DataType::Union(
                arrow_schema::UnionFields::empty(),
                arrow_schema::UnionMode::Sparse,
            ),
            run_end_encoded(DataType::Int32),
            run_end_encoded(DataType::Int8),
            DataType::ListView(Arc::new(Field::new("item", DataType::Int32, true))),
            DataType::LargeListView(Arc::new(Field::new("item", DataType::Int32, true))),
            // `MapArray::from` panics unless the entries are a two-field struct.
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(vec![Field::new("keys", DataType::Utf8, false)].into()),
                    false,
                )),
                false,
            ),
            // A negative width becomes a huge allocation in `new_null`.
            DataType::FixedSizeBinary(-1),
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, false)), -1),
            // `primitive_width` is blind to the time unit, so these reach
            // `make_array`, which has no arm for them.
            DataType::Time32(TimeUnit::Microsecond),
            DataType::Time32(TimeUnit::Nanosecond),
            DataType::Time64(TimeUnit::Second),
            DataType::Time64(TimeUnit::Millisecond),
        ] {
            assert!(
                minimal_non_null_array(&data_type).is_none(),
                "{data_type} must return None"
            );
            for nested in [
                DataType::Struct(Fields::from(vec![Field::new("f", data_type.clone(), true)])),
                DataType::List(Arc::new(Field::new("item", data_type.clone(), true))),
                DataType::FixedSizeList(Arc::new(Field::new("item", data_type, true)), 2),
            ] {
                assert!(
                    minimal_non_null_array(&nested).is_none(),
                    "{nested} must return None"
                );
            }
        }
    }

    #[rstest]
    #[case::null(DataType::Null)]
    #[case::utf8(DataType::Utf8)]
    #[case::float32(DataType::Float32)]
    #[case::struct_key(DataType::Struct(
        vec![Field::new("a", DataType::Int32, true)].into()
    ))]
    fn dictionary_with_a_non_integer_key_returns_none(#[case] key_type: DataType) {
        // No valid array can have such a key, but a `DataType` can spell it, and
        // `ArrayData::new_null` unwraps the key width, which `Null`, `Utf8` and
        // `Struct` do not have — so the screen has to run before arrow sees those,
        // nesting included. A `Float32` key has a width, and the checked build
        // rejects it instead.
        let dictionary = DataType::Dictionary(Box::new(key_type), Box::new(DataType::Utf8));
        let item = Arc::new(Field::new("item", dictionary.clone(), false));
        for data_type in [
            dictionary.clone(),
            DataType::Struct(Fields::from(vec![Field::new("d", dictionary, false)])),
            DataType::FixedSizeList(item.clone(), 3),
            DataType::List(item),
        ] {
            assert!(
                minimal_non_null_array(&data_type).is_none(),
                "{data_type} must return None"
            );
        }
    }

    #[test]
    fn interleaving_a_placeholder_adds_no_bytes() {
        // The property the updater depends on: splicing placeholders between
        // live rows leaves the output's byte payload equal to the live rows'.
        let live: ArrayRef = Arc::new(StringArray::from(vec!["alpha", "beta"]));
        let blank = minimal_non_null_array(&DataType::Utf8).unwrap();
        let out = arrow_select::interleave::interleave(
            &[live.as_ref(), blank.as_ref()],
            &[(0, 0), (1, 0), (0, 1), (1, 0)],
        )
        .unwrap();
        let out = out.as_string::<i32>();
        assert_eq!(out.len(), 4);
        assert_eq!(out.value(0), "alpha");
        assert_eq!(out.value(1), "");
        assert_eq!(out.value(2), "beta");
        assert_eq!(out.value(3), "");
        assert_eq!(
            out.value_offsets().last().copied().unwrap() as usize,
            "alpha".len() + "beta".len(),
            "placeholders must not consume offset space"
        );
    }
}
