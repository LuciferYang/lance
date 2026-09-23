// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// A dictionary column whose value type has a colon-bearing logical type
// (e.g. Decimal128) used to write successfully and then panic the reader,
// because the manifest logical type string could never be parsed back. The
// write must fail up front instead.
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt8Type;
use arrow_array::{Decimal128Array, Int32Array, RecordBatch, RecordBatchIterator};
use arrow_cast::cast;
use arrow_schema::{DataType, Field, Schema as ArrowSchema};

#[tokio::test]
async fn decimal_valued_dictionary_write_is_rejected() {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "d",
        DataType::Dictionary(
            Box::new(DataType::UInt8),
            Box::new(DataType::Decimal128(10, 2)),
        ),
        true,
    )]));

    let values = Decimal128Array::from(vec![Some(100), Some(200), Some(300)])
        .with_precision_and_scale(10, 2)
        .unwrap();
    let keys = Int32Array::from(vec![Some(0), Some(1), Some(2), Some(1)]);
    let keys = cast(&keys, &DataType::UInt8).unwrap();
    let dict = arrow_array::DictionaryArray::<UInt8Type>::new(
        keys.as_primitive().clone(),
        Arc::new(values),
    );

    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(dict)]).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

    let dir = tempfile::tempdir().unwrap();
    let path = format!("file://{}", dir.path().to_str().unwrap());

    let result = lance::Dataset::write(reader, &path, None).await;
    let err = result.expect_err("decimal-valued dictionary must be rejected at write");
    assert!(
        err.to_string().contains("Dictionary value type"),
        "unexpected error: {err}"
    );
}
