// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{collections::HashMap, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, criterion_group, criterion_main};
use lance_encoding::{
    encoder::{EncodingOptions, default_encoding_strategy, encode_batch},
    version::LanceFileVersion,
};

fn bench_encode_compressed(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("encode_compressed");

    const NUM_ROWS: usize = 5_000_000;
    const NUM_COLUMNS: usize = 10;

    // Generate compressible string data - high cardinality but compressible
    // (unique values to avoid dictionary encoding, repeated prefix for compression)
    let array: Arc<dyn arrow_array::Array> = Arc::new(arrow_array::StringArray::from_iter_values(
        (0..NUM_ROWS).map(|i| format!("prefix_that_compresses_well_{}", i)),
    ));

    for compression in ["zstd", "lz4"] {
        let mut metadata = HashMap::new();
        metadata.insert(
            "lance-encoding:compression".to_string(),
            compression.to_string(),
        );
        // Disable dictionary encoding to ensure we hit the compression path
        metadata.insert(
            "lance-encoding:dict-divisor".to_string(),
            "100000".to_string(),
        );
        // Force miniblock encoding (the path that benefits from compressor caching)
        metadata.insert(
            "lance-encoding:structural-encoding".to_string(),
            "miniblock".to_string(),
        );
        let fields: Vec<Field> = (0..NUM_COLUMNS)
            .map(|i| {
                Field::new(format!("s{}", i), DataType::Utf8, false).with_metadata(metadata.clone())
            })
            .collect();
        let columns: Vec<Arc<dyn arrow_array::Array>> =
            (0..NUM_COLUMNS).map(|_| array.clone()).collect();
        let schema = Arc::new(Schema::new(fields));
        let data = RecordBatch::try_new(schema.clone(), columns).unwrap();

        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
        // V2_2+ required for general compression
        let encoding_strategy = default_encoding_strategy(LanceFileVersion::V2_2);

        group.throughput(criterion::Throughput::Elements(
            (NUM_ROWS * NUM_COLUMNS) as u64,
        ));
        group.bench_function(
            format!("{}_strings_{}cols", compression, NUM_COLUMNS),
            |b| {
                b.iter(|| {
                    rt.block_on(encode_batch(
                        &data,
                        lance_schema.clone(),
                        encoding_strategy.as_ref(),
                        &EncodingOptions::default(),
                    ))
                    .unwrap()
                })
            },
        );
    }
}

/// Benchmark Decimal128 inline-bitpacking encode across the four non-trivial
/// kernel arms of the per-chunk u128 dispatch (NarrowU32 / NarrowU64 /
/// SequentialU128 / Memcpy). Symmetric to `bench_decode_decimal128` in
/// `decoder.rs`: for each `bit_width`, every chunk's `Stat::BitWidth` is
/// forced to the target arm so the encoder exercises the corresponding
/// pack kernel.
fn bench_encode_decimal128(c: &mut Criterion) {
    use arrow_array::Decimal128Array;

    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("encode_decimal128");

    const NUM_ROWS: u64 = 5_000_000;
    /// Fixed seed for reproducibility — see `decoder.rs::bench_decode_decimal128`.
    const SEED: u64 = 0xDEAD_BEEF;

    group.throughput(criterion::Throughput::Bytes(
        NUM_ROWS * std::mem::size_of::<i128>() as u64,
    ));

    let cases: &[(&str, u32)] = &[
        ("bw024_narrow_u32", 24),
        ("bw040_narrow_u64", 40),
        ("bw100_sequential_u128", 100),
        ("bw128_memcpy", 128),
    ];

    for &(label, bw) in cases {
        let values: Vec<i128> = generate_decimal128_values(NUM_ROWS as usize, bw, SEED);
        let array: Arc<dyn arrow_array::Array> = Arc::new(
            Decimal128Array::from(values)
                .with_precision_and_scale(38, 0)
                .unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "d",
            DataType::Decimal128(38, 0),
            false,
        )]));
        let data = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
        let encoding_strategy = default_encoding_strategy(LanceFileVersion::V2_2);

        group.bench_function(label, |b| {
            b.iter(|| {
                rt.block_on(encode_batch(
                    &data,
                    lance_schema.clone(),
                    encoding_strategy.as_ref(),
                    &EncodingOptions::default(),
                ))
                .unwrap()
            })
        });
    }
}

/// See `decoder.rs::generate_decimal128_values` — kept in sync. Forces every
/// 1024-element chunk's `Stat::BitWidth` to the target arm so the encoder's
/// `pack_u128_chunk` dispatch routes deterministically to NarrowU32 (1 ≤ bw
/// ≤ 32), NarrowU64 (33 ≤ bw ≤ 64), SequentialU128 (65 ≤ bw ≤ 127), or
/// Memcpy (bw = 128). For the Memcpy case, values are bounded negatives in
/// `[-(2^64), -1]`: the sign bit alone guarantees `Stat::BitWidth = 128`,
/// while the bounded magnitude stays within Decimal128(38, 0) precision
/// regardless of arrow's data-validation policy.
fn generate_decimal128_values(num_rows: usize, target_bw: u32, seed: u64) -> Vec<i128> {
    let mut state = seed | 1;
    (0..num_rows)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let lo = state as u128;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let hi = state as u128;
            let raw_u128: u128 = (hi << 64) | lo;
            if target_bw == 128 {
                -1i128 - ((raw_u128 as u64) as i128)
            } else {
                let mask: u128 = (1u128 << target_bw) - 1;
                let high_bit: u128 = 1u128 << (target_bw - 1);
                ((raw_u128 & mask) | high_bit) as i128
            }
        })
        .collect()
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
    targets = bench_encode_compressed, bench_encode_decimal128);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_encode_compressed, bench_encode_decimal128);

criterion_main!(benches);
