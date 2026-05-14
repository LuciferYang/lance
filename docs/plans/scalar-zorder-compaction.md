# Implementation plan: native Z-order/Hilbert compaction for scalar columns

## Background

`lance.dataset.optimize.compact_files` today **does not reorder data** (`rust/lance/src/dataset/optimize.rs:732`: "Compacts the files in the dataset without reordering them"). The only existing reorder primitive is `HilbertSorter` at `rust/lance-index/src/scalar/rtree/sort/hilbert_sort.rs:29`, which is wired exclusively into the r-tree spatial-index build path (`rust/lance-index/src/scalar/rtree.rs:898`) and only operates on geo columns.

Downstream consumers (notably `lance-spark`'s zonemap-based Dynamic File Pruning) want **multi-column scalar clustering** so each fragment's zonemap min/max becomes tight on more than one dimension simultaneously. Without this, single-column lex clustering only narrows the primary dimension; secondary dimensions remain wide and DFP can't prune on them.

This plan adds a **scalar Z-order compaction mode** to `compact_files` so a dataset can be re-clustered along multiple scalar columns in one pass.

## Scope

### In scope (MVP)

- Generalize the `Sorter` trait so it lives outside the rtree namespace.
- New `ZOrderSorter` for **scalar tuples** of two or more integer-typed columns (`Int32`, `Int64`, `UInt32`, `UInt64`, `Date32`, `Date64`, `Timestamp*`). Other scalar types via cast.
- New `CompactionMode::ZOrder { columns }` variant; wire into `compact_files`.
- Per-task: scan rows, compute a transient Morton-encoded column, sort by it, write the sorted batch as a new fragment.
- Unit tests: encoder correctness, ordering monotonicity, null handling.
- Integration test: round-trip `compact_files` with `ZOrder { columns: ["date_sk", "item_sk"] }`, verify (a) row count preserved, (b) per-fragment zonemap on `item_sk` tightens vs. baseline, (c) per-fragment zonemap on `date_sk` does not regress catastrophically.

### Out of scope (later phases)

- **Hilbert variant** — same shape as Z-order, different encoder. Defer until Z-order is validated; the Hilbert encoder is materially more work and the curve-quality gain is modest for typical 2D clustering at sf=100.
- **String columns** as cluster keys (would need prefix encoding to fixed width).
- **Java JNI binding** for the new mode — separate PR after the Rust side lands.
- **lance-spark `DfpClusterRebuilder` consumer flag** — separate lance-spark PR after JNI lands.
- **Persisting the morton column on disk** — transient virtual column only; recomputing from source columns is cheap.

## Detailed design

### Step 1: Generalize the `Sorter` trait

Current (`rust/lance-index/src/scalar/rtree/sort.rs:11-13`):
```rust
#[async_trait]
pub trait Sorter {
    async fn sort(&self, data: SendableRecordBatchStream) -> Result<SendableRecordBatchStream>;
}
```

**Move** to a more general home — `rust/lance-index/src/sort.rs` or `rust/lance-core/src/sort.rs`. Keep the trait shape unchanged; just stop nesting it inside the rtree-scalar namespace.

Update `HilbertSorter` to import from the new location. No behavior change.

### Step 2: New module `rust/lance-index/src/cluster/`

```
rust/lance-index/src/cluster/
├── mod.rs       — public re-exports of ZOrderSorter, ColumnSpec
├── zorder.rs    — ZOrderSorter impl + morton encoder
└── encode.rs    — per-type encoders (Int32, Int64, Date32, Timestamp*) → u32/u64 normalized to bit-width
```

#### `ColumnSpec` (in `cluster/mod.rs`)

```rust
pub struct ColumnSpec {
    pub name: String,
    /// Number of bits to use from this column when computing the morton code.
    /// Higher → more resolution on this dimension. Default heuristic: log2(distinct_count) rounded up.
    pub bits: u32,
}
```

#### `ZOrderSorter` (in `cluster/zorder.rs`)

```rust
pub struct ZOrderSorter {
    columns: Vec<ColumnSpec>,
}

impl ZOrderSorter {
    pub fn new(columns: Vec<ColumnSpec>) -> Result<Self> {
        // Validate: at least 2 columns, total bits ≤ 64.
    }
}

#[async_trait]
impl Sorter for ZOrderSorter {
    async fn sort(&self, data: SendableRecordBatchStream) -> Result<SendableRecordBatchStream> {
        // Mirror HilbertSorter::sort structure (rust/lance-index/src/scalar/rtree/sort/hilbert_sort.rs:40):
        // 1. Add a transient `_morton` UInt64 column via ProjectionExec + a new ZOrderUDF.
        // 2. SortExec by `_morton` ascending.
        // 3. ProjectionExec back, dropping `_morton`.
    }
}

struct ZOrderUDF { columns: Vec<ColumnSpec> }
impl ScalarUDFImpl for ZOrderUDF {
    // For each row, normalize each column value to its [0, 2^bits) range,
    // bit-interleave into a u64, return.
    // For null inputs: treat as 0 (option for the future — NULL FIRST/LAST).
}
```

**Encoder details** (`cluster/encode.rs`): for each `(value, bits)` pair, normalize to `[0, 2^bits)`:
- Integer types: rescale `(value - min) * (2^bits / range)` where `min` and `range` are computed once per batch via Arrow `min/max` kernels (or pre-known for monotonic types like `Date32`).
- Cast non-integer scalars to `i64` first; reject unsupported types.

**Morton interleaver**: standard bit-interleave of N column codes into one u64. For 2 columns of 32 bits each, well-known SIMD-friendly tables (e.g. `0x5555555555555555` masks). Cite: <https://en.wikipedia.org/wiki/Z-order_curve#Coordinate_values>.

### Step 3: `CompactionMode` extension

`rust/lance/src/dataset/optimize.rs` (around line 152). Current shape (paraphrased):
```rust
pub enum CompactionMode {
    Reencode,
    TryBinaryCopy,
    ForceBinaryCopy,
}
```

**Add**:
```rust
pub enum CompactionMode {
    Reencode,
    TryBinaryCopy,
    ForceBinaryCopy,
    /// Re-encode rows after sorting by a Z-order curve over the named columns.
    /// Each fragment that's re-written gets its rows in the new globally-sorted
    /// order; per-fragment zonemap min/max tightens on every column listed.
    ZOrder { columns: Vec<ColumnSpec> },
}
```

Add the parsing branch in the `CompactionMode::FromStr`-like impl (around line 140) so users can pass `"zorder:col1,col2"` from CLI tools.

### Step 4: Wire into `compact_files`

`rust/lance/src/dataset/optimize.rs:743` (`compact_files` → `rewrite_files` → batch stream). Track down `rewrite_files` — likely in `rust/lance/src/dataset/optimize/rewrite.rs` based on the `mod rewrite;` import at line 80-something. At the point where `SendableRecordBatchStream` is built from `scanner.try_into_stream()` (~line 908):

```rust
let mut data: SendableRecordBatchStream = ...; // existing
if let Some(CompactionMode::ZOrder { columns }) = &options.compaction_mode {
    let sorter = ZOrderSorter::new(columns.clone())?;
    data = sorter.sort(data).await?;
}
// proceed with existing write path
```

Constraint: `ZOrder` is incompatible with `TryBinaryCopy` / `ForceBinaryCopy` (those modes skip re-encoding entirely; we need re-encoding to reorder). Reject the combination up front.

### Step 5: Tests

#### Unit (`rust/lance-index/src/cluster/zorder.rs::tests`)

1. **Encoder correctness**: feed a tiny batch with `(date_sk, item_sk) = [(1, 5), (1, 6), (2, 0), (2, 1)]`, assert the produced `_morton` column is monotonically increasing.
2. **Ordering monotonicity**: feed a randomized batch of 10k rows; assert that after `sort`, every consecutive pair `(curr, prev)` satisfies `morton(curr) >= morton(prev)`.
3. **Null handling**: feed a batch with nulls in one of the columns; assert sort completes and nulls land at one consistent end.
4. **Bit-width clamping**: feed values that exceed the declared `bits`; assert they're clamped to the top range without panicking.

#### Integration (`rust/lance/src/dataset/optimize.rs::tests`)

1. Build a 10k-row dataset with `(date_sk: 0..100, item_sk: 0..100)` × density 1, written into 4 fragments via the default writer.
2. Call `compact_files` with `mode = ZOrder { columns: ["date_sk", "item_sk"] }`.
3. Re-open the dataset, scan each fragment, and assert:
   - **Row count preserved** (10k total, across whatever fragment count emerges).
   - **Per-fragment min/max of `date_sk`** is at most 1/sqrt(num_fragments) of the global range (within a constant tolerance).
   - **Per-fragment min/max of `item_sk`** is also at most 1/sqrt(num_fragments) of the global range.
   - Compare against a baseline `compact_files` without reorder: at least one of the two columns should have per-fragment range ≥ 50% of global (the unsorted dimension).

#### End-to-end zonemap interaction

Optional bonus: after the integration test's reorder, build a zonemap on `item_sk` using the existing `ZoneMapIndexBuilder`, and assert `getZonemapStats("item_sk")` returns zone min/max bounds that match the per-fragment bounds asserted above. This proves the reorder actually narrows zonemap pruning.

## Implementation phasing

Suggested order for the executing agent:

| Phase | What | Effort | PR scope |
|---|---|---|---|
| **1** | Promote `Sorter` trait out of rtree namespace | 0.5 day | Trivial refactor; no behavior change |
| **2** | New `cluster/zorder.rs` + `cluster/encode.rs` + unit tests | 1.5 days | All scalar Z-order machinery, no compaction wiring yet |
| **3** | `CompactionMode::ZOrder` + `compact_files` integration + integration test | 1 day | The user-visible feature |
| **4** | Documentation + `compaction.md`-style doc page | 0.5 day | Coverage |

Total: **~3.5 engineering days** for the MVP Rust PR.

Each phase is independently mergeable. Phase 1 is a refactor that should land first to keep the diff small; phases 2–4 can be one PR or stacked PRs depending on review preference.

## Acceptance criteria (for the MVP PR)

- [ ] `Sorter` trait lives outside `scalar::rtree::sort`; rtree code still compiles and tests pass.
- [ ] `ZOrderSorter::sort` passes the four unit tests above.
- [ ] `cargo clippy --all-targets -- -D warnings` clean.
- [ ] `cargo fmt --all --check` clean.
- [ ] `cargo test -p lance -p lance-index` green.
- [ ] Integration test verifies per-fragment range tightening on both columns after `compact_files` with `ZOrder`.
- [ ] `CompactionMode::ZOrder` incompatible with `TryBinaryCopy`/`ForceBinaryCopy` raises `invalid_input` early.
- [ ] A new doc page (or section of an existing one) explains when to use the mode and what the trade-offs are.

## Open questions the implementing agent should resolve

1. **Bit-width auto-selection**: should `ColumnSpec.bits` default to `log2(distinct_count_estimate)` (cheap but inexact), or require the caller to specify? My lean: require explicit. The default is too easy to get wrong, and the existing Lance APIs all require column-shape parameters explicitly.
2. **Pre-pass min/max computation**: the encoder needs `(min, range)` per column to normalize. Options: (a) one Arrow `min/max` kernel pass over the input stream before the sort (doubles I/O), (b) pass `min`/`max` as part of `ColumnSpec` (caller's responsibility), (c) approximate via reservoir sample. My lean: (b) — the caller (compaction planner) already has dataset stats.
3. **Nulls**: where do they land? `Date32` and `Timestamp*` columns sometimes carry nulls in TPC-DS-like workloads. Match the existing `SortExec` default (nulls last) and document it.
4. **Mixed-fragment lex vs. global Z-order**: the integration test should clarify whether `compact_files` runs **per-task** sort (each task sorts only its input fragments) or **global** sort across all input fragments. Per-task is what the existing pipeline does; global would require a shuffle stage we don't have. **MVP: per-task**, documented as such — users who want a global sort should call `compact_files` once with `target_rows_per_fragment` set so all fragments compact into one task, then re-split downstream.
5. **Interaction with `defer_index_remap`**: reordering invalidates row-id-based indexes if there are any. Confirm whether the existing index-remap path handles reordered fragments correctly; if not, the MVP should refuse to run when indexes are present and `defer_index_remap=false`.

## Anchors for the agent

Concrete files to read first:

- `rust/lance-index/src/scalar/rtree/sort/hilbert_sort.rs` — closest existing implementation; the `sort()` method body at line 40 is the template.
- `rust/lance-index/src/scalar/rtree/sort.rs` — the `Sorter` trait definition.
- `rust/lance/src/dataset/optimize.rs:152` — `CompactionOptions` shape and the `compaction_mode` enum.
- `rust/lance/src/dataset/optimize.rs:743` — `compact_files` entry point and the `rewrite_files` dispatch.
- The rewrite-time data flow: search for `SendableRecordBatchStream::from(scanner.try_into_stream())` in `optimize.rs` and follow it to where the write happens.
