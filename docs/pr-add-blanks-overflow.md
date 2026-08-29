# fix(updater): fill blank rows with each field's minimal value

## What broke

Rewriting a fragment that has deleted rows failed with

```
LanceError(Arrow): Failed to add blanks: Offset overflow error: 2148272826
```

`add_blanks` has to emit one row for every physical row, so it splices a placeholder into each deleted row's slot. It built that placeholder by copying row 0 of the batch, which means a column holding a large value paid for that value once per deleted row. A 32-bit offset column then crossed `i32::MAX` as soon as the copies added up: 17 blanks over a 128 MiB value is 2.125 GiB.

## The fix

A blank row now carries each field's minimal value, which is a null where the field is nullable and otherwise the smallest non-null value its type allows. The rows are spliced with one `interleave_batches` call instead of a per-column `take`. `interleave` charges only the source row's own bytes, so a blank adds no payload at all to a byte or list column.

Nullability is honored at every level, because a writer can dispatch on whether a child is present rather than on what it contains. A blob descriptor is `Struct<{data: LargeBinary?, uri: Utf8?, ...}>`, and a non-null empty `uri` there reads as an external reference to `""` while a null `uri` reads as the inline blob it is. So a nullable child stays null even inside a non-nullable parent. That leaves one case for the blob writer to settle: a present descriptor that names neither data nor a uri is an empty blob rather than an absent one. Without that, a non-nullable blob column could not be rewritten at all once its fragment had deletions. It is the only change outside the updater.

Dictionary keys have to keep indexing the values array they arrived with. `interleave` merges its sources' dictionaries and renumbers every key, while the v1 writer persists a dictionary's values once from the schema and writes each batch's keys as they come. Dictionaries are therefore replaced by a cheap stub before the interleave, since arrow's merge cannot address more values than the key type holds and panics inside `MutableArrayData` when it overflows. Every dictionary leaf is then rebuilt from the live batch with `take`. The rebuild walks down to the leaves instead of taking whole columns, because `take` copies everything under the node it is handed: restoring a whole `Struct<{Dictionary, Binary}>` would put the binary payload back on the blank rows and bring the overflow with it.

`add_blanks` also rejects offsets that go backwards or run past the batch. Those used to wrap and then over-allocate.

## Tests

`cargo test -p lance-arrow --lib blank` (46), `cargo test -p lance --lib dataset::updater` (16), `cargo test -p lance --lib dataset::blob` (96), `cargo test -p lance --lib` (3162), `cargo clippy --all --tests --benches -- -D warnings`, `cargo fmt --all --check`.

The regression test for the reported failure allocates about 384 MiB, so it is `#[ignore]`d. Run manually it passes, and with the old copy-row-0 behaviour restored it fails with `Offset overflow error: 2147483648`. `add_blanks_does_not_grow_binary_payload` asserts the same property at 1 MiB in CI.

Every property above has a test that fails when its mechanism is removed: the payload not growing for a dictionary's sibling at two nesting depths, keys surviving under `Struct`, `List`, `LargeList`, `Map` and `FixedSizeList`, a saturated `UInt8` dictionary, a sliced batch with a null row, and a nullable child staying null.

## Related

Two open PRs touch `add_blanks`, both orthogonal to this one: #7318 (empty input batches and oversized restored batches) and #8612 (deferring blanks when a fully deleted batch has no row to copy). Whichever lands second needs a small conflict resolution in that function.

## Not fixed here

Rewriting a dictionary column through `FileFragment::update_columns` on a Legacy dataset is already broken, independently of blank rows: the file's dictionary values come from the fragment schema while the keys written come from the right-hand stream's own dictionary. With no deletions at all, so with no blank row in play, a four-row `Dictionary(UInt8, Utf8)` column reads back `[beta, gamma, alpha, delta]` where it should read `[alpha, beta, omega, delta]`. That is out of scope here and left alone.
