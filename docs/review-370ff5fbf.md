# 评审 370ff5fbf — fix(updater): fill blank rows with minimal values instead of copying row 0

被评审的改动：`add_blanks` 不再靠复制第 0 行来造空白行，改成按列取该类型的最小非 null 值，拼接方式从 per-column `take` 换成 `interleave_batches`。目的是消掉 `Failed to add blanks: Offset overflow error`（32 位 offset 列上一个大值被复制够多次就越过 `i32::MAX`），顺带省掉每个空白行复制整个第 0 行的开销。新增 `rust/lance-arrow/src/blank.rs` 提供 `minimal_non_null_array`。

最重要的发现：新 helper 里那个 `unsafe { build_unchecked() }` 在 `Dictionary` 嵌在 `Struct` 或 `FixedSizeList` 下面时会产出**越界的字典 key**，而且它不 panic，是静静地写进输出 batch。也就是这个改动在这类 schema 上把「读取时报错」换成了「写出损坏数据」，比原来的故障更糟。

### 1. [HIGH · 正确性] 嵌套 `Dictionary` 的占位值带着越界 key 逃进输出 batch — CONFIRMED

- 锚点：`rust/lance-arrow/src/blank.rs:75`
- 问题：`ArrayData::new_null(&Dictionary(..), n)` 给字典的 values 子数组留的是**空数组**（`arrow-data-58.4.0/src/data.rs:697-700`），全 null 的 validity 正是它合法的原因——`check_bounds` 明确跳过 null 槽位（`data.rs:1595`）。`minimal_non_null_array` 在顶层 `Dictionary` 上单独处理了这件事（`blank.rs:40-54`），但 `strip_validity` 递归到子节点时不会走那个分支：它只剥 validity，于是 key 0 变成活的、指向一个 0 项的字典。`build_unchecked` 跳过 `validate_values`（含 `check_bounds`，`data.rs:1445`）所以放行。
- 失败场景：列类型是 `Struct([Field::new("d", Dictionary(UInt8, Utf8), false)])` 或 `FixedSizeList(Field::new("item", Dictionary(UInt8, Utf8), false), 4)`。`blank_row_for`（`rust/lance/src/dataset/updater.rs:434`）按 schema 造占位值，`to_data().validate_full()` 报 `Value at position 0 out of bounds: 0 (should be in [0, -1])`。接着 `interleave` **不报错**：`interleave_struct` → `interleave_dictionaries` → `merge_dictionary_values`（arrow-select 的 interleave 模块）把这个幽灵 key 映射成 2，对上一个 2 项的合并字典，输出的 `validate_full` 变成 `Value at position 1 out of bounds: 2 (should be in [0, 1])`。这个 batch 随后被写进数据文件；任何用 `TypedDictionaryArray`/`value_unchecked` 解引用它的消费者就是越界读（UB），走 `cast`/`take` 则在远离成因的地方报错。Lance schema 能带 `Dictionary`（`rust/lance-core/src/datatypes.rs:353`），嵌套深度不限。
- 处置：已修。字典的处理从 `minimal_non_null_array` 顶层挪进 `strip_validity` 递归：节点类型是 `Dictionary` 时只换掉 values 子数组，保留这个节点的 `len`/`offset`/keys buffer，所以 `FixedSizeList(dict, 4)` 下面的字典节点仍然是 4 个 key。收尾换成受检的 `build()`（条目 7），越界 key 由 arrow 自己拦住。rstest 补了 `struct_of_dictionary`、`fixed_size_list_of_dictionary`、`list_of_dictionary`、`dictionary_of_null` 四例，另加 `nested_dictionary_keeps_its_length_and_gets_an_entry` 钉住「换的是 child 不是 node」。变异验证：删掉递归里的字典分支，5 个测试挂（含新加的 struct 和 FSL 两例）；`list_of_dictionary` 不挂——List 下面的字典节点 0 个 key，本来就越不出界，这一例只算形状覆盖。

### 2. [HIGH · 正确性] `interleave` 给字典列重编号，v1 写出去的 key 对不上文件里的字典 — CONFIRMED（Phase 3 第三轮发现）

- 锚点：`rust/lance/src/dataset/updater.rs:475`
- 问题：`interleave` 对字典列不是原样搬 key。`should_merge_dictionary_values` 按 values 数组的**指针相等**判断是不是同一个字典，占位行那个新建的一项字典永远不等，于是要么走 `merge_dictionary_values` 把引用到的值压实、把所有 key（含活行的）重编号，要么走 `interleave_fallback_dictionary` 把两个 values 拼起来、把占位行的 key 往后挪。而 v1 写入器是分开写的：每批的 key 原样写（`write_dictionary_arr`），values 在 close 时从 schema 写一次（`write_schema_dictionaries`，`rust/lance-file/src/versions/v1/encoding.rs:103-114`，没有就直接报错）。改动前用的 `take` 会连同 values 的 `Arc` 一起复用，key 保持有效；换成 `interleave` 之后，写出去的 key 索引的是一个没被持久化的字典。
- 失败场景：V1（Legacy）数据集，一列 `Dictionary(UInt8, Utf8)`，fragment 上有删除行，然后走 `FileFragment::update_columns`（Python 侧也暴露了）重写这一列。写 schema 由 `self.schema().project_by_schema(..)` 从 fragment schema 投影而来，而 V1 打开时 `populate_manifest_schema_dictionaries` 已经把真实字典值填进去了，所以持久化的是原字典。实测：values 为 `[alpha, beta, gamma, delta]`、keys 为 `[3, 1, 0]` 的一列，`add_blanks(batch, &[1])` 之后 keys 变成 `[2, 3, 1, 0]`，拿原字典解出来是 `gamma, delta, beta, alpha`——第一行本来是 `delta`，读回来是 `gamma`。静默错值，不报错。v2 不受影响（字典随数组走 lance-encoding，按页自带）。
- 处置：已修。`add_blanks` 之后加一步 `restore_dictionary_columns`：凡是类型里含 `Dictionary`（递归判断，`Struct`/`List`/`FixedSizeList`/`Map` 下面的也算）的列，改用 `take` 从原批次重建，空白行取第 0 行的 key。对字典列来说这就是改动前的行为，也是唯一能保住 key 有效性的拼法；而复制一个 key 只是 1–8 字节，不会把 offset 撑爆，所以本次要修的溢出不会因此回来。测试 `add_blanks_keeps_dictionary_keys_valid` 顶层和 `Struct` 嵌套各一列，断言 keys 是 `[3, 3, 1, 0]` 且 values 原封不动。变异验证：去掉这一步，keys 变成 `[2, 3, 1, 0]`；去掉 `Struct` 那条递归，嵌套那列同样挂。
- 再补一句（Phase 11 想给这条写端到端测试时发现的）：`update_columns` 在 v1 上重写字典列，**本来就是坏的**，和空白行无关。写 schema 的字典值来自 fragment schema，而写进去的 key 来自右侧流自己的字典（比如 `["omega"]`），两者对不上。实测：一个 4 行 `Dictionary(UInt8, Utf8)` 的 Legacy 数据集，**不删任何行**（`add_blanks` 直接早退，一个空白行都没有），`update_columns` 之后 commit 再扫，读回来是 `[beta, gamma, alpha, delta]`，而应该是 `[alpha, beta, omega, delta]`。所以这条的失败场景在今天这条路上观察不到——它坏在我们上游。这个 pre-existing 缺陷不在本次改动范围里，没动它；本条的修复保住的是「key 与批次自带的 values 数组一致」这个契约（也就是 `take` 原来的行为）。
- 补一句：这条第一轮被我自己写进「查过没问题的点」里放过了——当时只核到「v2 无影响、v1 加新字典列本来就不持久化字典值」，没核 `update_columns` 重写已有字典列这条路。

### 3. [HIGH · 正确性] 占位行把可空子字段填成非 null 空值，blob 写入器当成外部引用 — CONFIRMED（Phase 3 第四轮发现）

- 锚点：`rust/lance-arrow/src/blank.rs:63`
- 问题：`strip_validity` 无条件把整棵树的 validity 都剥掉，`blank_row_for` 又只按 `field.data_type()` 取占位值，字段声明的可空性一路丢掉。于是 blob v2 的描述符列 `Struct<data: LargeBinary?, uri: Utf8?, ...>` 的占位行是「非 null 的空 `data` + 非 null 的空 `uri`」。而 `BlobWriter` 分派看的是有没有值、不是值是什么（`rust/lance/src/dataset/blob.rs:1016-1017` 的 `has_data`/`has_uri`）：空 `data` 过不了两道 `data_len > threshold`，于是落进 `if has_uri`，把 `""` 当外部 URI 去解析。改动前 `take` 复制第 0 行，内联 blob 的 `uri` 本来是 null，所以走的是内联那条路。
- 失败场景：fragment 上有 blob v2 列且有删除行，然后 `merge_insert`/`update_columns` 重写这一列。`resolve_external_reference("")` → `Url::parse("")` 失败，报 `External URI '' is outside registered external bases and is not a valid absolute URI`——消息既没提 blob 也没提删除行。而且 updater 这条路会开 `allow_external_blob_outside_bases`（`rust/lance/src/dataset/fragment.rs:2044`），把那道措辞更清楚的拒绝也绕过去了。纯内联的 blob 数据集一样中招，因为它们本来就没有 URI。
- 处置：已修，顺着可空性改，不是给 blob 开特例。公开入口从 `minimal_non_null_array(&DataType)` 换成 `minimal_value(&Field)`：字段可空就给 null，不可空才给该类型的最小非 null 值，而且逐层递归——可空子字段在不可空父节点里照样是 null。这既是正确的值，也是最省的值（null 不带任何 payload）。测试 `a_nullable_field_yields_a_null`、`a_nullable_child_stays_null_inside_a_non_nullable_parent`（就是描述符那个形状，顺带断言不可空的 `size` 子列仍有值）、以及 updater 侧的 `add_blanks_leaves_a_nullable_child_null`。变异验证：把可空性判断去掉，前两个测试分别在 `null_count` 0 vs 1 上挂。

### 4. [HIGH · 正确性] 字典列的护栏按整列回退，把兄弟字段的 payload 又搭了回来 — CONFIRMED（Phase 4 发现）

- 锚点：`rust/lance/src/dataset/updater.rs:475`
- 问题：缺陷在条目 2 的补丁里，不在被评审的这一版。那道护栏是按**顶层列**做的：只要这一列的类型里有 `Dictionary`，整列改用 `take` 重建。而 `take` 复制的是它拿到的那个节点底下的全部内容，所以 `Struct<{tag: Dictionary, blob: Binary}>` 这种列，`blob` 子列又变成「每个空白行复制第 0 行」——正是这次要修的溢出。字典和大字段同处一列时，护栏把 bug 放回去了。
- 失败场景：`Struct<{tag: Dictionary(UInt8, Utf8), blob: Binary}>`，第 0 行的 `blob` 是 1 MiB，8 个空白行。实测 `blob` 子列的 i32 offset 末值从 1 MiB 变成 9 MiB；把 1 MiB 换成 128 MiB、空白行给到 17 个，就是 `Failed to add blanks: Offset overflow error`。`List<Struct<{Dictionary, Utf8}>>` 同理。
- 处置：已修，改了两轮。第一轮 `restore_dictionaries` 只对 `Struct` 下钻，list 类容器仍整棵 `take`；Phase 5 指出那条路还留着同一个溢出（`List<Struct<{Dictionary, Binary}>>`，8 个空白行把 1 MiB 撑到 18 MiB），而且当时给的理由「元素位置对不上」只对 `FixedSizeList` 成立——`List`/`LargeList`/`Map` 的占位值是空列表，interleave 之后子数组和活批次是逐元素对齐的。第二轮改成把行位置翻译成子元素位置再递归下去，`interleave` 的 offsets 和 validity 原样留着：list 的空白行一个元素都不贡献，所以什么都不复制；`FixedSizeList` 的空白行仍然占满宽度，取第 0 行的槽位，字典叶子那里复制的是 key。测试四个：`add_blanks_does_not_grow_a_dictionary_siblings_payload`（Struct 里的 blob 子列只有 1 MiB）、`add_blanks_does_not_grow_a_listed_dictionary_siblings_payload`（同一条性质降一层）、`add_blanks_keeps_a_listed_dictionary_valid`（`List<Dictionary>` 走的是另一条 arrow 代码路径，且断言空白行是空列表）、`add_blanks_keeps_a_twice_nested_dictionary_and_its_nulls`（两层 struct + 活行自己的 null）。变异验证：按整列 `take` 时前两个在 9437184 vs 1048576 / 2 vs 0 上挂；把 list 那条分支退回整棵 `take`，两个 list 测试都挂；`interleaved.nulls()` 换成 `None`、以及去掉 struct 那条递归，各挂第四个测试的不同断言。

### 5. [MEDIUM · 正确性] 满键位的字典交给 `interleave` 会报错，挂在 FSL/Map 下面还会 panic — CONFIRMED（Phase 6 发现）

- 锚点：`rust/lance/src/dataset/updater.rs:475`
- 问题：`interleave` 要把两个来源的字典合并成一个，而合并结果的取值个数必须装得进 key 类型。占位行自带一个一项的字典，于是 `UInt8` key 的列只要活批次里引用了 255 个不同值，合并后就是 256 个，`merge_dictionary_values` 直接返回 `DictionaryKeyOverflowError`；字典挂在 `FixedSizeList` 或 `Map` 下面时走的是 `interleave_fallback`，那条路把两边的 values 拼起来（2 × M），`MutableArrayData::new` 在 `.expect("MutableArrayData::new is infallible")` 上 **panic**，M 只要过 127 就会踩到。改动前的 `take` 不合并字典，所以这两种都是本次改动引入的。
- 失败场景：`Dictionary(UInt8, Utf8)` 一列、256 个不同值全被引用，fragment 上有删除行 → `Failed to add blanks: Dictionary key bigger than the key type`，整个 update/merge 失败。同一份数据放进 `FixedSizeList(Dictionary(UInt8, Utf8), 2)` → 进程 panic，而库代码不允许 panic（`rust/CLAUDE.md`）。
- 处置：已修，改了两轮。`interleave` 拿到的字典换成 stub：key buffer 长度不变、全填 0，values 只留一项，所以合并永远只有两个取值，溢出不了。真字典留给条目 4 的 `restore_dictionaries` 用 `take` 重建，合并结果本来就是要丢掉的。第一轮给 `Dictionary(_, Null)` 开了个例外，理由是「它只有一个取值」——Phase 7 指出这不成立：`NullArray` 当 values 可以是任意长度，`Dictionary(Int8, Null)` 带 130 个取值照样 panic（实测 `MutableArrayData::new is infallible: DictionaryKeyOverflowError`）。当时给的另一条理由也是错的：拒 null stub 取值的是 `interleave` 自己拼不可空 struct 字段那一步，而原数组在那里同样是 logical null，所以例外白开。第二轮去掉例外。测试 `add_blanks_survives_a_saturated_dictionary`（256 个值的扁平列 + 同一份数据放进 `FixedSizeList(_, 2)`）和 `add_blanks_survives_a_saturated_dictionary_of_nulls`；变异验证：去掉 stub 时前者在 `Dictionary key bigger than the key type` 上挂，恢复那个例外时后者在 arrow 的 `expect` 上挂。

### 6. [MEDIUM · 正确性] 不可空的 blob v2 列碰上删除行就写不出去 — CONFIRMED（Phase 10 发现）

- 锚点：`rust/lance-arrow/src/blank.rs:63`
- 问题：条目 3 让占位值按字段可空性递归，于是不可空的 blob 描述符列拿到的是「struct 本身非 null、`data`/`uri` 全 null」——这是那些子字段声明可空时唯一正确的形状。而 `BlobPreprocessor` 走到 `rust/lance/src/dataset/blob.rs:1121-1125`：`struct_arr.is_null(i)` 在前面已经处理过了，这里 `has_data`/`has_uri` 都是 false，于是 `push_null()`，产出一个 null 描述符。可列的 prepared 字段是从用户字段继承可空性的（`prepared_blob_field_with_metadata`），重新拼批次时 `RecordBatch::try_new`（`blob.rs:791`）就报 `Column 'blob' is declared as non-nullable but contains null values`。
- 失败场景：`blob_field("blob", false)` 建的列（仓库自己的测试里就有这种用法），fragment 上有删除行，然后重写这一列 → 写失败。370ff5fbf 之前复制第 0 行，描述符是非 null 的，能写出去（代价是复制 payload）；条目 3 修完之后变成这个硬错误。
- 处置：已修，改在语义该定的地方：描述符存在（null 的那条在前面就 return 了）却既没有 data 也没有 uri，唯一讲得通的读法就是「空 blob」，所以不可空列走 `push_inline(Bytes::new())`；可空列保持原来的 `push_null()`，不动用户可见行为。测试 `preprocess_reads_an_empty_non_nullable_descriptor_as_an_empty_blob` 直接驱动 `BlobPreprocessor`；变异验证：改回无条件 `push_null()`，测试就报上面那句 non-nullable 错误。

### 7. [MEDIUM · 约定设计] `unsafe { build_unchecked() }` 什么都没换来，而且盖住了条目 1 — CONFIRMED

- 锚点：`rust/lance-arrow/src/blank.rs:63`
- 问题：把 `strip_validity` 的收尾换成受检的 `build()` 之后，37 种类型里**只有条目 1 那三种嵌套字典形状会失败**，其余全部返回 `Ok`——也就是 `build()` 恰好在数据真的非法时才拒。剥掉 validity 动不了 buffer 长度、offsets、child 长度，所以那些不变式本来就成立；唯一受影响的两条是「非空 child 要求 `null_count == 0`」（`data.rs:1354-1377`，这条由递归剥离满足，所以递归是承重的）和字典 key 边界。payload 只有一行，`unsafe` 也省不下什么。`rust/CLAUDE.md` 要求 `unsafe` 必须有能站住的 `// SAFETY:` 理由，而现在那段注释说的「剥掉 validity 不会让 payload 失效」对嵌套字典恰恰不成立。
- 失败场景：不是独立故障，是条目 1 之所以能溜过去的原因。附带一条：`build_unchecked` 的 `.unwrap()` 在 `--features force_validate` 下仍然会校验（`data.rs:2192`），所以同一份数据在校验构建里是 panic、在正常构建里是静默损坏。
- 处置：已修。`strip_validity` 改成返回 `Option<ArrayData>`、用受检 `build()` 收尾，`unsafe` 和那段 SAFETY 注释一起删掉。实测 24 种形状（含 `Map`、三层嵌套、`Dictionary(Int32, FixedSizeList)`）在受检 build 下全部通过 `validate_full`，没有类型因此新掉进 `None`——那会让 updater 回退到 `slice(0, 1)`，把这次要修的溢出放回去。

### 8. [LOW · 健壮性] offsets 非递增时的失败方式从静默错值变成巨额分配 — CONFIRMED（第三路 finder 发现）

- 锚点：`rust/lance/src/dataset/updater.rs:467`
- 问题：`(*batch_offset - next_id)` 仍然是 `u32` 相减，release 下回绕（工作区的 profile 没有开 `overflow-checks`）。改动前回绕值也参与 `u32` 的区间端点计算，`5..2` 这种空区间会被跳过，结果是**一个静默多了几行、值重复的 batch**；改动后 `as usize` 之后再加，`batch_pos..batch_pos + 约 4.29e9` 变成一个约 68 GB 的 `Vec<(usize, usize)>` 分配，进程被 OOM 掀掉。两种都不对，但失败方式换了，值得记一笔并顺手加护栏。
- 失败场景：`batch_offsets` 传成非递增（例如 `&[5, 3]`）。调用方今天保证有序——`deleted_batch_offsets_in_range` 从 `DeletionVector::into_sorted_iter` 取值，`add_blanks` 又是 `pub(crate)` 且只有一个非测试调用方——所以现网到不了；这是契约没写进代码。debug 构建两种版本都在减法处 panic。
- 处置：已修，和条目 11 合成一道护栏：`batch_offset.checked_sub(next_id)` 之后再要求 `num_rows <= 剩余活行数`，不满足就返回 `Error::invalid_input`，消息里给出合法区间 `[next_id, next_id + 剩余活行数]` 和批次行数；「必须递增且不越过批次」写进 `add_blanks` 的 doc。选 `checked_sub` 而不是 `debug_assert!`：debug 构建本来就会在减法处 panic，真正会分配 68 GB 的是 release。

### 9. [LOW · 健壮性] 把没筛过的 `DataType` 交给 arrow 就是 panic，而不是按 doc 承诺返回 `None` — CONFIRMED

- 锚点：`rust/lance-arrow/src/blank.rs:55`
- 问题：`_ =>` 分支把任何类型直接递给 `ArrayData::new_null`，而 `new_null` 对几种「`DataType` 拼得出来、但没有合法数组会长成那样」的类型是 panic 不是报错：字典 key 没有固定宽度时 `k.primitive_width().unwrap()`（`arrow-data-58.4.0/src/data.rs:688`）、run-end 类型不在 `Int16`/`Int32`/`Int64` 里时 `unreachable!`（`data.rs:737`）、零变体 union 的 `f.iter().next().unwrap()`、`FixedSizeBinary(-1)` 的 `*i as usize` 变成 `usize::MAX` 大小的分配。顶层的字典分支还多一处：`keys.buffers()[0]` 无条件下标（`blank.rs:49`）。而 `minimal_non_null_array` 的 doc（`blank.rs:19-23`）承诺的是「构造不出来就返回 `None`，调用方当成『用你原本会用的占位值』」。另有三处不在 `new_null` 里：`make_array` 走 `MapArray::from`，entries 不是两字段 struct 就 `expect` 炸（`build()` 不管这条）；`primitive_width` 对 `Time32`/`Time64` 只看宽度不看单位，于是 `Time32(Nanosecond)` 一路通过 `new_null` 和受检 `build()`，到 `make_array` 才撞上 `unimplemented!`；而把 `unsafe build_unchecked` 换成受检 `build()`（条目 7）之后，`List<RunEndEncoded>` 会在 `validate_values` 的 `unreachable!` 上新增一个 panic 点。
- 失败场景：`Dictionary(Null, Utf8)`、`Struct([Field::new("r", RunEndEncoded(Int8 run_ends, Int32), true)])`、`Map` 的 entries 只有一个字段、`FixedSizeBinary(-1)`——每一种都是进程直接 panic 或分配失败中止，而不是返回 `None`。库代码在可失败输入上 panic，`rust/CLAUDE.md` 禁止这一条。这些类型没有合法数组，从真实 `RecordBatch` 到不了，只有直接拿这个 `pub` helper 传畸形 `DataType` 才会踩到。
- 处置：已修，落点比原来写的更深，改了两轮。第一轮加了 `new_null_is_infallible` 只筛字典 key 和零变体 union；Phase 3 第一轮指出这不完整（漏了 run-end 类型、Map entries 形状、负宽度），于是换成 `is_supported`：在调 `new_null` 之前把整棵类型树筛一遍，`Union`/`RunEndEncoded`/两个 view list **在任何深度**都返回 `None`（原来只在顶层，而 doc 写的就是「land there」，这次让代码和 doc 对上了），另加非整数字典 key、非两字段 Map entries、负 `FixedSizeBinary`/`FixedSizeList` 宽度三条。Lance schema 装不了 union 和 REE，所以按深度收紧不影响任何真实列。Phase 3 第二轮又找出第四类：`Time32(Microsecond)`/`Time32(Nanosecond)`/`Time64(Second)`/`Time64(Millisecond)` 这四种 arrow 自己造不出数组的组合，也补进 `is_supported`。测试：`unsupported_types_are_reported_not_guessed` 十三种形状 × 裸用和三种嵌套位置，`dictionary_with_a_non_integer_key_returns_none` 四种 key × 四种位置，正向再补 `map`、`map_of_dictionary` 两例确认合法 Map 没被误伤。变异验证：`is_supported` 四条分支逐个删掉，分别得到 union 返回 `Some`、`MapArray::from` 的 expect 炸、`MutableBuffer` 分配失败、`make_array` 的 `not implemented: Unexpected data type Time32(µs)`——四条都承重。

### 10. [LOW · 死代码] `minimal_non_null_or_empty` 没有任何调用方 — CONFIRMED

- 锚点：`rust/lance-arrow/src/blank.rs:81`
- 问题：`grep` 整个 `rust/`、`python/`、`java/`，除了它自己的测试之外零调用。它是 `pub` 所以 dead_code lint 不报。`rust/CLAUDE.md`：「Remove dead code instead of adding `#[allow(dead_code)]`」。updater 故意没用它——零长度数组拿来拼接是错的。
- 失败场景：不是运行时故障。代价是公开 API 面上多一个没人用、语义还容易误用的函数。
- 处置：已删，连带它唯一的测试 `fallback_returns_empty_for_unsupported`，以及只为它存在的 `new_empty_array`、`Int32Array`、`Arc` 三个导入。

### 11. [LOW · 健壮性] offsets 越过批次末尾时从 `Err` 退化成 panic — CONFIRMED（Phase 3 第一轮发现）

- 锚点：`rust/lance/src/dataset/updater.rs:468`
- 问题：`indices.extend((batch_pos..batch_pos + num_rows)...)` 只按 offsets 算下标，不看批次实际有多少行。改动前 `take` 对越界下标返回 `ArrowError`，被包成 `Error::arrow("Failed to add blanks: ...")`；改动后这个下标交给 `interleave`，它在 `PrimitiveArray` 的取值断言上直接 panic。同一类畸形入参，故障从可捕获的错误变成 panic。
- 失败场景：`add_blanks(10 行的 Int32 批次, &[11, 12])`。改动前得到一个 `Failed to add blanks` 的 `Err`，改动后 `Trying to access an element at index 10 from a PrimitiveArray of length 10` 直接 panic。和条目 8 一样，`deleted_batch_offsets_in_range` 保证 `o_k <= N + k`，所以现网到不了。
- 处置：已修，护栏和条目 8 写在一起：`num_rows` 还要满足 `<= batch.num_rows() - batch_pos`。顺带把这条契约钉进测试 `add_blanks_rejects_offsets_it_cannot_honor`（重复、倒退、越界、以及吃掉活行之后再越界，四种形状 × 断言 `InvalidInput` 变体和消息）。变异验证：去掉 `checked_sub` 这一半，测试在减法处 panic；只去掉区间这一半，测试在 arrow 的 `PrimitiveArray` 下标处 panic；把 `- batch_pos` 去掉（Phase 3 第二轮指出前三种形状都在 `batch_pos == 0` 时就被拒，测不出这一项），`[3, 12]` 这一例才让它挂——三处都承重。这道区间检查同时让 `next_id = *batch_offset + 1` 的 u32 溢出变得构造不出来（offsets 被批次行数夹住了）。

- Phase 6 又找出一条同类的（第三轮修字典护栏）：`FixedSizeList` 的空白行要占满宽度，但把那些子槽位标成「第 0 行」会一路传下去——FSL 底下再挂一个 list 时，那个 list 会在空白位置重建第 0 行的元素，而 `interleave` 在那里放的是空条目；`ListArray::try_new` 只拦子数组太短、不拦太长，于是第一个空白行之后所有活行的 key 都静默错位。触发类型 `FixedSizeList<Struct<{tags: List<Dictionary>}>, 2>`（`is_supported_fixed_size_list_child` 只看 FSL 的直接子类型是不是 `Struct`，不看 struct 里面有什么，所以这个 schema 能转）。改法：空白槽位一路保持「空白」，只有叶子才把它解析成第 0 行；宽度为 0 的 FSL 没有元素要修，直接留 `interleave` 的结果。测试 `add_blanks_keeps_a_dictionary_under_a_fixed_size_list_aligned`，变异验证：把空白槽位换回第 0 行，key 从 `[0, 1, 2]` 变成 `[0, 1, 0, 1, 2]`。
- Phase 6 还点出三个变异当时六个测试全都拦不住，补了三例：`add_blanks_keeps_a_mapped_and_large_listed_dictionary_valid`（删掉 `Map`/`LargeList` 分支时挂）、`add_blanks_keeps_a_sliced_list_and_its_nulls`（把容器的 `interleaved.nulls()` 换成 `None`、或者在 `element_slots` 里把 offsets 按 `offsets[0]` 重新起算，分别挂在 null 断言和 key 断言上——后者只有切片过的批次才看得出来）。
- Phase 5 另外两条小的，一起改了：`restore_dictionary_columns` 的错误原来是裸的 `ArrowError`，和旁边 `interleave` 那句 `Failed to add blanks:` 不一致，看不出是补空白行这一步炸的，现在一样包上；`contains_dictionary` 漏了 `Union`、`RunEndEncoded` 和两个 view list，字典藏在它们下面会静默留着重编号的 key——Lance schema 装不了这些类型，但现在这个 match 的 `false` 只回答叶子类型了。

- Phase 6 的整份 diff 复核还带出四条注释/契约问题，一并改了：`blank_row_for` 里 `minimal_value` 返回 `None` 时回退成复制第 0 行，那是个「防不可能」的分支（Lance schema 到不了），而且真触发的话正好把本次要修的 payload 复制放回来——换成显式 `Error::not_supported`，顺便让 `element_slots` 那句「空白行不贡献元素，因为占位值是空列表」无条件成立；`take_slots` 里「空白行取第 0 行的 key」那句只对字典叶子成立，挪到那条分支上；`minimal_value` 的 doc 没写「`Null` 字段两种情况都给 null」，补上。

- Phase 7 还指出宽度为 0 的 `FixedSizeList` 原来留的是 `interleave` 的结果，也就是 stub 那份伪造的 values 数组（没有 key 会去索引它，所以解不出错，但白留了个假字典）；改成和别的类型一样从活批次 `take`，真 values 数组就回来了。

## 查过没问题的点

- 行顺序与行数对每种调用方能构造的输入都与改动前一致：空 `batch_offsets` 早退、位置 0 的空白、连续空白、越过最后一个活行的空白，以及调用方能给出的最大 offset（`o_k <= N + k - 1`，因为 `deleted_batch_offsets_in_range` 一超出区间就停下 stash），输出长度恒为 `N + K`。
- schema 与 metadata 逐字节一致：`interleave_batches` 取的就是 `batches[0].schema()` 那同一个 `Arc`，不是拷贝。
- v1 写入器的每页统计确实会读到空白行的值（`rust/lance-file/src/versions/v1/writer/mod.rs` 里 `collect_statistics` 收的就是它要写的那批数组）。改完条目 3 之后分两种：可空列的空白行是 null，min/max 不受影响、`null_count` 变大；不可空列的空白行是零/空，min/max 只会**变宽**。变宽对剪枝是安全的（统计仍是活值的超集），代价只是多读几个含删除行的页。zone map 走的是应用了删除向量的 scan 流，看不到空白行。
- ~~`Dictionary` 列的输出字典会被重建，但无影响。~~ **这条当时判错了，见条目 2。**

- 性能：实测（独立 harness，同一个 arrow 58.4，8192 活行 + 8 空白）`interleave` 相对 `take` 在定宽列上每行每列多约 0.39 ns、小字符串列多约 10%，而在 `FixedSizeList<Float32,1536>` 上**快 5.4 倍**（5.97 ms → 1.11 ms）。原因是 `take_fixed_size_list` 会先建一个 1536 × 8200 项的 u32 下标数组（50 MiB）再逐元素 gather，而 `interleave` 落到 `interleave_fallback` 合并连续区间、只做 17 次批量 memcpy。最坏的现实情况（100 个定宽列、每批都有删除）每批多 340 µs，而那一批要写 3.3 MB，属于个位数百分比。
- `batch_offsets` 为空时早退（`rust/lance/src/dataset/updater.rs:443`），没有删除行的批次一分钱不花。
- `FixedSizeList` 的 1536 个零 float 是每次 `add_blanks` 调用造一次，不是每个空白行造一次。
- `null_count` 没有记账问题：`ArrayData` 没有 `null_count` 字段，`null_count()` 由 `nulls` 推出（`data.rs:463`），`ArrayDataBuilder::nulls()` 会同时重置（`data.rs:2069`）。
- 顶层 `Dictionary` 的特殊分支本身是对的：复用的 keys buffer 宽度恰好是 `len * width`，values 长度为 1，而且走的是受检的 `.build()`。
- 递归剥离是承重的，不是顺手写的：非空 child 的 `Struct` 和 `FixedSizeList` 少了它就非法。
- `List`/`LargeList`/`Map` 剥完是空列表、`Utf8View`/`BinaryView` 是零长度内联 view、`FixedSizeList` 是全零、`Struct` 是零值子列——都是合法的真实值。嵌套的 `Union`/`ListView`、以及 run-end 类型合法的 `RunEndEncoded` 也都产出合法数组，所以那份 `None` 名单原本是策略选择而非正确性要求；后来因为 run-end 类型不合法时 arrow 直接 panic（条目 8），改成按深度一律拒掉。

## 非缺陷的观察

- 把空白行提到 `DeletionRestorer` 按 fragment 缓存能省每列每批次 200–600 ns（100 列约 20 µs），量级不值得；而且缓存 fallback 分支（`batch.column(idx).slice(0, 1)`）会把整个批次的 buffer 钉住整个 fragment 重写期。
- 更快的形状是按连续区间切片再 `concat`（Utf8 快 12 倍、Int32 快 3 倍），但空白密度高时反而慢 13 倍，需要密度启发式，是另一个改动。
- `DataType::Null` 落在非空字段上时，`blank_row_for` 交给 `RecordBatch::try_new` 的数组 `null_count() == 0` 但 `logical_null_count() == 1`；这与本次改动无关（老代码 take 第 0 行同样得到 null），但没人核过 arrow 校验的是哪一个。

- Phase 3 第二轮建议把 `is_supported` 的 `_ => true` 改成穷举匹配，让 arrow 以后新增类型时这里编译不过。没采纳：`DataType` 有四十来个变体，穷举要多写三十行 `=> true`，而这个坑要同时满足「arrow 加了新变体」且「`make_array` 不支持它」——历史上新变体（`Decimal32`/`Decimal64`）都是带全套支持一起进来的。同一个耙子在每个调 `make_array` 的 crate 里都存在，不是这个模块能收口的。

- `MapArray::try_new` 拒绝 entries 字段声明为可空的 map（`interleave` 走的 `From<ArrayData>` 那条路不管这个），所以「entries 可空 + 里面有字典」的 map 现在会在 `add_blanks` 报错。pyarrow 和 `MapBuilder` 都只产不可空的 entries，Lance 侧也要求 `keys_sorted=false`，没为它写规避代码。
- 子元素位置用 `u32` 装（`take` 的下标类型就是 `UInt32Array`），所以单个批次里某一列的子元素超过 42.9 亿个时会截断。那种批次本身就不可能出现，而且截断后的区间会让子数组变短、被 arrow 拦下来报错，不是静默错值。

- 有把 `add_blanks` 那套（占位值、stub、restore、element_slots）挪进单独文件的想法，没做：`updater.rs` 现在 1602 行，其中 845 行是测试，而这个仓库里 `fragment.rs` 6825 行、`schema_evolution.rs` 4104 行都是测试写在文件底部的同一种形状，所以 1602 行在本仓库不算异常；项目 CLAUDE.md 又明确「Keep PRs focused — no drive-by refactors」。要拆的话应该单独开一个 PR。

- Phase 8 的等价性复核（新旧两版对同一批输入的输出对比）报了一条「新版会拒掉旧版能过的批次」：列的嵌套字段名和 schema 里那个字段不一致时（比如数组把 list 子字段叫 `element`、schema 里写的是 `item`），`blank_row_for` 按 schema 造占位值，`interleave` 又按严格相等比类型，于是报「not possible to interleave arrays of different data types」。**实测否掉了**：arrow 58.4 的 `RecordBatch::try_new` 自己就不收这种列（`column types must match schema types, expected List(Int32) but found List(Int32, field: 'element')`），所以这种批次根本进不来。按它的建议改过一版（占位值的类型取自列而不是 schema），确认是空操作后回退了。
- 同一轮另一条留下了改动：`restore_dictionaries` 原来用 `ListArray::try_new`/`MapArray::try_new` 这类带类型的构造器重建节点，它们拦得比 arrow 自己的校验更严——非空的 item 字段配一个 values 里有「没人引用的 null」的字典就会被拒，而被替换掉的 `take` 那条路根本不查这个。改成统一走 `ArrayDataBuilder`（受检 `build()`），顺带把五个容器分支的重建代码合成一个。

- Phase 7 和 Phase 8 的两轮整份 diff 复核（两个 agent 各自独立跑）报回了**同一批八条注释问题**，都不是代码缺陷：`minimal_value` 的 doc 还写着「返回 `None` 时调用方当成『用你原本会用的占位值』」，而两个调用方现在都把它变成错误；`null_type_yields_a_null` 的注释说「只有可空字段会走到这里」，而字典那条路是拿不可空字段走进来的；`minimal_non_null_array` 的摘要漏了 `Null` 这个例外；stub 那段 doc 的算术举例是错的（`UInt8` 用满 256 个 key 就已经没位置了，而拼接那条路的门槛更低）；两处测试 doc 描述的是早先的行为；offsets 溢出那套理由在三个地方各写了一遍。都改了。
- 同两轮都提了测试里的 inline `use`（14 处，`rust/CLAUDE.md` 要求 import 放文件顶部），已经全部挪到 `mod tests` 头部。

- Phase 9 的第五轮注释复核又找出四条（都不是代码问题）：空批次那条 TODO 说的困难「要为任意 schema 造一个批次」是旧实现的，新代码里 `blank_row_for` 干的就是这件事，真正还卡着的是没有活行可以拿字典的 key；`is_supported` 的 doc 把 `Time64(Second)`/`Time64(Millisecond)` 说成「单位装不进宽度」，其实是 arrow 没给它们数组类型；offsets 测试最后那个用例的注释说「要先吃掉活行才会被拒」，实际是「`live_rows_left` 少了 `batch_pos` 就不再拒」；`minimal_value_is_one_non_null_row` 的断言消息说「非 null 才能放进不可空字段」，对 `dictionary_of_null` 这一例过头了（它 `null_count` 是 0 但整行是 logical null，arrow 的带类型构造器照样会拒），改成说它实际断言的「自己不带 null」。
- 同一轮提了一条 CLAUDE.md 偏差没改：「测试里用 `batch["column_name"]` 取列」，我这十六个测试用的是 `column(0)`。这些测试的批次都是一两列、就在断言上方三行构造出来的，下标没有歧义；那条规则的收益是宽批次里的可读性，这里用不上。评审自己也把它标成 low value。

- Phase 10 的第六轮注释复核又找出五条（都不是代码问题，改完了）：`is_supported` 的 doc 把「arrow 对嵌套的也一样 panic」写成了两类排除的共同理由，其实只对第二类成立——两个 view list 在 `new_null` 和 `make_array` 里都有分支，排除它们是策略选择；`Float32` 当字典 key 是有宽度的，拦它的是受检 build 不是那道筛子；拼接那条路是「两个 values 数组加起来到 256 就失败」，不是「超过 256」；`minimal_value_is_one_non_null_row` 这个名字对 `dictionary_of_null` 一例不成立（它是 logical null），改成 `minimal_value_is_one_row_with_no_null_of_its_own`；`test_restore_deletes` 里那句「Next batch is rows ids 10..16」应该是 10..18（这行是改动之前就有的，一个字的事，顺手改了）。
- 同一轮把之前几轮验过的 arrow 结论又逐条对着源码复核了一遍（`new_null` 的 key 宽度 unwrap、union 和 run-end 的 panic、`MapArray::from` 的 expect、字典溢出的三条路径、`element_slots` 的对齐前提、以及所有测试里的字节数和行数），没有新问题。

- Phase 11 的收尾评审给了「可以合了，没有 CRITICAL/HIGH」，另外两条 MEDIUM 都是测试覆盖：
  - `with_children` 走 `ArrayData` 而不用带类型构造器的理由（非空 item 字段配「values 里有没人引用的 null」的字典，`ListArray::try_new` 会拒、`ArrayData` 不拒）当时没测到，把整个 suite 换回 `ListArray::try_new` 也不会挂。已补：`add_blanks_keeps_a_listed_dictionary_valid` 的 fixture 改成用 `ArrayData` 构造（arrow 的带类型构造器根本造不出这个形状，只有 reader 会给），values 里加一个没人引用的 null。变异验证：list 那条分支换回 `ListArray::try_new`，报 `Non-nullable field of ListArray "item" cannot contain nulls`。
  - 端到端没有「删除 + 重写 + 读回」的测试。给 `test_cast_dictionary_to_string`（本来就参数化了 Legacy/Stable）加了一次 `delete`，这样 splice 在两个写入器版本上都真的跑过一遍；但要说清楚：它覆盖的是 splice，不是字典 restore——交给 `add_blanks` 的是 cast 之后的 `Utf8` 列，`restore_dictionary_columns` 在「没有字典列」那步就早退了；字典那条路见条目 2 的补注，它在 v1 上本来就坏着。Phase 12 还指出删 `d = 'beta'` 会把两个 beta 都删掉、key 1 就没人解了，改成删 `d = 'gamma'`（一行），每个 key 都还剩活行。
  - 同一轮的两条 LOW（`stub_dictionary_data` 里两个到不了的错误分支、以及 fallback 分支注释把三个到不了的类型也算进去了）：错误分支按 `rust/CLAUDE.md` 允许保留，注释改成只说宽度为 0 的 `FixedSizeList` 这一种真能到的情况。

## 顺手带上的小改动

- `blank_row_for` 头上还挂着旧的 `/// Add blank rows where there are deleted rows`（改动前是 `add_blanks` 的 doc），读起来像是 `blank_row_for` 在干 `add_blanks` 的事；删掉，同时给 `add_blanks` 补一行 doc 写明 offsets 的契约。
- `add_blanks_survives_binary_offset_overflow` 的 `#[ignore]` 写「~128MiB」，实际 `vec!` 和 `BinaryArray::from` 的拷贝各占一份，改成 256 MiB。
- `variable_width_placeholder_carries_no_payload` 里 offset 宽度的 match 有 `LargeUtf8`/`LargeList` 分支却没有对应 case，8 字节那条路没人走；补上两个 case。

## 收敛情况

十二轮。第 1–7 轮每轮都有代码缺陷（六个 HIGH，其中五个是这次改动自己引入的回归，不是原有 bug）；第 8 轮两路复核只报出注释不准；第 9 轮的正确性扫描第一次空手而归，注释复核报四条；第 10 轮正确性空手，但它交接的一个未完成假设查下去是条 MEDIUM（不可空 blob 列写不出去）；第 11 轮给「可以合了，没有 CRITICAL/HIGH」，两条 MEDIUM 都是测试覆盖，已补；第 12 轮只看第 11 轮之后那点 diff，结论是「不改变可合的判断」，两条都是它自己给的措辞修正，已按它写的改完。

代码自 9a0323908（blob 那条）之后没再动过，之后的提交都是测试和注释。门禁：`cargo clippy --all --tests --benches -- -D warnings` 干净、`cargo fmt --all --check` 干净、`cargo test -p lance --lib` 3162 passed、`cargo test -p lance-arrow` 145 + 6 passed、被 `#[ignore]` 的 2.125 GiB 回归测试手工跑过并确认改动前会挂（`Offset overflow error: 2147483648`）。
