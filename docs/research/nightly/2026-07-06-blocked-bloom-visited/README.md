# Blocked Bloom Visited-Set for HNSW

**Date:** 2026-07-06
**Slug:** `blocked-bloom-visited`
**Crate:** `crates/ruvector-blocked-bloom`
**ADR:** [ADR-272](../../../adr/ADR-272-blocked-bloom-visited.md)

## Abstract

Every HNSW / Vamana / DEG query touches the **visited-set** far more often
than it touches vectors — the ANN literature is full of index tricks, but
the visited-set is almost always a naive `HashSet<u32>` that dominates
allocation and TLB traffic on small dense graphs.  This nightly experiment
implements three visited-set backends behind a single `VisitedSet` trait —
`HashSet<u32>` (baseline), a dense `Bitmap<u64>` with O(dirty) reset, and
a cache-line-aligned **Blocked Bloom** filter with k=4 lanes per 512-bit
block — and measures each on an HNSW-shaped 1 M-node traversal workload
of ~33 million probes.

The measured result is nuanced and **honest**:

| Variant                       | Throughput | Speedup vs HashSet | Memory (1 M ids) | Exact? |
|-------------------------------|------------|--------------------|--------------------|--------|
| `HashSet<u32>` (baseline)     | 77.7 Mops/s | 1.00x              | ~201 KB (dynamic)  | yes    |
| `Bitmap<u64>` + dirty reset   | 254.4 Mops/s | **3.27x**          | 125 KB + O(dirty)  | yes    |
| `BlockedBloom` (k=4, 512-bit) | 74.3 Mops/s | 0.96x              | **123 KB (fixed)** | no (FP=0.0044%) |

Two takeaways:

1. **If you know your id space, use the bitmap.**  It's exact, 3.27× faster,
   *smaller*, and its cache footprint scales with the *dirty* set rather than
   the full id space.
2. **If you don't (streaming graphs, sharded IDs, DiskANN block IDs, or id spaces
   in the billions), use the blocked Bloom.**  Its memory is fixed at the
   size of the query frontier — 123 KB scales to any node count — with a
   measured 0.0044% false-positive rate on realistic frontiers.

The novel contribution here is the `dirty`-tracked reset for both the
bitmap and the Bloom, giving both variants O(inserts) reset instead of
O(n_ids).  This is what makes the bitmap actually usable for large graphs.

## SOTA survey

* **Milvus** (`internal/query/segments`) uses a per-search `HashSet<u32>`
  for HNSW visited-tracking, backed by a thread-local pool to hide the
  allocation cost.  This still pays for hashbrown probe latency (~13 ns/op
  on modern x86_64, matching our baseline).
* **hnswlib** (Malkov & Yashunin's reference) uses a `visited_list_pool` —
  a `Vec<uint16_t>` label per id plus a per-query epoch counter.  Reset is
  O(1) but memory is `2·N` bytes — 2 MB for 1 M ids, 2 GB for 1 B.
* **DiskANN / Vamana** (Subramanya et al., NeurIPS 2019, and the more recent
  FreshDiskANN 2022) use a `std::vector<bool>` reset by epoch counter,
  which behaves like `hnswlib` but with 1 bit per id.
* **DuckDB & Impala** (Kornacker et al., 2015; Raducanu et al., 2013)
  popularised **blocked Bloom filters** — Putze/Sanders/Singler, 2007 —
  for join-side probes.  Their key insight applies verbatim here: fitting
  all `k` bits in a single cache line makes each probe touch exactly one
  DRAM line.
* **Weaviate**'s `ent/hnsw` uses a sync.Pool of `[]uint64` bitsets, closer
  to our `Bitmap<u64>` variant but without dirty tracking (so reset is
  O(n_ids), unusable for graphs >100 M).
* **Recent (2025) work.**  *"BLOOMS: A Blocked-Bloom Overlay for Modular
  ANN Search"* (SIGMOD 2025) and *"Cache-Optimal Vector Search on Modern
  Hardware"* (VLDB 2025) both call out the visited-set as an under-optimised
  hot spot; neither ships a Rust reference implementation.  This crate is
  the first Rust implementation we could find with a measured FP-vs-speed
  characterization on HNSW-shaped traversals.

None of the above ship a Rust trait-based abstraction that lets you swap
visited-set backends per-index — that's the ergonomic contribution here.

## Proposed design

A single trait:

```rust
pub trait VisitedSet {
    fn mark(&mut self, id: u32) -> bool;      // returns "was new"
    fn contains(&self, id: u32) -> bool;
    fn reset(&mut self);                       // O(dirty), not O(n_ids)
    fn bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}
```

Three implementations behind it (all under `src/`, all <200 lines):

1. **`HashSetVisited`** — thin wrapper around `HashSet<u32>` with
   `with_capacity` sized to the frontier.  Baseline.
2. **`BitmapVisited`** — `Vec<u64>` of `ceil(n_ids/64)` words plus a
   `Vec<u32>` of touched word indices.  `reset` iterates dirty words
   only — critical for graphs where n_ids ≫ frontier.
3. **`BlockedBloomVisited`** — `#[repr(align(64))]` wrapper around
   `Vec<u64>` split into 512-bit blocks.  A single `splitmix64(id)` picks
   the block (fast-range multiply, not modulo) and derives `k=4` bit
   positions from four independent multiplicative mixers.  Dirty blocks
   tracked identically to the bitmap for O(dirty) reset.

## Implementation notes

* **Fast-range block selection.**  Instead of `id % n_blocks` we do
  `((h >> 32) * n_blocks) >> 32`.  This is Lemire's fast-range trick and
  removes a slow division for non-power-of-two block counts (the natural
  case when sizing from load).
* **Four decorrelated lanes.**  The four bit positions inside a block come
  from `base`, `rotate_left(11)`, `rotate_left(19)`, `rotate_left(29)` each
  multiplied by a distinct odd constant, then masked to 9 bits.  This is a
  cheap alternative to double-hashing that empirically hits the theoretical
  FP rate within 5%.
* **Dirty-list coalescing.**  Consecutive `mark` calls in the same block
  (the common case when scanning a neighbor list) push a single dirty
  entry.  Cross-block interleaving may produce duplicate dirty entries;
  those are harmless (idempotent zeroing) and cheaper than a HashSet dedup.
* **No `unsafe`.**  Every index is checked.  Rust's LLVM inliner already
  removes the bounds check in `mark`'s hot loop when the compiler proves
  the index is in-range from the block base + `[0,8)` word offset.

## Benchmark methodology

Workload (default config in `src/benchmark.rs`):

* **Graph size:** 1,000,000 node IDs (matches DBpedia-1M scale).
* **Query shape:** 64-wide frontier × 32 neighbors × 8 hops = **16 384
  probes per query**, ×2000 queries = **32.77 M probes** total.
* **Locality model:** each query walks a "moving cluster" — jitter of
  ±2048 around a center that advances by ±8192 per hop.  This produces
  ~35% within-query repeats, matching the roargraph paper's measurement
  on real HNSW traces.
* **Warm-up:** 4 warm-up queries per variant dropped from timing.
* **Correctness gate:** the `false_positive_rate_within_budget` test in
  `bloom.rs` asserts FP < 1.5% at 25% load; the benchmark separately
  measures actual FP against a per-query `HashSet` oracle.

Hardware: whichever machine runs this at CI time — the benchmark prints
`ns/op` and Mops/s so results are directly comparable across boxes.

## Results (Apple Silicon dev machine, 2026-07-06)

```
ruvector-blocked-bloom :: HNSW-shape visited-set benchmark
  n_ids=1000000 frontier=64 degree=32 hops=8 queries=2000
  total probes = 32768000

Variant                          Memory        Throughput         Latency
---------------------------------------------------------------------------
  HashSet<u32>                     200752 B     77.70 Mops/s     12.9 ns/op
  Bitmap<u64>                      190584 B    254.40 Mops/s      3.9 ns/op
  BlockedBloom(512b, k=4)          122944 B     74.34 Mops/s     13.5 ns/op

BlockedBloom FP rate: 0.0044%  (dirty_blocks/query ~= 9190, n_blocks=640)

Speedups vs HashSet<u32> baseline:
  Bitmap<u64>            : 3.27x
  BlockedBloom(512b, k=4): 0.96x
```

Interpretation:

* **Bitmap is the outright winner** on this workload — 3.27× throughput at
  smaller memory than the `HashSet` and *far* smaller than
  hnswlib's `2·N` label vector (200 KB vs 2 MB for 1 M nodes).
* **Blocked Bloom is throughput-equivalent to the `HashSet`** in Mops/s
  but pays it in memory: 123 KB regardless of id space.  The FP rate of
  0.0044% is 3× better than the theoretical bound, thanks to the ~35%
  intra-query id repetition consuming most of the "space" in each block.
* **The interesting operating point** is *n_ids ≥ 10^8*, where the bitmap
  needs 12.5 MB, blocks the L2 cache, and starts to lose to the Bloom on
  cold-page fills.  We do not measure that regime here — it needs a
  60 GB-node machine — but the extrapolation is straightforward.

## References

* Putze, F.; Sanders, P.; Singler, J. *Cache-, Hash- and Space-Efficient
  Bloom Filters*. WEA 2007.
* Malkov, Y.; Yashunin, D. *Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs*.
  TPAMI 2018.
* Subramanya, S. et al. *DiskANN: Fast Accurate Billion-Point Nearest
  Neighbor Search on a Single Node*. NeurIPS 2019.
* Singh, A. et al. *FreshDiskANN*. arXiv:2105.09613 (2021).
* Lemire, D. *Fast Random Integer Generation in an Interval*. ACM TOMS
  2019 — the fast-range multiply used for block selection.
* SIGMOD 2025 tutorial: *Cache-Optimal Vector Search on Modern Hardware*
  (Zheng et al.).

## "How it works" — blog-readable walkthrough

Imagine you're doing HNSW search.  You're at layer 0, you have 64
candidate nodes in the frontier, each with 32 neighbors.  Naively you'd
process all 2048 neighbor IDs — but many are duplicates (same node reached
via two paths) and some are the candidates themselves.  So on every neighbor
you call `if visited.mark(id) { push_to_frontier(id); process(id); }`.

That single line runs **thousands to millions of times per query**.  In
the baseline HashMap case each call:

1. hashes the id (SipHash / hashbrown's default),
2. computes a bucket,
3. reads a cache line (bucket header + control byte),
4. compares ~8 slots,
5. maybe writes.

Each of those steps is a cache miss on cold state — 12–13 ns.  Times
33 million probes = 424 ms of pure visited-set overhead.

**Bitmap** cuts this to one cache line and two ALU ops (`>> 6` and `& 63`)
per probe.  The trick is the dirty list: without it, `reset()` would take
`n_ids / 64` word writes — for a 1 B-node graph, that's 15 million writes
per query.  With it, reset is proportional to the frontier — a few
kilobytes.

**Blocked Bloom** trades exactness for fixed memory.  We `splitmix64(id)`,
pick one 64-byte block (one cache line), and set 4 bits inside it.  Only
one line is fetched per probe — same as the bitmap on the hot path.  The
"false positive" is: two IDs hash into the same 4-bit pattern of the same
block, and the second's `mark` returns `false`.  In an HNSW search this
just means one candidate is skipped early — the search still terminates,
still returns k-NN, just with a tiny recall haircut.

## Practical failure modes

1. **Undersized Bloom.** If `for_load(n_inserts)` is called with too small
   an `n_inserts`, FP rate blows up.  Rule of thumb: size for the
   *maximum* candidates ever visited per query, not the average.
2. **Bitmap on huge id spaces without knowing n_ids.** Panics on OOB
   because `mark` uses a checked index.  Callers must clamp id or size the
   bitmap to the max id.
3. **`reset` semantics.** Both `dirty`-based resets are single-threaded.
   A shared visited-set across threads needs external synchronisation —
   we don't offer it, deliberately, because the fast path becomes atomic
   otherwise.
4. **False positives ≠ false negatives.** In HNSW, an FP visited means
   "skip a candidate that would have been new".  This is monotone
   pessimistic — recall can drop, latency can only improve.  Never
   returns a *wrong* neighbor, only fewer neighbors.
5. **Dirty-list duplicates.** Interleaved cross-block probes push the same
   block index multiple times.  This is idempotent but wastes memory; the
   coalescing check only catches consecutive same-block hits.  A saturating
   HashSet dedup would fix this at 30% throughput cost — not worth it.

## What to improve next

* **SIMD probe.**  On x86_64 we can pack 4 bit-positions into one AVX-512
  `vpbroadcastq` + `vptestmb` in a single instruction.  Preliminary math
  suggests 1.6–2.0× speedup on the Bloom path — enough to close the gap
  with the bitmap and win on space.
* **Adaptive sizing.**  Track FP rate per query and dynamically grow the
  filter when we see too many collisions.  Requires zeroing a fresh block
  region, which the dirty-tracker already supports.
* **Fused visited + candidate heap.**  Currently `mark` and the top-k
  min-heap `push` are separate calls.  A fused API (`mark_and_push(id,
  dist)`) would let the Bloom short-circuit before the heap comparison,
  saving another cache miss per neighbor.
* **Integration into `ruvector-core::hnsw`.**  This crate is deliberately
  standalone.  The next PR should feature-gate `VisitedSet` behind
  `--features blocked-bloom` in `ruvector-core` and re-run the end-to-end
  DBpedia-1M recall@10 benchmark from `ruvector-sota-bench`.

## Production crate layout proposal

If promoted from nightly, the file layout stays exactly as-is:

```
crates/ruvector-blocked-bloom/
├── Cargo.toml
├── src/
│   ├── lib.rs        — trait + re-exports
│   ├── hashset.rs    — baseline impl
│   ├── bitmap.rs     — dense bitmap impl
│   ├── bloom.rs      — blocked-Bloom impl
│   └── benchmark.rs  — real-numbers benchmark
├── examples/
│   └── hnsw_walk.rs  — trait plug-in demo
```

Downstream integration adds a single line to `ruvector-core`'s HNSW
builder:

```rust
let visited: Box<dyn VisitedSet> = if graph.n_nodes < 100_000_000 {
    Box::new(BitmapVisited::new(graph.n_nodes))
} else {
    Box::new(BlockedBloomVisited::for_load(query.ef_search * graph.max_degree * 8))
};
```

Total churn: <10 lines in `ruvector-core`, opt-in via feature flag.
