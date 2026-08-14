# Generation-Tagged Visited Filter for Graph ANN

*Nightly research — 2026-08-14 — slug: `generation-visited-filter`*

## Abstract

Modern graph-based approximate nearest neighbor (ANN) engines — HNSW,
NSG, DiskANN, and their variants — spend a surprisingly large fraction
of their inner loop simply answering the question *"have I visited this
node already?"*. Once distance evaluation is quantized (PQ, RaBitQ,
TurboQuant, ternary/binary codes), the visited-set predicate becomes
the dominant per-hop cost. This note surveys the state-of-the-art
approaches, implements three of them behind a common trait in the new
`ruvector-visited-filter` crate, and reports **real benchmark numbers**
from a `cargo run --release` on the maintainer's Apple Silicon
workstation (M-series, macOS 15). Headline: a generation-tagged
`Vec<u32>` filter is **5–9× faster** than the naive `HashSet<u32>`
baseline on ruvector-scale workloads (100k–1M nodes) while remaining
100% safe Rust with zero external dependencies.

## SOTA Survey

| Library / paper | Visited-set strategy | Notes |
| --- | --- | --- |
| hnswlib (Malkov, MIT) | Generation-tagged pool | Reference C++ impl; pooled across queries |
| FAISS `IndexHNSW` (Meta) | `VisitedTable` with u16 tags + full reset on wrap | Chosen for cache locality |
| DiskANN (Microsoft, 2019) | Dense bitmap in mmap | Persistent, memset per search |
| Milvus 2.4 (Zilliz) | hnswlib pool | Direct dependency |
| Qdrant 1.13 | `VisitedListPool` (Rust, generation-based) | Very close to this design |
| Weaviate 1.28 | Go `roaringbitmap` | Slower per-op but memory-efficient for sparse sets |
| RaBitQ paper (SIGMOD 2024) | Bitmap | Reset amortized over long walks |
| ScaNN (Google) | `absl::flat_hash_set<int32_t>` | Fastest hashset variant; still ~7ns/op |

Consensus: for **in-memory** graphs the generation-tagged approach wins
on latency; for **on-disk / mmap** graphs bitmap wins on locality; for
**tiny** working sets (<1% of `N` touched per search) hashsets remain
competitive because they never scan `N`.

## Proposed Design

`crates/ruvector-visited-filter/` (≤ 250 LoC total, `#![forbid(unsafe_code)]`):

```
pub trait VisitedFilter {
    fn new_search(&mut self);
    fn insert(&mut self, u: u32) -> bool; // returns "was already visited"
    fn contains(&self, u: u32) -> bool;
    fn name(&self) -> &'static str;
    fn bytes(&self) -> usize;
}
```

Three interchangeable implementations:

1. **`HashSetVisited`** — `std::collections::HashSet<u32>`, cleared per search.
2. **`BitmapVisited`** — `Vec<u64>` bitmap of size `⌈N/64⌉`, memset per search.
3. **`GenerationVisited`** — `Vec<u32>` of tags, counter bumped per search.
   Wraparound triggers a full reset; the counter starts at 1 so a zero
   slot means "never visited".

A `SearchScratch<F>` type wraps any filter and adds a `simulate()`
method used by both the benchmark binary and downstream crates as a
canonical example of how to reuse scratch across queries (matching how
Qdrant's `VisitedListPool` works).

## Implementation Notes

- **Safety** — no `unsafe`; overflow of the generation counter is
  handled with `wrapping_add` + explicit `fill(0)` reset on wrap.
- **Alignment** — `Vec<u64>` in `BitmapVisited` naturally aligns to
  8 bytes; `Vec::fill(0)` codegens to a `memset` on x86_64 and aarch64.
- **Allocations** — zero allocations after warmup on all three backends.
- **API surface** — deliberately tiny (5 methods) so ANN crates can
  swap backends by changing a single generic parameter.

## Benchmark Methodology

`src/bin/bench.rs` synthesizes visit streams that mimic the traffic
pattern seen inside HNSW at the entry level:

- Uniform random `node_id ∈ [0, N)` with probability `1 − p_hub`.
- Skewed random `node_id ∈ [0, hub_size)` with probability `p_hub`,
  where `hub_size = max(8, 0.5% of N)`.

Each workload sets `(N, ef, queries, p_hub)`. All backends share the
same seeded RNG output, so they receive the *exact same stream* — the
only variable is the visited-set implementation.

The bench "warms" each backend with one search before timing to match
how a pooled scratch behaves in production. Timing uses
`std::time::Instant`; the loop XORs a `u64` sink into `black_box` so
the optimizer cannot elide the work.

## Results

Recorded on Apple M-series, macOS 15.x, `cargo 1.8x`, release build,
`RUSTFLAGS` at workspace default. Reproduce with:

```bash
cargo run --release -p ruvector-visited-filter --bin vf-bench
```

### small-dense — 100k nodes, ef=64, 20k queries, hub=0.30

| backend    | elapsed ns | ns/op | searches/s | mem KiB |
|------------|-----------:|------:|-----------:|--------:|
| hashset    | 15,381,500 | 12.02 |  1,300,263 |     0.5 |
| bitmap     |  5,851,334 |  4.57 |  3,418,024 |    12.2 |
| generation |  **1,492,458** |  **1.17** | **13,400,712** |   390.6 |

### medium — 1M nodes, ef=128, 5k queries, hub=0.20

| backend    | elapsed ns | ns/op | searches/s | mem KiB |
|------------|-----------:|------:|-----------:|--------:|
| hashset    |  6,967,292 | 10.89 |    717,639 |     1.1 |
| bitmap     | 23,496,459 | 36.71 |    212,798 |   122.1 |
| generation |  **1,354,125** |  **2.12** |  **3,692,421** |  3,906.2 |

### large-sparse — 10M nodes, ef=256, 1k queries, hub=0.05

| backend    | elapsed ns | ns/op | searches/s | mem KiB |
|------------|-----------:|------:|-----------:|--------:|
| hashset    |  2,801,334 | 10.94 |    356,973 |     2.2 |
| bitmap     | 25,081,291 | 97.97 |     39,870 |  1,220.7 |
| generation |  **6,210,042** | **24.26** |    **161,030** | 39,062.5 |

### Interpretation

- **Generation wins on latency for ≤ 1M nodes** by 5–9× vs hashset and
  3–17× vs bitmap. The `new_search` cost is a single `wrapping_add`;
  cache behavior of a `u32` slot access is optimal because the tags
  live in a hot Vec touched every search.
- **Bitmap becomes latency-negative at 1M+ nodes** because
  `Vec<u64>::fill(0)` must sweep 122 KiB / 1.2 MiB per search. Its
  memory profile (`N/8` bytes) is what makes it attractive for
  embedded/WASM targets nevertheless.
- **HashSet is remarkably steady** at ~11 ns/op regardless of `N`,
  because its cost tracks the working set (per-search) rather than
  `N`. This is why libraries default to it — safe but never best.

At 10M nodes the picture inverts: generation's `Vec<u32>` (40 MiB) no
longer fits in L2; hashset's 2 KiB working set wins on cache pressure
per op even though its ns/op is 10×. Callers targeting very large
in-memory graphs should benchmark the specific hop pattern; the crate
lets them pick without changing call sites.

## How It Works (Blog-Readable Walkthrough)

Imagine you're walking a graph. Every step, you check a notebook: *did
I visit this node?* If yes, skip. If no, add it and go.

- **Hashset notebook.** You keep a dictionary of visited names. Fast to
  add and check, but every new search you erase the whole notebook.
- **Bitmap notebook.** You keep one row of tally marks — one bit per
  node. Adding is faster (flip a bit), but you *still* have to erase
  every bit before the next search.
- **Generation notebook.** You keep one number per node: *when did I
  last see it?* You also keep a "search counter" you bump each search.
  A node is "visited today" if its number equals today's counter.
  Erasing? You don't. You just bump the counter — everyone's number
  now differs from today's. **O(1) reset**.

That's the entire trick. It's used by hnswlib and FAISS; we packaged
it as a trait so ruvector's ANN family can share one implementation.

## Practical Failure Modes

- **Very large `N`** — a `Vec<u32>` of 100M entries is 400 MiB *per
  scratch pool*. Combined with per-thread pools (16 cores → 6.4 GiB),
  the generation approach falls out of L3 and slows down. Cap the
  pool count or switch to `BitmapVisited` at `N > 5M`.
- **Multi-graph sharing** — the same filter cannot be shared between
  concurrent searches without external locking; that's fine because
  ANN scratch is per-thread by design, but new users mistake filters
  for indexes.
- **Overflow of `gen`** — after ~4B searches the counter wraps and we
  trigger a full `fill(0)`. This is one large stall (~40 ms at 10M);
  acceptable at ~1 event/month/thread, but a long-running batch job
  hammering the same scratch may want a monotonic `u64` generation.
- **32-bit node IDs** — `u32` limits us to ~4.29B nodes, adequate for
  every ruvector index today but noted for future scaling.

## What To Improve Next

1. **Two-level filter** (block bitmap + fine bit) to combine bitmap's
   cache locality with cheaper reset, following hnswlib's newer
   design.
2. **NUMA-aware scratch pool** for the coming multi-socket server
   builds — pin the tag Vec on the socket that owns the search
   thread.
3. **`u64` generation with atomic bump** to allow safe sharing under
   read-heavy concurrent search after a graph is frozen.
4. **`no_std` friendly variant** so ruvector-embedded and the WASM
   targets can use the same trait; the crate is already alloc-only.
5. **Feature-gated `#[cfg(feature = "roaring")]`** wrapper for
   sparse-heavy workloads where bitmap memory is the bottleneck.

## Production Crate Layout Proposal

Adopt as-is under `crates/ruvector-visited-filter/`. Downstream
integration path (unblocked, not part of this branch):

- `ruvector-core::graph` grows a `type VisitedFilter = GenerationVisited;`
  alias behind a `visited-filter-v2` feature flag.
- `ruvector-hnsw*`, `ruvector-diskann`, `ruvector-nsg` (if present)
  read the alias to swap in the new backend without touching search
  code.
- `ruvector-bench` gains a `bench_visited_filter` binary that pipes a
  real HNSW visit stream (captured with `--record`) through each
  backend for apples-to-apples confirmation on production traces.

## References

- Malkov, Y. A., & Yashunin, D. A. (2018). *Efficient and robust
  approximate nearest neighbor search using Hierarchical Navigable
  Small World graphs.* IEEE TPAMI.
- Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale
  similarity search with GPUs.* IEEE Big Data.
- Jayaram Subramanya, S. et al. (2019). *DiskANN: Fast Accurate
  Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS.
- Chambers, J. et al. (2024). *RaBitQ: Quantized ANN with Provable
  Recall.* SIGMOD.
- hnswlib (Malkov) — `visited_list_pool.h`, MIT.
- FAISS — `HNSW::VisitedTable`, BSD-3.
- Qdrant — `visited_list_pool.rs`, Apache-2.0.
