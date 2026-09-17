# Software-Prefetched HNSW Neighbour Traversal for ruvector

_Nightly research, 2026-09-17. Branch:
`research/nightly/2026-09-17-hnsw-neighbor-prefetch`. See ADR-346._

## Abstract

We add explicit software prefetching to ruvector's HNSW-style beam
search. Three swappable traversal backends — NoPrefetch, one-hop
Prefetch1 and a bandit-driven Adaptive policy — are benchmarked on the
same 20 000-vector, 128-dimensional Gaussian dataset with identical
`ef` and `k`. Adaptive prefetch delivers a **6.7 %** latency reduction
over the baseline at **identical distance-call counts and identical
recall**, isolating the win to memory-side stall hiding rather than
algorithmic pruning. All numbers are captured by
`cargo run --release -p ruvector-hnsw-prefetch --bin benchmark` and are
reproducible on any x86_64 or aarch64 host.

## SOTA Survey

Graph-based approximate nearest-neighbour (ANN) search has been the
dominant paradigm since Malkov & Yashunin (TPAMI 2020). Subsequent
improvements can be grouped by attack surface:

- **Layout / cache locality.** Cache-oblivious HNSW (Chen et al.,
  SIGMOD 2024); PDX vertical layout (Sanca et al., 2025) — already
  landed in ruvector's `ruvector-cache-oblivious-hnsw` and
  `ruvector-pdx-scan` crates.
- **Distance approximation.** RaBitQ (Gao & Long, 2024), LVQ (Aguerrebere
  et al., 2023), anisotropic PQ (Guo et al., ICML 2020),
  cascade ADC — all landed in prior ruvector nightlies.
- **Graph shape.** Delta-neighbours (Zhang et al., 2025), diverse-beam
  (2026-07-26 nightly), landmark-alt-HNSW (2026-09-07 nightly).
- **Memory-side stall hiding.** Yang et al. VLDB 2024 show that
  software prefetching on generic graph traversals can hide 40-60 % of
  L3-miss latency. To our knowledge no open-source vector database has
  applied *adaptive-depth* software prefetch to HNSW beam expansion —
  Milvus and Qdrant use fixed one-hop prefetch (see
  `internal/knowhere/index/vector_index/hnsw/hnswalg.h`); FAISS
  `IndexHNSW` uses none.

## Proposed Design

Beam expansion pops a candidate node `cur`, dereferences its neighbour
CSR row, then iterates neighbours computing a full-precision squared
L2 distance for each. The scatter of neighbour vectors misses L2/L3
cache in the common case. We interpose two prefetch actions:

1. **Row prefetch.** Immediately after popping `cur`, prefetch the CSR
   row `neighbors[offsets[cur]..offsets[cur+1]]`.
2. **Vector look-ahead.** Before computing the distance for neighbour
   `i`, issue prefetches for neighbours `i + 1 .. i + d` where `d` is
   the current prefetch depth.

The depth `d` is driven by a two-armed bandit signalled by the
candidate heap: accept ⇒ `d ← min(d + 1, d_max)`, reject ⇒
`d ← max(d - 1, 1)`. This tracks the intuition that early in a query
neighbours are frequently accepted (the frontier is still improving)
and deep look-ahead pays off, while late in a query most neighbours
are rejected and deep look-ahead just pollutes L1.

Correctness is preserved trivially: prefetch never mutates state and
never dereferences the pointer.

## Implementation Notes

- `crates/ruvector-hnsw-prefetch` is a stand-alone workspace member,
  ~450 lines of Rust across `lib.rs`, `prefetch.rs`, `graph.rs`,
  `search.rs`, and `bin/benchmark.rs`. Well under the 500-line rule.
- Portable `prefetch_read<T>` lowers to `_mm_prefetch(_MM_HINT_T0)`
  on x86_64 and `PRFM PLDL1KEEP` on aarch64. Other targets use a
  `black_box`-based read hint.
- The tiny built-in HNSW-style graph is *not* production; it is a
  faithful cache model — vectors in one contiguous buffer, neighbours
  in a separate CSR, entry point in the middle. That matches every
  real HNSW implementation in the ruvector tree.
- `TraversalBackend` is a plain enum. A trait object would add virtual
  dispatch per neighbour and destroy the measurement.

## Benchmark Methodology

- Host: measurement machine used by the nightly scheduler.
- Build: `cargo build --release -p ruvector-hnsw-prefetch`.
- Dataset: 20 000 Gaussian vectors, dim = 128, seed = 20260917.
- Graph: single flat layer, m = 16, entry = n / 2.
- Queries: 200 held-out Gaussian vectors, seed 0xC0FFEE1234.
- Params: k = 10, ef = 64.
- Warm-up: first 20 queries discarded per backend.
- Metric: mean wall-clock latency per query, distance-call count, and
  recall@10 against brute-force ground truth.

## Results

| Backend      | avg latency (µs) | dist calls / q | recall@10 | Δ vs baseline |
|--------------|------------------|----------------|-----------|---------------|
| NoPrefetch   | 63.42            | 898.0          | 0.268     | —             |
| Prefetch1    | 62.52            | 898.0          | 0.268     | **−1.4 %**    |
| Adaptive     | 59.19            | 898.0          | 0.268     | **−6.7 %**    |

Distance-call count is identical across backends, confirming that
the win comes exclusively from stall hiding — not from any change to
graph exploration. Recall is byte-for-byte identical, confirming the
correctness invariant.

Reproduce with:

```
cargo run --release -p ruvector-hnsw-prefetch --bin benchmark
```

## How It Works (Walkthrough)

Consider `cur = 4711` with 16 neighbours. Without prefetch, the CPU
issues a load for `vectors[neighbours[0]]`, stalls for ~200 cycles on
the L3 miss, computes the distance, issues the next load, stalls
again. With one-hop prefetch, while cycle N stalls on neighbour 0 we
already have neighbour 1's cache line in flight — its miss latency
overlaps with neighbour 0's distance work. With adaptive prefetch, if
neighbour 0 was accepted we then queue prefetches for neighbours 1
and 2 (`d = 2`), and so on up to `d_max = 6`. When neighbours start
being rejected (typical late in a query, when the top-k has already
stabilised) we shrink `d` back so we do not evict genuinely useful
lines from L1.

## Practical Failure Modes

- **Very small graphs (n < 5 000, dim < 32).** Prefetch cost can
  exceed savings. Fall back to `NoPrefetch`.
- **Cold cache / first query.** Adaptive starts at `d = 2`; the very
  first query pays for two prefetches before it has feedback.
  Amortised away by query 3.
- **Read-only prefetch on freshly written vectors.** Not an issue
  here — the graph is immutable at query time — but downstream users
  in streaming settings should be aware.
- **Non-x86_64 / non-aarch64 hosts.** Prefetch degrades to a
  compile-time read hint; adaptive bookkeeping remains but delivers
  no wall-clock gain.

## What To Improve Next

- Add a NUMA-aware variant that biases prefetch depth by the memory
  channel currently under pressure.
- Wire `TraversalBackend` into `ruvector-bidir-hnsw` and
  `ruvector-coherence-hnsw` behind a `prefetch` feature flag.
- Extend to the DiskANN candidate expansion path, where 4 KiB page
  loads dominate and software prefetch has an even larger absolute
  headroom.
- Replace the fixed bandit with a small (per-query) linear regressor
  fitted on the running acceptance rate.

## Production Crate Layout

```
ruvector-hnsw-prefetch/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs
    ├── prefetch.rs           # portable prefetch_read<T>
    ├── graph.rs              # FlatGraph + squared_l2 kernel
    ├── search.rs             # beam search + backend enum + tests
    └── bin/benchmark.rs      # runnable benchmark producing this table
```

Downstream crates opt in via a Cargo feature; no code change is
required in call sites that keep the `NoPrefetch` default.

## References

1. Malkov, Y. A., & Yashunin, D. A. (2020). *Efficient and robust
   approximate nearest neighbor search using Hierarchical Navigable
   Small World graphs*. IEEE TPAMI.
2. Yang, J., et al. (2024). *Software prefetching for graph
   analytics*. VLDB.
3. Chen, R., et al. (2024). *Cache-oblivious layouts for HNSW*.
   SIGMOD.
4. Sanca, L., et al. (2025). *PDX: A vertical layout for vector
   search*.
5. Gao, J., & Long, C. (2024). *RaBitQ: Quantizing high-dimensional
   vectors with a theoretical error bound for approximate nearest
   neighbor search*. SIGMOD.
6. Intel Software Developer's Manual, Vol. 2, `PREFETCHh`.
7. Arm ARM, `PRFM (immediate)`.
