# ADR-305: Generation-Tagged Visited Filter for Graph ANN Search

**Status:** Proposed
**Date:** 2026-08-14
**Deciders:** ruvector core team (nightly research bot)
**Related:** ADR-303 (entropy-adaptive-ann), ADR-302 (streaming-qng)

## Context

Every graph-based ANN algorithm ruvector ships (HNSW, DiskANN, NSG-style,
coherence-HNSW, entropy-adaptive) drives an inner loop that asks *"have I
already visited node u?"* on each neighbor expansion. On modern hardware,
once distance evaluation is quantized (PQ / RaBitQ / TurboQuant), that
predicate becomes the single hottest instruction in the search path.

Today most implementations use a per-search `HashSet<u32>` (or
`hashbrown::HashSet`). Measurements against 1M–10M-node visit streams in
`crates/ruvector-visited-filter/src/bin/bench.rs` show hashset dwelling
around **10.9 ns/op**, dominating the loop for any workload where a hop
costs less than that.

Two alternatives exist in the ANN literature and in production libraries
(`hnswlib`, `FAISS::IndexHNSW`):

1. **Dense bitmap** — `Vec<u64>` sized to the node count, memset-cleared
   per search. O(1) per op, but O(N/64) per new search.
2. **Generation-tagged array** — `Vec<u32>` where each slot stores the
   last-search generation that touched it. `new_search` bumps a counter
   (O(1) except on 32-bit wraparound). A node is "visited" iff its tag
   equals the current generation.

ruvector currently reimplements a visited-set inline inside each ANN
crate, which (a) prevents cross-crate optimization, (b) means no single
crate is benchmarked against alternatives, and (c) blocks future work
on capability-gated visitors (ADR-296 family) that need a common trait.

## Decision

Ship `crates/ruvector-visited-filter` — a dependency-free, zero-unsafe
crate exposing:

- `VisitedFilter` trait (`new_search`, `insert`, `contains`, `bytes`, `name`)
- `HashSetVisited`     — baseline
- `BitmapVisited`      — dense `Vec<u64>` bitmap
- `GenerationVisited`  — generation-tagged `Vec<u32>` with safe overflow
- `SearchScratch<F>`   — reusable pool object that ANN crates hold on their
  `ThreadLocal` scratch

Downstream ANN crates SHOULD switch to `GenerationVisited` for
in-memory graphs up to ~5M nodes and to `BitmapVisited` for embedded /
WASM targets where the 4× memory savings matter. `HashSetVisited`
remains available for sparse iteration patterns where the working set
is a tiny fraction of `N`.

## Consequences

**Positive**

- `ns/op` for the visited predicate drops from **~11 ns → ~1.2–2.1 ns**
  (5–9× faster) on medium workloads (see benchmark numbers in the
  companion research doc).
- Single trait unlocks capability-gated visitors (ADR-025x) without
  each ANN crate re-implementing plumbing.
- Zero unsafe code, no external deps beyond `rand` for the bench binary.

**Negative / risks**

- `GenerationVisited` allocates `4 × N` bytes eagerly per search
  scratch. At `N = 10M` that is **40 MiB per scratch** — non-trivial
  when combined with per-thread pools. `BitmapVisited` at `N/8` bytes
  is the mitigation; the crate lets callers pick.
- Bitmap's `fill(0)` dominates at large `N` (measured **98 ns/op** at
  10M nodes vs generation's **24 ns/op**) — callers must be aware that
  the tradeoff inverts with graph size.
- Generation counter wraps at 2^32 searches — safe-handled by a full
  `fill(0)` reset. Long-running processes hit this roughly every 4B
  searches (~1 event / month at 1kQPS), a negligible amortized cost.

## Alternatives Considered

1. **`hashbrown::HashSet<u32>`** — marginal (~15%) improvement over
   `std::HashSet`, still ~9 ns/op. Rejected because it does not close
   the gap to bitmap/generation approaches.
2. **`roaring::RoaringBitmap`** — excellent for sparse sets but has per-op
   branching overhead (~30 ns/op measured on similar workloads in
   existing crates); rejected for the inner search loop.
3. **`Vec<bool>` reset per search** — equivalent to `BitmapVisited`
   except with 8× more memory bandwidth on the reset path. Strictly
   worse.
4. **Two-level (block-visited + bit-visited)** — the hnswlib pattern.
   Adds complexity without beating generation-tagged at ≤10M nodes;
   parked as future work if workloads grow.

## References

- Malkov & Yashunin (2018), *Efficient and robust ANN search using HNSW*.
- FAISS `HNSW::VisitedTable` implementation (generation-based).
- hnswlib `visited_list_pool.h` (generation-based, MIT license).
- ruvector companion doc: `docs/research/nightly/2026-08-14-generation-visited-filter/`
