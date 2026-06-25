# ADR-268: SeRF-style Range-Filtered ANN as a first-class ruvector backend

- **Status:** Proposed (PoC landed 2026-06-25)
- **Date:** 2026-06-25
- **Related:** ADR-265 (benchmark suite), ADR-264 (matryoshka coarse-fine),
  `ruvector-acorn` (categorical filtered HNSW).
- **Research:** [`docs/research/nightly/2026-06-25-serf-range-filtered-ann/README.md`](../research/nightly/2026-06-25-serf-range-filtered-ann/README.md)

## Status

Proposed. A working PoC ships in [`crates/ruvector-serf`](../../crates/ruvector-serf)
with three measured backends and a benchmark binary. Full SeRF snapshot
compression and iRangeGraph segment labels are deferred to a follow-up ADR.

## Context

ruvector already supports *categorical* attribute filtering via
`ruvector-acorn` (label-aware HNSW). It does **not** currently address
*numeric range* filters of the form `attr ∈ [lo, hi]`, which dominate
real-world workloads: time-windowed retrieval ("documents from last week"),
price bands, geographic latitude ranges, and any monotone-ish ordinal.

The naive solution is post-filtering on top of HNSW results. As selectivity
shrinks (narrow windows), post-filter recall and QPS both collapse — the
graph search burns its budget on out-of-range nodes, and the few hits that
survive filtering are unlikely to be the true nearest in-range neighbors.

The 2024 SOTA papers — SeRF (SIGMOD) and iRangeGraph (VLDB) — independently
proposed making the graph search itself range-aware. SeRF stores
ε-compressed HNSW snapshots over the sorted attribute axis; iRangeGraph
labels each edge with a segment-tree interval. Both deliver large speed-ups
and recall lifts on narrow windows.

## Decision

Adopt a SeRF-style range-filtered ANN crate (`ruvector-serf`) as a sibling
to `ruvector-acorn`. Ship a runtime-only edge-pruning variant first (this
PoC); plan snapshot compression and segment labels as follow-up work once
the runtime benefit is quantified against ruvector's HNSW substrate.

Key concrete choices:

1. **Trait-based design.** A `RangeAnn` trait with backends
   `LinearPrefilter`, `PostFilterNsw`, and `SerfIndex` lets new variants
   land without touching the bench harness or downstream call sites.
2. **k-NN graph substrate, swappable later.** The PoC uses a brute-force
   symmetric k-NN graph; the production crate will plug into the existing
   HNSW substrate (`ruvector-coherence-hnsw`) once the API is stable.
3. **In-range entry point + linear fallback.** SeRF's snapshot compression
   exists primarily to guarantee an in-range entry; we approximate it with
   `mid = (lo+hi)/2` (correct for monotone attributes) plus a small linear
   fallback when traversal under-fills the result set. This is honest about
   the PoC's limitations and matches what production hybrid systems do.
4. **Squared L2 first.** Add cosine and inner product after the
   range-aware traversal API stabilizes.

## Consequences

**Positive**

- ruvector gains a credible numeric-range filtered ANN path with measured
  44.6× speed-up over post-filter HNSW at narrow widths (1 %) and 0.997
  recall@10 — see `docs/research/nightly/2026-06-25-serf-range-filtered-ann/README.md`.
- Memory footprint is tiny: 1.30 MiB of adjacency for 10 000 × 64-d ⇒
  scales linearly with `n · avg_degree`.
- A trait-shaped crate keeps the door open for full SeRF and iRangeGraph
  without rewrites.

**Negative / trade-offs**

- The PoC's edge-pruning variant loses recall at *medium* widths (5 %
  recall@10 = 0.72 vs post-filter's 0.92). This is the regime where full
  SeRF snapshot compression actually matters; the PoC documents the gap.
- The entry-point trick assumes monotone attributes. Non-monotone
  attributes need a sorted-attribute side index, deferred to follow-up.
- Brute-force k-NN graph construction is O(n²d). Acceptable for nightly
  research at n = 10⁴; production needs HNSW.

**Open questions for the follow-up ADR**

- Should `ruvector-serf` re-use `ruvector-coherence-hnsw` or introduce a
  thin shared graph trait first? (Leaning thin shared trait.)
- Memory budget for snapshot compression: target ≤ 2× plain HNSW.
- Combined ACORN + SeRF predicate path: should the trait take a generic
  `Predicate` instead of a numeric range?

## Alternatives considered

1. **Just lift `ef` adaptively (SuperPostFilter, VLDB 2024).** Simpler but
   only mitigates recall loss; does not address the wasted compute on
   out-of-range traversal. Worth implementing as a fourth backend later
   for direct comparison.
2. **Pre-partition by attribute buckets.** Trivial but explodes memory and
   degrades recall at bucket boundaries. Already implicitly available via
   ACORN at the cost of treating ranges as categories.
3. **Linear scan only.** Wins at very low n and very narrow ranges (this
   PoC confirms it: 171 944 QPS at 1 %), but does not scale.
4. **DiskANN-Range port.** Heavier; defer until the graph trait stabilizes.

## Acceptance

PoC acceptance (this ADR):

- ✅ `cargo build --release -p ruvector-serf` succeeds.
- ✅ `cargo test --release -p ruvector-serf` — 12/12 tests pass.
- ✅ Benchmark binary prints real numbers (not mocks).
- ✅ At 1 % range width: SeRF ≥ 10× faster than post-filter at ≥ 0.95
  recall@10. (Achieved: 44.6× / 0.997.)
- ✅ Files under 500 lines (largest: `post_filter.rs` ≈ 150).

Full-feature acceptance (follow-up ADR): snapshot compression closes the
5–20 % recall gap to ≥ 0.95 within ≤ 2× memory of plain HNSW.
