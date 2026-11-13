# ADR-362 — IVF Progressive Shortlist (IPSL)

- **Status:** Accepted-as-opt-in (negative-leaning result on cheap scorers)
- **Date:** 2026-11-13
- **Crate:** `crates/ruvector-ivf-progressive-shortlist`
- **Research doc:** [`docs/research/nightly/2026-11-13-ivf-progressive-shortlist/README.md`](../research/nightly/2026-11-13-ivf-progressive-shortlist/README.md)

## Context

IVF-PQ and IVF-L2 scanners in ruvector currently do depth-first scanning:
pick the `nprobe` best centroids, then scan every residual in each list.
This wastes work when a list's residual tail cannot improve the running
top-k.

Two bounds help prune:
- **Cauchy–Schwarz residual-norm bound.** Within a list, residuals of
  norm `r` cannot score above `-‖q−c‖² + 2‖q−c‖·r − r²`.
- **BlockMax-style tile bound.** Per-tile max residual norm drops
  monotonically down a norm-sorted list, so the bound lets us abandon
  a tail outright.

We asked: can a **breadth-first** scheduler across lists, revisiting
only lists whose tail bound still beats the current top-k threshold,
beat depth-first at identical recall?

## Decision

Ship **IPSL** as a new `Scanner` implementation alongside two baselines
(Exhaustive, NormOrderedAbandon) behind a shared trait.  Do **not**
replace the default scanner.  Expose IPSL as opt-in for future PQ-ADC
and SIMD-rerank pipelines, where its architectural win (skip tail of
expensive-to-score lists) is expected to materialise.

### Specifically

- Add trait `Scanner { fn search(...) -> Vec<Candidate>; }` with three
  impls.
- Build-time preprocessing: per posting, store `order[]` (residual-norm
  desc permutation) and `max_norm_tail[]` (Phase-2 prune bound
  precomputed).
- Query-time: Phase 1 scans prefix `m_init` across all probed lists;
  Phase 2 revisits the list with the best remaining upper bound,
  advancing `m_init` at a time, until no list's UB beats τ.

## Consequences

**Positive**
- Trait makes the three scanners A/B-testable per-query — a production
  ruvector engine can route easy queries to Exhaustive and hard ones
  (slow scorers, large lists) to IPSL.
- Preprocessing is one-shot and <5 % of index build time; it adds only
  `(n_vectors × 4) + (n_vectors × 4)` bytes = 8 bytes/vector (one u32
  permutation + one f32 norm).
- Clean architectural slot for future PQ-ADC and SIMD backends.

**Negative / honest**
- On this 50k×64d benchmark with a plain f32 scorer, IPSL is **5–45 %
  slower** than Exhaustive at the same recall — the per-list
  bookkeeping cost exceeds the single dot product it's trying to skip.
- Recall is byte-for-byte identical; the loss is pure wall-clock.
- Even NormAbandon (B1) loses at larger `nprobe` because the bound
  rarely fires when `‖r_max‖ ≈ ‖q − c‖`.

**Measured** (dim 64, 200 queries, 256 cells; see research doc for
the full table):

| nprobe | B0 lat | B2 lat | Δ    |
|-------:|-------:|-------:|-----:|
| 8      |  46 µs |  55 µs | +19% |
| 16     |  74 µs | 103 µs | +40% |
| 32     | 161 µs | 201 µs | +25% |

## Alternatives considered

1. **Block-wise MaxScore (BlockMax-IVF).**  Tiles-of-32 inside each list
   with per-tile `max_norm`.  Equivalent prune power to B1, more cache
   friendly.  Deferred — the research doc lists it as next-step.
2. **Query expansion / Rocchio reprobe.**  Orthogonal to scheduling;
   changes recall, not latency.
3. **Replace default scanner with IPSL.**  Rejected on the
   wall-clock data above.
4. **Add IPSL only, skip baselines.**  Rejected — the trait + three
   impls is the whole deliverable; without baselines the comparison
   is not reproducible.

## Follow-ups

- ADR-363 (future): add PQ-ADC scorer behind `Scanner`; re-run this
  comparison and expect IPSL to win (per-code cost grows ~4×).
- Benchmark on SIFT1M (BIGANN subset) when a bench harness is available.
