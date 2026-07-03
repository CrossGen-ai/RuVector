# ADR-272: Ptolemaic Pivot Pruning for Exact k-NN

- **Status**: Proposed (PoC landed — `crates/ruvector-ptolemaic`)
- **Date**: 2026-07-03
- **Slug**: `ptolemaic-pivot-pruning`
- **Extends**: baseline metric-space pruning literature (Chávez, Navarro, Hetland)

## Context

`ruvector` has extensive quantisation and graph-index coverage (`ruvector-rabitq`,
`ruvector-anisotropic-pq`, `ruvector-diskann`, `ruvector-spann`, ...) but no
first-class **metric-space pivot pruning** primitive. Pivot pruning is a
classical family of exact k-NN techniques that use pre-computed distances from
each dataset point to a small number of "pivot" reference points to derive a
*lower bound* on `d(q, x)` — allowing candidates to be discarded without the
full distance evaluation.

The default lower bound is the **triangle inequality**:

```
d(q, x) >= | d(q, p) - d(x, p) |
```

**Ptolemy's inequality** — provably valid in all Euclidean spaces (and,
more generally, Ptolemaic metric spaces) — gives a two-pivot bound that is
never weaker and usually strictly tighter:

```
d(q, x) >= | d(q, p_a) * d(x, p_b) - d(q, p_b) * d(x, p_a) | / d(p_a, p_b)
```

This ADR proposes a small self-contained crate that implements both bounds
behind a common `KnnIndex` trait, giving the workspace a research-grade,
provably-exact pivot backend to compose with existing indexes (e.g. as an
exact rerank stage on top of an ANN candidate list).

## Decision

Ship `crates/ruvector-ptolemaic` with three swappable backends:

| Backend         | Lower bound          | Exact? | Purpose                              |
|-----------------|----------------------|--------|--------------------------------------|
| `LinearScan`    | none                 | ✅     | Baseline (correctness oracle)        |
| `TrianglePivot` | triangle over P pivs | ✅     | Standard AESA-style pivot pruning    |
| `PtolemaicPivot`| Ptolemy over P·(P-1)/2 pairs | ✅ | This crate's contribution           |

Design constraints:

- **Trait-based (`KnnIndex`)** so future backends (e.g. a Ptolemaic rerank
  stage over HNSW candidates, or a rotated-frame variant) drop in cleanly.
- **Two pivot selectors** (`select_random`, `select_farthest_first`) with
  farthest-first-traversal being the default per SOTA convention.
- **Real per-query statistics** (`SearchStats { dcos, pruned, bounds_checked }`)
  emitted by every backend — the currency of pruning research.
- **No mocks, no approximations**: all three backends return byte-identical
  top-k. A test (`triangle_and_ptolemaic_match_linear`) enforces this.

## Consequences

**Positive**

- First metric-space Ptolemaic bound in the workspace — a new tool for the
  filtered/re-rank pipelines (`ruvector-hybrid`, `ruvector-gnn-rerank`).
- Deterministic, unit-tested exactness — safe drop-in replacement for
  linear scan wherever true k-NN is required (small-N leaves in DiskANN
  IVF cells, exact-rerank shortlists).
- The `PivotTable` memory model (`4·P·(N + P + dim)` bytes) is trivially
  serialisable and cheap: ~780 KB for N = 20 000, P = 10, dim = 32.

**Negative / Honest**

- The Ptolemaic bound loop is `O(P²)` per candidate; measured wall-clock is
  worse than triangle in every tested config on M4 Max (auto-vectorised
  Euclidean is very cheap up to dim ≈ 128). See `bench_results.txt`.
- Ptolemaic pays off only when the true distance is expensive relative to
  a handful of f32 multiplies — i.e. very high dim, non-Euclidean metrics
  (DTW, EMD), or *exact rerank on a shortlist* where the DCO count is what
  matters, not the per-candidate constant.
- Bound tightness is monotonically better than triangle (proved in unit
  test `ptolemaic_dominates_triangle`), but the *marginal* DCO cut over
  triangle varies 1–15 % depending on dim / pivot count.

**Neutral**

- Pivot table build time is `O(N·P·dim)`, negligible against index build
  time of any downstream ANN structure.

## Alternatives Considered

1. **Skip; the workspace has quantisation covered.** Rejected: pruning
   theory and quantisation theory answer different questions (bound-based
   vs. approximation-based). Both belong in the same toolbox.
2. **Fold into `ruvector-rairs`.** Rejected: RaIRS is IVF-scoped; pivot
   pruning is a metric-space primitive independent of partitioning.
3. **Implement only the tighter bound (Ptolemaic).** Rejected: triangle
   is the standard control; benchmarks without it are unfalsifiable.
4. **Full AESA / LAESA construction** (materialise all N² distances).
   Rejected as PoC scope creep — the two-pivot Ptolemaic bound is the
   novel contribution; AESA is a straight extension of `PivotTable`.

## References

- Hetland, M.L. (2009). *Ptolemaic Access Methods: Challenging the
  Reign of the Metric Space Model*. SISAP.
- Chávez, E., Navarro, G. (2001). *A Metric Index for Approximate
  String Matching*. LATIN.
- Bustos, B., Skopal, T. (2011). *Non-metric similarity search
  problems in very large collections*. ICDE tutorial.
- Yianilos, P.N. (1993). *Data structures and algorithms for
  nearest neighbor search in general metric spaces*. SODA.
