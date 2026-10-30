# ADR-356: Successive Halving IVF-PQ Probing (SH-IVF)

- Status: **Rejected (negative result — kept as research reference)**
- Date: 2026-10-30
- Deciders: nightly research pipeline
- Prototype: `crates/ruvector-sh-ivf/`
- Research doc: `docs/research/nightly/2026-10-30-sh-ivf-probing/README.md`

## Context

IVF-PQ scans with large `nprobe * avg_posting` candidate pools spend most
of their time in Asymmetric Distance Computation (ADC): M table lookups
plus a sum per candidate. The literature on anytime search and
successive halving (Jamieson & Talwalkar 2016; Nardini et al., "ADC
early abandon", 2024) suggests cascading cheap-to-expensive distance
bounds and pruning after each stage: the first few PQ subspaces give a
noisy but usable estimate of the full ADC distance, so the pool can be
halved at a fraction of the cost.

We wanted to measure whether a three-stage successive-halving cascade
(M/2 prefix → 3M/4 prefix → full M) delivers a Pareto improvement over
single-stage full-M ADC at matched recall, inside the ruvector IVF-PQ
stack.

## Decision

**Do not fold SH-IVF probing into ruvector-core yet.** On the honest
benchmark described below, SH-IVF is Pareto-dominated by just lowering
`nprobe` on single-stage ADC. The crate stays in-tree as a research
reference and a honest teaching example of a well-known-but-fragile
technique.

## Benchmark (crates/ruvector-sh-ivf, release, Apple M-series)

Setup: n=50 000 synthetic clustered vectors (D=96, 128 generator
clusters, sigma=0.35), 256 IVF cells, M=12 PQ subspaces × K=256,
200 queries, k=10. Numbers from
`./target/release/sh-ivf-demo` on 2026-10-30.

| scanner | nprobe | recall@10 | avg µs/query | p95 µs |
|---|---|---|---|---|
| full ADC | 8 | 0.468 | 76.7 | 91.6 |
| 2-stage (M/2, keep=200k) | 8 | 0.468 | 84.1 | 107.7 |
| SH (M/2, 3M/4, keep=200k/100k) | 8 | 0.455 | 98.7 | 124.6 |
| full ADC | 16 | 0.468 | 145.2 | 174.1 |
| 2-stage (M/2, keep=160k) | 16 | 0.442 | 146.2 | 169.0 |
| SH tight | 16 | 0.401 | 160.6 | 194.4 |
| SH loose | 16 | 0.242 | 139.3 | 165.5 |
| full ADC | 32 | 0.468 | 279.2 | 326.1 |
| SH tight | 32 | 0.293 | 256.0 | 299.6 |
| SH loose | 32 | 0.206 | 243.6 | 290.7 |

Memory footprint: ~997 KB total (98 KB centroids, 98 KB codebooks,
800 KB postings including u32 ids).

**Interpretation.** Across every nprobe we tested, the full-ADC
single-stage baseline sits on the recall/latency Pareto front. SH-IVF
either matches recall at higher latency, or saves ~10–15% latency at a
30–60% recall loss. There is no (recall, latency) point where SH-IVF
beats single-stage full ADC.

### Why

The per-cell LUT build cost is `M·K·d_sub = 12·256·8 = 24 576`
multiply-adds. For 32 probed cells that's ≈ 786 k ops of fixed cost
*before* any ADC scan begins. Our ADC scan at nprobe=32 is only
`avg_posting · M ≈ 195·12·32 = 75 k` lookups. The scan is already
smaller than the LUT build, so compressing the scan with a cascade
cannot buy much. SH-IVF only pays off when the ADC phase dominates the
total query time, which requires either:

1. Much longer postings (10 k+ entries per cell, i.e. billion-scale
   collections with modest `nprobe`); or
2. A LUT16/FastScan backend (ADR-related: `ruvector-pq-fastscan-lut16`)
   whose SIMD scan is cheap enough that LUT build becomes negligible —
   but then full ADC is already so fast that the cascade's constant
   overhead (prefix sum, quickselect, re-score) eats the savings.

Neither regime is the mainstream ruvector workload today.

## Consequences

Positive:

- Negative result documented and reproducible; future nightlies that
  touch IVF-PQ scan ordering can cite this measurement before trying
  the same cascade.
- A working Rust implementation of the technique (trait-based, swappable
  scanner) is available in-tree for anyone wanting to re-benchmark on
  a different workload.
- Reinforces that LUT build cost — not ADC scan cost — is the dominant
  per-query term for ruvector's current PQ settings. Follow-up work
  should attack LUT amortization (shared LUTs across cell neighbours,
  precomputed centroid-residual LUTs) before adding scan cascades.

Negative:

- Adds ~500 lines of crate code and ~130 lines of benchmark/demo code
  that the project does not use in production.

## Alternatives considered

- **Just lower `nprobe`.** Baseline at nprobe=4 (not shown above) gives
  recall ≈ 0.44 at ~41 µs/query, Pareto-dominant over every SH config
  we measured.
- **Early-abandon ADC per candidate** (abort the inner M loop when
  partial sum exceeds the current top-k threshold). Simpler, in-place,
  and gives the same cheap-prefix idea without the quickselect
  overhead. We expect this would beat both baseline and SH on long
  postings; it is deferred to a follow-up nightly.
- **FastScan/LUT16 (SSE/NEON) ADC.** Orthogonal to this ADR; folds the
  scan down to ~1 ns/candidate, after which SH's constant overhead
  dominates. Already tracked in `ruvector-pq-fastscan-lut16`.

## References

See research doc for full citations:

- Jamieson, Talwalkar (2016), *Non-stochastic Best Arm Identification
  and Hyperparameter Optimization* (AISTATS). Successive halving.
- Jégou, Douze, Schmid (2011), *Product quantization for nearest
  neighbor search* (IEEE TPAMI). IVF-ADC baseline.
- André, Kermarrec, Le Scouarnec (2015), *Cache locality is not enough:
  High-performance nearest neighbor search with Product Quantization
  Fast Scan* (VLDB). LUT16/FastScan.
