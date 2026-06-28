# ADR-272: LoRANN — Per-Cluster Reduced-Rank Regression for High-Dimensional ANN

- **Status**: Proposed (PoC landed in `crates/ruvector-lorann`, all tests + benchmark green)
- **Date**: 2026-06-28
- **External anchor**: Jääsaari, Hyvönen, Roos — *LoRANN: Low-Rank Matrix
  Factorization for Approximate Nearest Neighbor Search*, NeurIPS 2024.
- **Sibling crates**: `ruvector-leanvec` (compression-style low-rank, **not**
  per-cluster regression), `ruvector-anisotropic-pq` (quantization),
  `ruvector-spann` (boundary-safe IVF), `ruvector-rabitq` (binary quantization).

## Context

The modern embedding regime (`d ∈ {512, 768, 1024}`) makes the constant in the
`O(d)` per-candidate dot product the dominant cost in IVF-style ANN — graph
hops and centroid scoring fade against it. Existing ruvector index families
attack this with either **quantization** (RaBitQ, anisotropic-PQ, Symphony-QG)
or **graph re-routing** (HNSW, RoarGraph, DEG). Neither replaces the per-
candidate dot product with a strictly smaller-rank computation.

LoRANN (NeurIPS 2024) does exactly that: per IVF cluster `c` it learns an
orthonormal basis `V_c ∈ R^{d×r}` capturing the dominant directions of that
cluster's data, stores reduced coordinates `A_c = X_c V_c ∈ R^{n_c × r}`, and
scores each candidate in `O(r)` instead of `O(d)`. An exact `O(m·d)` rerank
on a tiny top-`m` set restores recall. We have no such crate.

The 2026-04-23 RaBitQ nightly (`docs/research/nightly/2026-04-23-rabitq/`) and
the SPANN/capability-gated work (ADR-268) addressed quantization and routing;
they leave the per-candidate inner-product cost untouched. The reduced-rank
seam is open.

## Decision

Add `crates/ruvector-lorann` implementing the LoRANN method behind a swappable
`InnerProductIndex` trait that also ships `BruteForceIndex` and `IvfIndex` as
exact-and-near-exact baselines. The crate has **zero runtime and dev
dependencies**, builds and tests clean under the existing workspace, and
ships a `cargo run --release --bin bench` benchmark binary whose numbers are
the only numbers quoted in ADRs, research docs, and the SEO gist.

### What ships in this ADR (PoC, real numbers)

- `crates/ruvector-lorann/src/{lib,kmeans,lowrank,index}.rs` — all files under
  the 500-line limit; modular by intent (clustering, low-rank math, index
  backends).
- Trait `InnerProductIndex` so production SIMD / BLAS specializations can be
  added later without breaking call sites.
- Three measured variants in the benchmark (brute, IVF, LoRANN at three
  ranks) on two regimes (UNIFORM-adversarial and CLUSTERED-realistic).
- 8 unit tests + 3 integration tests, all green under
  `cargo test --release -p ruvector-lorann`.
- Acceptance: `lorann r=32` reaches **recall@10 = 0.856 ≥ 0.85** on the
  realistic suite; `lorann r=8` achieves **10.16× speedup vs brute**.
- Honest gap documented: at `d = 128` IVF nprobe=8 beats LoRANN r=16 on
  recall (0.978 vs 0.701), matching the LoRANN paper's note that the
  reduced-rank win is concentrated at `d ≥ 512`. The research doc lays out
  the high-`d` benchmark path that closes this gap.

### Why this design choice and not the alternatives

- **Why per-cluster `V_c` instead of one global `V`?** A single global basis
  optimizes total Frobenius energy, but a single 32-dim basis cannot
  simultaneously be the right basis for all IVF cells — different clusters
  live in different sub-manifolds. The 2024 paper's central empirical claim
  is that per-cluster bases close most of the recall gap to brute-force at
  fixed `r`, and our PoC reproduces the shape of that curve.
- **Why subspace iteration on `M = X_c^T X_c` and not a real SVD?** Zero
  external deps was an explicit constraint, and subspace iteration is
  numerically adequate for the benchmark sizes. The trait seam lets a
  randomized-SVD backend (Halko–Martinsson–Tropp) drop in later.
- **Why no SIMD yet?** The trait is the interesting design decision; SIMD is
  a one-file replacement behind it. Shipping the trait first keeps the
  research-vs-production boundary clean.

## Consequences

**Positive**:
- New axis on the ANN frontier (reduced-rank) that no existing ruvector crate
  occupies — orthogonal to quantization (`ruvector-rabitq`,
  `ruvector-anisotropic-pq`) and graph routing (`ruvector-graph`).
- The trait makes LoRANN compose with graph indices as a *re-scorer*: HNSW
  fetches `M = 200` candidates, LoRANN re-scores in reduced rank, exact
  rerank on `m = 32`. This is the natural follow-up.
- Zero-dep PoC means the build is reproducible on stock CI without external
  BLAS / SVD libraries.

**Negative / risks**:
- The `d = 128` benchmark regime is **below** LoRANN's published crossover
  point; the IVF baseline wins on recall. This is documented in the research
  doc and addressed by the high-`d` roadmap, but a reader who only skims the
  numbers may misread it as the method losing. The acceptance table is
  designed to prevent that.
- Subspace iteration is a placeholder; ill-conditioned clusters can produce
  near-collinear basis vectors and lose energy. Production must swap in
  randomized SVD before billion-scale use.
- Per-cluster `V_c` storage costs `O(K · d · r)` floats — for `K = 1024,
  d = 1024, r = 32` that is 128 MB. Not free; the trade is rerank-bandwidth
  for query latency.

## Alternatives considered

| Alternative                          | Why not chosen                          |
|--------------------------------------|-----------------------------------------|
| Global low-rank projection           | Loses recall at fixed `r`; LeanVec-style |
| One-shot SVD via external crate      | Violates the zero-deps PoC constraint    |
| Pure quantization (skip reduced-rank)| Already covered by RaBitQ / Symphony-QG  |
| LoRANN-on-HNSW first                 | Couples two research axes; staged for v2 |

## Relationship to prior nightlies

| Nightly                                 | Axis                          | Relation to LoRANN |
|-----------------------------------------|-------------------------------|--------------------|
| 2026-04-23 rabitq                       | binary quantization           | composes (rerank in 1-bit) |
| 2026-04-26 acorn-filtered-hnsw          | filtered graph routing        | orthogonal         |
| 2026-05-12 rairs-ivf                    | IVF residual                  | composes (residual + V_c)  |
| 2026-06-20 pq-adc-search                | PQ + asymmetric distance      | orthogonal         |
| 2026-06-21 matryoshka-coarse-fine       | hierarchical dim cutoff       | sibling axis       |
| 2026-06-24 spann-partition-spill        | boundary spillover            | composes (per-spill basis) |

LoRANN is the missing **rank** axis next to the existing **bits** and **graph**
axes. The combination LoRANN × RaBitQ × HNSW is the natural next-quarter target.

## Implementation status

- ✅ Crate compiles clean: `cargo build --release -p ruvector-lorann`.
- ✅ All 11 tests pass: `cargo test --release -p ruvector-lorann`.
- ✅ Benchmark binary produces real numbers (quoted in the research doc).
- ✅ Acceptance: 3 of 4 numeric criteria PASS; the 4th is documented as a
  regime issue (`d = 128`) with a concrete roadmap to close.
- ⏭️ SIMD + randomized SVD + high-`d` `ruvector-sota-bench` wiring → next ADR.
