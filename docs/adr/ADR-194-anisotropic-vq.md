---
adr: 194
title: "Anisotropic Vector Quantization — score-aware PQ for MIPS recall"
status: accepted
date: 2026-05-16
authors: [nightly-research]
related: [ADR-193]
tags: [pq, quantization, mips, ann, vector-search, scann, nightly-research]
---

# ADR-194 — Anisotropic Vector Quantization (ruvector-anisotropic-pq)

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-05-16-anisotropic-vq` as
`crates/ruvector-anisotropic-pq`. `cargo build --release -p
ruvector-anisotropic-pq` is green; `cargo test -p ruvector-anisotropic-pq`
passes 5/5; `cargo run --release -p ruvector-anisotropic-pq --bin apq-bench`
produces the numbers cited in
`docs/research/nightly/2026-05-16-anisotropic-vq/README.md`.

## Context

ruvector already ships RaBitQ (1-bit rotated quantization), LeanVec
(dim-reduction + LVQ), AVQ (sample crate), and the new RAIRS IVF
(ADR-193). None of these implement **score-aware** quantization — i.e. a
codebook training loss whose objective is the *quality of the
inner-product estimate*, not the L2 reconstruction error.

Inner-product retrieval (MIPS) and cosine retrieval are the dominant
workloads for embedding-based search. For those, standard PQ wastes
representational capacity on the residual component orthogonal to the
query direction, because that component contributes negligibly to score
distortion at the relevant scale. Guo et al. (Google, ICML 2020) made
this observation precise and showed that re-weighting the per-point
quantization loss to penalise parallel error more heavily than
orthogonal error consistently improves recall at fixed compression
ratio. The technique is the core of ScaNN, which has been state of the
art on inner-product ANN benchmarks since 2020.

Despite the maturity of the algorithm there is no permissively-licensed
Rust implementation that ruvector can pull in. The reference ScaNN code
is C++ + TensorFlow with a non-trivial dependency footprint; FAISS has
no equivalent training mode upstream as of 2026; lance-core just landed
basic PQ but not the score-aware variant.

## Decision

We add a new crate `crates/ruvector-anisotropic-pq` that implements
three variants of product quantization behind a single `Quantizer`
trait:

| Variant   | Loss at training              | Encoding (assignment)        |
|-----------|-------------------------------|------------------------------|
| `Pq`      | `||x - c||²` (standard MSE)   | nearest by L2                |
| `Apq`     | `η · (r·u)² + (||r||² − (r·u)²)` | nearest by anisotropic loss |
| `OpqApq`  | same as `Apq`, after PCA-balanced rotation | rotate query, then APQ |

where `r = x_subvec − c`, `u = x_subvec / ||x_subvec||`, and `η ≥ 1` is
the parallel/orthogonal weight ratio.

The centroid update is a closed-form weighted-least-squares solve
inside each subspace: `(N·I + λ·Σᵢ uᵢuᵢᵀ) c* = Σᵢ xᵢ + λ·Σᵢ (xᵢ·uᵢ) uᵢ`
with `λ = η − 1`. We solve the `sub_dim × sub_dim` system by Gauss-
Jordan with partial pivoting (sub_dim is small — 4 to 16 in normal PQ
regimes — so this is essentially free).

Asymmetric distance computation (ADC) at query time is identical to
standard PQ: precompute one `m × k` table of sub-vector inner products,
then approximate `<q, x>` as a sum of `m` table lookups. The PoC scans
the full database, since the focus is the **quantization quality**, not
the index layout; integration with `ruvector-rairs` (IVF) and
`ruvector-core` (HNSW) is straightforward and listed under "Roadmap".

## Consequences

**Positive.**

- We pick up a measurable recall@10 improvement at fixed compression
  ratio (see research doc for numbers) over the standard PQ baseline,
  with **no change to query-time cost** — the per-query path is byte-
  identical to vanilla PQ.
- Closes a gap relative to ScaNN/Vertex AI's online vector matching.
- Provides a foundation for `IVF-APQ` (anisotropic in-list quantization)
  on top of `ruvector-rairs`.

**Negative.**

- Training is 8–12× slower than standard PQ in this PoC. The bottleneck
  is the per-iteration assignment-step inner loop, which now computes a
  scalar projection in addition to the squared L2. Production needs
  SIMD inner-loop kernels (a known optimisation).
- The PoC OPQ rotation uses a one-shot PCA-balanced assignment of
  eigenvectors, not the iterated rotation-codebook joint optimisation
  from Ge et al. CVPR 2013. In the PoC OPQ slightly *regresses* recall
  on the synthetic dataset — likely because the data is already
  variance-balanced after L2 normalisation. The full iterative OPQ
  variant is on the roadmap.

## Alternatives considered

- **Stay with vanilla PQ.** Cheapest path, but leaves recall on the
  table at every compression ratio.
- **Implement RVQ / Residual VQ instead.** Upstream just merged
  `research/nightly/2026-05-16-residual-vq` so this is covered.
- **Adopt LVQ (Locally-Adaptive VQ from LeanVec).** ruvector-leanvec
  already has it; LVQ is a *per-vector scale + offset* approach,
  orthogonal to APQ. They compose.
- **Wait for an upstream Rust APQ crate.** None exist; the algorithm is
  five years old and there is no sign of one.

## File layout

```
crates/ruvector-anisotropic-pq/
  Cargo.toml
  src/
    lib.rs        ~  175 lines  -- trait + tests + recall metric
    pq.rs         ~  165 lines  -- baseline PQ
    apq.rs        ~  250 lines  -- anisotropic PQ + Gauss-Jordan solver
    opq.rs        ~  170 lines  -- PCA-balanced rotation + APQ wrapper
    distance.rs   ~   45 lines  -- ADC topk
    synthetic.rs  ~   55 lines  -- mixture-of-Gaussians dataset
    main.rs       ~  150 lines  -- benchmark runner
  benches/
    apq_bench.rs  ~   45 lines  -- criterion query-latency benchmark
```

All files are well under the 500-line cap. No file lives at the repo
root. No mocks. No `unimplemented!()`.
