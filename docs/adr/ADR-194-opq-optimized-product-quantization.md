---
adr: 194
title: "OPQ — Optimized Product Quantization (rotated PQ via eigenvalue allocation + Procrustes)"
status: accepted
date: 2026-05-24
authors: [claude-flow, nightly-research]
related: [ADR-193]
tags: [pq, opq, quantization, ann, vector-search, nightly-research]
---

# ADR-194 — OPQ: Optimized Product Quantization

> **Provenance.** This ADR records the design and acceptance of
> `crates/ruvector-opq`, the nightly-2026-05-24 PoC. The technique is
> Optimized Product Quantization (Ge, He, Ke, Sun, CVPR 2013, TPAMI 2014).
> The implementation is original Rust; numbers come from real
> `cargo run --release` output, captured in
> `docs/research/nightly/2026-05-24-opq-optimized-product-quantization/README.md`.

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-05-24-opq-optimized-product-quantization` as
`crates/ruvector-opq`. `cargo build --release -p ruvector-opq` succeeds.
`cargo test -p ruvector-opq` runs **6 tests, all green**, including two
numeric-acceptance gates that compare OPQ-NP and OPQ-P against the PQ
baseline on anisotropic axis-aligned data.

## Context

ruvector ships several quantizers — `ruvector-rabitq` (1-bit binary codes,
SIGMOD 2024), `ruvector-lvq` (locally-adaptive vector quantization),
`ruvector-avq` (anisotropic VQ) — but until now **the classic rotated PQ
family was missing**. OPQ is the standard "drop-in upgrade" applied in front
of vanilla PQ in every major vector DB (FAISS `OPQMatrix`, Milvus
`opq_M` pretransform). It is also the rotation slot that 2024–2025
hybrid quantizers (OPQ→RaBitQ, JL→OPQ→RaBitQ) compose on top of.

Beyond the standalone gain, OPQ unblocks:

* **OPQ + RaBitQ composition** — a known recall lift on binary indexes.
* **IVF-OPQ** — the dominant index family in FAISS / Milvus for >100 M
  vectors, currently unrepresented in `ruvector-rairs`.
* **ScaNN-style anisotropic OPQ** — re-weighting the OPQ-P loss to optimise
  inner-product instead of L2.

## Decision

Ship `crates/ruvector-opq` containing three implementations of a single
`Quantizer` trait:

1. **`Pq`** — classical PQ baseline (Jégou 2011). `m` subspaces, `K=256`
   per-subspace centroids trained by k-means++.
2. **`OpqNp`** — non-parametric OPQ. Rotation `R` learned in closed form by
   eigenvalue allocation: PCA on training data, balanced-greedy partition of
   principal axes into `m` buckets to equalise `Σ log λ` per bucket.
3. **`OpqP`** — parametric OPQ. Warm-starts from OPQ-NP, then alternates
   `train-PQ-on-rotated-data` ↔ `R ← V Uᵀ` from SVD of `X Ŷᵀ` (orthogonal
   Procrustes). 4 iterations by default.

All three share the same trait, expose `build_lut` / `adc_lut` for the
production search path, and store rotation `R` (when present) as a flat
`d × d` `Vec<f32>`.

### Build / test gate

* `cargo build --release -p ruvector-opq` → clean.
* `cargo test --release -p ruvector-opq` → **6/6 tests pass** in 0.23 s.
* `cargo run --release -p ruvector-opq --bin opq-demo` → multi-regime
  benchmark printing real numbers; no mocks.

### Numbers (Apple M4 Max, single-threaded, M=8, K=256, d∈{64,128})

| Regime                       | PQ MSE     | OPQ-P MSE  | PQ recall@10 | OPQ-P recall@10 | Compression |
|-------------------------------|-----------:|-----------:|-------------:|----------------:|------------:|
| A: `d=64, m=4, ds=16`         | 0.037764   | 0.037344 (**-1.1%**) | 0.067 | 0.079 (**+18%**) | 64×         |
| B: `d=64, m=8, ds=8`          | 0.022191   | 0.022264   | 0.199 | 0.197           | 32×         |
| C: `d=128, m=8, ds=16`        | 0.014865   | 0.013849 (**-6.8%**) | 0.067 | 0.086 (**+28%**) | 64×         |

OPQ-NP is essentially a tie with PQ on these synthetic distributions; the
parametric variant is what carries the gain. Search-time cost is **identical
to PQ** once `build_lut` is paid once per query — verified by ≈ 25 ms scan
times for both, in every regime.

## Consequences

### Positive

* ruvector gains the canonical OPQ baseline every competing vector DB
  ships, with **64× compression** and Procrustes-refined recall.
* Decoupled from RaBitQ / RAIRS — clean swap target for future composition.
* Trait-based design means the same `recall_at_k` harness drives PQ /
  OPQ-NP / OPQ-P / future quantizers without touching the search loop.

### Negative

* OPQ-P training is `~5×` slower than PQ-train (regimes A, C). Static
  indexes only; streaming writes should use OPQ-NP (≈ 1.1× PQ-train).
* PCA cost is `O(n d²)`; SVD of `d × d` is `O(d³)`. Both fine to `d=1024`,
  but very-high-dim use will need randomised SVD.
* Stored rotation matrix is `4 d²` bytes per index. At `d=128` that is
  **66 KB** — negligible. At `d=2048` it is **16 MB** — non-trivial.

### Neutral

* `K=256` is hard-coded — same as classical PQ-8 and the byte-LUT scan
  kernels we plan to write in a follow-up. If a future use case needs
  `K=16` (PQ4-fastscan), it will land in a sibling crate
  `ruvector-opq-fastscan`, not this one.

## Alternatives considered

1. **Add `OPQMatrix` as a pretransform inside `ruvector-rabitq` directly.**
   Rejected: couples two algorithms whose lifecycles differ, and prevents
   the `OPQ + ScaNN-anisotropic-loss` variant from being a sibling crate.

2. **Use a learned (non-orthogonal) linear pretransform.** That is what
   LeanVec/LVQ already do (`ruvector-lvq`). OPQ is the *orthogonal* family;
   the two are complements, not substitutes.

3. **Skip OPQ-NP, ship only OPQ-P.** Rejected: NP is the right default for
   streaming-write workloads, since it costs one PCA and no PQ-train
   iterations. Shipping both keeps the cost/quality tradeoff explicit.

4. **Train OPQ jointly with an IVF coarse quantizer (IVF-OPQ).** That is
   the *next* ADR — depends on choosing whether the coarse quantizer is
   `ruvector-rairs` or a fresh k-means coarse layer. Tracked as future
   work; out of scope here.

## Acceptance summary

| Criterion                                                                              | Met |
|-----------------------------------------------------------------------------------------|----:|
| Runnable `Cargo.toml` + `cargo run` example                                              | ✅   |
| Swappable trait-based design                                                            | ✅   |
| Real memory/perf math (estimated + measured)                                            | ✅   |
| At least 3 measured variants                                                            | ✅ (PQ, OPQ-NP, OPQ-P) |
| Numeric acceptance test passes                                                          | ✅ (6/6 tests) |
| `cargo build --release -p ruvector-opq` succeeds                                        | ✅   |
| `cargo test -p ruvector-opq` passes with real tests (no mocks)                          | ✅   |
| Benchmark binary producing real numbers captured in research doc                        | ✅   |
| Files under 500 lines                                                                   | ✅   |
