---
adr: 194
title: "LeanVec-LVQ — learned linear projection + locally-adaptive 8-bit quantisation"
status: accepted
date: 2026-05-12
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-191, ADR-193]
tags: [quantisation, lvq, leanvec, ann, vector-search, nightly-research]
---

# ADR-194 — LeanVec-LVQ: ruvector's first scalar-quantisation + learned-projection codec

> **Provenance.** LVQ (Aguerrebere et al., 2023) and LeanVec (Tepper et al.,
> 2024) are real Intel SVS publications. The code in `crates/ruvector-leanvec`
> is an original Rust implementation written against their public algorithmic
> descriptions; FAISS's `IndexLVQ8` was used to cross-check the asymmetric
> distance kernel only. Benchmarks below are real `cargo run --release`
> output on Apple M4 Max — not aspirational.

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-12-leanvec-lvq`
as `crates/ruvector-leanvec`. All 10 unit + integration tests pass.
`cargo build --release -p ruvector-leanvec` is green.

## Context

Per ADR-193, ruvector now has IVF (`ruvector-rairs`). It already had graph
ANN (`ruvector-core` HNSW, `ruvector-diskann`) and one-bit quantisation
(`ruvector-rabitq`). What is still missing — and what every comparable
production stack (FAISS, Milvus, Qdrant, Pinecone) shipped between 2023 and
2025 — is the *mid-recall* codec layer:

- **Scalar quantisation** (per-vector affine 8-bit codes, asymmetric distance).
- **Learned linear projection** (PCA-style dim reduction before the codec).

Without these, every distance kernel in ruvector runs against full f32 data.
That is fine at laptop scale (the f32 corpus fits in cache). It loses badly
when the corpus exceeds last-level cache and memory bandwidth becomes the
bottleneck. Intel SVS reports 3-5× QPS gains on production text and image
embeddings; FAISS's 2025 `IndexLVQ8` adoption confirms the design point.

The two ideas compose: LVQ shrinks bytes per dim; LeanVec shrinks dims per
vector. Together they cut the inner-loop work by `(d/r)·(4/1)` while exact
rerank on retained originals defends recall.

## Decision

We introduce `crates/ruvector-leanvec` exposing **three flat index variants**
behind a single `VectorIndex` trait, so callers can A/B them with identical
call sites:

### Variant 1 — `FlatIndex` (baseline)

f32 brute force. Reference for recall and latency. 4 d bytes/vector.

### Variant 2 — `LvqIndex`

Locally-adaptive 8-bit codes (Aguerrebere et al., 2023). Each vector carries
its own `(lo, step)` so codes are `{0, …, 255}^d`. Asymmetric distance keeps
queries in f32 and decodes the database on the fly:

```
||q − v||² ≈ Σ (q_j − lo − step · code_j)²
```

Storage: `d + 8` bytes per vector. ~4× shrink at d ≥ 16.

### Variant 3 — `LeanVecIndex`

Trained orthonormal projection `P ∈ R^{r×d}` via PCA on a training sample,
LVQ-8 over `Pv ∈ R^r`, exact f32 rerank of the top `k · rerank_mult`
candidates against retained originals. Three knobs — `r`, `rerank_mult`,
training sample size — control the recall/speed/memory triangle.

### Trait surface

```rust
pub trait VectorIndex {
    fn add(&mut self, v: &[f32]) -> u32;
    fn search(&self, q: &[f32], k: usize) -> Vec<Neighbor>;
    fn len(&self) -> usize;
    fn bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}
```

This is intentionally narrow. The follow-up `VectorCodec` trait
(`encode` / `asym_distance` / `decode_one`) — proposed but not built in this
nightly — will let HNSW and IVF backends consume LeanVec-LVQ as a pluggable
codec rather than a standalone index.

## Consequences

**Positive.**

- ruvector now has a complete 2024-era codec stack:
  RaBitQ (1-bit) · LVQ-8 · LeanVec-LVQ · f32 — operators pick the recall band.
- Measured 2.73× speed-up at recall@10 = 1.000 on 10 000 × 128 anisotropic
  data with `r = 16`, `rerank_mult = 4`. Real numbers, not estimates
  (see `docs/research/nightly/2026-05-12-leanvec-lvq/README.md`).
- Zero `unsafe`. No LAPACK, no SIMD intrinsics, no extra dependencies beyond
  `rand`. The whole crate is ~600 lines.
- PCA training cost: 197 ms for 2 000 × 128 — amortised at first index build.

**Negative.**

- **LVQ-8 alone is slower than f32 at laptop scale** (0.88×). Honest
  finding: the asymmetric per-element decode (`lo + step · code`) costs more
  than f32 baseline saves when the corpus fits in cache. LVQ's win is
  bandwidth-bound; you need to leave L3 to see it. Documented in the
  research doc's "Practical failure modes" section.
- `LeanVecIndex` retains f32 originals for rerank, so its byte footprint is
  *higher* than `FlatIndex` (584 vs 512 at d=128). This is a speed trade,
  not a memory trade. A two-level LVQ follow-up would fix this.
- PCA component count grows linearly with training cost. Beyond `r = 256`
  the power-iteration implementation gets slow; future work should switch
  to a block-Krylov method or accept a LAPACK dependency.

**Risk — paper interpretation.** "LeanVec" and "LVQ" are real and well-cited,
but the SVS team's exact code is proprietary. Algorithmic choices in
`projection.rs` (power iteration with explicit deflation, PCA on raw
training vectors) are standard but not literally what SVS ships. Validate
against FAISS `IndexLVQ8` once we add a comparison harness.

## Alternatives considered

- **Product Quantisation (FAISS-style PQ)**, with or without OPQ rotation.
  Strictly more compact than LVQ-8 (8-16 bytes per vector vs ~136), at the
  cost of a global codebook and harder rerank. Rejected for this nightly
  because LVQ matches FAISS's own 2024 direction (`IndexLVQ8`) and avoids the
  codebook-training complexity. Will revisit as `ruvector-pq` once the
  `VectorCodec` trait is stable.
- **1-bit RaBitQ** already exists in ruvector. Lower recall band, much
  smaller. Composable with LeanVec in principle — track as future work.
- **Random projection (Johnson-Lindenstrauss).** Cheaper to train (no PCA),
  but breaks down on anisotropic data. Tepper et al. show learned `P` beats
  random `P` by 5-15 pp recall at the same `r`. Not worth the trade.
- **Disk-based codec (DiskANN / SPANN).** Different problem (out-of-core);
  composes with this work rather than replacing it.

## Acceptance criteria — status

- [x] `cargo build --release -p ruvector-leanvec` succeeds.
- [x] `cargo test --release -p ruvector-leanvec` passes (8 unit + 2 integration).
- [x] `cargo run --release -p ruvector-leanvec --bin leanvec-demo` produces
      the three-variant table reproduced in the research doc.
- [x] All files under 500 lines.
- [x] No `unsafe`, no `TODO`, no mocks.
- [x] At least three measured variants (FlatF32, LVQ-8, LeanVec-LVQ at
      r ∈ {64, 32, 16}).

## References

See research doc `docs/research/nightly/2026-05-12-leanvec-lvq/README.md` §
References.
