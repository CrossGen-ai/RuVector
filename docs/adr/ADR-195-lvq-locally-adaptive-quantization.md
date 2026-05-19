---
adr: 195
title: "LVQ — Locally-Adaptive Vector Quantization for streaming ANN"
status: accepted
date: 2026-05-19
authors: [ruvnet, claude-flow]
related: [ADR-193]
tags: [quantization, ann, vector-search, lvq, streaming, nightly-research]
---

# ADR-195 — LVQ: Locally-Adaptive Vector Quantization

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-05-19-lvq-quantization` as `crates/ruvector-lvq`.
`cargo build --release -p ruvector-lvq` is green; `cargo test -p ruvector-lvq`
passes 4/4; `cargo run -p ruvector-lvq --release --bin lvq-demo` produces
real numbers (see ADR §Consequences).

## Context

ruvector compresses vectors today via:

| Crate                  | Technique             | Profile                                |
|------------------------|-----------------------|----------------------------------------|
| `ruvector-rabitq`      | 1-bit rotation+code   | Extreme compression, theoretical bounds|
| `ruvector-leanvec`     | Dim. reduction        | Pre-quantization step                  |
| `ruvector-rairs`       | SQ8 (per-dim affine)  | Built into IVF candidate scan          |

There is **no per-vector adaptive quantizer**. SQ8 fits one `(lo, delta)` pair
*per dimension*, globally over the dataset — vectors that live in a narrow
slice of that range get half the resolution. For streaming workloads (the
ruflo memory subsystem, retrieval-augmented agents, RuVocal) where vectors
arrive continuously and codebook re-training is unacceptable, this is the
right gap to fill.

Intel's Locally-Adaptive Vector Quantization (Aguerrebere et al., NeurIPS
2023; arXiv:[2402.02044](https://arxiv.org/abs/2402.02044)) addresses this:
center against a global mean, then fit `(scale, bias)` *per vector* before
uniform B-bit quantization. The per-vector overhead is 12 bytes (3 × f32)
regardless of `d`, and the inner distance loop becomes a single dot product
against the raw code — strictly faster than SQ8.

## Decision

Introduce `crates/ruvector-lvq` exposing a `Quantizer` trait with three
backends:

- **`Sq8`** — global per-dimension affine. Baseline only.
- **`LvqOne`** — single-level LVQ at 4 or 8 bits. Per-vector
  `(Δ, lo, ‖x̂‖²)`. Asymmetric L2 via
  `‖q‖² − 2(Δ⟨q,c⟩ + lo·Σq + ⟨q,mean⟩) + ‖x̂‖²`.
- **`LvqTwo`** — primary LVQ + residual LVQ at independent bit widths.
  Recovers near-fp32 recall at 2.4× compression.

The crate is deliberately small (7 source files, all < 200 LOC) and has no
runtime SIMD intrinsics — LLVM autovectorization is sufficient at this
stage. A Turbo-LVQ-style packed layout is left for a follow-up ADR.

### Code shape

```rust
pub trait Quantizer: Send + Sync {
    fn fit(&mut self, training: &[Vec<f32>]) -> Result<(), LvqError>;
    fn encode(&self, v: &[f32]) -> Result<Encoded, LvqError>;
    fn decode(&self, e: &Encoded, out: &mut [f32]);
    fn distance(&self, q: &[f32], q_sq_norm: f32, e: &Encoded, metric: Metric) -> f32;
}
```

### Bit-width choices

- **8-bit primary** is the default — matches SQ8 in footprint, beats it on
  recall and throughput.
- **4-bit primary** is the disk/RAM-savings tier — 6.74× compression,
  recall@10 = 0.832 on isotropic Gaussian d=128.
- **8 + 4 residual** is the high-fidelity tier — recall@10 = 0.9995.

## Consequences

### Measured (Apple Silicon, release, N=20 000, d=128, k=10):

| Quantizer   | Bytes/vec | Index MB | Recall@10 | Scans/s     |
|-------------|-----------|----------|-----------|-------------|
| f32         | 512       | 9.77     | 1.0000    | 30 650 020  |
| SQ8         | 140       | 2.67     | 0.9745    |  8 795 532  |
| **LVQ1-8**  | **140**   | **2.67** | **0.9885**| **13 457 928** |
| LVQ1-4      |  76       | 1.45     | 0.8320    | 11 305 267  |
| LVQ2-8x4    | 212       | 4.04     | 0.9995    |  3 912 945  |

**LVQ1-8 strictly dominates SQ8** at the same byte budget: +1.4 pp recall,
+53% scan throughput.

### Positive

- Fills a real gap in the ruvector quantization story.
- Trait-based so other indexes (`ruvector-graph`, `ruvector-diskann`,
  `ruvector-rairs`) can adopt it incrementally.
- Streaming-friendly: per-vector parameters mean inserts never invalidate
  existing codes.
- All numbers are reproducible from a single `cargo run`.

### Negative

- f32 dense brute-force is still faster than any quantizer at this scale on
  Apple Silicon (LLVM autovectorizes `Σ(a-b)²` extremely well). LVQ's wins
  only materialise at larger N, in graph indexes where each query touches a
  small set of nodes, or when memory pressure forces compression.
- No SIMD intrinsics yet; Turbo-LVQ layout is a follow-up.
- Mean fitting is one-shot; streaming drift requires an EMA re-fit
  (trivial follow-up).

### Alternatives considered

1. **Extend SQ8 with per-vector range.** Equivalent in spirit but with
   ad-hoc API; LVQ formalises it.
2. **Adopt PQ instead.** Better compression at fixed recall, but codebook
   retraining is incompatible with streaming inserts — the explicit
   motivation for LVQ.
3. **RaBitQ-only.** Already present; complements but does not replace LVQ.
   RaBitQ shines at 1-bit; LVQ wins the 4–8 bit regime.

## References

- arXiv:2402.02044, "Locally-Adaptive Quantization for Streaming Vector
  Search", Aguerrebere et al.
- <https://github.com/intel/ScalableVectorSearch>
- US Patent Application 20240020308.
- ADR-193 (RAIRS IVF) — sibling work, will benefit from LVQ candidate scan.
