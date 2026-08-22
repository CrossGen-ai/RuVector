# ADR-334: Multi-Stage Residual Vector Quantization (RVQ) for ruvector

- **Status**: Proposed
- **Date**: 2026-08-22
- **Deciders**: RuVector Nightly Research
- **Related**: RaBitQ (crate `ruvector-rabitq`, 1-bit rotated quantization), PQ-ADC (nightly 2026-06-20 `ruvector-pq-search`), SOAR-IVF (`ruvector-soar-ivf`, anisotropic quantization). RVQ is orthogonal to all three — it slots underneath a coarse quantizer or filter.
- **Tags**: quantization, compression, residual, ann, retrieval, storage

## Context

ruvector already ships several compact-vector representations:

- **PQ (Product Quantization)**: splits the vector into `M` subspaces and quantizes each independently.
- **RaBitQ**: 1-bit-per-dim asymmetric quantization with theoretical bounds (SIGMOD 2024).
- **SOAR**: anisotropic loss for IVF residuals (Google 2023).

None of these expose a **stage-tunable additive quantizer** where the same code can trade off precision vs. bytes-per-vector by simply adding more stages. Residual Vector Quantization — the standard FAISS `IndexResidualQuantizer` / `ResidualCoarseQuantizer` recipe (Chen et al. 2010, refined by Babenko & Lempitsky 2014, and shipped in ScaNN's residual path) — fills that gap. Modern usage (2024–2026) also includes RVQ codebooks as the coarse layer for two-stage neural codecs (SoundStream, Encodec's residual VQ), so a first-class RVQ is a substrate we'll want to reuse well beyond retrieval.

## Decision

Introduce a new crate `crates/ruvector-rvq/` with a trait-based, backend-swappable multi-stage RVQ implementation:

1. **Trait `Quantizer`** — minimal `encode / decode / dim / code_bytes` seam so future backends (additive QR, LSQ, OPQ+RVQ) drop in without touching call sites.
2. **`Rvq` struct** — greedy stage-wise k-means (k-means++ seeded, deterministic under fixed seed). `L` stages × `K ≤ 256` centroids × `D`-dim. Encoded vectors occupy exactly `L` bytes.
3. **LUT-based scan** — per-query precompute `stages × K` inner-product LUT; per-candidate cost is `L` byte loads + `L` FP adds. Squared-L2 estimator adds a query-independent centroid-norm LUT cached once at index build.
4. **`RvqIndex::search_l2_rerank`** — canonical production recipe: coarse-filter top-`N` with RVQ, re-score with the full-precision vectors, return top-`k`. This is how FAISS `RQ+IndexRefine` and ScaNN `AH + reordering` are actually deployed.

The crate is fully self-contained (no ruvector-core coupling) so downstream crates opt in per feature. rayon parallelism is native-only; wasm32 falls back to sequential.

## Consequences

**Positive**

- **Storage**: `L` bytes/vector. On `D=128` embeddings, 8-stage RVQ is `64×` smaller than `f32` — an entire 10K corpus fits in 80 KB of codes.
- **Query hot-path**: measured 5.4× faster than a naïve `f32` L2 scan on `n=10k, d=128` (88 µs vs 477 µs, single-threaded, Apple Silicon).
- **Recall with reranking**: 96.8% recall@10 at 128× compression on clustered synthetic data with rerank pool of 500. Users pay for exactly the recall floor they need by choosing pool size.
- **Composable**: trait-based, so IVF or HNSW can hold RVQ codes in the leaves without a rewrite.
- **Deterministic**: k-means++ + fixed seed → bit-identical codebooks across runs. Important for reproducible research.

**Negative**

- **Training is greedy**, not jointly optimal. Additive-Quantization (Babenko 2014) reaches ~10–20 % better recall at the same code size at ~2× training cost. Left for a follow-up ADR.
- **Cross-stage inner products dropped** in the L2 estimator (same approximation FAISS makes). For deeply correlated residuals this can bias the ranking by up to ~5 % of the L2 gap; reranking cancels it.
- **Recall on i.i.d.-uniform data is intrinsically low** (measured 3 %). No quantizer can beat this on distributions with no cluster structure — the top-`k` and top-`k+1` distances collapse. This is a distributional property, not a bug; documented in the research doc.
- **k-means training is O(n·k·d·iters)** — for large corpora we'll want mini-batch k-means (roadmap).

## Alternatives Considered

1. **Extend `ruvector-rabitq`** — RaBitQ is fundamentally 1-bit; stage-tunable is not its story. Different design axis.
2. **Extend `ruvector-pq-search`** — Product Quantization is orthogonal-subspace, not residual. Codes are the same size but recall/precision curves differ. Both should live side-by-side.
3. **Additive Quantization directly** — better recall, but joint optimization is more code, more failure modes, and RVQ is the strictly-simpler baseline it should be compared against. Ship RVQ first; add AQ later on the same trait.
4. **OPQ preconditioner** — an orthogonal rotation before quantization improves recall on axis-aligned distributions. Additive; can layer on later without a code shape change.

## Acceptance Test (numeric, measured)

Run: `cargo run --release -p ruvector-rvq --bin rvq-demo`

| Config | Compression | Query µs (pure scan) | Speedup vs f32 flat |
|---|---|---|---|
| 4-stage, K=256 | 128× | 88.1 | 5.4× |
| 8-stage, K=256 | 64× | 130.8 | 3.6× |
| 16-stage, K=256 | 32× | 218.8 | 2.2× |
| baseline flat-f32 | 1× | 476.7 | 1.0× |

Run: `cargo run --release -p ruvector-rvq --example rvq_recall` (clustered `n=8000, d=128, k=10`):

| Stages | Rerank pool | Recall@10 |
|---|---|---|
| 4 | 500 | 0.968 |
| 8 | 500 | 0.972 |
| 16 | 500 | 0.987 |

Tests: `cargo test --release -p ruvector-rvq` — 4/4 pass, including monotone-MSE-in-stages and non-degenerate recall.

## References

- Chen, Guan, Wang (2010): *Approximate Nearest Neighbor Search by Residual Vector Quantization*
- Babenko & Lempitsky (2014): *Additive Quantization for Extreme Vector Compression*, CVPR
- Ge, He, Ke, Sun (2013): *Optimized Product Quantization for Approximate Nearest Neighbor Search*, CVPR
- Guo et al. (2020): *Accelerating Large-Scale Inference with Anisotropic Vector Quantization* (ScaNN), ICML
- Gao & Long (2024): *RaBitQ*, SIGMOD (already implemented in `ruvector-rabitq`)
- FAISS: `IndexResidualQuantizer`, `IndexRefine` (implementation reference)
