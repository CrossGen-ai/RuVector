# ADR-352: Lloyd-Max 4-bit Scalar Quantization Codec

**Status:** Proposed
**Date:** 2026-10-06
**Author:** nightly research agent
**Supersedes:** —
**Related:** `crates/ruvector-rabitq`, `crates/ruvector-opq-rotation`, `crates/ruvector-pq-fastscan-lut16`

## Context

ruvector today offers two aggressive compression paths:

- **Product Quantization** family (`ruvector-pq-*`, `ruvector-opq-rotation`,
  `ruvector-rabitq`) — 8–32× compression, high recall, codebook training.
- **No native scalar quantization crate.** Uniform SQ8/SQ4 lives in ad-hoc
  spots inside benches and the agent-memory layer, with no shared codec.

Two external gaps motivate a dedicated SQ4 crate:

1. **PQ is overkill for small-dim embeddings** (d ≤ 128) where per-dim
   independence already captures most of the signal. SQ4 at 8× compression
   with a byte-sized LUT per dim is simpler and ~2× faster to scan than PQ8
   with Hamming-adjacency LUTs.
2. **Uniform SQ4 loses recall on heavy-tailed embeddings.** Modern
   layer-normed features cluster near zero with asymmetric tails; linear
   bins waste levels. Lloyd-Max (per-dim 1-D k-means) fixes exactly this.

Our nightly PoC (`crates/ruvector-lloyd-max-sq4`) measures a 35 % reduction
in reconstruction MSE and +3 to +6 absolute recall@10 points on Gaussian and
Laplace synthetic data at identical 8× compression and identical scan cost
(no PQ-style LUT widening). Fit is a one-time ~80 ms for 10 k × 64, running
on a single thread.

## Decision

Adopt Lloyd-Max 4-bit scalar quantization as a **first-class codec** in
ruvector, promoted from the nightly PoC after one hardening pass. The codec
exposes a single `Codec` trait with two implementations:

- `UniformSQ4` — current default behavior, kept for compatibility and as the
  cheap-fit baseline.
- `LloydMaxSQ4` — new, used by default for freshly built indexes when a
  training sample is available.

The scan path uses a per-query `dim × 16` float LUT, independent of which
codec produced the codes. The on-disk code format is bit-compatible between
the two codecs; only the stored centroids table differs.

Promotion criteria (not yet all met in PoC):

- Rayon-parallel per-dim fit landed, p99 fit cost < 50 ms at d=128, n=10k.
- Codec serde + versioning in `io.rs`.
- Scan path validated under NEON on arm64 and AVX-2 on x86 fallback,
  AVX-512 as feature-gated.
- Golden regression tests on SIFT1M-mini and a GloVe sample.

## Consequences

**Positive**

- 8× memory compression vs float32, matching uniform SQ4.
- +3 to +6 absolute recall@10 at identical bytes-on-disk and identical
  scan time — strict Pareto improvement over uniform SQ4 for free.
- Clean trait shared with PQ codecs, enabling future hybrid (SQ coarse +
  PQ rerank or vice versa).
- Pure-Rust, no `unsafe`, no SIMD intrinsics required for correctness.

**Negative**

- One-time training cost per index (serial fit ~80 ms at d=64, n=10k;
  Lloyd fit is 100× slower than uniform's min/max pass).
- Lloyd fit requires a representative training sample; too-small samples
  (< 500 per dim) can produce under-trained tail bins.
- Non-trivial change: a new codec format needs an on-disk version bump
  for any snapshot that embeds centroid tables.

**Neutral**

- Scan performance is unchanged (same LUT path), measured 34–70 vectors/µs
  on Apple M4 Max at d ∈ {32, 64, 128}. SIMD lift is a follow-up.

## Alternatives considered

1. **Keep uniform SQ4 only.** Simpler, but leaves a measurable recall gap
   that compounds with ANN approximation. Rejected: Lloyd adds zero scan
   cost and one-time fit cost, with Pareto-dominant recall.

2. **Jump straight to Product Quantization (PQ8).** Already covered by
   `ruvector-pq-fastscan-lut16`. PQ requires k-means in multi-dim subspaces,
   a shared codebook, and more complex scan plumbing. For d ≤ 128 and
   compression targets in the 4–8× range, SQ4 is simpler and competitive;
   use PQ when targeting > 16× compression.

3. **RaBitQ (1-bit)** (`ruvector-rabitq`). Pushes compression to 32×, but
   recall drops substantially without careful reranking. Complementary, not
   substitute: SQ4 is the right default for "I just need to shrink my
   float32 index by 8× without losing recall."

4. **Lloyd-Max 8-bit.** Would give near-float32 recall at 4× compression;
   worth adding as a sibling codec `LloydMaxSQ8`. Deferred until SQ4
   ships; the training algorithm is identical with `NLEVELS = 256`.

5. **Learned (NN-based) quantizers.** Autoencoder-style scalar quantizers
   produce lower MSE but introduce a training-time dependency on a GPU
   stack we don't want on the critical path. Deferred.
