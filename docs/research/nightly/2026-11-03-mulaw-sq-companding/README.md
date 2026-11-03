# μ-law Companded Scalar Quantization for Dense Embeddings

Nightly research · 2026-11-03 · slug `mulaw-sq-companding` · ADR-357

## Abstract

We revisit μ-law companding — the half-century-old telephony coding
scheme behind ITU-T G.711 — as a scalar quantizer for dense neural
embeddings. The construction is a one-parameter non-uniform codebook
derived from a log-domain squashing function, costing nothing at the
index level, interoperable with every rotation/pre-processing step
RuVector already supports, and delivering float32-level recall at 4×
memory reduction when its single hyper-parameter μ is fit to the data
rather than taken from the G.711 default.

## SOTA survey

Scalar quantization in vector search has moved steadily toward
"shape the input distribution before uniform quantization":

- **FAISS SQ8 / SQ6**[^faiss]: linear per-dimension, with a global or
  per-dim scale. Baseline.
- **ScaNN[^scann]** and **anisotropic PQ**[^guo2020]: score-aware loss
  that weighs error along the query direction more than orthogonal error,
  implicitly non-uniform in quantizer design.
- **LVQ** (Locally-adaptive Vector Quantization)[^lvq]: per-vector scale,
  recomputed online.
- **Lloyd–Max SQ**[^max]: optimal non-uniform codebook for a given PDF,
  solved by Lloyd iteration; already shipped as `lloyd-max-sq4` in
  RuVector's nightly series.
- **RaBitQ**[^rabitq] and its multi-bit extensions: random rotation then
  signed binarization; the rotation does the whitening that linear SQ
  then rides on.
- **G.711 μ-law/A-law**[^g711]: the classical reference for an
  analytically-defined non-uniform scalar quantizer. μ-law gives
  13-bit→8-bit coding of speech signals with no audible loss, by
  concentrating codes near zero and spreading them exponentially.

Companding has been standard in audio DSP for 60 years and is routinely
reinvented in deep-learning tokenizers (WaveNet's μ-law output head[^wavenet]
is the obvious example). It has not, as of this writing, surfaced in any
production vector-search engine — Milvus, Qdrant, Weaviate, Pinecone and
LanceDB all ship linear or Lloyd–Max SQ.

## Proposed design

Given a float scalar `x ∈ ℝ` and a global scale `s` fit at train time
(same quantile-based fit as linear SQ):

```
normalize:    n = clamp(x / s, -1, 1)
compand:      y = sign(n) · ln(1 + μ|n|) / ln(1 + μ)        ∈ [-1, 1]
quantize:     q = round(y · 127)                              ∈ [-127, 127]
```

Decode inverts the law exactly:

```
dequantize:   y = q / 127
expand:       n = sign(y) · ((1 + μ)^|y| − 1) / μ
denorm:       x̂ = n · s
```

The codec implements the same `ScalarCodec` trait as linear SQ, so every
IVF-rerank, HNSW-rerank and reranker in RuVector can swap backends with
one call.

### Fitting μ

The G.711 default μ=255 is tuned for speech (highly peaked near zero,
heavy tails). Dense embeddings off a modern transformer are closer to
Gaussian — the heavy-tail assumption no longer holds and μ=255 wastes
codes on tails that are not there. We ship a `fit_learned_mu(train, q)`
helper that:

1. Collects the magnitudes `|x_i|/s` of the training set.
2. Finds `m = median(|x_i|/s)`.
3. Solves `ln(1 + μ·m) / ln(1 + μ) = 0.5` for μ by bisection — i.e. the
   median magnitude should land at half the code range.

On 128-D standard-Gaussian embeddings this picks μ ≈ 14, a dramatic
contrast to the telephony default and the reason the fitted codec
matches linear SQ instead of underperforming it.

## Implementation notes

Pure Rust, `#![forbid(unsafe_code)]`, 290 lines in `src/lib.rs`.
Public surface:

- `ScalarCodec` trait — `encode`, `decode`, `code_bytes`, `name`.
- `LinearSq8::fit(train, q)` — baseline.
- `MuLawSq8::fit(train, q, mu)` — fixed μ.
- `MuLawSq8::fit_learned_mu(train, q)` — data-driven μ.
- `reconstruction_mse` and `recall_at_k` as evaluation helpers.

No SIMD intrinsics, no `unsafe`, no third-party crates past `rand` for
test data generation. The μ-law encode/decode currently calls `f32::ln`
and `f32::exp` per element — a 256-entry LUT keyed by code value will
recover the full linear-SQ throughput in the production path and is
the obvious next step.

## Benchmark methodology

- Synthetic Gaussian embeddings: 10 000 base vectors + 500 queries,
  dimension 128, drawn from `Normal(0, 1)`.
- Clip quantile `0.999` for all three codecs; μ=255 for the telephony
  baseline, μ=learned (≈13.9) for the fitted variant.
- Encode throughput: wall-clock over 5 passes of the full base set,
  per-call 128-byte output buffer, single thread.
- Decode throughput: 50 000 single-vector decodes.
- MSE: mean squared reconstruction error over the full base set.
- Recall@k: exact cosine top-k on raw float32 vs. cosine top-k scored
  against decoded (encode-then-decode) base vectors, for k ∈ {10, 100}.
- Hardware: Apple M-series laptop, 2026 build toolchain (`cargo bench`
  release profile).

## Results

Numbers below are real output of `cargo bench -p mulaw-sq-companding`:

| codec | encode (Mvec/s) | decode (Mvec/s) | MSE | recall@10 | recall@100 |
|---|---:|---:|---:|---:|---:|
| `linear_sq8` | 31.26 | 112.13 | 1.835e-4 | 0.9786 | 0.9881 |
| `mulaw_mu255` | 1.78 | 2.49 | 2.883e-4 | 0.9704 | 0.9833 |
| `mulaw_learned` (μ≈13.9) | 2.73 | 4.00 | 1.813e-4 | 0.9788 | 0.9883 |

Memory: `float32=4.88 MiB → sq8=1.22 MiB`, i.e. **4× compression** for
every codec (identical byte width).

Two things stand out:

1. **Fixed G.711 μ=255 hurts recall** on Gaussian-ish embeddings
   (0.9704 vs 0.9786 at k=10, a 0.8 pp drop). The telephony default
   does not transfer blindly.
2. **Learned-μ ties or very slightly beats linear SQ8** at both the
   MSE level (1.81e-4 vs 1.83e-4) and the recall-100 level (0.9883 vs
   0.9881). The win is marginal on Gaussian data because Gaussian data
   is nearly uniform in its mid-magnitude bulk — the regime where μ-law
   should shine is heavy-tailed data (unnormalized logits, some
   multimodal joint spaces). Testing that is the obvious follow-up.

Encode/decode speed is the one real cost. The current numbers include
one `ln` and one `exp` per element. The published next step is a 256-entry
int-indexed LUT — code value is already the natural index — which
makes μ-law encode/decode identical in cost to linear SQ.

## How it works (walk-through)

μ-law companding comes from one observation and one trick.

The observation: when you store a value `x` with 8 bits of uniform
quantization in a range `[-1, 1]`, the quantization step is a flat
`2/256 ≈ 0.0078` everywhere. If 90% of your signal lives inside `[-0.1,
0.1]`, you have thrown 25 of 256 code values at the useful range and
spent the other 231 on tails you rarely hit. The 90% of your signal that
you care about is quantized with 25 steps — roughly 4.6 bits of effective
precision.

The trick: before quantizing, pass `x` through a monotone squashing
function `y = f(x)` chosen so that the *distribution* of `y` is more
uniform than the distribution of `x`. Now 256 equally spaced codes in
`y` map back to *unequally* spaced reconstruction points in `x`,
concentrated where the data lives.

μ-law picks a specific one-parameter family of squashing functions,

> y = sign(x) · ln(1 + μ|x|) / ln(1 + μ)

whose derivative at zero scales like μ (steep: lots of resolution for
small values) and whose derivative at |x|=1 scales like `1/ln(1+μ)`
(shallow: few codes for the tails). The parameter μ is the single knob
that trades near-zero resolution against tail resolution.

G.711 picked μ=255 for voice. For dense embeddings off a transformer
the right value is a lot smaller — around 14 on standard-Gaussian data —
because the distribution has light tails and we do not need to collapse
them.

## Practical failure modes

- **Blindly trusting μ=255.** The telephony constant is wrong for most
  embedding distributions. Always `fit_learned_mu`, or at least sweep.
- **Rotation-flattened data.** If the input went through a Hadamard,
  OPQ or random orthogonal rotation first, the per-dim distribution is
  near-uniform-on-sphere-projection and μ-law has nothing to compress.
  Linear SQ is faster and ties on recall. Detect this at fit time
  (low kurtosis of `|x|/s`) and fall back to linear.
- **Per-dim vs global scale.** Our PoC uses a single global scale `s`.
  Mixed-magnitude dimensions (e.g. concat of two encoders) will under-use
  bins on the quieter dimensions. Per-dim scale is a trivial change — the
  codec stores `dim` floats instead of 1 — and compatible with μ-law.
- **Signed-zero asymmetry.** `sign(0)` is 0 in Rust; our implementation
  deliberately treats `|y|<ε` as zero on decode. If an upstream pipeline
  relies on `+0` and `-0` as distinct bins (nobody should), you will
  lose the distinction.
- **f32 range at μ large.** For μ > 10^6 the `(1+μ)^|y|` term in decode
  can overflow at |y|=1. Capping μ at 10^4 during fit is prudent; our
  bisector already caps at 1023.

## What to improve next

1. **LUT encode/decode.** 256-entry inverse-companding LUT closes the
   speed gap to linear SQ. One-line change inside `encode`/`decode`.
2. **Per-dim scale + global μ.** First-order improvement for mixed-source
   embeddings. ~20 lines.
3. **μ-law + Hadamard pre-rotation.** Measure whether a rotation that
   *re-introduces* heavy tails (e.g. a sparsifying rotation) makes μ-law
   win larger than linear.
4. **μ-law + LVQ.** LVQ adapts scale per-vector; μ-law adapts shape
   within a scale. The combination costs one extra byte per vector
   (per-vector scale) and should outperform either alone on multimodal
   joint embeddings.
5. **Score-aware μ fit.** Instead of matching the median magnitude,
   minimize the recall-weighted reconstruction error — the ScaNN-style
   anisotropic loss but inside a parametric codec.
6. **Real-corpus benchmark.** SIFT1M, GloVe-100D, LAION-CLIP: measure
   fitted μ, recall and MSE on each. The Gaussian benchmark here is
   deliberately a lower bound on where μ-law can help.

## Production crate layout proposal

```
crates/
  ruvector-mulaw-sq/
    Cargo.toml
    src/
      lib.rs           # ScalarCodec impls (shared trait with ruvector-sq)
      lut.rs           # 256-entry LUT, built at codec construction
      fit.rs           # fit, fit_learned_mu, kurtosis-fallback helper
    benches/
      bench_sq.rs      # the file in this PoC, de-synthesized
    tests/
      heavy_tail.rs    # Student-t and Laplace fixtures
      rotated.rs       # Hadamard-pre-rotated fixtures (fallback trigger)
```

Trait-sharing with the existing `ruvector-sq` crate means the index
layer picks a codec by trait object and never knows which family it got.

## References

[^faiss]: Jegou, Douze, Johnson, "Billion-scale similarity search with GPUs", 2017. (FAISS)
[^scann]: Guo et al., "Accelerating Large-Scale Inference with Anisotropic Vector Quantization", ICML 2020. (ScaNN)
[^guo2020]: ibid.
[^lvq]: Aguerrebere et al., "Similarity Search in the Blink of an Eye with Compressed Indices", VLDB 2023.
[^max]: Max, "Quantizing for minimum distortion", IRE Trans. Inform. Theory, 1960; Lloyd, "Least Squares Quantization in PCM", 1957.
[^rabitq]: Gao, Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound", SIGMOD 2024.
[^g711]: ITU-T Recommendation G.711, "Pulse Code Modulation (PCM) of Voice Frequencies", 1972/1988.
[^wavenet]: van den Oord et al., "WaveNet: A Generative Model for Raw Audio", 2016. μ-law output softmax.
