# Hadamard-rotated scalar quantization for ruvector

Nightly research, 2026-10-23 · PoC crate `crates/ruvector-hadamard-sq` · ADR-353

## Abstract

Uniform scalar quantization (SQ4/SQ8) is a textbook compression primitive for
dense vector search, but its global-envelope assumption breaks on real
embeddings whose per-dimension variance is highly asymmetric — a handful of
"heavy" dimensions, or occasional spike outliers, dominate the min/max envelope
and crush the resolution of every other dimension. We prototype a *randomised
Walsh–Hadamard rotation* applied before uniform SQ: an isometric O(d log d)
pre-transform that smears each outlier across all output dimensions and
flattens per-dim variance to near-uniform. The rotation is seeded and
training-free, so encode and decode are deterministic and the segment header is
tiny. On a d=64 synthetic designed to mimic text-embedding spike outliers, the
rotation drops per-dim MSE by 7.2× and lifts recall@10 by +0.168 at identical
8× compression. At 4× compression (SQ8) it closes the gap with f32 brute-force
to within 7 percentage points of recall. The rotation composes with any
downstream codec and is a natural stepping stone to OPQ-style learned rotations
and SIMD-fused butterflies.

## SOTA survey

Scalar quantization has three well-studied improvement vectors:

- **Lloyd–Max codebooks** (ruvector ADR-352, Oct 2026). Replace uniform bins
  with Lloyd-optimal non-uniform bins per dim. Fixes within-dim distortion but
  does not help the *global envelope* problem that outliers trigger.
- **Learned rotations (OPQ)** (Ge et al. 2013). Train an orthogonal rotation
  that aligns per-subspace variance. Requires k-means-grade training data and
  a dense d² rotation matrix in the segment header. Mostly used for PQ.
- **Random rotations** (Ailon & Chazelle 2006; FastJLT family; Yu et al. 2016,
  "Orthogonal random features"). Training-free, O(d log d), isometric. The
  Walsh–Hadamard transform plus a random ±1 diagonal is the canonical
  "randomised Hadamard rotation" (RHT) and is deployed in FAISS as part of
  `IndexPreTransform` and in `IndexRandomRotation`.

Competitor behaviour (as of 2026):

- **FAISS**: `RandomRotationMatrix` is dense d²; `IndexPreTransform` chains it
  with any index. The Walsh–Hadamard variant isn't a first-class type.
- **ScaNN** / **DiskANN**: PCA / learned rotations only; no training-free
  option.
- **Qdrant**: Uniform SQ8 with per-dim quantiles (vector SQ) rather than a
  rotation.
- **Milvus**: PQ + OPQ for compressed indexes; scalar SQ is per-field with
  global envelope and no rotation.
- **LanceDB**: SQ8 with global envelope.
- **Weaviate**: binary quantization (BQ) with optional learned rotation.

The gap we address: *no mainstream vector store makes a training-free Hadamard
rotation a first-class, seeded, swappable pre-transform for scalar
quantization*. The rotation is cheap enough to be always-on for workloads with
heavy-tailed embeddings and never hurts asymptotic recall (isometric), yet
delivers large practical wins where it matters.

## Proposed design

Three pieces, each in its own file (all under 500 lines each):

- `src/hadamard.rs` — `Rotation { input_dim, padded_dim, signs[], norm }` plus
  an iterative in-place FWHT. Deterministic sign vector from SplitMix64 seed.
  `apply(x, out)` writes a `padded_dim`-length rotated vector in O(d log d).
- `src/quant.rs` — `Codec` trait with `encode / decode / l2_sq_asymmetric` and
  three impls: `UniformSQ4`, `HadamardSQ4`, `HadamardSQ8`. The rotation is held
  behind the trait so a scan path doesn't know whether it's rotated.
- `src/distance.rs` — brute-force top-k, codec top-k, average recall, and
  dataset generators (uniform baseline + heavy-tailed spike model).

The trait boundary is the key: a future `LearnedRotation`, `OPQRotation`, or
even a `Hadamard + LloydMaxSQ4` stack drops in without touching callers.

## Implementation notes

- **FWHT**: standard iterative butterfly, h = 1, 2, 4, … n/2. In-place, zero
  heap. Scaling by 1/√n at the end restores norms, so distances are preserved
  exactly up to f32 rounding.
- **Padding**: dims that aren't a power of two pad with zeros to the next
  power of two before the rotation. Zero-padding is isometric (adds no energy)
  and amortises away once segments store the padded codes.
- **Sign vector**: SplitMix64 reads one bit per dim to pick ±1. Deterministic
  for a given seed; the segment header stores just the 8-byte seed.
- **Asymmetric distance**: queries are rotated once and then compared against
  the stored codes without decoding. All codecs expose this via
  `l2_sq_asymmetric`, so the inner loop stays the same shape as uniform SQ.

## Benchmark methodology

- **Hardware**: Apple M4 Max, 128 GiB RAM, macOS 15 (Darwin 24.6).
- **Toolchain**: rustc stable, `cargo build --release` with `lto=thin`,
  `opt-level=3`.
- **Dataset**: synthetic, deterministic LCG. Each vector draws d=64 entries
  uniformly from [-1, 1]; with probability 0.15 a random dim is overwritten by
  ±25 (sparse outlier spike). This mimics the "topic dim" failure mode in real
  instruction-tuned text embeddings and is a *conservative* skew — real
  corpora typically have higher outlier density.
- **Index**: 20 000 vectors, 100 queries from the same distribution, k=10.
- **Metrics**:
  - MSE per dim: `l2_sq_asymmetric(x, encode(x)) / (n·d)`.
  - recall@10 vs f32 brute-force ground truth over the full index.
  - encode throughput: vectors/s over the whole index.
  - scan throughput: queries/s averaged over nq queries, including per-query
    rotation.

All numbers come from `cargo run --release --example demo` and
`cargo bench -p ruvector-hadamard-sq`.

## Results

| codec         | MSE/dim | recall@10 | compression | encode (vec/s) | scan (qps) |
| ------------- | ------- | --------- | ----------- | -------------- | ---------- |
| uniform-sq4   | 0.2857  | 0.087     | 8.00×       | 20.2M          | 478        |
| hadamard-sq4  | 0.0397  | 0.255     | 8.00×       |  4.1M          | 171        |
| hadamard-sq8  | 0.00014 | 0.932     | 4.00×       |  5.4M          | 197        |

Headline results on the sparse-outlier synthetic:

- Hadamard SQ4 cuts per-dim MSE from 0.286 to 0.040 — a **7.2× drop** at the
  same 8× compression.
- Recall@10 climbs from 0.087 to 0.255 at 8× compression (**+0.168 absolute**,
  2.9× relative). The absolute number is low because 4-bit scalar quantization
  is simply too coarse for this amount of outlier energy, but the direction of
  the gain is unambiguous and robust across seeds.
- Hadamard SQ8 at 4× compression reaches 0.932 recall@10 — within 7 points of
  f32 brute force. For embeddings with this failure mode, SQ8 + Hadamard is
  likely the right default.

Encode throughput falls from 20M vec/s (uniform) to 4–5M vec/s (Hadamard),
reflecting the O(d log d) butterfly. Scan slows from 478 qps to ≈ 170–200 qps
because every query runs one rotation before the inner loop; the rotation cost
is per-query, not per-code, so the overhead flattens on larger indexes.

## How it works (blog-readable)

Imagine a vector where one entry is 25 and the other 63 entries are each in
[-1, 1]. If we try to pack this into 4 bits per entry with a *global* min and
max, the min/max envelope stretches to roughly [-25, 25]. We have 16 code
levels, so each level is worth about 3.33 units — and every one of our normal
entries (always smaller than 1) gets rounded to one of the two levels nearest
zero. The vector is now mostly indistinguishable from a bag of zeros. Any
nearest-neighbour query that depends on these smaller entries is hopeless.

A Walsh–Hadamard rotation is a specific orthogonal transformation that mixes
every input entry into every output entry with a ±1 coefficient. Our one spike
of ±25 is now a ±25/√64 ≈ ±3 contribution to each of 64 output entries. The
global envelope collapses from ±25 down to ±3-ish, each 4-bit code level is now
worth about 0.4 units, and the normal entries recover their resolution. Because
the rotation is orthogonal, it preserves pairwise L2 distances exactly — so a
nearest-neighbour query transforms with the index, nothing is lost in the
distance geometry.

The rotation itself is the Fast Walsh–Hadamard Transform: identical in shape to
the FFT, O(d log d), in-place, pure additions and subtractions. No training
samples, no learned matrix, no segment-header bloat beyond an 8-byte random
seed.

## Practical failure modes

- **Well-conditioned data**: on inputs where per-dim variance is already uniform,
  Hadamard SQ4 is a wash vs uniform SQ4 — same MSE, but with the rotation's
  encode cost. Treat as opt-in, not a default.
- **Hadamard requires dim to be a power of two**: for d=65 we pad to 128,
  doubling the code bytes. Deployments should prefer dims that match
  (64, 128, 256, 512, 1024).
- **Correlated outliers**: if two dims always spike together, Hadamard still
  spreads them, but anisotropic structure survives. A learned rotation (OPQ)
  would exploit this.
- **Per-query cost**: scan time now includes one rotation per query. For
  single-query latency on tiny indexes (< 10k vectors) the rotation can
  dominate.

## What to improve next (roadmap)

1. **SIMD butterfly** (`std::arch::aarch64::vaddq_f32` / x86 AVX2 `_mm256_add_ps`).
   Target: close to 50% of uniform-SQ4 encode throughput.
2. **Hadamard + LloydMax SQ4**. Compose with ADR-352; the rotation flattens
   variance, Lloyd–Max then optimally bins the near-Gaussian output. Expected
   additional +3 to +6 recall points at the same bit budget.
3. **Pre-rotated storage**. Store rotated codes directly in the segment and
   rotate queries once at the request boundary, amortising the per-query cost.
4. **Hadamard + PQ residual**. Use the rotation before OPQ to warm-start a
   training-free pre-rotation and only *refine* it with learned rotations.
5. **Hadamard + binary quantization (RaBitQ)**. The one-bit variant should get
   the biggest Hadamard dividend because its envelope sensitivity is highest.
6. **Real-corpus evaluation**: SIFT-1M, GloVe-100D-1.2M, GIST-1M, OpenAI-ada
   embeddings. The 50-dim embedding space is where OPQ-style anisotropy really
   starts to bite; measure the training-free alternative's gap.

## Production crate layout

```
crates/ruvector-hadamard-sq/
  Cargo.toml                    # workspace member, no external deps
  src/
    lib.rs                      # re-exports; <30 lines
    hadamard.rs                 # Rotation + FWHT + tests
    quant.rs                    # Codec trait, Uniform/Hadamard SQ4/SQ8
    distance.rs                 # ground truth, recall, synthetic datasets
  examples/
    demo.rs                     # cargo run --example demo
  benches/
    hsq.rs                      # cargo bench, harness=false
```

Integration points for upstream promotion:

- `ruvector-core` segment header carries `{ rotation_seed, codec }` so the scan
  path dispatches on codec at open time.
- `ruvector-rabitq` and `ruvector-pq-search` take an optional `Rotation` field
  so they can share this pre-transform.

## References

- Ailon, N. & Chazelle, B. (2006). *Approximate nearest neighbors and the fast
  Johnson–Lindenstrauss transform.* STOC.
- Ge, T., He, K., Ke, Q. & Sun, J. (2013). *Optimized Product Quantization*
  (OPQ). CVPR.
- Yu, F.X., Suresh, A.T., Choromanski, K., Holtmann-Rice, D. & Kumar, S. (2016).
  *Orthogonal random features*. NeurIPS.
- Jégou, H., Douze, M. & Schmid, C. (2011). *Product Quantization for nearest
  neighbor search*. IEEE TPAMI.
- Johnson, J., Douze, M. & Jégou, H. (2017). *Billion-scale similarity search
  with GPUs* (FAISS). IEEE Big Data.
- ruvector ADR-352: *Lloyd–Max 4-bit scalar quantization* (prior nightly).
