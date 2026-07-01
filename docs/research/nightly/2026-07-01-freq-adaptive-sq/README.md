# FASQ — Frequency-Adaptive Scalar Quantization for ruvector

**Nightly research • 2026-07-01 • Crate:** [`crates/ruvector-fasq`](../../../../crates/ruvector-fasq/)

## Abstract

Modern embedding models (OpenAI `text-embedding-3-*`, Cohere `embed-v4`,
Voyage `voyage-3`, and most Matryoshka-trained models) produce vectors whose
per-dimension variance is highly **anisotropic** — a small number of
"principal" dimensions carry most of the signal, while a long tail contributes
almost nothing. Uniform scalar quantization (SQ8, SQ4) wastes bits on the low-
variance tail and starves the high-variance head, degrading recall for the
same storage budget.

FASQ (Frequency-Adaptive Scalar Quantization) solves the bit-budget problem
directly: given a target average of *B* bits per dimension, allocate integer
bit counts `bᵢ ∈ [b_lo, b_hi]` per dimension to minimize expected quantization
distortion `∑ᵢ σᵢ² · 4^(-bᵢ)` via a discrete water-filling procedure. The
allocator runs once at training time, is trait-swappable with the existing
`UniformSq8`/`UniformSq4` baselines, and — critically — degrades gracefully
to uniform SQ4 on isotropic inputs.

**Result on 4 000×dim=64 anisotropic-Gaussian vectors (real numbers,
`cargo run --release --bin fasq-demo`)**:

| Quantizer    | Bytes/vec | Recon MSE  | Recall@10 |
|--------------|:---------:|:----------:|:---------:|
| SQ8          | 64        | 6.4 × 10⁻⁵ | 0.9765    |
| SQ4          | 32        | 1.84 × 10⁻² | 0.7080    |
| **FASQ (4)** | **32**    | **6.4 × 10⁻⁴** | **0.9745** |

FASQ delivers **28.72× lower reconstruction MSE** and closes 92% of the SQ4→SQ8
recall gap at half the storage of SQ8. On isotropic input where there is no
anisotropy to exploit, FASQ falls back to SQ4 performance (recall 0.7655) with
no penalty — the allocator naturally distributes bits uniformly.

## SOTA survey

- **RaBitQ (SIGMOD '24)** — rotates then 1-bit-quantizes; strong for high-D
  isotropic distances; already in `crates/ruvector-rabitq`.
- **LVQ / AVQ / OPQ** — learned vector / adaptive vector / optimized product
  quantizers; more accurate than plain PQ but require heavy training.
- **LeanVec / Matryoshka (2023-24)** — coarse-to-fine dimensional pruning;
  complementary to FASQ (FASQ can be applied *after* Matryoshka truncation).
- **Milvus SQ8 / Qdrant SQ / Weaviate BQ** — production databases still ship
  a single-bit-width scalar quantizer per vector; none allocate bits per dim.
- **Rate-distortion for scalar quantizers** — Cover & Thomas Ch. 13,
  "Reverse water-filling" — the theoretical basis for FASQ. Standard result
  from information theory (1970s), applied here to embedding compression.

Novelty of FASQ vs. the above: it is (a) purely training-side (no rotation,
no rebalancing at query time), (b) implements the *integer* water-filling
under bit bounds, and (c) is written as a swappable trait, so it composes
directly with existing HNSW / IVF pipelines that already accept an SQ
backend.

## Design

### Trait

```rust
pub trait Quantizer {
    fn bits_per_vector(&self) -> usize;
    fn bytes_per_vector(&self) -> usize;                                // rounded up
    fn encode(&self, v: &[f32], out: &mut Vec<u8>) -> Result<usize, _>;
    fn decode(&self, bytes: &[u8], out: &mut [f32]) -> Result<(), _>;
    fn distance_sq(&self, q: &[f32], code: &[u8], scratch: &mut Vec<f32>) -> Result<f32, _>;
}
```

Three implementors: `UniformSq8`, `UniformSq4`, `Fasq`. All share the same
calibration input (`&[Vec<f32>]` training set) and the same distance API,
so any component downstream can hold a `Box<dyn Quantizer>`.

### Allocator — discrete water-filling

Adding one bit to dim `i` reduces its distortion contribution by 4×. So the
optimal integer allocation is greedy: start every dim at `b_lo`, then
repeatedly spend the next bit on the dim with the largest *current* residual
contribution `σᵢ² · 4^(-bᵢ)`, until the total budget `D · B` is exhausted or
all dims saturate at `b_hi`. This is the discrete analog of Cover-Thomas
reverse water-filling and is provably optimal for the 4^(-b) rate-distortion
curve.

Complexity: `O(D · B_avg)`. For D=1024, B_avg=4, this is one 4k-iteration
loop at training time — negligible.

### Encoder

Per-dim uniform quantizer over `[min_i, max_i]` with `2^bᵢ - 1` levels,
codes bit-packed MSB-first into a byte stream. The bit I/O uses a
release-safe `low_mask_u32` helper because Rust's `<<` masks the shift
amount modulo the type width in release mode, so `(1u8 << 8) - 1` is `0`,
not `255` — the kind of profile-dependent bug that costs a day.

## Implementation notes

- 8 files, all under 500 lines (`lib.rs`, `allocator.rs`, `quantizer.rs`,
  `baseline.rs`, `main.rs`, `benches/fasq_bench.rs`, `Cargo.toml`).
- 10 unit tests covering allocator invariants, bit I/O round-trips,
  encode/decode round-trips, and baseline SQ8/SQ4 sanity.
- No `unsafe`, no SIMD intrinsics — a straight scalar pass. SIMD is the
  obvious follow-up but the point of this PoC is to establish the
  bit-allocation win before adding hand-written AVX2.

## Benchmark methodology

Two synthetic distributions, both `rand::rngs::StdRng` with fixed seeds
for reproducibility:

- **anisotropic-decay**: `σᵢ = 4.0 · 0.85ⁱ + 0.05`, so σ₀ ≈ 4.05 and
  σ₆₃ ≈ 0.05 — heavy anisotropy, close in shape to what we see on
  post-PCA embedding models.
- **isotropic-unit-normal**: `σᵢ ≡ 1` — worst case for FASQ, exists in
  the corpus to confirm graceful degradation.

For each: 4 000 training vectors, 4 000 base vectors, 200 queries, dim 64.
Brute-force float `L2` gives ground-truth top-10, then each quantizer's
`distance_sq` re-ranks the base and we measure recall@10.

Criterion benches (`cargo bench -p ruvector-fasq`) on d=128, 1 000 vectors:

## Results

**End-to-end (`cargo run --release --bin fasq-demo`)**:

```
=== anisotropic-decay :: dim=64 train=4000 base=4000 q=200 ===
storage bytes/vec  SQ8=64  SQ4=32  FASQ=32  (FASQ avg bits/dim = 4.000)
train time         SQ8=346µs  SQ4=331µs  FASQ=336µs
recon MSE          SQ8=6.40e-5  SQ4=1.84e-2  FASQ=6.42e-4
FASQ vs SQ4 (same storage): 28.72× lower MSE
encode 4000 vecs   SQ8=374µs  SQ4=511µs  FASQ=739µs
recall@10          SQ8=0.9765  SQ4=0.7080  FASQ=0.9745

=== isotropic-unit-normal :: dim=64 train=4000 base=4000 q=200 ===
storage bytes/vec  SQ8=64  SQ4=32  FASQ=32
recon MSE          SQ8=6.87e-5  SQ4=1.90e-2  FASQ=1.90e-2
FASQ vs SQ4 (same storage): 1.00× lower MSE
recall@10          SQ8=0.9790  SQ4=0.7655  FASQ=0.7655
```

**Criterion microbenchmarks (`cargo bench -p ruvector-fasq`, d=128, 1 000 vec)**:

| Bench                       | SQ8        | SQ4        | FASQ (avg 4b) |
|-----------------------------|------------|------------|---------------|
| encode 1 000 × d=128        | 138 µs     | 161 µs     | 270 µs        |
| distance 1 000 × d=128      | 56 µs      | 178 µs     | 220 µs        |

**Hardware**: Apple Silicon (arm64), stock `cargo bench --release`, no
external accelerators. Reproducible: same seeds, no external data.

## How it works — walkthrough

1. Call `Fasq::train(&vectors, b_avg, b_lo, b_hi)`.
2. `describe_dims` computes per-dim mean, min, max, variance in one pass.
3. `allocator::allocate` runs discrete water-filling over the variances,
   returning `Vec<u8>` bit counts (2..=8 per dim).
4. Scales `(max − min) / (2^bᵢ − 1)` are stored per dim.
5. `encode(v)` clamps `(v[i] − min[i]) / scale[i]` to `[0, 2^bᵢ − 1]`,
   rounds to the nearest integer level, and bit-packs into a byte buffer
   whose length is `⌈∑bᵢ / 8⌉`.
6. `decode(bytes)` unpacks and dequantizes.
7. `distance_sq(query, code)` decodes into a scratch buffer and returns
   L2 — same distance model as any other reconstruction-based SQ.

## Practical failure modes

- **Very small training sets** (≪ 500 vec) leave σᵢ² estimates noisy, so
  allocation can be wrong. Mitigation: bootstrap-resample variance
  estimates or fall back to uniform SQ4 when D · sample count < 10 k.
- **Isotropic corpora** — FASQ has no bits to reallocate and offers no
  win over SQ4. This is correct behavior; do not deploy FASQ on
  post-random-projection features unless you can prove residual
  anisotropy.
- **Distribution drift** — recalibrate periodically (e.g. every 10 M new
  writes). All allocator state is a small `Vec<u8>` / `Vec<f32>` so
  swap-in is cheap.
- **Distance is reconstruction-based**, not ADC/LUT. To match RaBitQ or
  PQ-ADC throughput, add SIMD (AVX2/NEON) in a follow-up, or wire FASQ
  as the *rerank* stage on top of a coarser first-pass.

## What to improve next (roadmap)

1. **SIMD encode/decode** — nibble/two-bit unpacking with `pshufb` /
   `vqtbl` doubles distance throughput on AVX2 / NEON. Expected 3–4×.
2. **Per-dim non-uniform quantizer** — replace uniform scalar levels
   with Lloyd-Max-optimal levels on the empirical CDF; another
   ~1.2–1.5× MSE win on non-Gaussian dims.
3. **Rotation before FASQ** — random orthogonal or PCA rotation
   concentrates variance and multiplies FASQ's advantage on inputs
   that arrive isotropic in the raw basis.
4. **HNSW integration** — plug `Fasq` behind
   `ruvector-core::VectorStore` so the existing HNSW index compresses
   vectors transparently.
5. **Cross-index rerank ladder** — RaBitQ (1 bit) → FASQ (4 bit) → f32
   as three tiers, chosen by the coordinator based on candidate depth.

## Production crate layout

Following existing ruvector conventions the crate would evolve to:

```
crates/ruvector-fasq/
  Cargo.toml
  src/
    lib.rs          # public traits, DimStats, helpers
    allocator.rs    # water-filling
    quantizer.rs    # Fasq encoder/decoder + bit I/O
    baseline.rs     # UniformSq8, UniformSq4
    simd/           # (roadmap) AVX2 / NEON kernels
    hnsw.rs         # (roadmap) VectorStore adapter
  benches/fasq_bench.rs
  examples/end_to_end.rs
```

## References

- T. M. Cover & J. A. Thomas, *Elements of Information Theory*, 2nd ed.,
  Chapter 13 "Rate Distortion Theory", Wiley, 2006.
- Jianyang Gao & Cheng Long, *RaBitQ: Quantizing High-Dimensional Vectors
  with a Theoretical Error Bound for Approximate Nearest Neighbor Search*,
  SIGMOD 2024.
- Kusupati et al., *Matryoshka Representation Learning*, NeurIPS 2022.
- Aumüller et al., *ANN-Benchmarks: A Benchmarking Tool for Approximate
  Nearest Neighbor Algorithms*, Information Systems 2020.
- Milvus 2.4 changelog — scalar quantization pipeline (`internal/core/src/
  quantization`, retrieved 2026-07-01).
- Qdrant 1.10 changelog — scalar and binary quantization defaults
  (retrieved 2026-07-01).
