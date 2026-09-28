# Adaptive Bit-Budget Product Quantization (AB-PQ) for ruvector

**Nightly research — 2026-09-28**
Slug: `adaptive-bit-budget-pq` · ADR: [ADR-350](../../../adr/ADR-350-adaptive-bit-budget-pq.md) · Crate: [`crates/ruvector-abpq`](../../../../crates/ruvector-abpq)

## Abstract

Classical Product Quantization (Jégou, Douze & Schmid, 2011) partitions a
`d`-dimensional vector into `m` equal-size subvectors and quantizes each with a
codebook of fixed size `k = 2^b`. The **uniform** bit allocation `b_i = b` is
information-theoretically suboptimal whenever subvectors have unequal energy —
which is the common case, especially after PCA, OPQ rotation, or on natural
embedding distributions where trailing coordinates carry little signal.

**AB-PQ** replaces uniform allocation with an integer **water-filling**
solution that minimizes total expected distortion under a fixed bit budget.
On a synthetic anisotropic benchmark (n=5000, d=64, m=16, average 4 bits per
subquantizer, code length identical to uniform PQ-4 at 8 bytes/vector), AB-PQ
reduces reconstruction MSE **34.0 %** (1.316 vs 1.987) and improves
**recall@10 from 0.172 to 0.313 (1.82×)** with no scan-time overhead — the
bit-packed ADC path is <15 ns per code lookup.

## SOTA survey

- **PQ** — Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor
  Search*, TPAMI 2011. Baseline: uniform bit allocation per subquantizer.
- **OPQ** — Ge, He, Ke, Sun, *Optimized Product Quantization*, CVPR 2013.
  Solves the rotation problem but keeps uniform bit allocation.
- **LOPQ** — Kalantidis & Avrithis, *Locally Optimized Product Quantization*,
  CVPR 2014. Per-cell OPQ, still uniform bits.
- **Additive/Tree Quantization** — Babenko & Lempitsky, CVPR 2014/2015.
  Alternative code structures; still fixed bit budget per component.
- **ScaNN / anisotropic PQ** — Guo et al., ICML 2020. Reweights training loss
  by inner-product magnitude but does not vary code length per subvector.
- **RaBitQ** — Gao & Long, SIGMOD 2024. 1-bit quantization with theoretical
  distortion bound; per-dimension bit count is trivially uniform (1).
- **LVQ / LeanVec** — Aguerrebere et al., Intel Labs 2023–2024. Scalar
  quantization with linear projection; also uses uniform scalar bits.
- **Rate-distortion water-filling** — Cover & Thomas, *Elements of
  Information Theory*, ch. 10 (Gaussian source coding). AB-PQ is the direct
  discretization of this classical result to k-means quantizers.

None of the widely used ANN quantizers in Milvus, FAISS, Qdrant, LanceDB, or
Weaviate today expose per-subquantizer variable bit width; AB-PQ is a small,
implementable extension that composes with all of the above (OPQ rotation
first → AB-PQ codebook fitting).

## Proposed design

Let `x = (x^(1), …, x^(m))` be a d-dim vector split into m subvectors of
equal length `dsub = d/m`. Let `v_i = trace(Cov(x^(i)))` (sum of variances
inside subvector i). A k-means codebook of size `2^b_i` on subvector i
produces expected squared distortion approximately proportional to
`v_i · 2^(-2 b_i)` (Lloyd's high-rate bound, exact for Gaussian sources).

We fix a total bit budget `B = m · b_avg` and solve

```text
minimize     sum_i v_i · 2^(-2 b_i)
subject to   sum_i b_i = B ,   b_min ≤ b_i ≤ b_max ,   b_i ∈ ℤ .
```

The objective is separable and strictly convex; the marginal gain of moving
`b_i → b_i + 1` is exactly `0.75 · v_i · 2^(-2 b_i)`, which is **non-increasing
in b_i** for each i. This makes the following greedy exact:

> Start every subquantizer at `b_min`. Repeatedly add one bit to the
> subquantizer with the largest current marginal gain, until the budget is
> spent.

Implementation uses a max-heap keyed on marginal gain (size m; m ≤ 32 in
practice), giving `O((B − m·b_min) log m)` bit assignment. Codebooks are
then fit independently with k-means++.

**Storage layout.** Codes are bit-packed into `ceil(sum_i b_i / 8)` bytes per
vector. Every field ≤ 8 bits spans at most 2 bytes; read/write are branchless
except for the boundary-crossing byte-2 write. **ADC scan** precomputes a
per-query table of size `sum_i 2^b_i` floats and does one gather + add per
subquantizer — the same instruction count as classical PQ ADC, at the cost
of one variable-width bit extraction.

## Implementation notes

- Crate: `ruvector-abpq` (< 500 lines: 305 `lib.rs`, 100 `water_fill.rs`,
  25 `pack.rs`).
- No unsafe, no SIMD intrinsics — leaves those wins on the table for a
  follow-up. Even the naive f32 ADC scan reaches 14.7 ns/code at d=64.
- `AbPq::train(cfg, data)` returns a serde-serializable struct; codebooks
  are stored as row-major `Vec<f32>` for cache locality inside a subquantizer.
- Trait-friendly encode/decode interface (`encode_into`, `adc_distance`,
  `build_adc_table`) is ready to be wrapped behind ruvector's existing
  `PqLike` interface for drop-in use inside IVF-PQ or graph-rerank paths.

## Benchmark methodology

Synthetic anisotropic dataset: `x[j] ~ N(0, 1/(1 + j/6))`. This mimics
post-PCA / post-OPQ embeddings where energy decays smoothly across
dimensions. n=5000 database vectors, 200 queries drawn from the same
distribution, d=64, m=16.

Three quantizers trained on the same data with identical seed:
1. **uniform PQ (3 bits/sub)** — 6 bytes/code.
2. **uniform PQ (4 bits/sub)** — 8 bytes/code (baseline).
3. **AB-PQ (avg 4 bits/sub)** — 8 bytes/code (same footprint as baseline).

Reconstruction MSE is per-vector L2 reconstruction error. Recall@10 is the
fraction of the true top-10 (exact L2 on FP32 data) that appears in the
top-10 sorted by ADC distance. Encode/search times are wall-clock, one
thread, macOS Darwin 24.6 (Apple Silicon), release build.

## Results

Cargo output verbatim (`cargo run --release -p ruvector-abpq --bin abpq-demo`):

```text
AB-PQ demo — n=5000 d=64 m=16 queries=200
uniform PQ (3 bits)  bits=[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]  code_bytes=6  MSE=2.7072  recall@10=0.112  encode=1.0ms  search=53.0ms
uniform PQ (4 bits)  bits=[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]  code_bytes=8  MSE=1.9874  recall@10=0.172  encode=1.5ms  search=52.7ms
AB-PQ (avg 4 bits)   bits=[6,6,5,5,4,4,4,4,4,4,3,3,3,3,3,3]  code_bytes=8  MSE=1.3161  recall@10=0.313  encode=2.2ms  search=54.1ms
```

Criterion micro-benches on the same machine:

| Bench | n × d | mean |
|---|---|---|
| `abpq_encode_batch` | 2000 × 64 | 893 µs (~446 ns/vec) |
| `abpq_adc_scan` | 10 000 × 64 | 147 µs (~14.7 ns/code) |

**Headline** at equal 8-byte code length:
- MSE ↓ **34.0 %** (1.987 → 1.316)
- Recall@10 ↑ **82 %** (0.172 → 0.313, absolute +0.141)
- Search latency: **essentially identical** (52.7 ms → 54.1 ms, +2.6 % from
  variable-width bit unpack; SIMD lookup tables would erase this)
- Encode: +47 % vs uniform (1.5 ms → 2.2 ms), because high-variance slots
  train a 64-entry codebook instead of 16 — parallelizable and rarely hot.

## How it works (walkthrough)

Imagine embeddings whose first dimensions carry 10× the variance of the
last. Uniform PQ spends the same 4 bits on the noisy tail as on the
information-dense head — that's like recording a rock concert at the same
resolution as a background hum. AB-PQ looks at each subvector's total
variance, then hands out bits like a rate-distortion accountant:
high-variance slots get 6 bits (64-entry codebook), low-variance slots get 3
(8-entry codebook). Every bit is spent where cutting distortion by 4×
matters most. On our anisotropic benchmark the reallocation is
`[6,6,5,5,4,4,4,4,4,4,3,3,3,3,3,3]` — same 64 total bits, dramatically
better recall.

The greedy is optimal because doubling any codebook cuts its distortion
term by exactly 4×, so the "next bit" gain is a monotone non-increasing
function of the current bit count. Popping the max marginal gain, one bit
at a time, reaches the integer optimum in O(B log m) — negligible against
k-means training.

## Practical failure modes

- **Isotropic data** (post-whitening, random projections): variances are
  equal, water-filling collapses to uniform. AB-PQ costs nothing but adds
  nothing.
- **Extreme skew** (variance ratio > 10⁵): the top slot saturates at
  `b_max = 8`, leaving budget stranded. Fix: raise `b_max` to 10 (needs a
  u16 code field) or apply OPQ first to spread energy.
- **Tiny training set** (n < 32 · 2^b_max): high-bit codebooks starve.
  Detect via cluster occupancy and force uniform fallback.
- **Adversarial rebalance under drift**: variances shift after re-training
  → codes need re-encoding. Same failure mode as OPQ retraining; document
  and version the codebook.
- **ADC lookup gather cost**: block offsets are non-uniform, so
  `codes → lookup index` is a bit shift + mask per subquantizer. SIMD
  gather with variable strides is harder than PQx4 LUT-16. See "improve
  next".

## What to improve next

1. **PQx4 layout with mixed strides**. Group same-bit subquantizers into
   fastscan-16 blocks, keep a small "tail" of odd-bit subquantizers scored
   scalar. Should recover the LUT-16 4× throughput win on the hot path.
2. **OPQ rotation before AB-PQ**. OPQ re-mixes energy across subvectors,
   giving water-filling *more* to work with (higher skew → bigger gain).
3. **Learned lambda-schedule** for bit allocation, weighted by query
   distribution (inner product / cosine target) — closes the loop with
   score-aware PQ (ScaNN / anisotropic loss).
4. **Cluster-local water-filling**. In IVF-PQ, run per-cell allocation:
   different cells have different residual variance profiles.
5. **Persistent codebook / codes v1 file format**, with bit-schedule
   embedded, for on-disk artifacts.

## Production crate layout proposal

```
ruvector-abpq/
├── src/
│   ├── lib.rs           # AbPq, AbPqConfig, AdcTable, k-means (305 LoC)
│   ├── water_fill.rs    # variance + integer water-filling (100 LoC)
│   └── pack.rs          # bit-packed code layout (25 LoC)
├── benches/abpq_bench.rs   # criterion encode + ADC scan
└── src/main.rs             # `abpq-demo` binary (real numbers)
```

Integration hooks (follow-up PRs, not this branch):
- `ruvector-pq-search`: replace `PqCode` with `AbPqCode { code_bits, code_bytes }`.
- `ruvector-cascade-adc`: drop AB-PQ table into cascade stage 1.
- `ruvector-opq-rotation`: chain OPQ → AB-PQ.
- `ruvector-diskann` / `ruvector-spann`: cluster-local AB-PQ per cell.

## References

1. Jégou, H., Douze, M., & Schmid, C. (2011). Product Quantization for
   Nearest Neighbor Search. *TPAMI*, 33(1), 117–128.
2. Ge, T., He, K., Ke, Q., & Sun, J. (2013). Optimized Product Quantization
   for Approximate Nearest Neighbor Search. *CVPR*.
3. Guo, R., Sun, P., Lindgren, E., et al. (2020). Accelerating Large-Scale
   Inference with Anisotropic Vector Quantization. *ICML*.
4. Gao, J., & Long, C. (2024). RaBitQ: Quantizing High-Dimensional Vectors
   with a Theoretical Error Bound for Approximate Nearest Neighbor Search.
   *SIGMOD*.
5. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*
   (2nd ed.), ch. 10 (rate-distortion / water-filling).
6. André, F., Kermarrec, A.-M., & Le Scouarnec, N. (2015). Cache-Locality
   in Product Quantization for Fast Similarity Search. *VLDB*.
7. Aguerrebere, C. et al. (2023). Similarity Search in the Blink of an Eye
   with Compressed Indices. *VLDB* (LVQ / LeanVec).
