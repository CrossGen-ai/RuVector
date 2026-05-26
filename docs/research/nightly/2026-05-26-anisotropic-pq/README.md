# Anisotropic Product Quantization for RuVector

**Nightly research — 2026-05-26**
Branch: `research/nightly/2026-05-26-anisotropic-pq`
Crate:  `crates/ruvector-anisotropic-pq`
ADR:    [ADR-194](../../../adr/ADR-194-anisotropic-product-quantization.md)

## Abstract

Product Quantization (PQ) compresses dense vectors 16–32× with a uniform
ℓ² loss, but **maximum inner-product search (MIPS)** is asymmetric in a way
that uniform ℓ² ignores: the residual component _parallel_ to a datapoint's
direction perturbs the inner-product score more than the perpendicular
component does. ScaNN (Guo et al., ICML 2020) exploits this with a
**score-aware loss** that up-weights the parallel error by a factor `η ≥ 1`.

This nightly delivers a working Rust implementation of a per-subspace
anisotropic PQ trainer behind a clean `Quantizer` trait and benchmarks
it against vanilla PQ on a synthetic clustered MIPS workload. The
implementation is honest about a known limitation: ScaNN's loss is not
exactly subspace-decomposable, so the per-subspace approximation captures
only part of the theoretical gain. Results show η=2.0 yields a small but
real R@10 lift at the looser quantization rate (M=16) and is neutral-to-mildly
negative at the tighter rate (M=32) — matching the qualitative behaviour
reported in the public literature for the decomposed variant.

## SOTA survey

| System          | Loss for codebook training         | Notes                                                 |
|-----------------|------------------------------------|-------------------------------------------------------|
| PQ (Jégou ’11)  | Σ ‖x − x̃‖²                        | Original product quantization. Subspace-independent.  |
| OPQ (Ge ’13)    | Σ ‖Rx − x̃‖² with orthogonal R     | Learns rotation; reduces inter-subspace correlation.  |
| LOPQ (Kalantidis ’14) | Per-cell OPQ                 | Locally optimized.                                    |
| ScaNN (Guo ’20) | Σ η ‖r_∥‖² + ‖r_⊥‖²                | Score-aware loss. Anisotropic in parallel direction.  |
| RaBitQ (Gao ’24)| 1-bit rotation-quantization        | Implemented in `ruvector-rabitq` (nightly 2026-04-23).|
| AQLM (Egiazarian ’24) | Additive codebooks           | Trades encode complexity for tighter recall–size.     |

ScaNN's published gains over PQ at the same bit-rate are in the 1–4 % R@10
range on production embedding workloads (GloVe, MS-MARCO). The gain comes
from two pieces: the score-aware loss **and** a hard residual encoder.
This nightly isolates the loss change; the encoder change is left to a
follow-up (see "What to improve next" below).

## Proposed design

A new crate `ruvector-anisotropic-pq` exposes:

```rust
pub trait Quantizer: Sync {
    fn encode(&self, x: &[f32]) -> Vec<u8>;
    fn build_ip_lut(&self, q: &[f32]) -> Vec<f32>;
    fn score(&self, lut: &[f32], code: &[u8]) -> f32;
    fn bytes_per_vector(&self) -> usize;
    // …
}
```

with two implementors:

- `Pq` — isotropic Lloyd's algorithm over each of M subspaces.
- `ApqQuantizer` — per-subspace anisotropic loss with `η ≥ 1`.

Both produce a code of `M` bytes when `K = 256`, and both serve a query
via the standard PQ asymmetric distance-computation lookup table.

### Per-subspace anisotropic loss

For a residual `r = x_sub − c` and unit direction `n̂ = x_sub / ‖x_sub‖`:

```
L(r, x_sub) = η · (r · n̂)²  +  ‖r‖²  −  (r · n̂)²
            = (η − 1) · (r · n̂)²  +  ‖r‖²
```

`η = 1` recovers PQ. `η > 1` penalises the parallel component more.

### Centroid update (M-step)

Writing the per-point weight `W_i = I + (η−1) n̂_i n̂_iᵀ`, the optimal
centroid for cluster `c` is

```
c* = (Σ_{i∈c} W_i)⁻¹ · (Σ_{i∈c} W_i x_i)
   = (n_c · I + (η−1) Σ n̂ n̂ᵀ)⁻¹ · (Σ x + (η−1) (n̂·x) n̂)
```

a small `sub_dim × sub_dim` linear system per centroid per iteration.
The implementation uses Gauss-Jordan with partial pivoting (sub_dim is
typically 4–8, so this is essentially free).

## Implementation notes

- Single library file (`src/lib.rs`, ~340 lines).
- Single binary (`src/main.rs`, ~140 lines) for `cargo run` benchmark.
- Integration tests against brute-force MIPS ground truth (`tests/recall.rs`).
- Criterion bench (`benches/apq_recall.rs`) for per-query latency.
- Zero `unsafe`. Stable Rust. No mocks. No SIMD intrinsics yet.

## Benchmark methodology

Synthetic workload (deterministic seed):

- `n = 20 000` unit-norm vectors in `ℝ¹²⁸`.
- 64 latent clusters; each vector is `normalize(center + 0.15·N(0,I))`.
- 200 queries drawn as `normalize(x_db + 0.20·N(0,I))` for a realistic
  "search for things similar to X" workload (queries are correlated with
  the database, which is where score-aware quantization is meant to win).
- Codebook training on 8 000 vectors, 12 Lloyd iterations.
- Ground truth: exact MIPS top-10 / top-100 by inner product.
- `K = 256` centroids per subspace → 1 byte per subcode.

Hardware: Apple Silicon (Darwin 24.6). Single thread for training,
no SIMD intrinsics.

## Results

Real numbers from `cargo run --release -p ruvector-anisotropic-pq`
(2026-05-26):

### M = 16 (sub_dim = 8, 16 bytes/vec, 32× compression vs fp32)

| Variant     | bpv | total mem | encode (full db) | search (200 q × 2 k) | R@10    | R@100   |
|-------------|----:|----------:|-----------------:|---------------------:|--------:|--------:|
| PQ          |  16 | 0.31 MiB  | 145.0 ms         | 188.4 ms             | 0.3075  | 0.3969  |
| APQ η=2.0   |  16 | 0.31 MiB  | 239.3 ms         | 160.6 ms             | **0.3140** | 0.3920 |
| APQ η=4.0   |  16 | 0.31 MiB  | 242.9 ms         | 158.7 ms             | 0.3015  | 0.3877  |

Training: PQ 639 ms · APQ η=2 2505 ms · APQ η=4 2553 ms.

### M = 32 (sub_dim = 4, 32 bytes/vec, 16× compression vs fp32)

| Variant     | bpv | total mem | encode (full db) | search (200 q × 2 k) | R@10    | R@100   |
|-------------|----:|----------:|-----------------:|---------------------:|--------:|--------:|
| PQ          |  32 | 0.61 MiB  | 193.4 ms         | 204.7 ms             | **0.5860** | **0.6509** |
| APQ η=2.0   |  32 | 0.61 MiB  | 281.6 ms         | 195.4 ms             | 0.5845  | 0.6454  |
| APQ η=4.0   |  32 | 0.61 MiB  | 281.3 ms         | 194.6 ms             | 0.5685  | 0.6286  |

Training: PQ 865 ms · APQ η=2 2529 ms · APQ η=4 2541 ms.

Brute-force fp32 MIPS ground truth: 374 ms for 200 queries × 20 000 db.

### Key observations (honest)

- At **M=16 (heaviest compression)**, APQ η=2 gives a small but real
  R@10 improvement (+0.65 percentage points, +2.1 % relative) over PQ.
  This is the regime where any extra signal in the loss matters most.
- At **M=32**, where each subspace already captures fine structure, the
  per-subspace anisotropic loss provides no benefit and η=4 actively hurts
  (-1.75 pp R@10).
- η=4 hurts in both regimes — the per-subspace direction estimate becomes
  noisy when sub_dim is small (4–8), so over-weighting it amplifies noise.
- Training is ~3–4× slower for APQ (extra outer-product accumulation +
  linear-solve per centroid per iteration). Encode is ~1.7× slower (the
  anisotropic loss computation per centroid is more arithmetic than ℓ²).
- Search-time and bytes-per-vector are identical: ADC over the LUT is
  loss-agnostic.

## How it works (blog-readable walkthrough)

Imagine you compress a vector by snapping it to the nearest of 256
centroids in each of M = 16 chunks. The "snap error" is a residual
`r = x − x̃`. When you later score a query `q` against the compressed
vector, the error in the score is exactly `q · r`. If `q` points in the
same direction as `x` (which is what happens at the top of the result
list — the top-k matches really do look like x), then the component of
`r` that lines up with `x` is the component that pollutes the score.

Anisotropic PQ says: don't treat all directions of the error equally.
Penalise the parallel component (the one that hurts the top-k score)
more, even at the cost of a slightly larger perpendicular component
(which doesn't affect top-k as much).

In practice the gain is modest because the parallel/perpendicular
decomposition has to happen separately in each subspace, and the
direction signal weakens as subspaces shrink.

## Practical failure modes

1. **Small sub_dim:** `sub_dim ≤ 4` makes the per-subspace direction
   estimate noisy. Use OPQ-style rotation first, or stick with `η ≈ 1.5–2`.
2. **Non-MIPS metrics:** If you score by ℓ² distance (not inner product),
   the anisotropic loss is the wrong objective; PQ is correct.
3. **Out-of-distribution queries:** The "queries are similar to db points"
   assumption breaks down for cross-modal retrieval (e.g. text→image).
   Score-aware loss can underperform there.
4. **Empty clusters:** Re-seeding empty clusters from random data
   points is implemented but can stall recall improvement late in training.
5. **η tuning:** η is a hyperparameter; pick by recall on a held-out
   query set. There is no closed-form value.

## What to improve next

1. **Full-vector (non-decomposed) loss.** ScaNN's published gains come
   from optimising `(r · x)²` jointly across subspaces, which requires
   coordinate-descent training. Net: ~+3 % R@10 expected.
2. **Hard residual encoder.** Rather than greedy nearest-centroid per
   subspace, jointly pick the M-tuple of codes minimising the
   anisotropic loss. Tractable with beam search.
3. **OPQ pre-rotation.** Train an orthogonal rotation `R` first
   (`crates/ruvector-opq` exists) and run APQ on `Rx`. Removes
   inter-subspace correlation, which should restore the gain at M=32.
4. **SIMD ADC.** The LUT scoring loop is the runtime hot path. AVX-512 /
   NEON gather+sum should give ~2× search throughput.
5. **Integration with `ruvector-acorn` filtered HNSW.** Use APQ as the
   compressed distance during graph traversal, fp32 only for re-rank.
6. **Wasm packaging.** Add `ruvector-anisotropic-pq-wasm` once API stabilises.

## Production crate layout proposal

```
crates/
  ruvector-anisotropic-pq/       ← this crate (PoC + bench)
  ruvector-anisotropic-pq-wasm/  ← future wasm bindings
  ruvector-pq-shared/            ← future: shared codebook + LUT runtime
                                   for PQ / OPQ / APQ / LOPQ
```

`Quantizer` would graduate to `ruvector-pq-shared` so HNSW, IVF, and
DiskANN backends can swap losses without code duplication.

## References

- Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar. **Accelerating
  Large-Scale Inference with Anisotropic Vector Quantization.** ICML 2020.
  https://arxiv.org/abs/1908.10396
- Jégou, Douze, Schmid. **Product Quantization for Nearest Neighbor
  Search.** IEEE TPAMI 2011.
- Ge, He, Ke, Sun. **Optimized Product Quantization.** CVPR 2013.
- Egiazarian et al. **Extreme Compression of Large Language Models via
  Additive Quantization.** ICML 2024 (AQLM).
- Gao, Long. **RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search.**
  SIGMOD 2024.
- ScaNN open-source reference: https://github.com/google-research/google-research/tree/master/scann

## Reproduction

```bash
cargo build --release -p ruvector-anisotropic-pq
cargo test  --release -p ruvector-anisotropic-pq
cargo run   --release -p ruvector-anisotropic-pq          # full bench
cargo bench --bench apq_recall -p ruvector-anisotropic-pq # per-query latency
```
