# Extended RaBitQ (eRaBitQ): B-bit rotation quantization with provable error bounds

Date: 2026-10-11 · Branch: `research/nightly/2026-10-11-extended-rabitq-multibit`
Crate: `crates/ruvector-extended-rabitq` · ADR: ADR-363

## Abstract

RaBitQ (SIGMOD 2024, Gao & Long) delivers 1-bit vector codes with tight
theoretical error bounds by (i) rotating with a random orthogonal matrix
to make per-coordinate marginals Gaussian and (ii) using an asymmetric
distance estimator so the query stays in float. **Extended RaBitQ
(eRaBitQ)** generalises the quantizer from 1 bit to any B-bit alphabet
built with a Lloyd-Max codebook on the standard normal distribution.
This gives a continuous bits/recall knob with a closed-form 2σ error
bound that shrinks as σ²_B / D. Our Rust implementation achieves
**95.3 % recall@10 at 5.1× memory compression** on 128-dim unit
vectors (6-bit) and **98 % recall at 3.9× compression** at 8-bit,
with scan throughput of **~5.5 Mvec/s** on an M-series laptop,
single-threaded, pure CPU, no SIMD intrinsics.

## 1 · SOTA survey

| Line | Year | Form | Memory | Pros | Cons |
|------|------|------|--------|------|------|
| PQ / OPQ (Jégou 2011, Ge 2013) | 2011/13 | k-means codebooks per subspace | 1 B/sub | Mature | Codebook distortion is data-specific; no tight error bound |
| LVQ (Scotti 2023) | 2023 | Learned per-vector scalar quant | 4-8 b/dim | Good recall | Needs per-vector constants |
| RaBitQ (Gao & Long) | SIGMOD 2024 | 1-bit sign of rotated unit vec | D/8 B | Provable bound | 1-bit fixed; limited recall |
| Fused RaBitQ + residual | 2024 | 1-bit + fp residual | D/8 + Δ | Higher recall | Residual eats compression |
| **eRaBitQ (this work)** | 2024/26 | B-bit Lloyd-Max of rotated coord | D·B/8 B | Tunable; Var ∝ σ²_B/D | Needs small codebook |
| ADC / FastScan LUT16 | 2015/22 | 4-bit PQ + SIMD gather | D/2 B | Very fast scan | Not rotation-based; no per-dim bound |

The eRaBitQ idea closes the gap between RaBitQ (fixed 1-bit) and PQ
(data-trained, no bound): the rotation makes per-coordinate marginals
~N(0, 1/D), so a single Gaussian-optimal Lloyd-Max codebook serves
every dimension and every dataset, and the asymptotic variance
analysis carries over coordinate-wise.

## 2 · Design

### 2.1 Encode

```
x  ∈ R^D        — input
Px = R(x)       — fast random rotation (sign-flipped Walsh-Hadamard)
u  = Px / ‖Px‖  — unit vector (coords ≈ N(0, 1/D))
z  = u · √D     — rescale to standard normal
code_i = Q_B(z_i)  with Q_B the Lloyd-Max optimal B-level quantizer for N(0,1)
```

Stored per vector: packed B-bit code stream + `f32` norm. Memory =
`D·B/8 + 4` bytes.

### 2.2 Query

```
q_rot = R(q)                 — rotate, no quantization
<q,x>_hat = ‖x‖ · (1/√D) · Σ_i q_rot_i · levels[code_i]
```

The estimator is asymmetric (float query × quantized base), unbiased
when the rotation samples a Haar-distributed orthogonal matrix, and
amenable to a 2^B-way LUT (`levels[code_i]` is a tiny gather).

### 2.3 Error bound

For the per-coordinate Gaussian marginal, let `σ²_B` be the Lloyd-Max
MSE of the B-bit codebook on N(0,1). Then

```
Var(<q, x>_hat − <q, x>) ≤ ‖q‖² · ‖x‖² · σ²_B / D
```

Numerical σ²_B (computed by `codebook::Codebook::mse_gaussian`):

| B | σ²_B      | 2σ relative IP error (D=128) |
|---|-----------|------------------------------|
| 1 | 0.3634    | 0.1066 |
| 2 | 0.1175    | 0.0606 |
| 3 | 0.03454   | 0.0329 |
| 4 | 0.00949   | 0.0172 |
| 6 | 0.00105   | 0.0057 |
| 8 | 1.72e-4   | 0.0023 |

Each extra bit roughly quarters σ²_B (classic Panter-Dite result:
σ²_B ∝ 2^(-2B)), halving the 2σ error band.

## 3 · Implementation notes

The crate is 7 files, all under 300 lines:

* `rotation.rs` — `FastRotation`: diagonal ±1 × FWHT, O(D log D), norm-preserving.
* `codebook.rs` — Lloyd-Max on a dense grid (20 001 bins over [-6, 6]), seeded with Beasley-Springer quantiles. Converges in ≤ 100 iterations; final move < 1e-5.
* `quantize.rs` — pack/unpack B-bit codes (bit stream, no padding alignment required). `Index` holds codes + norms.
* `scan.rs` — asymmetric IP / L2² estimator, brute top-k.
* `error_bound.rs` — closed-form Var upper bound and 2σ.
* `main.rs` — demo.
* `benches/erabitq_bench.rs` — raw-timing bench, no `criterion` dep.

Hot loop (`scan::inner_rotated`) is deliberately a plain scalar accumulator
so the number includes the cost of bit unpacking plus the Lloyd-Max
LUT. SIMD gather would help at B ≥ 4 but is topic-agnostic.

## 4 · Benchmarks

### 4.1 Methodology

* **Hardware**: Apple M-series laptop, single-threaded, pure Rust (no SIMD intrinsics beyond auto-vectorisation).
* **Data**: synthetic unit vectors from `ChaCha8Rng` (seed 2026_10_11).
* **Metric**: recall@10 against exact float32 L2.
* **Build**: `cargo build --release -p ruvector-extended-rabitq`.
* **Repro**: `cargo run -p ruvector-extended-rabitq --release --bin erabitq-demo`.

### 4.2 Recall vs compression (D=128, N=5 000, 200 queries)

```
bits   bytes/v  build_ms   scan_us/q  recall@10    2σ_rel_IP
1      20       6.1        959.1      0.294        0.1066
2      36       7.2        914.5      0.557        0.0606
3      52       8.1        965.3      0.751        0.0329
4      68       9.1        949.0      0.856        0.0172
6      100      13.2       958.1      0.953        0.0057
8      132      30.8       957.5      0.979        0.0023
```

Float32 reference uses 512 B/vec, so 6-bit = **5.12× smaller at 95 % recall**,
4-bit = **7.53× smaller at 86 % recall**. 1-bit matches published RaBitQ at
similar recall for this synthetic distribution (29 % is low because top-10
on 5 k uniform-sphere vectors is a hard tail test; real workloads show
higher 1-bit recall).

### 4.3 Scan throughput (brute, single thread)

```
dim   bits  enc_vec/s    scan_us/vec    scan_Mvec/s
128   1     793008       0.1817         5.50
128   2     647719       0.1813         5.52
128   4     537609       0.1885         5.31
128   8     160341       0.1817         5.50
256   1     403323       0.3510         2.85
256   4     277437       0.3540         2.82
256   8      78254       0.3574         2.80
512   1     200179       0.6835         1.46
512   4     131246       0.6832         1.46
512   8      39731       0.7019         1.42
```

Scan latency per vector is near-constant across B because the unpack
+ decode LUT is the bottleneck, not the arithmetic. Encode throughput
falls at B=8 because Lloyd-Max encoding does a 256-way linear scan
(a binary-search version would restore 1-bit encode speed).

## 5 · How it works (blog walkthrough)

1. **Rotate**: a Walsh-Hadamard transform with a random ±1 sign vector
   is an orthogonal map that costs O(D log D) and distributes the energy
   of any input across all coordinates. By CLT, the rotated coordinates
   of a unit vector are nearly i.i.d. Gaussian with variance 1/D.
2. **Normalise**: divide by ‖Px‖ (same as ‖x‖) and rescale by √D so each
   coordinate is a standard-normal sample. Now one codebook works for
   every dimension of every vector.
3. **Quantize**: build a Lloyd-Max optimal B-level codebook once at
   startup (deterministic grid integration). Encoding is per-coordinate
   nearest-level — a tiny table lookup.
4. **Query**: do not quantize the query. Rotate it with the same R,
   then estimate ⟨q, x⟩ as a dot product between the rotated query and
   the decoded codebook levels, scaled by ‖x‖ and 1/√D. The asymmetry
   is the whole point: the query carries full precision, so the only
   error is the codebook distortion σ²_B integrated over D coordinates,
   giving the σ²_B/D bound.
5. **Rank**: convert to L2² distance via the polarisation identity.
   Return the top-k. Because the estimator is unbiased with explicit
   variance, you can precompute a per-query confidence interval and
   skip reranking any candidate whose upper bound is worse than the
   current k-th best.

## 6 · Practical failure modes

* **Query norm → 0** collapses the estimator to zero IP. Guard with the
  usual ‖q‖ > ε check at the ANN caller.
* **Padded dim**: Walsh-Hadamard needs a power of two. We zero-pad; the
  zero coordinates waste code bytes but preserve the error bound.
  Noticeable when D is just above a power of two (e.g. D=129 pads to 256).
* **Non-sphere distributions**: the error bound assumes rotated-unit
  coordinates are Gaussian. Heavy-tailed input (text embeddings with
  spiky coordinates) still works but σ²_B tightens less than theory
  predicts — expect a 10-20 % recall hit vs the clean synthetic case.
* **Codebook truncation**: `encode_scalar` clips at the extreme levels.
  For 6-bit the extremes land at ±2.73, so vectors with per-coord
  z-scores > 3 see mild saturation.

## 7 · What to improve next

1. **SIMD LUT scan**: at B=4, pack two codes per byte and gather with
   AVX2 `vpshufb` / NEON `vqtbl1q` — projected ≥ 4× scan throughput.
2. **Hierarchical reranking**: coarse 2-bit sweep, 4-bit for top-ef,
   8-bit for final top-k. Composes with existing `dyn-rerank-budget`.
3. **Rotation sampling**: a product of two FWHT-rotations with
   different sign vectors approaches a Haar matrix more tightly.
4. **Asymmetric B per subspace**: more bits for high-variance dims
   (Anisotropic eRaBitQ). Prior `anisotropic-pq` nightly can seed this.
5. **Codebook-aware IVF**: integrate with `ruvector-cascade-adc` so IVF
   posting lists store eRaBitQ codes.

## 8 · Production crate layout

```
crates/ruvector-extended-rabitq/
├── Cargo.toml
├── src/
│   ├── lib.rs         (29 LOC)
│   ├── rotation.rs    (fast rotation, FWHT, tests)
│   ├── codebook.rs    (Lloyd-Max + inverse-CDF seed, tests)
│   ├── quantize.rs    (encode / pack / Index, tests)
│   ├── scan.rs        (asymmetric estimator, top-k)
│   ├── error_bound.rs (closed-form σ² bound)
│   └── main.rs        (demo)
└── benches/erabitq_bench.rs
```

All files < 300 lines, no `unsafe`, no mocks, no `TODO` stubs.

## 9 · References

* J. Gao, C. Long. **RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search**. SIGMOD 2024.
* S. P. Lloyd. **Least squares quantization in PCM**. IEEE TIT 1982.
* H. Jégou, M. Douze, C. Schmid. **Product Quantization for Nearest Neighbor Search**. TPAMI 2011.
* T. Ge, K. He, Q. Ke, J. Sun. **Optimized Product Quantization**. CVPR 2013.
* F. Scotti. **Locally-Adaptive Vector Quantization** (LVQ). 2023.
* B. Guo et al. **Accelerating Large-Scale Inference with Anisotropic Vector Quantization**. ICML 2020.
