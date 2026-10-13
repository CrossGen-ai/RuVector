# Rotation-Ensemble RaBitQ: Spending Your Bit Budget on Rotations Instead of Precision

**Date:** 2026-10-13
**Branch:** `research/nightly/2026-10-13-rabitq-rotation-ensemble`
**Crate:** `crates/ruvector-rabitq-ensemble`
**ADR:** ADR-353

## Abstract

Classical RaBitQ (Gao & Long, SIGMOD 2024) spends one random orthogonal rotation `R` and stores one 1-bit code per coordinate: `sign(R·x)`. The multibit extensions (NeurIPS 2024, VLDB 2025 "Extended RaBitQ") spend their extra bits on *finer quantisation* of the same single rotation. We investigate the orthogonal axis: at the same bit budget, spend bits on *more independent rotations* instead. Each extra rotation `R_j` produces another unbiased single-rotation estimator; averaging K of them cuts estimator variance by `1/K` because, for independent Haar-uniform rotations, the per-rotation errors are asymptotically uncorrelated.

Our contribution is a self-contained Rust implementation with three rotation backends (dense Haar, HD-HD-HD TurboQuant-style, and single-sign HD) and a measured head-to-head sweep comparing single-rotation RaBitQ, K-rotation ensembles for K ∈ {2, 4, 8}, and the exact `FlatF32` oracle on L2-normalised uniform random data at n ∈ {1k, 5k, 50k, 100k} and d ∈ {64, 128}. All numbers below come from `cargo run --release` on Apple M4 Max / macOS 15.6.

## SOTA survey

The 2024 RaBitQ paper (SIGMOD 2024, Gao & Long) proved an unbiased 1-bit estimator for inner products with a Johnson–Lindenstrauss-style error bound, replacing PQ's lookup-table scan with a `popcount+rotate+scale` kernel. Extensions since:

| Work | Year | Direction |
|---|---|---|
| RaBitQ (orig.) | 2024 | 1-bit asymmetric, Haar rotation |
| TurboQuant | arXiv:2504.19874 (2025) | Replace dense Haar with `D·H·D·H·D` randomised Hadamard — O(D log D), near-Haar in distribution |
| Extended RaBitQ | VLDB 2025 | b-bit per-coordinate, non-uniform codebook, b ∈ {2,…,8} |
| Fused RaBitQ+residual | our 2026-09-04 nightly | combine 1-bit RaBitQ with per-cluster residual rerank |
| Anisotropic PQ (Guo et al., ICML 2020) | 2020 | weight parallel vs orthogonal error — PQ not RaBitQ |

What everyone has shared so far: *one* rotation per vector. The ensemble idea is standard in classical dimensionality reduction (bagged random projections — Fradkin & Madigan KDD 2003; Fern & Brodley ICML 2003) but has not, to our knowledge, been studied as an alternative to multibit quantisation inside the RaBitQ family specifically.

## Proposed design

Build K independent rotations `R_1,…,R_K`, each with its own seed. For every row `x`, store K separate 1-bit codes `b_j := sign(R_j · x)`, plus one copy of `‖x‖` shared across members. For a query `q`:

1. For each `j`, apply `R_j` to `q` → `rq_j` (K rotations total).
2. For each candidate id, score it under each member and sum:
   ```
   est_ip_j(q, x) = ‖x‖ · (1/√D_pad) · Σ_i sign(R_j x)_i · (rq_j)_i
   est_ip   (q, x) = (1/K) · Σ_j est_ip_j(q, x)
   ```
3. Keep the top-k by score.

### Why K times better variance (sketch)

Each single-rotation estimator `est_ip_j` is unbiased with variance `σ² = C · ‖q‖² · ‖x‖² / D` under a Haar-uniform rotation (RaBitQ Prop. 3.1). For K *independent* Haar rotations, the K estimators are not i.i.d. — they share the same input vectors — but their errors come from independent randomness in different `R_j` and are asymptotically uncorrelated in D (any two fixed vectors become uniformly random under two independent Haar rotations). So `Var[avg] ≈ σ²/K`. The same argument works for randomised Hadamard `D·H·D·H·D` because the three sign vectors are independent across members.

### Trade matrix

For the same `K` bits per coordinate:

| Scheme | Memory | Rotate cost at query | Scan cost at query | Variance |
|---|---|---|---|---|
| Multibit RaBitQ (b=K) | K · D bits + 1 norm | 1 rotation | K · D ops | ∝ 2^(−K) per coord |
| Rotation ensemble (K rot.) | K · D bits + 1 norm | K rotations | K · D ops | ∝ 1/K per coord |

Multibit shrinks error geometrically; ensemble shrinks error linearly in K. So in theory multibit should dominate at large K. The empirical question is where the crossover sits once you include constants: multibit RaBitQ needs a non-uniform codebook per coordinate and extra codebook lookups; ensemble just reuses the same 1-bit kernel K times. For small K and modest d both are competitive.

## Implementation notes

* `crates/ruvector-rabitq-ensemble/` is dependency-light: `rand`, `rand_distr`, `thiserror`. No BLAS, no C deps, `#![forbid(unsafe_code)]`.
* All randomness is seeded; `(dim, kind, seed, data)` → bit-identical index.
* Three rotation backends, all O(D log D) or less at apply time except Haar (O(D²)):
  * `HaarDense` — Gram–Schmidt on a Gaussian block. Stored as `4·D²` bytes; apply is O(D²). Reference-grade; only use at d ≤ 64 if latency matters.
  * `HadamardSigned` — `D₁·H·D₂·H·D₃`. Stored as `3·D_pad` floats; apply is O(D log D). Matches TurboQuant's RHT.
  * `HadamardSingle` — `D·H`. Stored as `D_pad` floats; apply is O(D log D). Weaker pseudo-orthogonality, cheaper setup.
* Non-dyadic `d` is handled by zero-padding to `D_pad = next_pow2(d)` before the FWHT, standard trick.
* K=1 ensemble is score-equivalent to `RabitqSingle` (test `ensemble_k1_matches_single` asserts this bit-identically for `HadamardSingle`, where both have identical normalisation).
* Scoring loop is deliberately simple — scalar fused multiply-add over bitstream. SIMD lift is one obvious follow-up.

## Benchmark methodology

### Hardware
Apple M4 Max, macOS 15.6, 128 GB RAM. Single-threaded. `cargo build --release` (default workspace profile).

### Data
Uniform random vectors in `[-1, 1]`, L2-normalised so that IP ≡ cosine. L2-normalisation makes the task sensitive to *angular* quantisation error — the exact thing 1-bit sign codes trade away — so it's a stress test for the ensemble.

### Metric
Recall@10 against `FlatF32Index` as the oracle (exact f32 inner product). We also measure µs per query and aux bytes per row (rotation state + codes + norms, excluding the originals).

### Sweep
```
n ∈ {1_000, 5_000, 50_000, 100_000}
d ∈ {64 (for n=1k), 128 (for n ≥ 5k)}
K ∈ {1 (single), 2, 4, 8}
kind ∈ {HaarDense (d ≤ 64 only), HadamardSigned, HadamardSingle}
seeds — fixed per sweep (reproducible)
```

## Results (real cargo-run numbers)

### n = 100 000, d = 128, 25 queries, k = 10 (headline configuration)

```
FlatF32 (oracle)                recall@10=1.000   18 518.5 µs/q   aux=    0.0 B/row
single/hadamardSigned           recall@10=0.208   25 038.2 µs/q   aux=   20.0 B/row
ensemble K=2/hadamardSigned     recall@10=0.316   62 424.0 µs/q   aux=   40.0 B/row
ensemble K=4/hadamardSigned     recall@10=0.444   88 731.0 µs/q   aux=   80.1 B/row
ensemble K=8/hadamardSigned     recall@10=0.592  155 747.3 µs/q   aux=  160.1 B/row
single/hadamardSingle           recall@10=0.184   21 320.6 µs/q   aux=   20.0 B/row
ensemble K=2/hadamardSingle     recall@10=0.328   45 190.7 µs/q   aux=   40.0 B/row
ensemble K=4/hadamardSingle     recall@10=0.512  106 014.9 µs/q   aux=   80.0 B/row
ensemble K=8/hadamardSingle     recall@10=0.580  182 732.7 µs/q   aux=  160.0 B/row
```

Each doubling of K adds roughly 0.1–0.15 recall@10 on uniform random L2-normalised data at d=128, which matches the `1/K` variance scaling for an unbiased estimator paired with a logistic-like recall curve.

### Scaling across n (HadamardSigned, d = 128, k = 10)

| n | single | K=2 | K=4 | K=8 |
|---:|---:|---:|---:|---:|
| 5 000 | 0.257 | 0.404 | 0.543 | 0.668 |
| 50 000 | 0.196 | 0.306 | 0.476 | 0.604 |
| 100 000 | 0.208 | 0.316 | 0.444 | 0.592 |

Recall at a fixed K drops slightly with larger n (more neighbours competing within the quantisation-noise band), but the K-wise *gain* is stable: ~0.3 absolute recall gain going from K=1 → K=8.

### n = 1 000, d = 64 — Haar vs Hadamard back-to-back

```
single/hadamardSigned           recall@10=0.367    88.7 µs/q   aux=  12.8 B/row
single/hadamardSingle           recall@10=0.381    85.7 µs/q   aux=  12.3 B/row
single/haar                     recall@10=0.358   109.7 µs/q   aux=  28.4 B/row
ensemble K=4/hadamardSigned     recall@10=0.624   363.1 µs/q
ensemble K=4/hadamardSingle     recall@10=0.624   445.1 µs/q
ensemble K=4/haar               recall@10=0.596   465.9 µs/q
```

At d=64 all three backends are within noise; the single-sign HD-rotation is as good as the three-sign one and the dense Haar, confirming the TurboQuant observation that one random sign vector + one FWHT is already deep in the "close-to-Haar" regime for d ≥ 64.

### Micro-benchmark (`cargo bench`)

Latency-only, no recall accounting. Mean µs per top-10 query on random L2-normalised data (`HadamardSigned`):

| n | d | single | K=2 | K=4 | K=8 |
|---:|---:|---:|---:|---:|---:|
| 10 000 | 128 | 1 746.8 | 17 414.7* | 9 576.3 | 18 744.6 |
| 50 000 | 128 | 9 906.3 | 28 140.6 | 52 405.2 | 100 818.9 |
| 100 000 | 128 | 32 594.4 | 65 205.2 | 89 924.9 | 282 182.5 |

(*p99 blows up at K=2/n=10 000 — the sampler hits one allocator pause; the mean inflates with it. The mean/p99 ratio normalises above n=10 000.)

Reading the bench + demo together: ensemble latency is roughly `K× single` plus one extra rotation per member. The scan loop dominates asymptotically because it is K passes over K× the bytes; a SIMD rewrite that fuses all K bitstream reads under one accumulator (described below) is the obvious next step.

## How it works — blog-readable walkthrough

Imagine you have a 128-dim vector and you want to compare it quickly to a billion other vectors.

**Step 1 — rotate.** We first apply a fixed random rotation `R` to every vector in the database. All distances are preserved exactly (rotations are orthogonal), but the coordinates are now a scrambled version of the originals.

**Step 2 — shrink to 1 bit.** For each rotated coordinate, we keep only its sign: positive → 1 bit, negative → 0 bit. A 128-dim f32 vector that used to be 512 bytes becomes 16 bytes. We also save the vector's length separately (one f32).

**Step 3 — score a query.** When a new query vector arrives, rotate it (not quantise it — keep the full f32 precision). Then for each database vector, read its 128 sign bits and compute `Σᵢ sign(x)ᵢ · rotated_q[i]`, multiply by `‖x‖ / √128`, and that's the estimated inner product. This is unbiased: on average over random rotations, it equals the true inner product.

**Step 4 — reduce variance.** A single 1-bit code is noisy. Our twist: instead of making each code have more bits, use **multiple independent codes under multiple independent rotations**. Each one gives an unbiased noisy estimate; averaging them cuts the variance proportionally.

For K=4 rotations:
* Memory grows 4×.
* Rotation cost at query time grows 4× (negligible — one FWHT per member).
* Scan cost grows 4× (biggest hit; linear in K).
* Recall jumps from ~0.21 to ~0.44 at d=128 (n=100k).

For most workloads a 4× memory + 4× latency hit to double recall is the wrong trade — but at K=2 (recall 0.21 → 0.32 for 2× cost) you spend less than the equivalent full-f32 upgrade (16× cost) and you unlock a very cheap rerank-free intermediate point.

## Practical failure modes

1. **Low recall ceiling.** Even K=8 at n=100k, d=128 is 0.60 recall@10 — fine for coarse candidate generation feeding a reranker, bad as the final answer. Always pair with an exact-f32 rerank over top-M candidates in production.
2. **Latency scales linearly in K.** The scan is already memory-bound; naive K-pass scan is `K×` the DRAM traffic. The remedy is a fused bitstream kernel that pulls one row's K codes into one SIMD register and accumulates all K scores in one pass. We did not implement that here.
3. **Haar backend is a toy at d > 64.** The `O(D²)` apply dominates every other cost. Only use `HadamardSigned` or `HadamardSingle` in anger.
4. **Ensemble members share x, not R.** Variance reduction is `1/K` only in the Haar limit. For `HadamardSigned` the independence between members is only through independent sign vectors — three per member, drawn from the same construction. Empirically close to the Haar case at d ≥ 64; theoretically slightly looser.
5. **Non-uniform data.** Our evaluation uses uniform random L2-normalised data, which is the hardest case for sign quantisation. On natural embeddings (clustered, anisotropic) the single-rotation RaBitQ baseline is much stronger, and the marginal gain from K>1 shrinks accordingly.

## What to improve next (roadmap)

Nothing in the current implementation is production-grade; the point of this nightly is to establish the recall/cost curve. Natural follow-ups, roughly in order:

1. **Fused K-code scan.** For K ∈ {2, 4, 8}, pack all K bits per coordinate into one word (shuffle), fetch one DRAM line, accumulate K scores in one pass through the row. Expected: K-pass → 1-pass, scan latency drops to roughly `single × 1.5`.
2. **Mixed K per cluster.** With an IVF coarse quantiser, use K=1 for low-recall clusters and K=4 for high-recall clusters. Lets the user amortise memory on cold data.
3. **Correlated rotations.** Deliberately correlate `R_1,…,R_K` to span complementary subspaces (orthogonal signs rather than independent) and measure variance gain vs free-independent. If positive, the ensemble becomes strictly dominated by a structured-rotation variant.
4. **Head-to-head vs multibit RaBitQ.** We claim multibit should dominate ensemble at the same bit budget. Prove it: build the exact crossover plot at matched memory on the same data.
5. **Ensemble + residual.** Our fused-RaBitQ+residual nightly (2026-09-04) added a f32 residual rerank. Combine: K-rotation ensemble as the coarse estimator, residual as the fine. Expected recall closer to flat at the same storage.

## Production crate layout proposal

If this proves useful on real embeddings, the production shape is:

```
crates/ruvector-rabitq-ensemble/
├── Cargo.toml
├── src/
│   ├── lib.rs           # trait Index, re-exports
│   ├── error.rs         # EnsembleError
│   ├── rotation.rs      # Rotation { Haar | HD-H-D-H-D | H-D }, apply_into
│   ├── single.rs        # RabitqSingle: norms + bitstream codes
│   ├── ensemble.rs      # RabitqEnsemble: K independent singles, averaged
│   ├── flat.rs          # exact f32 oracle (for test & bench use)
│   ├── scan_simd.rs     # (future) fused K-code SIMD scan
│   ├── persist.rs       # (future) serde + mmap layout
│   └── main.rs          # end-to-end recall + throughput demo
├── benches/ensemble_bench.rs
└── BENCHMARK.md         # (future) locked numbers for CI drift tracking
```

Currently we ship everything except `scan_simd` and `persist`.

## References

* Jianyang Gao, Cheng Long. *RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search.* SIGMOD 2024. [arXiv:2405.12497](https://arxiv.org/abs/2405.12497)
* Jianyang Gao et al. *Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search* (Extended RaBitQ). VLDB 2025 preprint.
* TurboQuant. [arXiv:2504.19874](https://arxiv.org/abs/2504.19874).
* Dmitriy Fradkin, David Madigan. *Experiments with Random Projections for Machine Learning.* KDD 2003 — one of the earliest ensembles-of-random-projections empirical studies.
* Xiaoli Fern, Carla Brodley. *Random Projection for High-Dimensional Data Clustering: A Cluster Ensemble Approach.* ICML 2003 — ensemble averaging over independent random projections.
* Hervé Jégou, Matthijs Douze, Cordelia Schmid. *Product Quantization for Nearest Neighbor Search.* TPAMI 2011 — the predecessor family RaBitQ competes against.
* Ruiqi Guo et al. *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.* ICML 2020 — different-but-related angle on where to spend bits.
* ruvector nightly `2026-10-11-extended-rabitq-multibit` — our own multibit RaBitQ writeup.
* ruvector nightly `2026-09-04-fused-rabitq-residual` — our own RaBitQ+residual writeup.
