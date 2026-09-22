# ruvector-opq-rotation

Optimized Product Quantization (OPQ) with a learned block-Householder rotation, in pure safe Rust.

## What it does

Plain Product Quantization (PQ) splits a `d`-dim vector into `m` contiguous
sub-vectors and runs k-means per sub-space. When the input distribution has
**uneven variance across dimensions** (essentially every embedding model),
plain PQ pays a big recall tax because one or two sub-spaces dominate error.

OPQ (Ge et al., CVPR 2013) learns an orthogonal rotation `R` that redistributes
variance across sub-spaces *before* encoding. Query time cost is one extra
matmul (~10 µs at d=64) shared across every code in the batch — effectively free.

This crate provides three swappable backends via the `Quantizer` trait:

| Backend             | Rotation                                      | Bytes/code | Notes |
|---------------------|-----------------------------------------------|------------|-------|
| `PlainPq`           | none                                          | `m`        | baseline |
| `RandomRotPq`       | fixed random orthogonal (Haar-like)           | `m`        | free variance smoothing |
| `OpqHouseholder`    | learned via alternating Kabsch / PQ retrain   | `m`        | full OPQ |

Kabsch step is closed-form via polar decomposition of the cross-covariance,
computed as `C · (CᵀC)^{-1/2}` where `(CᵀC)^{-1/2}` comes from a symmetric
Jacobi eigendecomposition — no SVD dependency.

## Benchmark (real numbers, this repo)

Ran on Apple M-series, `cargo run --release --bin opq-benchmark`.
Synthetic Gaussian data with 10× variance skew between first-half and
second-half dimensions.

```
d=64  n=4000  n_q=100  k=10  m=8  ks=256  (8 bytes/code, 32× compression)

PlainPq         recall@10 = 0.309   per_query = 0.07 ms
RandomRotPq     recall@10 = 0.445   per_query = 0.07 ms   (+44% recall)
OpqHouseholder  recall@10 = 0.486   per_query = 0.07 ms   (+57% recall)
```

Training-time cost is one-shot: PlainPq ≈ 275 ms, OpqHouseholder ≈ 1160 ms
(3 outer sweeps × PQ retrain + Kabsch).

## Layout

- `src/lib.rs` — traits, PlainPq, RotatedPq, RandomRotPq, OpqHouseholder,
  Kabsch/Procrustes via Jacobi eigendecomp.
- `src/util.rs` — brute-force ground truth and recall@k helper.
- `src/main.rs` — `opq-benchmark` binary.

## Run

```bash
cargo build --release -p ruvector-opq-rotation
cargo test  --release -p ruvector-opq-rotation
cargo run   --release -p ruvector-opq-rotation --bin opq-benchmark
```

## References

- Ge, He, Ke, Sun. *Optimized Product Quantization*. CVPR 2013.
- Norouzi, Fleet. *Cartesian K-Means*. CVPR 2013.
- Jégou, Douze, Schmid. *Product Quantization for Nearest Neighbor Search*. TPAMI 2011.
- Kabsch. *A solution for the best rotation to relate two sets of vectors*. Acta Cryst. 1976.
