# ruvector-anisotropic-pq

Anisotropic Product Quantization (APQ) for ruvector: score-aware PQ
codebooks in pure safe Rust, plus a learned orthonormal rotation
(non-parametric OPQ) so PQ subspaces get balanced variance.

Three swappable [`Quantizer`] implementations:

| Variant             | Loss                                | Rotation | Bytes/code |
|---------------------|--------------------------------------|----------|------------|
| `PlainPQ`           | `\|\|r\|\|^2`                        | none     | `m`        |
| `AnisotropicPQ`     | `η·\|\|r_par\|\|^2 + \|\|r_orth\|\|^2` | none   | `m`        |
| `AnisotropicPQR`    | same as APQ                          | learned  | `m`        |

## Quick start

```bash
cargo run --release -p ruvector-anisotropic-pq --bin anisotropic-pq-bench
```

## Design highlights

- No BLAS, no unsafe, deterministic given seed.
- Anisotropic loss splits each residual into components parallel and
  orthogonal to the datapoint direction, upweighting the parallel term by
  `η` — the exact factor Guo et al. (ScaNN, 2020) show governs MIPS recall.
- Rotation is fit from the data covariance via an in-crate Jacobi
  eigendecomposition, then eigenvectors are round-robin binned so each PQ
  subspace has balanced variance.
- All three variants implement the same trait; downstream code (IVF,
  DiskANN, HNSW rerank) can swap backends via generic bounds.

## References

Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector
Quantization*, ICML 2020.
Ge et al., *Optimized Product Quantization*, TPAMI 2013.
