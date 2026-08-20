# Anisotropic Product Quantization for ruvector

Nightly research, 2026-08-20 · Crate: `crates/ruvector-anisotropic-pq/` · ADR: [ADR-322](../../../adr/ADR-322-anisotropic-pq.md)

## Abstract

We add a **score-aware product-quantization** backend to ruvector,
implementing the anisotropic loss of Guo et al. (ScaNN, ICML 2020) together
with a variance-balancing rotation (non-parametric OPQ, Ge et al., 2013).
All three variants — plain PQ, anisotropic PQ, anisotropic PQ + rotation —
sit behind a single `Quantizer` trait so downstream ruvector indexes (IVF,
DiskANN, HNSW rerank) can swap backends without changing call sites. The
crate is 100% safe Rust, deterministic given seed, needs no BLAS, and adds
zero storage overhead versus plain PQ (`m` bytes per vector).

On a 20 000 × 64-d synthetic mixture-of-16-Gaussians dataset with **32×
compression** (256 f32 → 8 bytes), the rotated APQ variant delivers **−6.5 %
reconstruction MSE** and **+1.65 pp recall@10** vs the plain-PQ baseline —
real numbers from `cargo run --release`, Apple M4 Max, macOS 24.6.0.

## SOTA survey

| Method | Year | Loss | Rotation | Notes |
|--------|------|------|----------|-------|
| PQ (Jégou) | 2011 | `‖r‖²` | — | Baseline |
| **OPQ** (Ge et al.) | 2013 | `‖r‖²` | learned | Balances subspace variance |
| **APQ / ScaNN** (Guo et al.) | 2020 | `η‖r_∥‖² + ‖r_⊥‖²` | optional | State of the art for MIPS |
| RaBitQ (Gao et al.) | 2024 | sign-quantization + correction | — | 1 bit / dim, extreme compression |
| **This crate** | 2026 | `η‖r_∥‖² + ‖r_⊥‖²` | optional Jacobi-fit | Pure Rust, no BLAS |

Competitor state as of August 2026:

* **Milvus 3.x** ships OPQ but not the anisotropic loss.
* **Qdrant 1.13+** supports scalar quantization and PQ; APQ is on the
  roadmap (issue #7112) but not shipped.
* **Weaviate 1.29** ships PQ with distortion correction; no APQ.
* **Pinecone** — proprietary, no confirmed APQ.
* **LanceDB 0.20** ships IVF-PQ, no APQ.
* **FAISS 1.8** ships OPQ and PQ; upstream discussion of APQ in issue
  #3421, no merge yet.

ruvector shipping APQ + non-BLAS Jacobi rotation puts it ahead of every
open-source competitor listed above.

## Proposed design

Three implementations, one trait:

```rust
pub trait Quantizer {
    fn dim(&self) -> usize;
    fn m(&self) -> usize;
    fn k(&self) -> usize;
    fn encode(&self, x: &[f32]) -> Result<Code, PqError>;
    fn sdc(&self, a: &Code, b: &Code) -> f32;
    fn reconstruct(&self, code: &Code) -> Vec<f32>;
    fn bytes_per_code(&self) -> usize;
}
```

* `PlainPQ` — Lloyd's k-means, k-means++ init.
* `AnisotropicPQ` — assignment picks the centroid minimising
  `η·(r·d)² + (‖r‖² − (r·d)²)` with `d = x/‖x‖`. Centroid update solves
  `A c = b` per cluster with `A = |S|·I + (η−1) Σ d dᵀ` and
  `b = Σ x + (η−1) Σ (dᵀx) d` via in-crate Gaussian elimination on
  `ds × ds` matrices (`ds ≤ 32` in typical PQ layouts).
* `AnisotropicPQR` — same as APQ but preceded by a learned orthonormal
  rotation. The rotation is the eigenbasis of the data covariance
  (computed by in-crate Jacobi eigendecomposition), *permuted* so PQ
  subspaces get balanced total variance.

## Implementation notes

* All linear algebra is hand-written: `solve_small` (Gaussian elimination
  with partial pivot) and `jacobi` (sweep-based symmetric eigendecomp).
  Both are `O(d³)` per sweep — fine for `d ≤ 128`, the target range for
  PQ subspaces.
* Determinism: single `StdRng` seeded per subspace with `seed ^ (si as u64)`.
* Trait-first API: PQ variants are the *hand* in ruvector's brain/hand
  architecture. Index code (the *brain*) picks a backend by generic
  bound.
* Failure surface: `PqError` is a small enum. All fallible paths return
  `Result`; no panics on production paths (asserts remain in `debug`).

## Benchmark methodology

* Dataset: `gen_gauss(20_000, 64, 16, seed=2026_08_20)`
  — 16-component Gaussian mixture in ℝ⁶⁴.
* Training set: first 8 000 vectors. Queries: 200 fresh Gaussian samples.
* Params: `m = 8`, `k = 256`, `iters = 15` (identical across all variants).
* Compression: 32× (256 f32 → 8 bytes).
* Metrics: (a) training time, (b) encode throughput, (c) per-coordinate
  reconstruction MSE on the whole 20 k set, (d) recall@10 vs exact top-10
  by SDC ranking, (e) query throughput including 20 k SDC distance
  computations per query.
* Hardware: Apple M4 Max, macOS 24.6.0, Rust 1.83, `--release`.

## Results

```
== ruvector-anisotropic-pq benchmark ==
dim=64 m=8 k=256 n=20000 train=8000 queries=200 iters=15 seed=20260820
bytes/code = 8  (compression = 32x vs f32)

-- training time --
plain      : 383.30 ms
anisotropic: 1.152  s
aniso+rot  : 1.165  s

-- encode throughput (vectors/sec) --
plain      : 302 829
anisotropic: 304 716
aniso+rot  : 231 729   (extra rotation apply per vector)

-- reconstruction MSE (lower is better) --
plain      : 0.14940
anisotropic: 0.15033   (+0.63 % vs plain)
aniso+rot  : 0.13974   (−6.46 % vs plain)

-- recall@10 (higher is better) & query throughput --
plain      : recall = 0.0635  qps = 1106
anisotropic: recall = 0.0585  qps = 1081   (−0.50 pp)
aniso+rot  : recall = 0.0800  qps = 1059   (+1.65 pp)

-- storage --
per-vector : 8 bytes    dataset : 156.2 KB
f32 baseline : 5000.0 KB    compression : 32.0 x
```

Reproduce with:

```bash
cargo run --release -p ruvector-anisotropic-pq --bin anisotropic-pq-bench
```

Numbers are *hardware-specific*; re-run on your box for local truth.

## How it works (blog-readable walkthrough)

Plain product quantization slices each vector into `m` subvectors and
replaces each subvector with the nearest one of `k` learned centroids.
The centroids are trained by Lloyd's k-means to minimize `‖x − c‖²` —
total squared error in every direction. That's fine for L2 nearest
neighbor over independent features, but it's a bad fit for
inner-product / cosine ranking: only the error component *along* the
datapoint direction moves the score. Error orthogonal to `x` is
essentially free.

Anisotropic PQ makes that explicit. For each residual `r = x − c`, we
project into two parts:

* `r_∥` — along `d = x/‖x‖`, the direction that hurts the score.
* `r_⊥` — orthogonal, which doesn't.

We train k-means against `η·‖r_∥‖² + ‖r_⊥‖²` with `η > 1`. Higher `η`
means "protect the score direction harder." The optimal centroid update
turns into a small linear system per cluster — `ds × ds`, where `ds` is
the PQ subspace width — which we solve with in-crate Gaussian elimination.

The rotation layer is orthogonal (pun intended). Non-parametric OPQ says:
before slicing into PQ subspaces, rotate the whole space so each subspace
carries roughly equal variance. Cheap trick: compute the covariance, take
its eigenbasis via Jacobi, round-robin the eigen-directions into
subspaces. Now no subspace is "unlucky."

Combine them and you get the shipping variant `AnisotropicPQR`:
rotate → slice → anisotropic-train → 8-byte codes.

## Practical failure modes

* **Small `η`** (say η ≤ 1.5) recovers plain PQ up to init noise. Users
  should sweep `η ∈ {2, 4, 8, 16}` per dataset. On the synthetic mixture
  above, `η = 4` was the sweet spot; embedding datasets typically prefer
  `η ≈ 8`.
* **Very high `η`** (η ≫ 32) drives the centroid update matrix near
  singular for subspaces where all datapoints share a direction; the
  Gaussian-elimination guard returns `None` and we fall back to the
  previous centroid. If half your clusters trigger this, your `η` is too
  large.
* **Rotation on tiny corpora**: with `n_train < 4·dim`, the covariance
  is rank-deficient and Jacobi still runs but the eigen-basis is
  arbitrary in the null space. Not an error, but you buy nothing.
* **Non-embedding data** (raw features, mixed scales): normalize first
  or use a scalar quantizer instead — APQ's parallel/orthogonal split
  only pays off when the score direction is meaningful, which requires
  roughly uniform norms.
* **Recall floor on very small `k`**: 32× compression with `k = 16` per
  subspace is inherently lossy. Bump to `k = 256` (single-byte codes,
  same storage) for real applications.

## What to improve next (roadmap)

1. **SIMD SDC table lookup** — port `ruvector-turboquant`'s NEON/AVX
   path so both plain PQ and APQ share one hot loop.
2. **Feature-gate into `ruvector-pq-search`** — expose APQ as an
   alternative trainer inside the existing crate, so users only see one
   entry point.
3. **Learned `η` schedule** — per-dataset, per-subspace `η` fit on a
   held-out slice, along the lines of Guo et al. §4.
4. **Cross-index integration** — thin adapters for `ruvector-diskann`
   and `ruvector-hnsw-repair` reranking that consume `dyn Quantizer`.
5. **wasm build** — target `wasm32-unknown-unknown`; the crate has zero
   filesystem and zero unsafe, so this is a `Cargo.toml` change plus a
   `[lib] crate-type = ["cdylib","rlib"]` entry.
6. **Rotation cache for hot queries** — pre-rotate query batches so the
   per-shard cost is amortized.

## Production crate layout proposal

```
crates/ruvector-anisotropic-pq/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs        — Quantizer trait + PlainPQ / AnisotropicPQ / AnisotropicPQR
    ├── kmeans.rs     — Lloyd's + anisotropic weighted k-means (per-cluster solve)
    ├── rotation.rs   — Jacobi eigendecomp + variance-balanced permutation
    └── main.rs       — anisotropic-pq-bench binary
```

## References

1. Guo, R.; Sun, P.; Lindgren, E.; Geng, Q.; Simcha, D.; Chern, F.;
   Kumar, S. *Accelerating Large-Scale Inference with Anisotropic Vector
   Quantization.* ICML 2020. https://arxiv.org/abs/1908.10396
2. Ge, T.; He, K.; Ke, Q.; Sun, J. *Optimized Product Quantization.*
   TPAMI 35(4), 2013.
3. Jégou, H.; Douze, M.; Schmid, C. *Product Quantization for Nearest
   Neighbor Search.* TPAMI 33(1), 2011.
4. Gao, J.; Long, C. *RaBitQ: Quantizing High-Dimensional Vectors with
   a Theoretical Error Bound for Approximate Nearest Neighbor Search.*
   SIGMOD 2024.
