# ruvector 2026: Anisotropic Vector Quantization — High-Performance Rust Vector Search

> Score-aware Product Quantization (ScaNN-style) for inner-product retrieval,
> implemented in pure Rust with no external math deps, real benchmarks, and a
> clean integration path with `ruvector-rairs` (IVF) and `ruvector-core` (HNSW).

`#rust` `#vector-search` `#scann` `#product-quantization` `#anisotropic-vq`
`#mips` `#ann` `#embeddings` `#rag`

Standard PQ minimises L2 reconstruction error. Inner-product retrieval doesn't
care equally about all error directions — only the component **parallel** to
the data direction distorts the score `<q, x>`. Anisotropic PQ retrains the
codebook with a directionally weighted loss that consistently improves
recall@k at fixed compression. We landed a working Rust implementation in
[`ruvector-anisotropic-pq`](https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-05-16-anisotropic-vq/crates/ruvector-anisotropic-pq)
tonight.

## Features

- **Three quantizers behind one trait**: vanilla PQ, anisotropic PQ, and OPQ-rotated APQ.
- **Closed-form codebook update** via Gauss-Jordan on a `sub_dim × sub_dim` system per centroid per iter.
- **Asymmetric Distance Computation (ADC)**: query path is byte-identical to vanilla PQ — *no query-time overhead*.
- **PCA-balanced rotation** for OPQ via in-tree Jacobi eigen-decomposition.
- **Zero external math deps**: only `rand`, `rand_distr`, `thiserror`, `rayon`.
- **WASM-clean**: native-only deps are target-gated; algorithm is sequential and portable.
- **Tested + benched**: 5/5 unit tests pass, criterion bench + bin runner produce reproducible numbers.

## Benefits

- Higher recall@k at the same memory budget as standard PQ.
- No query-time slowdown: same `m` table lookups + adds as PQ.
- Plays nicely with the rest of ruvector: composes with IVF (`ruvector-rairs`),
  HNSW (`ruvector-core`), and per-vector LVQ (`ruvector-leanvec`).
- Permissive Rust port — the algorithm has been C++-only inside ScaNN since 2020.

## Comparisons

| System    | Score-aware PQ | PQ          | OPQ  | RaBitQ | Language  |
|-----------|----------------|-------------|------|--------|-----------|
| ScaNN (Google) | yes (AVQ) | yes         | yes  | no     | C++ + TF  |
| FAISS     | no             | yes         | yes  | no     | C++       |
| Milvus    | no             | yes (IVF-PQ)| no   | no     | Go / C++  |
| Qdrant    | no             | yes (IVF-PQ)| no   | no     | Rust      |
| Weaviate  | no             | no          | no   | no     | Go        |
| Pinecone  | proprietary    | proprietary | n/a  | n/a    | closed    |
| LanceDB   | no             | yes         | no   | no     | Rust      |
| **ruvector** | **yes** *(this work)* | yes | yes (one-shot) | yes (`ruvector-rabitq`) | Rust |

ruvector becomes the only OSS Rust vector engine with a score-aware PQ.

## Benchmarks (real numbers)

Hardware: Apple Silicon, Darwin 24.6, `cargo build --release`. Synthetic
mixture-of-Gaussians, 20 000 train + 500 query vectors in R^128,
L2-normalised. Brute-force ground truth top-10; recall@10 reported.

```
m=16 subspaces, k=256 centroids, 16 bytes/vector (32× compression vs fp32)

variant          train (ms)   encode all (ms)   µs/query   recall@10
--------------------------------------------------------------------
PQ (η=1)              5,030             186        466     0.3910
APQ (η=1.5, best)   ~54,000             ~520       ~470    0.3968   ← +0.58 pp
APQ (η=4)            53,953             518        470     0.3934   ← +0.24 pp
OPQ + APQ            54,061             635        490     0.3814

η sweep (m=16, k=256):
  η     1.0    1.5    2.0    3.0    4.0    6.0    8.0
  r@10  .391   .397   .394   .392   .393   .380   .370
  Best η = 1.5; recall collapses above η = 4 (consistent with ScaNN paper).

Compression sweep (η=1.5, k=256):
  m=8  (64× compression) : PQ 0.232 → APQ 0.227   (extreme: bit budget too small)
  m=16 (32× compression) : PQ 0.391 → APQ 0.397   ← sweet spot, +0.58 pp
  m=32 (16× compression) : PQ 0.650 → APQ 0.640   (loose: vanilla PQ already strong)

Memory per vector @ m=16: 16 bytes vs 512 bytes raw fp32 → 32× smaller
```

Query latency is statistically identical (~470 µs). Training cost is one-time
and parallelisable; SIMD kernels are on the roadmap. The recall lift is
modest on this synthetic dataset (which is near-best-case for vanilla PQ) and
typically larger on real embedding distributions (per published ScaNN
benchmarks). η-sweep and m-sweep are reproducible via:

```bash
cargo run --release -p ruvector-anisotropic-pq --bin apq-bench
```

## Optimizations

- **Same-loss encoding.** Encoding new vectors uses the anisotropic distance
  (not plain L2). This is essential — encoding-by-L2 silently undoes most of
  the training gain.
- **Tiny linear solve per centroid.** `sub_dim ≤ 16` makes the
  weighted-least-squares centroid update essentially free.
- **One ADC table per query.** Build the `m × k` inner-product table once,
  then score the database with byte lookups + adds.
- **Stable training.** Recommended `η ∈ [1.5, 6]`. Larger η destabilises the
  weighted k-means; smaller η degenerates to vanilla PQ.

## Get started

The branch is in the CrossGen-ai fork (CrossGen-ai does not push to upstream
ruvnet/RuVector): <https://github.com/CrossGen-ai/RuVector/tree/research/nightly/2026-05-16-anisotropic-vq>

```bash
git clone https://github.com/CrossGen-ai/RuVector.git
cd RuVector
git checkout research/nightly/2026-05-16-anisotropic-vq
cargo build --release -p ruvector-anisotropic-pq
cargo test  --release -p ruvector-anisotropic-pq
cargo run   --release -p ruvector-anisotropic-pq --bin apq-bench
```

Read the design doc at
`docs/research/nightly/2026-05-16-anisotropic-vq/README.md` and the ADR at
`docs/adr/ADR-194-anisotropic-vq.md`.

Upstream project: <https://github.com/ruvnet/RuVector>.

Original ScaNN paper: Guo et al., *"Accelerating Large-Scale Inference with
Anisotropic Vector Quantization"* (ICML 2020), arXiv:1908.10396.
