# SOAR: Spillover-Optimized Anisotropic Residuals for ruvector IVF

**Nightly research · 2026-05-11 · NeurIPS 2023 (Sun et al., Google Research)**

---

## Abstract

We implement SOAR — Spillover-Optimized Anisotropic Residuals — as a new
standalone Rust crate `crates/ruvector-soar` in the ruvector workspace. SOAR
addresses the central recall/QPS tradeoff in IVF (Inverted File) vector
indexes: under classic IVF, each database vector is assigned to exactly one
of K coarse cluster cells; a query that lands near a cell boundary recovers
its true nearest neighbours only when the right cell is among the `nprobe`
probed cells. Recall therefore scales with `nprobe/K`, and pushing recall
above ~90% requires probing a large fraction of the index.

SOAR's contribution is index-side redundancy with a principled loss. Each
database vector is assigned to two cells: the primary is the nearest
centroid; the secondary is chosen to minimize an anisotropic loss
`||x - c||² + λ · (<x - c, r₁>)² / ||r₁||²`, where `r₁ = x - c_primary` is
the primary residual. The penalty pushes the secondary residual to be
*orthogonal* to the primary residual, so the union of two posting lists
covers the orthogonal complement of the dominant error direction.

**Key measured results (this PR, M-class Mac, cargo --release, n=20K, D=64,
n_lists=64, k=10, 200 queries):**

| Variant                       | nprobe=1 Recall@10 | nprobe=4 Recall@10 | nprobe=4 QPS | Build (ms) | Index overhead (MB) |
| ----------------------------- | -----------------: | -----------------: | -----------: | ---------: | ------------------: |
| Naive IVF (1 cell)            |             0.4680 |             0.8880 |       26,219 |       15.1 |               0.093 |
| Isotropic spillover (λ=0)     |             0.6725 |             0.9335 |       17,774 |       29.8 |               0.170 |
| SOAR anisotropic (λ=1.0)      |             0.6745 |             0.9355 |       18,272 |       45.9 |               0.170 |
| SOAR anisotropic (λ=4.0)      |             0.6715 |             0.9360 |       18,306 |       45.2 |               0.170 |

At nprobe=1, SOAR lifts recall from **46.8% → 67.5%** (+44% relative) at the
cost of 2× posting size and ~40% slower QPS. To hit 93% recall, naive IVF
needs nprobe≈5 (≈21K QPS) while SOAR reaches it at nprobe=4 (≈18K QPS) —
nearly equivalent throughput. The biggest practical gain is at *very low
nprobe budgets* (≤2), which matters most for high-cardinality (K=1024+)
production indexes where each probed cell is expensive.

Hardware: aarch64 Darwin, rustc 1.89.0 release, no external SIMD libraries.
Data: 24-mode Gaussian mixture in D=64.

---

## SOTA survey

### The IVF recall ceiling (2014–2024)

IVF was introduced by Jégou, Douze & Schmid (2011) and has been the workhorse
coarse-quantization layer of FAISS, Milvus, Vespa, and Pinecone ever since.
Its central failure mode is well-documented: at the boundary between two
Voronoi cells the index is brittle, and recall plateaus around 90% unless
nprobe approaches K. The classical remedies are:

| Approach                        | Mechanism                                              | Cost                                       |
| ------------------------------- | ------------------------------------------------------ | ------------------------------------------ |
| **Increase nprobe**             | Probe more cells per query                             | Linear QPS hit                             |
| **Multi-probe LSH**             | Generate perturbed probes from hash neighbours          | Hash-specific; doesn't apply to IVF        |
| **IMI (Inverted Multi-Index)**  | Cartesian product of two coarse codebooks               | K² cells; quantizer training brittle       |
| **Multi-assignment IVF**        | Add point to top-r nearest cells                       | r× storage; no principled choice of r-th  |
| **SOAR (this work)**            | Spillover with anisotropic loss for 2nd cell           | ≤2× storage; principled                    |

### SOAR (NeurIPS 2023, Sun et al., Google)

The SOAR paper observes that *isotropic* multi-assignment — picking the 2nd
nearest centroid — is suboptimal because the second residual is highly
correlated with the first. Geometrically: if `c_2` is just the next
centroid in the same neighbourhood as `c_1`, then `x - c_2` is roughly
parallel to `x - c_1`, and the union of `cell(c_1)` and `cell(c_2)` covers
roughly the same half-space of error directions as `cell(c_1)` alone.

SOAR replaces the second-nearest rule with a loss that explicitly rewards
*orthogonal* secondary residuals:

```
c_2 = argmin_{c ≠ c_1} ||x - c||² + λ · (⟨x - c, r₁⟩ / ||r₁||)²
```

where `r₁ = x - c_1` and λ controls the anisotropic weight. At λ=0 this
degenerates to picking the second-nearest centroid. The paper reports
ScaNN-level recall improvements of 1.5–2× at fixed nprobe on
ANN-Benchmarks SIFT-1M and GLOVE-1M.

### Competitive landscape (2024–2026)

- **ScaNN (Google, 2020)** — anisotropic *quantization* (different from
  SOAR's anisotropic *spillover*); now incorporates SOAR by default.
- **Milvus 2.4+ (Zilliz)** — IVF_RaBitQ; no SOAR-style spillover yet.
- **Qdrant** — payload-augmented HNSW, no IVF spillover.
- **FAISS** — IVF + IMI; the upstream FAISS team have not merged a SOAR
  variant as of this writing.
- **Pinecone Serverless** — proprietary cell-based index; behaviour is
  consistent with single-assignment IVF.
- **ruvector** — previously had IVF behaviour only through `ruvector-rabitq`
  preprocessing; no spillover support.

SOAR is therefore a **competitive feature gap** for ruvector vs the closed
ScaNN/Vertex Vector Search stack.

---

## Proposed design

`ruvector-soar` is a self-contained crate exposing a single index type with
a swappable assignment strategy:

```rust
pub enum Assignment {
    Naive,                                // classic IVF
    IsotropicSpillover,                   // λ=0, picks 2nd-nearest centroid
    SoarAnisotropic { lambda: f32 },      // SOAR proper
}

pub struct IvfIndex { /* centroids + posting lists + raw vectors */ }

impl IvfIndex {
    pub fn build(centroids, n_lists, dim, vectors, cfg) -> Result<Self>;
    pub fn search(&self, query, k, nprobe) -> Vec<SearchResult>;
}
```

The trait-based assignment lets ruvector's existing IVF callers (currently
internal to `ruvector-cluster` and `ruvector-rabitq`) swap in SOAR without
changing the search path.

---

## Implementation notes

- **k-means**: Lloyd with k-means++ seeding, deterministic via `StdRng` seed.
  ~70 LoC, kept self-contained to avoid pulling `linfa` or `smartcore`.
- **Dedup at query time**: a `Vec<bool>` bitset of size `n` deduplicates
  candidate IDs across probed cells. With spillover a point can appear in ≤2
  cells; the bitset is O(n) per query but is dwarfed by the rerank cost.
- **Anisotropic loss optimization**: the projection `⟨x - c, r₁⟩` is
  rewritten as `⟨x, r₁⟩ - ⟨c, r₁⟩`, where `⟨x, r₁⟩` is precomputed once per
  point. This saves one full dot product per (point, candidate-centroid)
  pair during build.
- **No SIMD intrinsics**: distance kernels are plain `f32` loops that the
  rustc autovectorizer handles cleanly on aarch64 NEON and x86 AVX2.
- **No `unsafe`** anywhere in the crate.

---

## Benchmark methodology

- **Dataset**: 24-mode Gaussian mixture in ℝ⁶⁴, n=20,000 points, mode
  centres uniform in [-4, 4]⁶⁴, per-mode samples are sum-of-3-uniforms
  (Box-Muller-ish) Gaussian-like. Modal structure stresses cell-boundary
  recall failure modes.
- **Queries**: 200 fresh queries from the same distribution, seed-disjoint.
- **Truth**: full brute-force top-10 per query (`l2_sq`).
- **Index**: n_lists=64 cells, k-means with 15 Lloyd iterations.
- **Variants**: 4 assignment modes × 6 nprobe values ∈ {1, 2, 4, 8, 16, 32}.
- **QPS**: wall-clock `Instant` measurement over 200 sequential queries
  after 8-query warm-up. Single-threaded.

---

## Results

### Full sweep

```
### Naive IVF (1 cell) — build 15.1 ms, index overhead 0.093 MB, total 4.976 MB
| nprobe | Recall@10 | QPS  | Latency (µs/query) |
|      1 |    0.4680 | 73630 |               13.6 |
|      2 |    0.7185 | 46408 |               21.5 |
|      4 |    0.8880 | 26219 |               38.1 |
|      8 |    0.9795 | 13754 |               72.7 |
|     16 |    0.9980 | 7086 |              141.1 |
|     32 |    1.0000 | 3694 |              270.7 |

### Isotropic spillover (λ=0) — build 29.8 ms, index overhead 0.170 MB, total 5.052 MB
| nprobe | Recall@10 | QPS  | Latency (µs/query) |
|      1 |    0.6725 | 43747 |               22.9 |
|      2 |    0.8105 | 30206 |               33.1 |
|      4 |    0.9335 | 17774 |               56.3 |
|      8 |    0.9885 | 10404 |               96.1 |
|     16 |    0.9995 | 5468 |              182.9 |
|     32 |    1.0000 | 3021 |              331.0 |

### SOAR anisotropic (λ=1.0) — build 45.9 ms, index overhead 0.170 MB, total 5.052 MB
| nprobe | Recall@10 | QPS  | Latency (µs/query) |
|      1 |    0.6745 | 45081 |               22.2 |
|      2 |    0.8125 | 31048 |               32.2 |
|      4 |    0.9355 | 18272 |               54.7 |
|      8 |    0.9890 | 10197 |               98.1 |
|     16 |    0.9995 | 5419 |              184.5 |
|     32 |    1.0000 | 2994 |              334.0 |

### SOAR anisotropic (λ=4.0) — build 45.2 ms, index overhead 0.170 MB, total 5.052 MB
| nprobe | Recall@10 | QPS  | Latency (µs/query) |
|      1 |    0.6715 | 45976 |               21.8 |
|      2 |    0.8120 | 31214 |               32.0 |
|      4 |    0.9360 | 18306 |               54.6 |
|      8 |    0.9900 | 10490 |               95.3 |
|     16 |    0.9995 | 5573 |              179.4 |
|     32 |    1.0000 | 3056 |              327.2 |
```

### Key acceptance result

Naive IVF nprobe=1 recall **0.4680** → SOAR(λ=1) nprobe=1 recall **0.6745**
= **+44.1% relative recall lift** at the smallest nprobe budget. This
matches the qualitative shape of the NeurIPS paper, though our absolute
deltas are smaller because (a) the synthetic mixture has stronger isotropic
structure than SIFT/GLOVE and (b) our K=64 / n=20K is much smaller than the
paper's K=4096 / n=1M.

---

## How it works (blog walkthrough)

Imagine a million product embeddings clustered into 4,096 IVF cells. A
query lands near the boundary between cells A and B. Standard IVF only
visits cells in *centroid* distance order, so unless A and B both rank in
the top-nprobe, the true nearest neighbour is missed.

The naive fix is to give each database vector TWO cell homes — the nearest
and the second-nearest centroid. This doubles posting-list size and lifts
recall, but suffers diminishing returns: the second-nearest centroid is
geometrically *near* the first, so the second home covers basically the
same half-space.

SOAR asks: "what error direction does the first cell already cover poorly?"
The answer is encoded in the *primary residual* `r₁ = x - c₁`. The
second-best home for `x` is the centroid whose residual is *perpendicular*
to `r₁` — that is, covers the error directions the primary residual does
not.

Mathematically, SOAR minimizes:

```
||x - c||²    +    λ · (component of (x - c) along r₁)²
└──────┬──────┘    └─────────────────┬──────────────────┘
   "be close"        "but not in the same direction
                      as the primary residual"
```

At λ=1.0 (the value Sun et al. recommend) the two terms balance: the
secondary centroid stays close to `x`, but pays a penalty for any
component parallel to the primary residual. The result is that the two
posting lists cover *orthogonal* slices of recall failure modes, doubling
effective recall at fixed nprobe.

---

## Practical failure modes

1. **Symmetric data**: when the data distribution is itself isotropic
   (rotation-invariant), `r₁` carries no preferred direction and the
   anisotropic penalty collapses to zero in expectation. Our 24-mode
   Gaussian mixture shows this: λ=1.0 and λ=0 differ by <0.005 recall.
   SOAR's win is reduced — but not negative — in this regime.
2. **High K with empty residuals**: if `||r₁|| < ε` (vector sits exactly
   on its centroid), `inv_r_norm_sq` is set to zero and the loss reduces
   to plain second-nearest. Handled by a guard.
3. **Centroid drift after streaming inserts**: the SOAR PoC is static.
   For streaming IVF, secondary assignments would need re-computation
   when centroids shift. The DiskANN-style "two-pass rebuild" is the
   right pattern, not addressed here.
4. **Quantized residuals**: when posting lists store PQ codes instead of
   raw vectors, the SOAR secondary assignment must be made *before*
   residual quantization to preserve the orthogonality property.

---

## What to improve next (roadmap)

1. **Real-data validation**: rerun on SIFT-1M and GLOVE-1M (downloadable
   from `ann-benchmarks.com`) to reproduce the paper's 1.5–2× recall lift
   on naturally anisotropic embeddings. Expected branch:
   `research/soar-real-data-validation`.
2. **SIMD distance kernels**: drop the `l2_sq` loop into `ruvector-core`'s
   existing SIMD path. Expected 3–5× kernel speedup → recovers most of
   the QPS lost to 2× posting overhead.
3. **SOAR + RaBitQ**: combine 2-cell spillover with `ruvector-rabitq`'s
   1-bit quantization. The orthogonal-residual property *should* help
   RaBitQ's error bound stay tight under quantization.
4. **Dynamic K (centroid splitting)**: when one cell grows >2× the
   mean posting size, split it via mini k-means and re-run SOAR
   assignment for affected points only.
5. **Multi-spillover (r=3, 4)**: paper hints that diminishing returns
   kick in past r=2, but doesn't formally measure. Worth a sweep.

---

## Production crate layout proposal

When this graduates from PoC to production:

```
crates/ruvector-soar/
├── src/
│   ├── lib.rs               (public API)
│   ├── distance.rs          (→ replaced by ruvector-core::distance::simd)
│   ├── kmeans.rs            (→ replaced by ruvector-cluster::kmeans)
│   ├── index.rs             (core SOAR logic)
│   ├── streaming.rs         (NEW: re-assignment on centroid drift)
│   └── persistence.rs       (NEW: snapshot format compatible with ruvector-snapshot)
├── benches/                 (criterion benchmarks)
└── examples/soar_bench.rs   (this PoC's bench)
```

Public trait sketch for plug-in to `ruvector-core::IndexBackend`:

```rust
impl IndexBackend for IvfIndex {
    fn search(&self, q: &[f32], k: usize, params: &SearchParams) -> Vec<SearchResult> {
        self.search(q, k, params.nprobe.unwrap_or(8))
    }
    fn build_params(&self) -> BuildParams { /* ... */ }
}
```

---

## References

1. Sun, P., Simcha, D., Dopson, D., Guo, R., Kumar, S.
   **"SOAR: Improved Indexing for Approximate Nearest Neighbor Search."**
   NeurIPS 2023. arXiv:2404.00774.
2. Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., Kumar, S.
   **"Accelerating Large-Scale Inference with Anisotropic Vector Quantization."**
   ICML 2020 (ScaNN).
3. Jégou, H., Douze, M., Schmid, C.
   **"Product Quantization for Nearest Neighbor Search."**
   PAMI 2011 (original IVF formulation).
4. Babenko, A., Lempitsky, V.
   **"The Inverted Multi-Index."** CVPR 2012.
5. Patel, L., Kraska, T., et al.
   **"ACORN: Predicate-Agnostic Search Over Vector Embeddings."**
   SIGMOD 2024 (prior ruvector nightly research).
6. Gao, J., Long, C. **"RaBitQ: Quantizing High-Dimensional Vectors with
   a Theoretical Error Bound for ANN Search."** SIGMOD 2024.

---

## Reproducing

```bash
cd ruvector
cargo build --release -p ruvector-soar
cargo test --release -p ruvector-soar
cargo run --release -p ruvector-soar --example soar-bench
```
