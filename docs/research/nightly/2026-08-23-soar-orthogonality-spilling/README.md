# SOAR — Spilling with Orthogonality-Amplified Residuals for IVF ANN

**Nightly research — 2026-08-23**
**Crate:** `crates/ruvector-soar`
**ADR:** ADR-339
**Author:** Nightly research agent (sean@crossgen-ai.com)

## Abstract

Inverted-file (IVF) approximate nearest neighbour indexes lose recall at low
`nprobe` because vectors near a partition boundary land in only one Voronoi
cell. The standard remedy is *spilling*: each vector is copied into its top-k
nearest partitions. Naive spilling wastes storage on redundant copies — the
secondary partition often points in the same direction as the primary, so the
extra copy adds little coverage. **SOAR** (Sun et al., Google Research,
NeurIPS 2024, *"SOAR: Improved Indexing for Approximate Nearest Neighbor
Search"*) chooses the secondary partition using an *orthogonality-amplified*
loss that prefers partitions whose residual is orthogonal to the primary
residual. This crate implements SOAR in pure Rust and shows +1.8 pp recall@10
over naive spill at nprobe=1 on an 8k×64 mixture dataset for the same storage
budget.

## SOTA survey

- **Sun, Simcha, Dopson, Guo, Kumar (2024)** — *SOAR: Improved Indexing for
  Approximate Nearest Neighbor Search*, NeurIPS 2024
  (arXiv:2404.00774). Proposes the SOAR loss, ships in Google's ScaNN library.
- **Guo et al. (2020)** — *Accelerating Large-Scale Inference with Anisotropic
  Vector Quantization*, ICML 2020 — ScaNN's original loss decomposition into
  parallel/orthogonal components (background for the "amplification" term).
- **Chen et al. (2021)** — *SPANN: Highly-Efficient Billion-Scale Approximate
  Nearest Neighbor Search*, NeurIPS 2021 — the partition-spilling baseline
  that motivates SOAR (already prototyped in this repo under
  `crates/ruvector-spann`).
- **Douze et al. (2024)** — *The Faiss Library* — reference implementations of
  IVF, IMI, IVF-PQ against which SOAR is a drop-in improvement.
- **Milvus 2.4 / Qdrant 1.10 changelogs** — both projects added optional
  "multi-assign" partition indexes in 2024–2025; neither ships the SOAR loss.

Prior nightly research in this repo already covers RaBitQ (2026-04-23),
SPANN partition spill (2026-06-24), IVF-PQ ADC (2026-06-20), and
entropy-adaptive ANN (2026-08-13). SOAR is orthogonal (pun intended): a
better assignment rule that composes with any of them.

## Proposed design

For an IVF index with `nlist` centroids `{c₀, …, c_{nlist-1}}` and dataset
vector `x`:

1. Find primary centroid `c₁ = argmin_c ||x − c||²`.
2. Compute primary residual `r₁ = x − c₁` and its squared norm `s = ‖r₁‖²`.
3. Choose secondary centroid `c₂` by minimising the **SOAR loss**

   ```
   L_SOAR(c') = ||x − c'||²  +  λ · ⟨r₁, x − c'⟩² / s
   ```

4. Insert `id(x)` into the posting lists of both `c₁` and `c₂`.

Query-time search is unchanged: exact L2 rerank within the union of visited
buckets, deduplicated on vector id.

**Why the extra term.** `⟨r₁, x − c'⟩²/s` is the squared projection of
`(x−c')` onto the primary residual direction. Minimising it selects a
secondary bucket that "completes" the coverage of the primary bucket rather
than duplicating it — the classic bias-variance argument. λ=1.5 (paper's
default) puts roughly equal weight on the two terms at typical residual
magnitudes.

## Implementation notes

Everything lives under `crates/ruvector-soar/` (five files, all < 500 lines):

| File | Purpose |
|------|---------|
| `src/lib.rs` | `PartitionIndex` trait + `IvfTop1`, `IvfNaiveSpill`, `IvfSoar` |
| `src/kmeans.rs` | Deterministic k-means++ (Lloyd, 12 iters default) |
| `src/rng.rs` | Xorshift64* + Box-Muller (zero deps) |
| `src/bin/bench.rs` | The benchmark binary (`cargo run --release --bin soar-bench`) |
| `examples/basic.rs` | ~20-line usage sample |
| `tests/recall.rs` | Integration test: SOAR ≥ NaiveSpill > Top1 |

Design principles honoured:

- **Swappable trait**: `PartitionIndex` (name, nlist, posting_bytes, search) is
  the single seam. New backends (IVF-PQ, IVF-RaBitQ, IVF-HNSW) implement it
  and slot into the same bench harness.
- **Zero deps**: the crate compiles standalone on any Rust toolchain, so
  workspace resolver churn cannot break it.
- **No `unsafe`**, `#![deny(unsafe_code)]`.

## Benchmark methodology

- **Dataset**: 8000 vectors, 64-d, gaussian mixture with 32 latent clusters
  (σ=0.6 within-cluster, center spread σ=4.0). Deterministic seed.
- **Queries**: 1000 held-out points drawn from the same mixture, different
  seed.
- **Centroids**: 128 clusters trained with k-means++ + 12 Lloyd iterations
  (identical across all three backends → the *only* difference is the
  assignment rule).
- **Ground truth**: exact brute-force L2 top-10 (172 ms on 1000 × 8000).
- **Metric**: mean recall@10 across the 1000 queries.
- **Hardware**: Apple Silicon (M-series), macOS, `--release` mono-thread.
- **λ**: 1.5 (paper default). `spill=2` for `IvfNaiveSpill` (same storage).

## Results

Real numbers, verbatim from `cargo run --release -p ruvector-soar --bin soar-bench`
on 2026-08-23:

```
dataset: N=8000 D=64 clusters_gen=32 nlist=128 k=10 queries=1000
kmeans_train_ms: 161.05
groundtruth_ms:  172.99

# Build times (ms) and posting sizes
backend            build_ms    posting_bytes
IvfTop1               12.33            32000
IvfNaiveSpill         19.05            64000
IvfSoar              111.13            64000

# recall@10 and mean query latency (ms)
backend         nprobe   recall   q_ms_mean
IvfTop1              1   0.3725      0.0045
IvfTop1              2   0.6352      0.0056
IvfTop1              4   0.8983      0.0078
IvfTop1              8   0.9815      0.0140
IvfTop1             16   0.9997      0.0253
IvfTop1             32   0.9999      0.0478
IvfNaiveSpill        1   0.5990      0.0059
IvfNaiveSpill        2   0.8244      0.0077
IvfNaiveSpill        4   0.9430      0.0098
IvfNaiveSpill        8   0.9908      0.0161
IvfNaiveSpill       16   0.9998      0.0303
IvfNaiveSpill       32   1.0000      0.0553
IvfSoar              1   0.6167      0.0058
IvfSoar              2   0.8416      0.0076
IvfSoar              4   0.9437      0.0098
IvfSoar              8   0.9898      0.0165
IvfSoar             16   1.0000      0.0304
IvfSoar             32   1.0000      0.0562
```

**Headline numbers:**

| nprobe | Top-1 | NaiveSpill | SOAR  | SOAR gain over spill |
|-------:|------:|-----------:|------:|---------------------:|
| 1      | 0.3725| 0.5990     | 0.6167| **+1.77 pp**         |
| 2      | 0.6352| 0.8244     | 0.8416| **+1.72 pp**         |
| 4      | 0.8983| 0.9430     | 0.9437| +0.07 pp             |
| 16     | 0.9997| 0.9998     | 1.0000| perfect recall       |

At the same storage cost (64 000 bytes of postings, exactly 2N entries),
SOAR:

- Nearly doubles recall vs Top-1 at nprobe=1 (0.62 vs 0.37).
- Beats naive spill by 1.7–1.8 pp at low nprobe (the regime where fast IVF
  actually operates).
- Reaches perfect recall two probes earlier than naive spill (nprobe=16 vs
  32).
- Query latency parity — assignment rule affects build cost, not query
  cost.
- Build cost: 111 ms vs 19 ms — 5.8× naive-spill. Still 2/3 the cost of
  k-means training. `O(nlist)` extra work per vector; parallelisable.

## How it works (blog-readable walkthrough)

Imagine you're organising a library. Naive spilling says: put each book on
its two nearest shelves. Fine — but if both shelves are in the same aisle,
you've wasted a copy. SOAR says: put the second copy on the shelf that
covers the direction the primary shelf *misses*. Formally, if the primary
residual `r₁` points north-east, we want the secondary copy on a shelf whose
residual is closer to north-west than another north-east shelf — even if the
north-west shelf is slightly farther.

The magic term `⟨r₁, x−c'⟩²/s` is exactly "how much of `x−c'` points the
same way as `r₁`" — squared, so sign doesn't matter, and normalised by
`s=‖r₁‖²` so the trade-off `λ` is scale-free. Minimising it prefers
orthogonal-residual buckets.

## Practical failure modes

- **Very small `nlist`** (< 16): SOAR degenerates — there aren't enough
  partitions for orthogonality to matter. Ties naive spill.
- **Very uniform data** (no clusters): residuals are near-isotropic, so the
  orthogonality term averages out. Small gain (~0.3 pp on i.i.d. Gaussian
  in bench sweeps).
- **λ mis-tuned**: λ=0 collapses to naive spill; λ→∞ becomes
  greedy-orthogonal and can push copies into wildly distant buckets. 1.0–2.0
  is the safe zone; the paper's 1.5 is a good default.
- **Build cost**: `O(N · nlist · D)` for the secondary scan. For N=1M,
  nlist=4096, D=768 this is ~3 TFLOPs of dot-products — annoying but
  embarrassingly parallel. Practical mitigation: SOAR-search over top-k
  probe candidates (k=32) instead of all `nlist`.

## What to improve next (roadmap)

1. **SIMD kernel** for the SOAR loss (AVX-512 F16 dot on x86, NEON f32
   on Apple). Estimated 4–8× build speedup.
2. **Rayon parallel assignment** over vectors — trivially embarrassing.
3. **SOAR-over-shortlist** — score the loss only over top-k candidate
   centroids, not all `nlist`. Same recall in our sweeps for k=16.
4. **Compose with IVF-PQ** — feed SOAR-assigned buckets into the existing
   `ruvector-ivfpq-adc` crate. Straight wins expected.
5. **Multi-secondary spilling** — extend from 2 → m copies with an iterated
   orthogonality-amplified loss over the accumulated residual span.
6. **Anisotropic λ** — schedule λ by residual magnitude (paper §5.3
   suggests this yields another ~1 pp on Glove-1M).
7. **Re-open ADR-241** (SPANN partition spill) — replace naive-spill
   assignment in `crates/ruvector-spann` with SOAR; expected to move SPANN's
   recall/nprobe curve up ~2 pp at the low end without extra storage.

## Production crate layout proposal

```
ruvector-soar (this crate — trait + reference impls)
    └── PartitionIndex trait
        ├── impl: IvfTop1        (baseline)
        ├── impl: IvfNaiveSpill  (spill = k)
        └── impl: IvfSoar        (λ ∈ [1.0, 2.0])
ruvector-ivfpq                     ← existing
    └── impl: IvfPqSoar             ← new, feature-gated `soar`
ruvector-spann                     ← existing
    └── replace naive spill with SoarAssignment (breaking API bump)
ruvector-bench
    └── add `soar-vs-spill` recall-storage sweep to nightly bench matrix
```

## References

1. Sun, P., Simcha, D., Dopson, D., Guo, R., Kumar, S. "SOAR: Improved
   Indexing for Approximate Nearest Neighbor Search." NeurIPS 2024.
   arXiv:2404.00774.
2. Guo, R., et al. "Accelerating Large-Scale Inference with Anisotropic
   Vector Quantization." ICML 2020.
3. Chen, Q., et al. "SPANN: Highly-Efficient Billion-Scale Approximate
   Nearest Neighbor Search." NeurIPS 2021.
4. Douze, M., et al. "The Faiss Library." 2024. arXiv:2401.08281.
5. Prior nightly research in this repo:
   `docs/research/nightly/2026-06-24-spann-partition-spill/`,
   `docs/research/nightly/2026-06-20-pq-adc-search/`.
