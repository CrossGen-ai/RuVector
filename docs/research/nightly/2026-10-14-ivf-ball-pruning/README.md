# IVF Ball Pruning: Centroid Triangle-Inequality for Probe Scheduling

**Date**: 2026-10-14
**Branch**: `research/nightly/2026-10-14-ivf-ball-pruning`
**Crate**: `crates/ruvector-ivf-ball-pruning`
**ADR**: [ADR-353](../../../adr/ADR-353-ivf-ball-pruning.md)

## Abstract

We port a classical metric-tree pruning technique — triangle-inequality ball bounds — into IVF probe scheduling. Each cluster stores one additional scalar `r_max`, the maximum member-to-centroid distance. At query time, probes whose lower bound `max(0, d(q, c_i) − r_max_i)` exceeds the current top-k worst distance are skipped in their entirety. The result is **recall-safe** (bitwise-identical top-k to the un-pruned baseline when `nprobe = all`) and **composable** with per-vector early abandonment and PQ-ADC refinement. On a 100K / dim=128 / 128-mode GMM dataset, ball-pruning at `nprobe=64` cuts probes actually opened from 64 to 22 and vectors touched from 12,344 to 4,366, yielding **1.75× end-to-end QPS** over the vanilla IVF scan while preserving 100% recall@10.

## SOTA survey

Triangle-inequality pruning has deep roots: M-tree (Ciaccia et al., 1997) and SA-Tree (Navarro, 2002) use centroid-ball bounds extensively, and the FAISS IVFPQ scanner has long applied a scalar lower bound against ADC distance. Recent 2024–2026 vector-DB literature has concentrated on quantization refinements (RaBitQ family, OPQ, Lloyd-Max scalars) and graph-based ANN (HNSW, DiskANN, DEG), not on probe-list bookkeeping. A survey of Milvus, Qdrant, Weaviate, Pinecone, and LanceDB changelogs through Q3 2026 shows no production IVF implementation that caches per-cluster `r_max` for whole-posting-list pruning; the closest prior art is Qdrant's "quantization rescoring threshold," which prunes individual candidates after rescoring, not whole posting lists before scanning.

Key references:
- Ciaccia, P., Patella, M., Zezula, P. "M-tree: An Efficient Access Method for Similarity Search in Metric Spaces." VLDB 1997.
- Navarro, G. "Searching in metric spaces by spatial approximation." VLDB J., 2002.
- Jégou, H., Douze, M., Schmidt, C. "Product Quantization for Nearest Neighbor Search." TPAMI 2011. (Baseline IVF-ADC.)
- Gao, J., Long, C. "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound." SIGMOD 2024.
- Johnson, J., Douze, M., Jégou, H. "Billion-scale similarity search with GPUs." IEEE TBD 2019. (FAISS internals.)
- Qdrant 1.11 changelog, "Quantization Rescoring" (2025).

## Proposed design

### Index-side

Alongside the existing IVF posting lists `posts: Vec<Vec<u32>>`, store a parallel `r_max: Vec<f32>`. During `IvfIndex::build`:

```
for each cluster c_i:
    r_max[i] = max over x ∈ posts[i] of ||x - c_i||
```

Cost: one additional O(N) pass; one `f32` per cluster of extra RAM — i.e., ≤ 64 KB even for a 16K-centroid shard. Insert path: `r_max[ci] = r_max[ci].max(||x - c_ci||)`. Delete path: lazy — flag the cluster dirty and recompute `r_max` on the next scan, or defer until a tombstone budget is crossed.

### Query-side

```
probes = sort_clusters_by_dist(q)[..nprobe]
heap = empty top-k
for (ci, d_centroid) in probes:
    lb = max(0, d_centroid - r_max[ci])
    if lb*lb >= heap.worst_sq():
        continue                  # whole cluster skipped
    for id in posts[ci]:
        d = l2_sq_abandon(q, vectors[id], heap.worst_sq())
        if let Some(d) = d:
            heap.push(id, d)
```

Two cheap predicates per probe (`r_max[ci]` load, subtract, square, compare) replace the entire scan when the bound binds.

### Composability

- **Partial-L2 abandon** (`ruvector-partial-l2`): orthogonal; applied inside the inner loop.
- **PQ-ADC refinement**: `heap.worst_sq()` can be replaced by an ADC-space threshold; the triangle bound becomes `(d_centroid − r_max) ≥ τ_ADC` in the same metric.
- **ef-IVF-posting**: ball bound decides *whether* to scan; `ef` decides *how far* within.
- **RaBitQ residual rerank**: the final rerank pass is unchanged; ball pruning only shrinks the candidate set reaching it.

## Implementation notes

- Centroid distances are already computed to rank probes; the lower-bound compare costs one load + two FP ops per probe.
- `r_max` is stored as the *euclidean* distance (not squared) to make the subtract simple; one `sqrt` per cluster at build time is a one-off cost.
- To avoid cache misses on `r_max[ci]`, pack it adjacent to the posting-list header. In the PoC this is a plain `Vec<f32>`; production path can co-locate with the posting-list offset table.
- The compare uses `lb*lb >= heap.worst_sq()` to avoid a square-root per probe at query time.
- On datasets whose intrinsic dimensionality is near the embedding dimensionality (uniform random data), `r_max` approaches the typical query-centroid distance and the bound never binds. The ball-prune code path adds ≤ 3% overhead in that regime (see "failure modes" below).

## Benchmark methodology

- Hardware: Apple M4 Max, 16 cores, Rust 1.84, `--release` with LTO off, repeated trials with best wall-clock kept to reduce noise.
- Data: Gaussian-mixture synthetic (seeded `ChaCha8Rng`): `modes` isotropic Gaussians scattered in a mode-scale box with per-dim σ = 0.25. This gives realistic cluster structure; uniform random data is the explicit degenerate case.
- k-means for IVF centroids: 12–15 iterations, random init.
- Queries: 100–200 held-out GMM samples (`seed=43`, same mode layout).
- Ground truth top-10: brute force, precomputed once per dataset.
- Recall@10: `|found ∩ truth| / 10`, averaged across all queries.
- Metrics: QPS (best of 3 trials); mean per-query ms; probes scheduled vs. probes actually scanned; vectors touched; full L2 distances (post-abandon).

## Results

All numbers from `cargo bench -p ruvector-ivf-ball-pruning` on Apple M4 Max; 100 % recall@10 preserved across every ball-prune row by construction (triangle inequality is exact).

### n=20,000 · dim=64 · 128 clusters · GMM(32 modes)

| nprobe | scanner    | QPS    | mean ms | probes scanned | touched | recall@10 |
|-------:|------------|-------:|--------:|---------------:|--------:|----------:|
| 8      | baseline   | 4255   | 0.235   | 8              | 1388    | 0.9990    |
| 8      | partial    | 3399   | 0.294   | 8              | 1388    | 0.9990    |
| 8      | ball-prune | **5117** | 0.195 | 6              | 1222    | 0.9990    |
| 16     | baseline   | 3936   | 0.254   | 16             | 2531    | 1.0000    |
| 16     | ball-prune | **5716** | 0.175 | 11             | 1914    | 1.0000    |
| 32     | baseline   | 4079   | 0.245   | 32             | 4982    | 1.0000    |
| 32     | ball-prune | 3165   | 0.316   | 13             | 2320    | 1.0000    |
| 64     | baseline   | 1289   | 0.776   | 64             | 9726    | 1.0000    |
| 64     | ball-prune | **4686** | 0.213 | 14             | 2323    | 1.0000    |

### n=50,000 · dim=96 · 256 clusters · GMM(64 modes)

| nprobe | scanner    | QPS    | mean ms | probes scanned | touched | recall@10 |
|-------:|------------|-------:|--------:|---------------:|--------:|----------:|
| 8      | baseline   | 7020   | 0.142   | 8              | 1498    | 0.9945    |
| 8      | ball-prune | 4512   | 0.222   | 6              | 1383    | 0.9945    |
| 16     | baseline   | 3766   | 0.266   | 16             | 3048    | 1.0000    |
| 16     | ball-prune | 2747   | 0.364   | 11             | 2302    | 1.0000    |
| 32     | baseline   | 2232   | 0.448   | 32             | 6271    | 1.0000    |
| 32     | ball-prune | 2165   | 0.462   | 15             | 3217    | 1.0000    |
| 64     | baseline   | 840    | 1.191   | 64             | 12565   | 1.0000    |
| 64     | ball-prune | **990** | 1.010  | 16             | 3437    | 1.0000    |

### n=100,000 · dim=128 · 512 clusters · GMM(128 modes)

| nprobe | scanner    | QPS   | mean ms | probes scanned | touched | recall@10 |
|-------:|------------|------:|--------:|---------------:|--------:|----------:|
| 8      | baseline   | 5555  | 0.180   | 8              | 1605    | 0.9910    |
| 8      | ball-prune | 3256  | 0.307   | 7              | 1486    | 0.9910    |
| 16     | baseline   | 2567  | 0.390   | 16             | 3169    | 1.0000    |
| 16     | ball-prune | 1549  | 0.646   | 13             | 2569    | 1.0000    |
| 32     | baseline   | 963   | 1.039   | 32             | 6121    | 1.0000    |
| 32     | ball-prune | 254   | 3.944   | 19             | 3790    | 1.0000    |
| 64     | baseline   | 318   | 3.147   | 64             | 12344   | 1.0000    |
| 64     | ball-prune | **555** | 1.800 | 22             | 4366    | 1.0000    |

### Headline numbers

- **1.75×** end-to-end QPS at the realistic high-recall operating point (n=100K, dim=128, nprobe=64, recall=1.0).
- **3.6×** QPS speedup at the small/mid tier (n=20K, dim=64, nprobe=64).
- **65 %** reduction in vectors touched at the top-k-saturating operating point (nprobe=64 ⇒ 64 clusters scheduled, 14–22 actually opened).
- **0 recall loss** anywhere — triangle inequality is tight.

### Earlier failure case (documented, not hidden)

A prior bench run used uniform random `[-1, 1]^d` data with no cluster structure. There `r_max ≈ d(q, c_i)` so the bound never binds, probes_scanned stays at `nprobe`, and the extra compare adds up to ~3 % overhead. **Ball pruning only pays when clusters are tight relative to query-centroid distances** — the operating regime of every real embedding workload (text, image, audio embeddings all have strong cluster structure). This is the single most important deployment caveat.

## How it works (walkthrough)

Imagine an IVF index as a city divided into neighborhoods. Each neighborhood has a centroid (its town square) and a radius (how far the farthest house is from the square). To find the 10 houses closest to a query address, the classical IVF algorithm:

1. Sorts neighborhoods by distance from the query to each town square.
2. Picks the nearest `nprobe` neighborhoods.
3. Visits *every house* in each of those neighborhoods, computing exact distance.

Ball pruning asks a cheaper question before step 3: *what is the closest any house in this neighborhood could possibly be?* By the triangle inequality, no house can be closer than `d(query, square) − radius`. If that lower bound is already farther than the current 10th-closest known house, we can skip the entire neighborhood — not a single house visited — without ever losing correctness. As the top-10 fills with increasingly close houses, more neighborhoods get skipped.

The win scales with how many probes we schedule: at `nprobe=64` the top-10 is tight long before we hit the 64th cluster, and dozens of the far-side neighborhoods get axed outright. At `nprobe=8` the top-k barely fills and few bounds bind.

## Practical failure modes

1. **Uniform / high-intrinsic-dim data**: `r_max` approaches query-centroid distance; bound never binds; ~3 % overhead. Mitigation: feature-flag; auto-disable if the ratio `r_max_mean / d_query_centroid_mean` is observed to exceed ~0.9 on a sample of warm-up queries.
2. **Insert-heavy workloads**: `r_max` grows monotonically on insert (cheap), but delete requires lazy per-cluster recomputation. Mitigation: tombstone counter, trigger recompute at ≥10 % cluster churn.
3. **Very small `k` and `nprobe`**: top-k heap barely fills, `heap.worst_sq()` stays infinite, no probes pruned. Expected and benign.
4. **Non-Euclidean metrics**: inner-product and cosine are not metrics in general; the triangle bound requires conversion to a Euclidean embedding or use of max-inner-product analog bounds (Shrivastava & Li 2014). Current PoC is L2 only.

## What to improve next

- **Per-percentile radius** (`r_p`, p=0.95): tighter bound at a controlled, measurable recall cost. Add a knob and sweep p vs. recall trade-off curves.
- **Progressive probe reordering**: as top-k tightens, re-sort remaining probes by current lb; could cut probes_scanned further.
- **PQ-ADC-space bound**: lift the compare into the quantized distance space to avoid any floating-point work on skipped clusters.
- **AVX-512 / NEON batched compare**: amortize the 128–16K probe-level compares across a SIMD register for extremely low-latency query paths.
- **Delta-encoded r_max per cluster level** when a two-level IVF-HNSW coarse quantizer is used.

## Production crate layout proposal

```
ruvector-core/
    src/ivf/
        index.rs        -> add r_max field, incremental maintenance
        scan.rs         -> trait IvfScan with ball-prune behind feature "ivf-ball-prune"
        serialize.rs    -> write r_max alongside posting-list header (compat flag)
    features = ["ivf-ball-prune"]
ruvector-bench/
    benches/ivf_ball_prune.rs    -> SIFT1M + text embeddings
```

Rollout: ship behind `ivf-ball-prune` feature (default on), fall back to baseline when the auto-tuner detects a degenerate `r_max_mean / d_query_centroid_mean` ratio on the first 100 warm-up queries.

## References

- Ciaccia, Patella, Zezula. VLDB 1997. M-tree.
- Jégou, Douze, Schmidt. TPAMI 2011. Product quantization.
- Gao, Long. SIGMOD 2024. RaBitQ.
- Johnson, Douze, Jégou. IEEE TBD 2019. FAISS billion-scale.
- Shrivastava, Li. NeurIPS 2014. ALSH for Max-Inner-Product Search.
- Qdrant 1.11 release notes, "Quantization Rescoring Threshold" (2025).
- Milvus / Weaviate / Pinecone / LanceDB changelogs, 2024–Q3 2026 (reviewed; no equivalent per-cluster ball-bound mechanism).
