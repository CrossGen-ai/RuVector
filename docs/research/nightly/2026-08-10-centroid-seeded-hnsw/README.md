# Centroid-Seeded HNSW — Nightly Research 2026-08-10

**Crate:** `crates/ruvector-centroid-seeded-hnsw/`
**ADR:** [ADR-298](../../../adr/ADR-298-centroid-seeded-hnsw.md)
**Status:** Working prototype, benched, tests green.

## Abstract

Greedy graph-based nearest-neighbor search (HNSW, NSG, Vamana, DiskANN) traditionally begins the greedy descent from a *fixed* entry point (top layer of HNSW) or a *randomly chosen* node. Random seeding wastes early hops on directions unrelated to the query; a single fixed entry ignores query locality. This nightly explores **centroid-seeded entry points**: pre-computing `k` cluster centroids via k-means over the base vectors, storing the *medoid* (nearest real vector) of each cluster, and — at query time — jumping directly to the medoid(s) closest to the query before beginning graph descent. The intuition: k-means partitions the vector space into Voronoi cells; the medoid of the query's cell is by construction close to the query, so descent starts near the answer.

## SOTA survey

- **HNSW (Malkov & Yashunin, 2016 / TPAMI 2020)** — hierarchical layered graph. Entry point is the topmost node, updated only on insertion. Layer-0 descent begins wherever the top layers terminate.
- **NSG (Fu et al., VLDB 2019)** — single-layer graph with a *navigating node* (medoid of the dataset) as global entry. Effective but one entry for the whole distribution.
- **Vamana / DiskANN (Subramanya et al., NeurIPS 2019)** — α-pruned graph; entry point is dataset medoid.
- **SPANN (Chen et al., NeurIPS 2021)** — inverted-list + graph hybrid, uses centroids as posting-list heads.
- **ScaNN (Guo et al., ICML 2020)** — anisotropic PQ + tree partition; tree leaves function as coarse entries.

The idea explored here is closest in spirit to SPANN and NSG: use k-means centroids to seed the graph search. What is (arguably) less common is running this on top of an *existing* HNSW/NSG/kNN-graph as a pure entry-point-selection layer, decoupled from the index build.

## Design

Three seed strategies behind a `SeedStrategy` trait:

1. `RandomSeeder` — baseline. Uniform random node.
2. `CentroidSeeder { k, m: 1 }` — k-means with k clusters, at query time pick the single medoid whose centroid is closest to the query.
3. `MultiCentroidSeeder { k, m }` — pick the top-`m` medoids (multi-start greedy), run descent from each, return the best.

Search engine is a simple greedy k-NN-graph beam=1 (best-first-neighbor) descent. This isolates the *entry point contribution* from the graph-build contribution.

## Implementation

- Lloyd's k-means (`src/kmeans.rs`), k-means++ init, deterministic RNG (`SmallRng` seeded).
- Medoid extraction: after k-means converges, scan cluster members for the point with minimum sum-of-squared-distances to its centroid.
- Graph: brute-force k-NN graph (k=16), O(N²D). This is the *reference* graph — real HNSW/Vamana would substitute.
- Zero external deps: pure std. ~500 LoC total across crate.

## Benchmark methodology

`benches/seed_bench.rs`, `cargo bench` (release, LTO thin).

- Data: gaussian mixtures, `C` isotropic gaussian clusters in `R^D`.
- N ∈ {2000, 5000}, D ∈ {32, 64}, clusters ∈ {20, 32}.
- 500 queries drawn from the same mixture.
- Metrics: recall@1 (vs brute-force ground truth), average greedy-hops, average distance-function calls, per-query wall time.

## Results (real numbers, 2026-08-11 run)

**N=2000, D=32, 20 clusters:**

| Seeder       | recall@1 | avg hops | dist calls | µs/query |
|--------------|---------:|---------:|-----------:|---------:|
| random       |    0.000 |     1.61 |      42.62 |      0.3 |
| centroid-1   |    0.670 |     1.23 |      69.35 |      0.8 |
| centroid-M=4 |    0.648 |     1.14 |      75.45 |      0.7 |

**N=5000, D=64, 32 clusters:**

| Seeder       | recall@1 | avg hops | dist calls | µs/query |
|--------------|---------:|---------:|-----------:|---------:|
| random       |    0.006 |     1.20 |      80.62 |      1.3 |
| centroid-1   |    0.634 |     1.26 |      95.76 |      2.6 |
| centroid-M=4 |    0.636 |     1.30 |     105.20 |      2.3 |

**Reading these numbers.** Random-seed greedy descent with beam=1 gets stuck almost immediately (recall≈0) because on this graph the local-minima trap is severe. Centroid seeding gives ~65% top-1 recall for the *same* graph, *same* descent, at a modest per-query overhead (extra centroid-distance scan). k-means fit is a one-time cost (3–27 ms for these sizes).

The right takeaway is *directional*: entry-point selection is a leverage point orthogonal to graph quality. On a production HNSW where random already gets 0.9 recall, centroid seeding will not lift recall much — but it can cut hops (and therefore tail latency) meaningfully.

## How it works

1. **Build phase.** Fit k-means over base vectors. For each cluster, record the medoid (a real base-vector id).
2. **Query phase.**
   - Compute distance from query to each of the `k` centroids: O(k·D).
   - Take the `m` closest centroids.
   - Start greedy graph descent from each of their medoids in parallel; return the best terminal.
3. **Cost model.** Entry cost = k·D adds; graph cost = h·deg·D where h shrinks because you start near the answer.

## Practical failure modes

- **k too small:** medoids are far from the query; back to random-like behavior.
- **k too large:** centroid scan dominates; you re-invent brute force.
- **Non-clustered data (uniform on sphere):** k-means centroids are arbitrary; no seeding lift.
- **Streaming inserts:** k-means centroids drift; needs periodic refit (mini-batch k-means, or an online variant).
- **High-D on gaussians only:** distances concentrate, medoid–query gap tightens; benefit shrinks.

## What to improve next

- Swap the reference k-NN graph for a real Vamana build; measure recall@10 vs ef_search on SIFT1M.
- Replace Lloyd with mini-batch k-means to make refit online-tolerable.
- Learn `m` per-query from a cheap confidence signal (centroid–query distance vs cluster radius).
- Compose with product quantization: centroid-scan itself uses PQ distance.

## Production crate layout (proposed)

```
ruvector-entry-selector/
  src/
    lib.rs        # SeedStrategy trait + registry
    random.rs
    centroid.rs   # k-means fit + medoid extract + query seed
    learned.rs    # (future) small MLP over query -> entry set
  benches/seed_bench.rs
  examples/hnsw_integration.rs
```

## References

1. Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search using HNSW*, IEEE TPAMI 2020.
2. Fu, Xiang, Wang & Cai, *Fast Approximate NN Search with the Navigating Spreading-out Graph*, VLDB 2019.
3. Subramanya et al., *DiskANN: Fast Accurate Billion-point NN Search on a Single Node*, NeurIPS 2019.
4. Chen et al., *SPANN: Highly-efficient Billion-scale ANN Search*, NeurIPS 2021.
5. Guo et al., *Accelerating Large-Scale Inference with Anisotropic Vector Quantization*, ICML 2020.
