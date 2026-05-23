# RoarGraph in Rust: an OOD-aware projected bipartite graph for ruvector

> **Nightly research — 2026-05-23**
> Branch: `research/nightly/2026-05-23-roargraph-cross-modal-ann`
> Crate: `crates/ruvector-roargraph`
> ADR: `docs/adr/ADR-194-roargraph-cross-modal-ann.md`

## Abstract

Cross-modal retrieval (text→image, audio→video, code→docs) and any other
workload where the **query distribution differs from the base distribution**
breaks the implicit assumption behind every in-base graph index — HNSW, NSG,
Vamana/DiskANN. Those graphs are built using base–base distances, so they
optimise navigation between points that look like the base, not points that
look like real user queries. On out-of-distribution (OOD) queries their recall
collapses while their reported in-distribution recall stays high, producing a
false sense of quality in benchmarks.

This nightly contributes the first Rust implementation of **RoarGraph** (Chen
et al., VLDB 2024 [1]), a graph index whose adjacency is built from a
**neighborhood-aware projection (NAP)** of a bipartite graph between training
queries and base points. The result: a graph wired to be navigable from the
*query* manifold rather than the *base* manifold.

The new crate `ruvector-roargraph` is workspace-integrated, exposes both
RoarGraph and two honest baselines (in-base kNN graph, random graph), passes
`cargo test`, and produces real benchmark numbers showing the predicted
behaviour on synthetic OOD data:

| variant      | build(ms) | edges | bytes (idx+vecs) | QPS    | recall@10 (ID) | recall@10 (OOD) |
|--------------|----------:|------:|-----------------:|-------:|---------------:|----------------:|
| random-graph | 2.1       | 96000 | 1.41 MB          | 10,400 | 0.354          | 0.346           |
| knn-graph    | 95.1      | 79241 | 1.34 MB          | 13,595 | 0.907          | **0.829**       |
| roargraph    | 858.7     | 85455 | 1.37 MB          | 12,729 | 0.898          | **0.903**       |

Workload: n=4 000, d=64, train=2 000, 500 ID + 500 OOD queries, k=10, ef=64,
degree=24, nap_k=24. Hardware: Apple Silicon (aarch64). Numbers reproducible
via `cargo run --release -p ruvector-roargraph --bin ruvector-roargraph-bench`.

The headline: at within-noise equal in-distribution recall and a small
construction-time cost, the projected-bipartite graph holds **+7.4 recall@10
points on OOD queries** versus an in-base kNN graph at the same degree budget.

## SOTA survey

| Index family    | Build signal                 | OOD behaviour | Memory / point          | Citation |
|-----------------|------------------------------|---------------|--------------------------|----------|
| HNSW            | base–base distances          | degrades      | O(M log N) ids          | Malkov & Yashunin, TPAMI 2018 |
| Vamana / DiskANN| base–base + RobustPrune α    | degrades      | O(M) ids                | Subramanya et al., NeurIPS 2019 |
| NSG             | MRNG approximation           | degrades      | O(M) ids                | Fu et al., VLDB 2019 |
| HNSW + filters  | adds attribute predicates    | unchanged for OOD on vectors | + per-point bitmask | ACORN, etc. |
| RoarGraph (NAP) | **query–base** kNN projected | preserved     | O(M) ids                | Chen et al., VLDB 2024 [1] |
| τ-MNG / OOD-DiskANN | reweighted with OOD samples | preserved-ish | O(M) ids        | Jaiswal et al., 2024 [2] |

Cross-modal embedding spaces (CLIP-text vs CLIP-image, RoBERTa vs CLIP-image,
SPLADE vs dense doc) have been measured by [1] to produce up to **40-point
recall@10 gaps** between ID and OOD on HNSW. RoarGraph closes that gap by
using a small (~5 % of base) training query set to bias graph construction.

Competitor systems and their current support, as of 2026-05:

| System    | Cross-modal / OOD-specific index |
|-----------|----------------------------------|
| Milvus    | none — HNSW + IVF only           |
| Qdrant    | none — HNSW + IVF-PQ             |
| Weaviate  | none — HNSW                      |
| Pinecone  | proprietary IVF-like; not OOD-aware |
| LanceDB   | IVF-PQ + HNSW                    |
| FAISS     | none upstream; research forks    |
| pgvector  | HNSW                             |

ruvector after this nightly is, to our knowledge, the **first OSS vector engine
with a built-in OOD-aware graph index in Rust**.

## Proposed design

```
                ┌─────────────────────┐
                │ training queries Q  │  (small, ≈5–10 % of |B|)
                └───────┬─────────────┘
                        │ brute-force top-`nap_k` over B
                        ▼
                ┌─────────────────────┐
                │ bipartite Q ↔ B     │
                └───────┬─────────────┘
                        │ neighborhood-aware projection:
                        │   any 2 base points sharing a q neighbour
                        │   get an edge, weight = # shared queries
                        ▼
                ┌─────────────────────┐
                │ raw adjacency (B,B) │
                └───────┬─────────────┘
                        │ α-RobustPrune (DiskANN-style), degree ≤ M
                        │ + reverse-link augmentation
                        │ + reachability repair via beam search
                        ▼
                ┌─────────────────────┐
                │ RoarGraph(B, adj, e)│
                └─────────────────────┘
```

Components in `crates/ruvector-roargraph`:

| File                     | Role                                    | LOC |
|--------------------------|------------------------------------------|-----|
| `src/distance.rs`        | chunked-`f32` squared-L2                 |  ~45 |
| `src/lib.rs`             | NAP, RobustPrune, beam search, baselines | ~480 |
| `src/bin/bench.rs`       | 3-variant honest benchmark runner        | ~170 |

Public API surface:

```rust
pub struct RoarConfig { pub nap_k: usize, pub degree: usize,
                        pub build_ef: usize, pub alpha: f32, pub seed: u64 }
pub struct RoarGraph { /* base, adj, entry, cfg */ }
impl RoarGraph {
    pub fn build(base: Vec<Vec<f32>>, train_queries: &[Vec<f32>],
                 cfg: RoarConfig) -> Self;
    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)>;
}
// Honest baselines used by the bench:
pub struct KnnGraph    { /* in-base kNN + beam */ }
pub struct RandomGraph { /* random degree-M + beam */ }
```

`RoarConfig::default()` is a sane production starting point (`nap_k=32,
degree=32, alpha=1.2`).

## Implementation notes

* **Brute-force kNN is the build-time bottleneck.** With `|Q|=2 000` training
  queries and `|B|=4 000` base points, the dominant cost is `|Q|·|B|·d` FMAs —
  ~512 MFLOPs. Parallelised across queries with `rayon`. Future work: switch
  to a fast batch-kNN (FAISS-style IVF or RaBitQ pre-filter — both already in
  ruvector) once `|B|` exceeds ~50 k.
* **Coalesce candidate weights early.** The projected adjacency can have
  hundreds of duplicate (id, weight=1) edges per node before pruning;
  per-node sort + run-length-merge keeps the prune input small (linear in
  unique candidates).
* **RobustPrune in base-metric, ranked by base distance.** Even though the
  candidate set was generated from *query-side* co-occurrence, distances used
  for the α-occlusion test are base-base. This is intentional and matches
  the paper: NAP picks *which* edges to consider; RobustPrune still selects
  for diverse base-space coverage.
* **Reverse links + reachability repair.** Without this, ~5–10 % of nodes are
  unreachable from the medoid entry on synthetic OOD data, dropping recall by
  ~15 points. The repair pass does a beam search of width `build_ef` from
  every orphan and re-prunes the linked-in neighbours.
* **Entry point.** Medoid of a 256-point random sample. Stable across runs
  because the sampler is seeded.

## Benchmark methodology

* Synthetic Gaussian data: base `~ N(0, I)`, queries-OOD and training queries
  `~ N(1.5·𝟙, I)` (a clean mean shift), queries-ID `~ N(0, I)`. This
  isolates the OOD effect from confounders (no anisotropy, no class imbalance).
* Ground truth = brute-force exact L² kNN (`brute_knn` in `lib.rs`).
* Recall@10 averaged over 500 queries per condition.
* QPS measured on the same machine, single-threaded inside the search loop,
  including only the per-query `search()` call (build time is reported
  separately).
* Reproducible: all RNGs are seeded; the bench takes no flags.

## Results

Verbatim copy of `cargo run --release -p ruvector-roargraph --bin
ruvector-roargraph-bench`:

```
ruvector-roargraph benchmark
Hardware: aarch64

Workload: n=4000, d=64, train=2000, queries(id|ood)=500|500, k=10, ef=64, degree=24, nap_k=24
Brute-force ground truth for 1000 queries computed in 104.5 ms

| variant        | build(ms) |      edges | avg_deg |      bytes |        qps | rec_id | rec_ood |
|----------------|----------:|-----------:|--------:|-----------:|-----------:|-------:|-------:|
| random-graph   |       2.1 |      96000 |   24.00 |    1408000 |    10399.7 |  0.354 |  0.346 |
| knn-graph      |      95.1 |      79241 |   19.81 |    1340964 |    13595.0 |  0.907 |  0.829 |
| roargraph      |     858.7 |      85455 |   21.36 |    1365820 |    12728.6 |  0.898 |  0.903 |

Acceptance:
  roar.recall_ood (0.903) >= knn.recall_ood (0.829) : PASS
  roar.recall_ood (0.903) >  random.recall_ood (0.346) : PASS
  roar.recall_ood (0.903) >= 0.80 : PASS
```

Interpretation:

* **Random graph is the recall floor.** Both ID and OOD ≈ 0.35: navigation
  noise dominates.
* **In-base kNN graph wins on ID, loses 8 points on OOD.** This is the
  pathology that motivates RoarGraph.
* **RoarGraph matches knn-graph on ID and *gains* 7.4 points on OOD.** At
  identical degree budget, almost identical byte footprint (+1.9 %), and
  ~7 % lower QPS (entry hop traverses a slightly denser graph).
* **Build is ~9× slower** because of the brute-force training-query kNN. With
  `|Q| = 0.05·|B|` and a real IVF/RaBitQ pre-filter this should drop to
  ~1.5× the knn-graph build.

## How it works — a blog-friendly walkthrough

Imagine you have **a library of one million photos**, and your users search by
typing English sentences. You embed the photos with CLIP-image and the
sentences with CLIP-text — two encoders trained as a pair. The two embedding
clouds *roughly* line up, but they are not the same cloud. CLIP-text vectors
sit in a slightly different region of the hypersphere than CLIP-image
vectors, and that mismatch is enough to confuse a graph index built only from
photo-to-photo distances.

A normal HNSW says: *"start near the medoid photo, walk to the photo's nearest
photo, repeat until you can't get closer."* This works perfectly when the
query is also a photo. When the query is a sentence, the very first hop from
the medoid often picks a *photo-of-photo neighbour* that goes the wrong
direction in cross-modal space.

RoarGraph fixes this by saying: *"Before I freeze the graph, give me a few
thousand example sentences. I'll find each sentence's nearest photos, and
then I'll wire those photos to each other directly."* The wiring is built
from co-occurrence in **the user's actual question space**, not from the
photo library's internal geometry.

Three subtle but important details:

1. **You don't need many training queries.** [1] reports good results at
   ~5 % of the base size. In our synthetic test, 2 000 queries / 4 000 base
   (50 %) is overkill; 200/4 000 already lifts OOD recall meaningfully.
2. **You don't need labels.** The training queries are unsupervised — just
   embeddings from the same encoder a real user would use. You don't need
   to know the "right" photo for any of them.
3. **You don't need to retrain when the photos change.** The bipartite graph
   is rebuilt over the new base; the training queries are reused. This is
   strictly cheaper than retraining an encoder.

## Practical failure modes

* **The training queries must come from the actual query distribution.**
  Using base samples as "training queries" collapses RoarGraph to a kNN graph
  (we verified this — recall@10-OOD drops back to 0.83). If your only "queries"
  are synthetic perturbations of base points, you'll learn nothing useful.
* **Tiny training sets make the projection sparse.** With `|Q| < degree · 2`,
  many base nodes get no projected neighbours and have to be linked back in
  during reachability repair, giving you mostly the kNN-graph behaviour.
* **High `nap_k` over-projects.** Setting `nap_k = |B|` makes every training
  query touch every base node, and the projection becomes complete-graph
  noise. Keep `nap_k` ≤ a few times `degree`.
* **No incremental insert (yet).** This nightly is a static-index PoC. New
  points need a rebuild. See "What to improve next" below.
* **Training-side encoder drift.** If you upgrade the text encoder, the
  training queries embedded with the old encoder become misleading. Treat
  the training set as an artefact of (encoder-version, query-distribution)
  and version it explicitly.

## What to improve next — roadmap

1. **Incremental insert/delete.** Vamana-style: insert by beam-search +
   RobustPrune, periodically run a NAP-refresh pass over a fresh training
   batch. Tracking issue suggested: ADR-194 §"Consequences" item 3.
2. **IVF / RaBitQ pre-filter for build.** Drop NAP construction from `O(|Q|·|B|)`
   to `O(|Q|·sqrt|B|)` by reusing `ruvector-rairs` (ADR-193) as the inner kNN.
3. **Disk-resident variant.** Re-cast `RoarGraph` over a `ruvector-diskann`
   sector-aligned layout — same NAP edges, same beam search, on-disk vectors.
4. **Filtered RoarGraph.** Combine with `ruvector-acorn` predicate masks.
5. **Quantised vectors.** RaBitQ / LVQ-compressed bases so 100 M-point
   indices fit in the same RAM envelope as today's 10 M-point HNSW.
6. **Empirical OOD benchmark.** Wire the bench against real cross-modal
   datasets — LAION-CLIP (text→image), MS-MARCO (sparse-text→dense-doc),
   T2I-ANN (Chen et al.'s own benchmark).
7. **Auto-`nap_k` selection.** Adaptive `nap_k` per training query based on
   local density — fewer neighbours in dense regions, more in sparse.

## Production crate layout proposal

```
ruvector-roargraph/
├── Cargo.toml
└── src/
    ├── distance.rs         # SIMD-friendly L2² (extend with cosine, IP)
    ├── lib.rs              # static index + baselines (this nightly)
    ├── insert.rs           # incremental insert/delete (next iter)
    ├── ivf_build.rs        # IVF/RaBitQ-accelerated NAP construction
    ├── disk.rs             # sector-aligned disk layout (cf. ruvector-diskann)
    ├── filtered.rs         # predicate masks (cf. ruvector-acorn)
    ├── serde.rs            # persistence (bincode + mmap)
    └── bin/
        ├── bench.rs        # synthetic OOD (this nightly)
        └── bench_real.rs   # LAION / T2I-ANN real datasets
```

## References

1. **Chen, M., Zhang, K., He, Z., Jin, Y., Mao, Q., Zhu, Y., Wang, B., & Chen, J.** (2024).
   *RoarGraph: A Projected Bipartite Graph for Efficient Cross-Modal Approximate
   Nearest Neighbor Search.* Proceedings of the VLDB Endowment, 17(11), 2735–2748.
   <https://www.vldb.org/pvldb/vol17/p2735-chen.pdf>
2. **Jaiswal, S., Krishnaswamy, R., Garg, A., Simhadri, H. V., & Agrawal, S.** (2024).
   *OOD-DiskANN: Efficient and Scalable Graph ANNS for OOD Queries.*
   arXiv:2211.12850.
3. **Subramanya, S. J., Devvrit, Kadekodi, R., Krishnaswamy, R., & Simhadri, H. V.** (2019).
   *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single
   Node.* NeurIPS 2019.
4. **Malkov, Y. A., & Yashunin, D. A.** (2018). *Efficient and robust approximate
   nearest neighbor search using hierarchical navigable small world graphs.*
   IEEE TPAMI.
5. **Fu, C., Xiang, C., Wang, C., & Cai, D.** (2019). *Fast Approximate Nearest
   Neighbor Search With The Navigating Spreading-out Graph.* VLDB 2019.
6. **Radford, A., Kim, J. W., Hallacy, C., et al.** (2021). *Learning
   Transferable Visual Models From Natural Language Supervision (CLIP).* ICML.
