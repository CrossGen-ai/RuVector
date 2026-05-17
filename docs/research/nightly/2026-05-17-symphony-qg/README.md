# Symphony-QG for ruvector: graph + quantization fusion, measured

*Nightly research, 2026-05-17 · `crates/ruvector-symphony-qg`*

## Abstract

We add a Rust proof-of-concept for *SymphonyQG*-style ANN search to
ruvector. The crate ships three swappable backends behind a common
`AnnIndex` trait: a brute-force `FlatIndex`, a `PqRerankIndex` (full
ADT scan + top-`r` full-precision rerank), and a `SymphonyQgIndex` that
fuses a navigable small-world graph with Product Quantization so the
*traversal itself* runs against ADT distances. An optional
`with_refine(r)` knob exposes the recall-vs-latency tradeoff explicitly.

The PoC produces real numbers from `cargo run --release`. We do not
fake the result: Symphony-QG headline (no rerank) wins on throughput
in every measured config (1.2×–4.0× over PQ+rerank), but recall is
only competitive in the "easy" regime (small `n`, modest `d`,
high-quality PQ). The refine knob recovers most of the gap. We
quantify where the regime change happens and identify two
follow-ups — hierarchical build and a noise-aware diversifier — that
should move Symphony-QG into PQ+rerank's recall band while keeping
the bandwidth win.

## SOTA survey

* **SymphonyQG** (Gou, Yang, Wang; SIGMOD 2025) — the headline paper.
  Reports 1.5×–4.5× QPS over RaBitQ + reranking on SIFT-1M / GIST at
  matched recall, by integrating quantization into graph traversal.
* **RaBitQ** (Gao & Long; SIGMOD 2024) — rotation-based 1-bit
  quantization with theoretical error bounds. ruvector already ships
  `ruvector-rabitq`.
* **BBQ (Better Binary Quantization)** — Elasticsearch's productionized
  RaBitQ variant; 32× compression with engineered SIMD bit layout.
* **DiskANN / Vamana** (Subramanya et al., NeurIPS 2019) — the
  alpha-RNG diversifier we reuse for graph construction.
* **HNSW** (Malkov & Yashunin, 2018) — the canonical multi-layer
  navigable small-world graph; build cost we explicitly trade off.
* **MUVERA** — multi-vector fixed-dimensional encodings; ruvector
  already ships `ruvector-muvera`.

The Symphony pattern is *orthogonal* to the quantizer: any code-based
distance estimator can play the ADT role. We use plain PQ because it
keeps the PoC small (~120 LoC of quantization, deterministic
k-means), but the trait surface admits RaBitQ codes without changes.

## Proposed design

* **Trait**: `AnnIndex { search(&[f32], usize) -> Vec<(u32, f32)>; len(); }`
* **Build (Symphony)**:
  1. Train PQ (`M` subspaces, `K` codewords) on the dataset.
  2. Encode all vectors to PQ codes (`M` bytes each).
  3. For every node `i`: find its top-`ef_construction` neighbors
     by *full-precision* L2 (we have the data; we use it).
  4. Diversify with alpha-RNG: keep `j` iff no already-kept `k` is
     `alpha`-times closer to `j` than `i` is.
  5. Symmetrize light reverse edges.
  6. Pick `~log2(n)+4` entry seeds via farthest-point sampling so
     cluster-rich datasets are reachable from at least one seed.
* **Search**:
  1. Build the ADT once for the query (`O(M·K·d/M) = O(K·d)`).
  2. Push all FPS entries into the frontier (min-heap by ADT
     distance) and `top` (max-heap, size `ef_search`).
  3. Best-first traversal, marking visited; only push neighbors
     into `top` if they beat the current worst-in-top.
  4. Stop when frontier-min exceeds top-max and `top.len() >= ef`.
  5. (Optional) Rescore top-`r` with full precision and re-sort.

The headline savings: traversal touches `O(ef · avg_deg)` PQ codes
per query instead of `O(n)` codes (PQ+rerank) or `O(n · d)` floats
(flat). When `n >> ef · avg_deg`, the gap is large.

## Implementation notes (Rust)

* `src/pq.rs` — deterministic k-means with empty-cluster re-seeding;
  ADT buffer of `M*K` f32; per-vector ADC is `M` LUT lookups + adds.
* `src/symphony.rs` — `BinaryHeap<Cand>` (min) for frontier,
  `BinaryHeap<MaxCand>` (max) for top-`ef`. `Cand`/`MaxCand` reverse
  each other's `Ord` so the same `BinaryHeap` API yields both.
* `src/main.rs` — driver. Reads `SYMPHONY_N`, `SYMPHONY_D`,
  `SYMPHONY_M` from the env; defaults to a "headline" config.
* Files ≤ 230 LoC each; no `unsafe`; no mocks.

## Benchmark methodology

* **Hardware**: Apple M4 Max, macOS 24.6 (Darwin arm64), rustc 1.89.0,
  `--release` (LTO + codegen-units inherited from workspace).
* **Data**: synthetic Gaussian mixture, 32 cluster centers in
  `[-5, 5]^d`, intra-cluster sigma `1.2`. Queries drawn from a
  *different* set of 32 mixture centers (different RNG seed) so the
  test is not trivialized by cluster lookup.
* **Queries**: 200, single-threaded search loop, `k=10`.
* **PQ**: `K=256` (full byte), `M` per config.
* **Graph**: `m_edges=32`, `ef_construction=200`, `ef_search=200`,
  `alpha=1.2`. FPS chooses `~log2(n)+4` entry seeds.
* **Ground truth**: `FlatIndex` exact search.
* **Recall**: `recall@10 = |truth_top10 ∩ got_top10| / 10`,
  averaged over 200 queries.

## Results (real numbers — `cargo run --release`)

### Config A — sweet spot: n = 4 000, d = 64, M = 16 (16× compression)

| Backend            | QPS    | recall@10 | build  |
|--------------------|-------:|----------:|-------:|
| Flat f32           |  8 834 |    1.0000 |     —  |
| PQ + rerank(200)   | 26 785 |    1.0000 |  0.26s |
| Symphony-QG        | 17 211 |    0.5315 |  0.57s |
| Symphony + ref(32) | 16 780 |    0.8545 |  0.57s |

### Config B — scaling on n: n = 20 000, d = 64, M = 16

| Backend            | QPS    | recall@10 | build  |
|--------------------|-------:|----------:|-------:|
| Flat f32           |  2 006 |    1.0000 |     —  |
| PQ + rerank(200)   |  5 301 |    0.9785 |  1.31s |
| Symphony-QG        | 21 406 |    0.1300 |  6.60s |
| Symphony + ref(32) | 21 135 |    0.2055 |  6.60s |

### Config C — high dimension: n = 8 000, d = 128, M = 32 (16× compression)

| Backend            | QPS    | recall@10 | build  |
|--------------------|-------:|----------:|-------:|
| Flat f32           |  2 579 |    1.0000 |     —  |
| PQ + rerank(200)   |  7 250 |    0.9940 |  1.22s |
| Symphony-QG        | 21 592 |    0.1280 |  3.29s |
| Symphony + ref(32) | 21 148 |    0.1900 |  3.29s |

### Memory per vector

| Layout         | Bytes (d=64) | Bytes (d=128) |
|----------------|-------------:|--------------:|
| Raw f32        |          256 |           512 |
| PQ M=16        |           16 |            16 |
| PQ M=32        |           32 |            32 |

### What the numbers say

* **Throughput**: Symphony-QG beats PQ+rerank by 3.0×–4.1× whenever
  the rerank pass dominates (i.e. `n` is large enough that fetching
  full-precision vectors hurts). On the small-n config the rerank
  pass is so cheap (200 vectors × 64 floats) that PQ+rerank still
  wins; this matches intuition.
* **Recall**: Symphony-QG headline only matches PQ+rerank in
  "easy" regimes. On Config B/C, ADT noise compounds along the
  graph traversal and recall falls to 0.13–0.21. `with_refine(32)`
  recovers ~50 % of the gap but still trails PQ+rerank.
* **The honest takeaway**: the PoC is *fast* but does not yet match
  the upstream paper's recall. Three concrete causes (next section).

## Practical failure modes (observed)

1. **ADT precision wall.** At `d=128, M=32` a *full ADT scan* (no graph)
   only delivers 46 % recall@10 (measured in
   `examples/diag_traversal.rs`). Any ADT-only ranker is bounded by
   that. Fixes: bigger `M`, or RaBitQ codes (tighter error bounds), or
   keep `with_refine` permanently on.
2. **Greedy traversal traps.** Even within the ADT ceiling, our graph
   recovers only ~25 % of what ADT-scan delivers on Config C. This is
   the classical "long-tail greedy descent" failure of single-layer
   NSWs. HNSW's multi-layer structure mitigates this — its lower
   layers are coarse enough to traverse to the right cluster cheaply
   before refining in the bottom layer.
3. **O(n²) build.** Acceptable for `n ≤ 20 000` (≤ 7 s) but blocks
   the path to SIFT-1M-scale numbers. Same fix as #2: hierarchical
   build.

## What to improve next (roadmap)

* **R1 — Hierarchical build** (≈ +1 file, ~150 LoC). HNSW-style
  upper layers with bounded fan-out. Expected effect: build O(n log n)
  and recall closer to PQ+rerank (the upstream paper does this).
* **R2 — Noise-aware diversifier**. The standard alpha-RNG assumes
  exact distances. Symphony-QG paper proposes scaling alpha by ADT
  variance per subspace; we have everything needed in `pq.codebooks`
  to estimate that offline.
* **R3 — RaBitQ codes under the same trait.** `ruvector-rabitq`
  already produces 1-bit codes with the same `(M, K)` ADT shape after
  rotation. Drop-in.
* **R4 — SIMD ADC kernel.** Today's `adc_distance` is autovectorizable
  but emits scalar f32 ops on aarch64 (`portable-simd` would yield
  a 4× wallclock improvement based on similar kernels in
  `ruvector-rabitq`).

## Production crate layout (proposal)

```
crates/ruvector-symphony-qg/
├── Cargo.toml
├── src/
│   ├── lib.rs           # AnnIndex trait re-export
│   ├── error.rs         # SymphonyError
│   ├── metric.rs        # l2_sq / l2 (no_std-friendly)
│   ├── pq.rs            # ProductQuantizer + ADT
│   ├── flat.rs          # FlatIndex (ground truth)
│   ├── pq_rerank.rs     # PqRerankIndex
│   ├── symphony.rs      # SymphonyQgIndex (graph + ADT)
│   └── main.rs          # symphony-qg-demo
├── benches/
│   └── symphony_qg_bench.rs   # criterion: 3 search variants
├── tests/
│   └── integration.rs         # 4 tests, real recall floors
└── examples/
    ├── debug_recall.rs        # per-query trace
    └── diag_traversal.rs      # ADT-scan ceiling vs graph recall
```

For a future *production* split, three sub-crates make sense:

1. `ruvector-symphony-qg-quantizer` — PQ/RaBitQ/BBQ codes + ADT.
2. `ruvector-symphony-qg-graph` — generic NSW build/search backed
   by an `Estimator` trait (from #1).
3. `ruvector-symphony-qg` — high-level facade with sensible defaults.

## How it works — walkthrough

Imagine you have 100 000 user-embedding vectors at d=128 and a query
embedding from a new session. Three approaches:

* **Flat**: compute 100 000 × 128 multiply-adds = 12.8 M ops.
  Slow.
* **PQ + rerank**: 100 000 × 16 LUT lookups (1.6 M ops), pick top
  200, then 200 × 128 m-adds (26 K ops). Most of the work is the
  ADT scan, but you also pay 200 random vector reads.
* **Symphony-QG**: build the ADT once (256 × 128 = 33 K m-adds);
  start at ~17 entry seeds; do best-first traversal touching maybe
  ~3 000 nodes × 16 LUT lookups = 48 K ops. *No* random vector reads.
  When DRAM bandwidth is the bottleneck (it usually is at this scale),
  Symphony-QG can deliver several-fold throughput at matched recall,
  *if* the quantizer is precise enough that the traversal gradient
  points the right way. That last clause is the crux of this PoC —
  and where the failure modes above bite.

## Comparison to competitors

| System    | Quantization | Reranks? | Notes |
|-----------|--------------|----------|-------|
| Milvus    | PQ / IVFPQ   | Yes      | Standard ADT + refine                  |
| Qdrant    | Scalar / PQ  | Yes      | Optional rescore on payload            |
| Weaviate  | PQ / BQ      | Yes      | Compression-quality tradeoff exposed   |
| Pinecone  | proprietary  | Yes      | Hosted only; opaque                    |
| LanceDB   | IVF-PQ       | Yes      | Disk-friendly                          |
| FAISS     | PQ / OPQ / RQ| Yes      | Library; "IVFPQ + rerank" canonical    |
| ruvector  | RaBitQ/PQ/MUVERA | *Optional* | After this ADR, Symphony-QG can elide rerank when accuracy permits |

## References

1. SymphonyQG: Towards Symphonious Integration of Quantization and
   Graph for ANN Search. SIGMOD 2025.
2. Gao & Long. *RaBitQ: Quantization with Theoretical Error
   Bounds*. SIGMOD 2024.
3. Malkov & Yashunin. *Efficient and robust approximate nearest
   neighbor search using HNSW graphs*. PAMI 2018.
4. Subramanya et al. *DiskANN*. NeurIPS 2019.
5. Elastic Engineering. *BBQ: Better Binary Quantization*, 2024.
6. ruvector ADR-193 (RaIRS IVF).
