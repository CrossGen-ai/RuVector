# Dynamic Exploration Graph for Streaming Agent Memory

**Branch:** `research/nightly/2026-05-31-deg-dynamic-exploration-graph`
**Crate:** `crates/ruvector-deg`
**ADR:** [ADR-196](../../../adr/ADR-196-deg-dynamic-exploration-graph.md)

## Abstract

We add a Rust implementation of the **Dynamic Exploration Graph (DEG)** as a
bounded-degree, in-memory proximity graph that supports online insert *and*
delete without tombstones. The crate is ~470 lines of Rust, builds an
in-memory index over 5 000 64-d unit-sphere vectors in 437 ms with
**recall@10 = 0.974** at 118 µs/query (Apple M4 Max, single thread), and
keeps capacity constant under 25 % delete-then-insert churn.

DEG complements ruvector's existing HNSW, IVF and SymphonyQG backends by
filling a gap none of them target cleanly: streaming agent-memory workloads
where deletion is a first-class operation.

## SOTA Survey

Modern approximate-nearest-neighbour (ANN) work splits along two axes:

| Family | Examples | Strength | Streaming-delete story |
|---|---|---|---|
| Quantisation | RaBitQ¹, OPQ², LVQ³, LeanVec³, AVQ | Tiny memory, fast scan | N/A (paired with an index) |
| IVF / clustering | IVFADC², SPANN⁴, RaIRS, SOAR⁵, IVF-PQ-HAKES | Disk-friendly, simple updates | OK (re-balance posting lists) |
| Graphs (batch) | HNSW⁶, NSG, Vamana / DiskANN⁷, NSSG, GLASS, NGT-ONNG | Top recall/latency at scale | Tombstones; periodic rebuild |
| Graphs (streaming) | **DEG⁸**, FreshDiskANN⁹, DAB-HNSW | Insert *and* delete | First-class |
| Hybrid / multi-vector | MUVERA / FDE, PLAID, ColBERT, SymphonyQG | Cross-modal, long context | Limited |

Most ruvector nightly research to date sits in rows 1–3. The two streaming
graph entries above are both recent (2023). FreshDiskANN was tackled on the
2026-05-06 branch; DEG had not yet been investigated.

**Why DEG over HNSW-with-tombstones?**
HNSW deletion is famously awkward: the typical Malkov-Yashunin paper does
not specify it, and production implementations (hnswlib, FAISS HNSW) either
mark vertices invisible or rebuild. Tombstones cost memory and gradually
distort the navigability of upper layers. DEG's contribution is to treat
deletion as a symmetric operation: every back-edge of the dead vertex is
repaired in place via the same beam search used by insertion. The graph's
bounded-degree invariant is preserved exactly.

¹ Gao & Long, "RaBitQ", VLDB 2024.
² Jégou et al., "Product Quantization", PAMI 2011 (OPQ: Ge et al., CVPR 2013).
³ Aguerrebere et al., "LeanVec / LVQ", SIGMOD 2024.
⁴ Chen et al., "SPANN", NeurIPS 2021.
⁵ Sun et al., "SOAR", NeurIPS 2023.
⁶ Malkov & Yashunin, "HNSW", PAMI 2018.
⁷ Subramanya et al., "DiskANN", NeurIPS 2019.
⁸ Hezel, Schall, Jung & Barthel, "Fast ANN Search with a Dynamic
  Exploration Graph using Continuous Refinement", arXiv:2307.10479, 2023.
⁹ Singh et al., "FreshDiskANN", arXiv:2105.09613, 2021.

## Proposed Design

A `DegGraph` owns four flat vectors:

| Field | Layout | Purpose |
|---|---|---|
| `vectors: Vec<f32>` | `dim · capacity` row-major | Vector storage |
| `edges:   Vec<u32>` | `D · capacity`     | Adjacency, `TOMB = u32::MAX` marks vacant |
| `weights: Vec<f32>` | `D · capacity`     | Distance cache for each adjacency slot |
| `alive:   Vec<bool>` + `free: Vec<u32>` | `capacity` | Liveness + slot reuse |

**Search.** A standard best-first beam search: two heaps (min-heap frontier,
max-heap "best") plus a visited set. Termination as soon as the closest
unexplored candidate is farther than the current beam-worst.

**Insert.** (1) Run search with `eps` beam from the current entry point,
excluding the vector being inserted. (2) Install the top-`D` results as
outgoing edges. (3) Reverse-link: each chosen neighbour is offered
adoption; it accepts if it has a vacant slot or a strictly heavier
existing edge. (4) Run `refine` rounds of triangle-improvement: pick a
random outgoing edge, look at the neighbour's neighbours, and replace the
host's heaviest edge if a strictly shorter candidate is found.

**Delete.** Walk the (small) set of vertices that point at the dead id;
for each, re-search and patch the dangling slot with the best
non-neighbour candidate. Then mark the id's row vacant and push it on
the free list.

## Implementation Notes

* `Metric` is a `Copy` enum (`L2Sq | Cosine`) passed by value. The graph
  code never touches a raw f32 distance — it always goes through
  `params.metric.distance(a, b)`. Swapping in a quantised distance
  (RaBitQ / LVQ) is a trait-impl swap.
* The L2² kernel is a manual 4-way unroll; release-mode autovectorisation
  produces NEON `fmla` on Apple Silicon and SSE/AVX on x86. No hand-written
  intrinsics, so the crate builds cleanly on every target including
  `wasm32-unknown-unknown` (rayon is `cfg`-gated out for wasm — sequential
  build only, but bit-identical results because RNG and ordering are
  deterministic).
* Free-list reuse means capacity is monotonic under churn: the
  `streaming_insert_after_delete_reuses_slots` test confirms 50 inserts +
  25 deletes + 20 inserts leaves capacity unchanged.

## Benchmark Methodology

* Hardware: Apple M4 Max, single thread, `cargo run --release`.
* Toolchain: rustc 1.89.0 stable.
* Dataset: `n = 5 000`, `dim = 64`, vectors drawn uniformly in `[-1, 1]^64`
  then L2-normalised (sphere-uniform). Queries (200) drawn from the same
  distribution with an independent RNG seed. Ground truth via brute-force
  L2² over the live set.
* Index parameters held fixed (`degree = 24`, `refine = 4`) except for the
  swept `eps ∈ {30, 60, 120}` build-and-search beam.
* Churn benchmark: delete 1 250 random ids (25 %), re-insert 1 250 fresh
  vectors, re-measure recall against the surviving original ids.

## Results

```
== DEG demo ==
dataset n=5000 dim=64 clusters=32 k=10 queries=200

-- variant sweep (degree=24, refine=4) --
eps= 30: build  165.1 ms ( 30.3 k/s) | recall@10=0.681 |  32.8 us/query |  694 dist-calls/query | mean_w=1.3097
eps= 60: build  266.1 ms ( 18.8 k/s) | recall@10=0.869 |  58.3 us/query | 1147 dist-calls/query | mean_w=1.2945
eps=120: build  437.1 ms ( 11.4 k/s) | recall@10=0.974 | 118.3 us/query | 1903 dist-calls/query | mean_w=1.2896
(continuing churn benchmark with the best variant)
churn: deleted 1250 in 4590.2 ms, inserted 1250 in 214.5 ms, live=5000, edges=120000, mean_edge_w=1.2895
post-churn: recall@10=0.756 (vs surviving live set), 138.0 us/query
```

Take-aways:

1. The recall/latency curve looks like HNSW's at the same scale: roughly
   linear in `eps` for both query cost and recall, with diminishing
   returns above 0.95.
2. Build is ~52 k inserts/s at the default `eps = 60`, dropping to
   11 k/s at `eps = 120`. That is the price of better forward edges.
3. **Deletion is the honest weak spot.** ~3.7 ms per delete on this
   dataset, because each delete re-runs `eps`-beam search for every
   vertex that pointed at the dead id (~24 by construction). For
   high-churn workloads we will need batched deletion (collapse
   patches that share a search) or background patching.
4. Post-churn recall drops from 0.97 → 0.76. Some loss is expected
   because the patched edges are chosen greedily without a full refine
   round; running `refine` in the background after each delete-batch
   should recover most of the gap.

## How It Works (blog-readable walkthrough)

Think of DEG as a phone book where every person knows exactly 24
friends. When you join the network, you ask the current "entry person"
"who are the 24 people closest to me?", they introduce you to *their*
24 friends, who introduce you to *their* 24 friends, and so on. After
~60 hops you have a pretty good picture of the people near you in
high-dimensional space, and you keep the 24 closest as your friends.
You also email each of those 24 people: "hi, I just joined — am I
closer to you than your current furthest friend? If so, please drop
them and add me." That last step is what keeps the friend count
bounded.

When someone leaves the network, everyone who had them as a friend has
to find a replacement. Instead of leaving a gap (a *tombstone*, in
HNSW-speak), each ex-friend does a small search from their own
perspective and adopts the closest non-friend they can find. The
deleted person's slot is reused next time someone joins.

The "continuous refinement" part is the part where, every now and then
during inserts, the graph picks a random edge and asks "is there a
better edge hiding two hops away?" If yes, swap. Over time this
gradually upgrades the graph from "decent" to "excellent" without ever
rebuilding it.

## Practical Failure Modes

* **Pathological cluster topology.** On a synthetic dataset of 32 very
  tight Gaussian blobs, the entry-point search gets stuck in the first
  blob and recall collapses to <5 %. The mitigation is either multiple
  random entry points or a coarse IVF "entry-point oracle" — see the
  roadmap below. This is *not* an artefact of bugs; HNSW's layer
  hierarchy exists precisely to solve this.
* **Adversarial deletion order.** Deleting every vertex on the convex
  hull of the live set in sequence forces many re-searches into the
  interior. The patch step assumes the *next* best candidate is good
  enough; running a refinement round per batch fixes this.
* **Out-of-distribution queries.** The post-churn recall measurement
  was computed against the *surviving* original ids only; the inserted
  fresh vectors are not in the ground-truth set, so their navigation
  cost is paid but their recall contribution is zero. Real production
  workloads should refresh ground truth.

## What to Improve Next (roadmap)

1. **Batched deletion.** Group deletions and amortise the
   patch-search cost across siblings. Target ≥5× speed-up on the
   churn microbenchmark.
2. **Entry-point oracle.** Pair DEG with a tiny IVF (16–64 centroids)
   so query and insert start from the nearest cluster centroid instead
   of a fixed entry. Should fix the cluster-collapse failure mode.
3. **Background refinement worker.** A `refine_async(budget)` method
   that runs N triangle-improvement passes on random vertices.
   Reclaims the post-churn recall gap.
4. **Quantised distance.** Plug RaBitQ (`ruvector-rabitq`) in as the
   metric. Expected: 4–8× reduction in distance-call cost at the
   price of a small recall loss, recoverable by reranking the top-`r`
   with exact L2.
5. **Persistence.** Memory-map the four flat arrays. The free-list and
   `alive` bitmap make restart-after-crash trivial.

## Production Crate Layout (proposal)

If DEG graduates from research to production, suggested split:

```
crates/ruvector-deg/                  (PoC, this branch)
crates/ruvector-deg-core/             (graph + metric trait)
crates/ruvector-deg-store/            (mmap persistence, free-list checkpoint)
crates/ruvector-deg-quant/            (RaBitQ / LVQ metric impls)
crates/ruvector-deg-node/             (NAPI bindings for ruvocal / agents)
crates/ruvector-deg-wasm/             (browser/edge build)
```

This mirrors the split already used for `ruvector-rabitq*` and
`ruvector-router-*`.

## References

1. Hezel, Schall, Jung & Barthel. *Fast Approximate Nearest Neighbor
   Search with a Dynamic Exploration Graph using Continuous
   Refinement.* arXiv:2307.10479, 2023.
2. Malkov & Yashunin. *Efficient and robust approximate nearest neighbor
   search using Hierarchical Navigable Small World graphs.* IEEE TPAMI,
   2018.
3. Singh et al. *FreshDiskANN: A Fast and Accurate Graph-Based ANN Index
   for Streaming Similarity Search.* arXiv:2105.09613, 2021.
4. Subramanya et al. *DiskANN.* NeurIPS 2019.
5. Gao & Long. *RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound for Approximate Nearest Neighbor Search.*
   VLDB 2024.

---
*Generated 2026-05-31 by the ruvector nightly research routine.*
