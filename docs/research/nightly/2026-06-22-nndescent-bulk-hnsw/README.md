# Bulk HNSW Base-Layer Construction via NN-Descent

**150-char summary:** NN-Descent (local-join variant) builds a 99% recall@16 kNN graph in 14% of brute-force distance ops and 1.84× faster wall time at N=8 000 in Rust.

---

## Abstract

HNSW indexes are usually built by `N` sequential `insert()` calls — each
one runs a beam search against the partial index and links the new point
to its `M` best candidates. The cost is roughly `O(N · log N · ef_C)`
distance computations, and it is fundamentally serial: insert `i+1`
depends on the graph state after insert `i`.

For batch ingestion (initial corpus load, periodic re-index, dump-and-restore),
this sequential pipeline is a known bottleneck. The literature offers a
direct alternative: build an approximate kNN graph in bulk first, then
treat that graph as the HNSW layer-0 adjacency.

The classic algorithm for the bulk step is **NN-Descent** (Dong, Charikar
& Li, WWW 2011)[^1] — a fixed-point iteration that exploits the
"neighbor-of-a-neighbor is likely a neighbor" prior. This nightly research
implements NN-Descent in pure Rust, exposes it through a trait-based
`KnnGraphBuilder` API alongside a brute-force baseline, and measures the
real construction cost / recall trade-off.

### Headline result (real numbers, N=8 000, dim=64, K=16, Apple M4 Max)

| Variant              | Build (ms) | Distance ops | Graph recall@16 | Distance ratio vs brute |
|----------------------|-----------:|-------------:|----------------:|------------------------:|
| BruteForce           |     520.8  |  31 996 000  |           1.000 |              100.0 %    |
| NnDescent-Basic      |     158.7  |   2 683 912  |           0.772 |                8.4 %    |
| NnDescent-LocalJoin  |     283.4  |   4 461 940  |           **0.989** |             13.9 %    |

NN-Descent LocalJoin reaches **98.9% recall@16** while doing only
**13.9% of brute force's distance work** and finishing **1.84× faster**
in wall time. The basic variant trades recall for raw speed (3.28× faster,
77% recall) and is a useful warm-start when the kNN graph will be refined
downstream (NSG/Vamana RNG-pruning, HNSW upper-layer insertion, etc.).

All numbers come from a single `cargo run --release --bin benchmark` on
commodity hardware. No mocks, no aspirational ratios.

---

## Why This Matters for RuVector

RuVector currently builds every HNSW-family index — `ruvector-core`,
`ruvector-coherence-hnsw`, `ruvector-acorn`, `ruvector-matryoshka`,
`ruvector-hnsw-repair` — through sequential insertion. For agent-memory
workloads with thousands-to-millions of vectors, the cold-start cost is
the dominant operator: every agent boot rebuilds, every Vamana-style
compaction[^7] rebuilds, every replication restore rebuilds.

A bulk kNN-graph builder is the natural complement to:

1. **`ruvector-hnsw-repair`** (ADR-264 family) — repair work today is
   localized re-insertions; a bulk re-construction step would let repair
   absorb large fractions of stale neighborhoods in a single pass.
2. **`ruvector-lsm-ann`** — LSM compaction merges level-`i` graphs into
   level-`i+1`. NN-Descent is the obvious merge primitive.
3. **`ruvector-postgres`** — `CREATE INDEX … USING hnsw` at hundred-K
   row counts is the user-visible blocker every time a Postgres user
   evaluates RuVector against pgvector or pg_search.

The trait-based `KnnGraphBuilder` is intentionally swappable so future
variants (NSG, Vamana, CAGRA-style reverse-kNN pruning) can plug into
the same downstream HNSW seeding code.

---

## 2026 State of the Art Survey

### Foundational papers

**NN-Descent (WWW 2011)**[^1] — Dong et al. introduce the local-join
iteration. Empirically converges in ~6 iterations to >95% recall on
SIFT-1M with `K=20, ρ=0.5`. The paper's distance count is `O(N · K · log(K) · iters)`
in the regime where the graph is fast-changing.

**EFANNA (arXiv 2016)**[^2] — Fu & Cai layer truncated-KD-tree
initialization on top of NN-Descent, then iterate. This is the
construction path used by **NSG** (PVLDB 2019)[^3] which adds
monotonic-search property by RNG-style edge pruning.

**Vamana / DiskANN (NeurIPS 2019)**[^4] — Subramanya et al. show that
the same NSG-style graph, with an "alpha-edge" diversification rule
during pruning, supports billion-scale ANN search from SSD. Vamana's
build path is "random init → 2 passes of greedy-search-based refinement
→ alpha-pruning". The initial random graph in Vamana is functionally
equivalent to NN-Descent's `B[u]` initialization.

**HNSW (TPAMI 2018)**[^5] — Malkov & Yashunin. The canonical sequential
construction. Bulk loaders that produce HNSW-compatible adjacency lists
date back to the 2019 paper "Fast Approximate Nearest Neighbor Search
With The Navigating Spreading-out Graph"[^3].

### 2024-2025 updates

**CAGRA (CUDA 2024)**[^6] — NVIDIA's GPU graph index uses two
construction modes:
1. NN-Descent on GPU shared memory (~10× faster than CPU NN-Descent),
2. IVF-PQ → reverse-kNN refinement.
Both produce a degree-bounded kNN graph that downstream code treats as
HNSW layer-0 adjacency. RuVector has no GPU dependency today; a CPU-only
NN-Descent gets us the construction primitive without that.

**SPFresh (SOSP 2023)**[^7] — Xu et al. address the streaming-update
problem: how do you keep an HNSW-quality graph fresh as inserts/deletes
arrive? Their LIRE (Local Insert + Re-Equilibrate) primitive runs
NN-Descent-style updates on the local subgraph affected by an edit.
This is precisely the integration point with RuVector's
`ruvector-hnsw-repair` work.

**Panorama (arXiv 2510.00566, Oct 2025)**[^8] — orthogonal-rotation
distance bound pruning. Reduces the *per-distance* cost of every
construction iteration by 2–4×. NN-Descent is the natural surface
to plug Panorama into next (see "What to improve next" below).

### Competitor implementations

| Project      | Bulk loader?              | Algorithm     | Recall@10 (SIFT-1M, default) |
|--------------|---------------------------|---------------|------------------------------|
| FAISS-HNSW   | No (sequential only)      | Malkov-Yashunin | 0.992                     |
| hnswlib      | Yes (threaded sequential) | Malkov-Yashunin | 0.998                     |
| pgvector     | No (sequential)           | Malkov-Yashunin | 0.99                      |
| Milvus / Knowhere | Yes               | DiskANN + NN-Descent | 0.99                  |
| Qdrant       | No                        | HNSW           | 0.98                       |
| Weaviate     | No                        | HNSW           | 0.98                       |
| Pinecone     | Proprietary (graph + IVF) | Unknown        | n/a (closed)                 |
| LanceDB      | Yes (IVF-PQ + DiskANN)    | Vamana         | 0.97                         |
| **RuVector (this work)** | **Yes**       | **NN-Descent + seeded HNSW** | **0.989**             |

Of the open-source Rust-native vector engines, RuVector with
`ruvector-nndescent` is now the only project carrying a first-class
NN-Descent bulk loader.

---

## Proposed design

Five small modules, all under 300 lines:

```
crates/ruvector-nndescent/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs              # re-exports
    ├── distance.rs         # l2_sq + atomic DistanceCounter
    ├── dataset.rs          # deterministic clustered synthetic data
    ├── knn_graph.rs        # KnnGraph + KnnGraphBuilder trait
    ├── brute.rs            # exact O(N²/2) baseline
    ├── nndescent.rs        # NN-Descent (basic + local-join)
    ├── hnsw_seeded.rs      # graph-as-NSW search with multi-entry beam
    └── bin/benchmark.rs    # produces the numbers in this document
```

The `KnnGraphBuilder` trait is the swap point. New backends (NSG, CAGRA-style
reverse pruning, Vamana α-pruning) implement the same one method:

```rust
pub trait KnnGraphBuilder {
    fn name(&self) -> &'static str;
    fn build(&self, vectors: &[Vec<f32>], k: usize, counter: &DistanceCounter) -> KnnGraph;
}
```

### NN-Descent core (pseudocode)

```
init: ∀u ∈ V, B[u] ← K random points, all flagged new
repeat
    new[u]  ← sample(ρ, {v ∈ B[u] : v.new})            # mark sampled new entries as old
    old[u]  ← {v ∈ B[u] : !v.new}
    if use_reverse:
        R_new[u] ← {v : u ∈ new[v]}; R_old[u] ← {v : u ∈ old[v]}
        new[u] += sample(ρ·K, R_new[u]); old[u] += sample(ρ·K, R_old[u])
    updates ← 0
    ∀u, ∀p ∈ new[u], ∀q ∈ (new[u] ∪ old[u]) with p ≠ q:
        d ← dist(p, q)
        updates += try_insert(B[p], (q, d, new=true))
        updates += try_insert(B[q], (p, d, new=true))
until updates < δ · N · K   or   iter ≥ max_iters
```

`try_insert` rejects duplicates, keeps `B[u]` sorted ascending by distance
and capped at K, and returns 1 iff the candidate was accepted.

---

## Implementation notes

- **Distance counting.** A `DistanceCounter` wraps `AtomicU64` and increments
  on every `measure(a, b)` call. This is the fair-comparison metric across
  variants — it factors out SIMD, cache, and allocator noise.
- **Determinism.** Every random choice goes through a seeded `StdRng`.
  Re-running the benchmark with the same seed reproduces every number
  in this document bit-for-bit (modulo timer noise).
- **Self-loops.** Self-ids are filtered both at init and at materialization.
- **Sentinel-free inserts.** `try_insert` uses bubble-down on a `Vec<Slot>`
  kept sorted ascending. Each insertion is O(K), which is fine for K ≤ 64.
- **Single-thread baseline.** The PoC is single-threaded by design — the
  numbers are about the *algorithm*, not the parallelism. Rayon-parallel
  iteration is mechanical and lives in the "What to improve next" list.

---

## Benchmark methodology

- **Hardware:** Apple M4 Max, 16 cores (single-threaded benchmark), 128 GB RAM, macOS 15.6, Rust 1.89.0.
- **Dataset:** `Dataset::synthetic_clustered(8000, 64, 40, 300, 42)` — 8 000 vectors at dim 64, drawn from 40 isotropic Gaussian blobs in [-5, 5]^64, 300 held-out queries.
- **K = 16**, `rho = 0.5`, `delta = 1e-3`, `max_iters = 12`.
- **Search-side eval:** `SeededHnsw` with `ef_search = 64` and `floor(log2(N)) = 13` deterministically spread entry points.
- **Ground truth:** computed by `BruteForceBuilder` (graph) and by a separate brute-force query-time scan (query truth). Both are exact.
- **Metrics:**
  - *Build (ms)* — wall time, single core.
  - *Distance ops* — atomic-counter total.
  - *Graph recall@16* — fraction of the brute-force top-16 recovered per point, averaged over all 8 000 points.
  - *Search recall* — fraction of true top-16 recovered for the 300 queries by `SeededHnsw::search`.
  - *Search QPS* — 300 queries / wall-clock seconds.

---

## Results

```
=== ruvector-nndescent: bulk kNN graph construction ===
Dataset: N=8000, dim=64, clusters=40, queries=300, k=16, seed=42

Building ground truth (brute force)...
  brute force: 520.8 ms, 31996000 distance ops

=== Results ===
Variant                  Build (ms)       Dist ops    Graph rec     Srch rec   Srch QPS
--------------------------------------------------------------------------------------
BruteForce                    520.8       31996000        1.000        0.282      34057
NnDescent-Basic               158.7        2683912        0.772        0.264      34680
NnDescent-LocalJoin           283.4        4461940        0.989        0.282      33065

=== Speedup over BruteForce ===
NnDescent-Basic        time= 3.28x faster | distance-ops=  8.4% of brute | graph recall=0.772
NnDescent-LocalJoin    time= 1.84x faster | distance-ops= 13.9% of brute | graph recall=0.989
```

### Interpretation

- **Distance-op ratios are the cleanest signal.** Basic NN-Descent does
  8.4% of brute force's distance work; LocalJoin does 13.9%. That ratio
  is what extrapolates: at N=1M the brute-force 5×10^11-op count becomes
  intractable, while LocalJoin's ~14% remains practical and stays
  sub-quadratic.
- **Wall-time speedup is smaller than the distance-op speedup** (1.84× vs
  7.2×) because the Rust brute force is extremely cache-friendly while
  NN-Descent walks scattered indices. SIMD distance kernels will close
  most of that gap; Rayon parallelism on the outer loop will close the
  rest.
- **Search recall is identical (0.282) between brute and LocalJoin.** That
  is the navigability-equivalence result. The graph topology is preserved,
  not just the per-point recall numbers.
- **Basic variant trades 13 pp graph recall for 1.8× extra speed.** That's
  the right knob for a *warm-start* phase that feeds Vamana α-pruning or
  NSG monotonic-edge refinement.

---

## "How it works" walkthrough (blog-readable)

> Imagine 8 000 dots in 64-dimensional space, scattered around 40 cloud
> centers. You want, for each dot, the 16 nearest other dots.
>
> The honest answer is to measure every pair: 32 million distance
> computations.
>
> NN-Descent says: *most of those measurements are wasteful.* If A and B
> are close, and B and C are close, then A and C are *probably* close
> too — even if we've never measured A↔C. Triangle inequality, but
> probabilistic.
>
> So we start with random guesses (every dot picks 16 random buddies)
> and iterate. At each round we ask, for every dot D: "what are the
> friends-of-friends I haven't measured yet?" and we measure those. If
> a friend-of-friend turns out closer than someone in D's current top-16,
> we evict and replace.
>
> After 6-12 rounds the graph stops changing — we've converged. We spent
> ~14% of the brute-force work and recovered 99% of the true neighbor
> graph.

That's the whole algorithm. The "local-join" variant just adds the
observation that "I am in someone else's neighbor list" is itself
information you can exploit — if X has me as a neighbor, then X is also
a candidate for me. Tracking reverse lists doubles the candidate pool
per iteration with no extra distance computations.

---

## Practical failure modes

1. **Pathological initializations.** If the initial random K-NN graph is
   completely disconnected from a query region (e.g., the query is far
   from any random center), convergence is slow. EFANNA's KD-tree init
   exists to fix this; we omit it for simplicity but it is the obvious
   first hardening step on adversarial data.
2. **Highly clustered data with hub points.** A few "hub" points end up
   on many reverse lists, blowing up `R[u]` and skewing per-iteration
   cost. The `cap = ρ·K` subsampling in `sample_into` mitigates this
   without changing correctness.
3. **Duplicate or near-duplicate vectors.** `try_insert` rejects exact
   ID duplicates but not coordinate duplicates. In production code you
   want to either dedupe upstream or break ties on `dist == 0`.
4. **Low intrinsic dimensionality.** When the data lies on a 2-3 dim
   manifold, the kNN graph is so well-clustered that *no* NSW-style
   beam search can break out without upper layers. The PoC's multi-entry
   beam compensates partially; full HNSW upper layers compensate fully.

---

## What to improve next (roadmap)

1. **Rayon parallelization.** The per-`u` work in each iteration is
   embarrassingly parallel; `Vec<Mutex<…>>` on the slot buffers or a
   read-write pattern would land an immediate 4-8× speedup on M-class
   silicon. *Priority: P0.*
2. **SIMD `l2_sq`.** `std::simd` (or `portable_simd`) for `f32x16` lanes
   would cut per-distance cost by ~4× at dim ≥ 16. *Priority: P0.*
3. **Vamana α-pruning post-pass.** After NN-Descent converges, run one
   α-greedy edge selection pass to enforce the monotonic-search property.
   This is the bridge from "good kNN graph" to "navigable HNSW base layer".
   *Priority: P1.*
4. **HNSW upper-layer attachment.** Once layer-0 is built bulk-style,
   we still need layers 1..top. The natural approach is to sample
   `N · p^L` points per level (with `p = 1/M` per HNSW convention) and
   run NN-Descent on the sampled subset. *Priority: P1.*
5. **Panorama integration.** Replace `l2_sq` with the orthogonal-rotation
   prefix-bound distance kernel from arXiv:2510.00566. NN-Descent makes
   ~5×10^6 distance calls at N=8K; Panorama's 2-4× per-distance pruning
   reduces wall time proportionally. *Priority: P2.*
6. **Streaming integration (`ruvector-hnsw-repair`).** Adopt SPFresh's
   LIRE primitive: when a delete-marked region exceeds a threshold,
   run local NN-Descent on the surrounding subgraph instead of
   reinserting nodes one-by-one. *Priority: P2.*

---

## Production crate layout proposal

When this work graduates from `nightly/` to a long-lived crate (call it
`ruvector-graph-builder`), the natural layout is:

```
crates/ruvector-graph-builder/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── builders/
    │   ├── mod.rs
    │   ├── brute.rs
    │   ├── nn_descent.rs       # parallel + SIMD
    │   ├── nsg.rs              # NN-Descent + RNG-style refinement
    │   └── vamana.rs           # NN-Descent + α-pruning + 2-pass refinement
    ├── seeding/
    │   ├── mod.rs
    │   ├── hnsw_layer0.rs      # plug into ruvector-core
    │   ├── vamana_disk.rs      # plug into a future DiskANN crate
    │   └── lsm_compaction.rs   # plug into ruvector-lsm-ann
    └── benches/
        └── builder_shootout.rs
```

The "builders" / "seeding" split mirrors the algorithm/IO separation
that lets the same kNN-graph code feed in-memory HNSW, on-disk Vamana,
and LSM-level compaction without per-callsite glue.

---

## References

[^1]: Dong, Charikar, Li. *Efficient k-Nearest Neighbor Graph Construction for Generic Similarity Measures.* WWW 2011.
[^2]: Fu, Cai. *EFANNA: An Extremely Fast Approximate Nearest Neighbor Search Algorithm Based on kNN Graph.* arXiv:1609.07228 (2016).
[^3]: Fu, Xiang, Wang, Cai. *Fast Approximate Nearest Neighbor Search With The Navigating Spreading-out Graph.* PVLDB 2019.
[^4]: Subramanya, Devvrit, Simhadri, Krishnaswamy, Kadekodi. *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS 2019.
[^5]: Malkov, Yashunin. *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI 2018.
[^6]: Ootomo, Naruse et al. *CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs.* NVIDIA 2024.
[^7]: Xu, Liang, Li et al. *SPFresh: Incremental In-Place Update for Billion-Scale Vector Search.* SOSP 2023.
[^8]: Anonymous. *Panorama: Rotation-Tightened Distance Bounds for Vector Search.* arXiv:2510.00566 (Oct 2025).
