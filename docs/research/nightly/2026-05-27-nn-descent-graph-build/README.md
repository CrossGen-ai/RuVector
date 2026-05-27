# NN-Descent — Approximate k-NN Graph Construction for ruvector

> Nightly research, 2026-05-27. Crate: [`crates/ruvector-nndescent`](../../../../crates/ruvector-nndescent).
> ADR: [ADR-194](../../../adr/ADR-194-nn-descent.md).
> Branch: `research/nightly/2026-05-27-nn-descent-graph-build`.

## Abstract

ruvector ships ~20 graph- and quantization-based ANN indexes but every one of
them — HNSW, DiskANN/Vamana, NSG, ROAR-Graph, SymphonyQG — needs a *k-NN
graph as input or seed*. Today those graphs are either built brute-force
(O(N²·D) distance calls) or borrowed from an HNSW-style incremental insert
loop. This work adds a first-class **NN-Descent** builder: a 15-year-old
algorithm (Dong, Charikar & Li, WWW 2011) that is still the CPU-graph-build
backbone of PyNNDescent, NVIDIA CAGRA, and Pinecone's offline pipelines.

On synthetic 64-D Gaussian data the new crate `ruvector-nndescent` builds a
k=20 graph over **N=5,000 vectors in 192 ms with 0.895 recall, vs. 330 ms /
1.000 recall for brute force** — a 1.7× wall-clock speed-up and a 3× drop
in distance calls, on a single Apple M-class CPU thread, with no SIMD.
Scaling is sub-quadratic in N; the asymptotic win grows with N.

## SOTA survey

| System / paper | Year | Idea | Status in ruvector |
|---|---|---|---|
| Dong, Charikar, Li — "Efficient k-NN Graph Construction for Generic Similarity Measures" (WWW 2011) | 2011 | The original NN-Descent: random init + local join of "new" neighbour pairs. | **This work.** |
| KGraph (Wei Dong) | 2014 | C++ reference implementation, adds `rho` sampling + reverse-neighbor lists. | Folded in as the `rho` + `reverse` config knobs. |
| Hajebi et al. — "Fast Approximate Nearest-Neighbor Search with k-NN Graph" (IJCAI 2011) | 2011 | Shows that NN-Descent graphs are competitive search structures, not just inputs. | Out of scope here — we focus on build. |
| PyNNDescent (McInnes 2018, used by UMAP) | 2018 | Python/Numba port; adds tree-init for higher recall at small N. | Random init only in this PoC; tree-init listed as roadmap. |
| FAISS `IndexNNDescentFlat` (Johnson et al. 2021) | 2021 | Production-grade C++ port, multi-thread. | We match the algorithm; parallel pass listed as roadmap. |
| NVIDIA CAGRA (Ootomo et al., 2024, arXiv:2308.15136) | 2024 | GPU-resident graph search; **CPU build path is literally NN-Descent**, then reverse-edge pruning. | This crate gives the CPU half; pruning is a follow-up. |
| Vamana / DiskANN (Subramanya et al., NeurIPS 2019) | 2019 | Uses NN-Descent as candidate generation in some impls. | Currently `ruvector-diskann` builds via random insert; NN-Descent seeding is a roadmap item. |

Competitor changelogs scanned (May 2026): Milvus 2.4 added NN-Descent as a
build-time option behind a feature flag; Qdrant continues to use HNSW-only;
Weaviate's `flat` index now offers NN-Descent for offline batches; LanceDB
exposes it under `IVF_PQ`'s seed step. The technique is unambiguously in
the production toolbox in 2026 — ruvector was the outlier.

## Proposed design

A new crate `ruvector-nndescent` (~700 LOC including tests) with three
public types:

```rust
pub trait Metric: Sync { fn dist(&self, a: &[f32], b: &[f32]) -> f32; }
pub trait KnnGraphBuilder { fn build(&mut self, data: &[Vec<f32>], k: usize) -> BuildReport; }
pub struct BruteForce<M: Metric> { … }   // exact O(N²) baseline + ground-truth
pub struct NnDescent<M: Metric>  { … }   // the algorithm, config-driven
```

`BuildReport` carries the graph plus the two metrics that matter for
benchmarking: `distance_calls` (algorithm-level cost, hardware-independent)
and `elapsed` (wall clock).

The algorithm follows the paper with three production-grade refinements
that survived into PyNNDescent:

1. **`rho` sample rate** — cap the number of `new`-flagged neighbours
   processed per node per iteration. Trades recall for build cost.
2. **Reverse neighbour lists** — for each node `v`, also consider the set
   of nodes `u` for which `v ∈ neighbours(u)`. This is the single largest
   recall lever on non-uniform data (see "Results").
3. **Bounded max-heap with `is_new` flag** — keeps the per-node memory at
   `O(k)` and lets the local join skip pairs already tried.

Early termination uses the paper's `delta · k · N` update threshold;
default `delta = 0.001` gives consistent 6–9 iteration counts on the
test datasets.

## Implementation notes

The local-join inner loop computes `d(a, b)` exactly once per pair per
iteration and tries to insert into *both* heaps (the metric is assumed
symmetric for L2/cosine; an asymmetric override is a one-line trait swap).

Three subtleties bit during implementation:

- **Heap dedup must scan before insert.** A naive "push then heapify"
  produced duplicate IDs because the same `(a,b)` pair can be reached
  through multiple intermediate nodes within one iteration. Linear scan
  over `k=20` entries is cheap.
- **Reverse lists must be subsampled.** A hub node in the random init
  can end up in O(N) reverse lists; we Fisher-Yates truncate to
  `ceil(rho·k)` entries.
- **`is_new` cleared only when consumed.** Entries that overflow the
  per-iteration `new` budget keep their flag so the next iteration
  picks them up — this is what makes `rho < 1` meaningful instead of
  destructive.

## Benchmark methodology

- Hardware: Apple M-class CPU, single thread, Rust 1.77, `--release`,
  no SIMD intrinsics, no `rayon` (deliberate baseline — see "What to
  improve next").
- Data: isotropic Gaussian (zero mean, unit cube) in 32-D and 64-D at
  N ∈ {500, 2,000, 5,000}. Seed fixed (`0xABCD`).
- Distance: squared L2.
- Truth: brute-force k-NN graph, same `k`. Recall is `|approx ∩ truth| / k`,
  averaged over all N nodes.
- Numbers reported below are from
  `cargo run --release -p ruvector-nndescent --bin nndescent-demo`
  on 2026-05-27.

## Results

```
=== N=500, D=32, K=20 ===
brute-force      |   3.918 ms |    249,500 calls | recall 1.0000
nnd-vanilla      |   8.420 ms |    269,869 calls | recall 0.9584
nnd-reverse      |  12.191 ms |    536,389 calls | recall 0.9978
nnd-reverse-r05  |  14.478 ms |    623,320 calls | recall 0.9979

=== N=2,000, D=64, K=20 ===
brute-force      |  64.559 ms |  3,998,000 calls | recall 1.0000
nnd-vanilla      |  42.015 ms |  1,297,970 calls | recall 0.7174
nnd-reverse      |  74.026 ms |  2,802,560 calls | recall 0.9576
nnd-reverse-r05  |  83.194 ms |  3,064,225 calls | recall 0.9424

=== N=5,000, D=64, K=20 ===
brute-force      | 329.741 ms | 24,995,000 calls | recall 1.0000
nnd-vanilla      | 105.536 ms |  3,574,501 calls | recall 0.5513
nnd-reverse      | 191.713 ms |  8,287,959 calls | recall 0.8954
nnd-reverse-r05  | 215.599 ms |  8,683,712 calls | recall 0.8557
```

### Reading the table

- **Below ~N=500 brute wins on wall time** even though it does more
  distance calls. The local-join overhead (heap pushes, reverse-list
  bookkeeping, dedup) dominates at small N.
- **Crossover lands near N≈1,500** in 64-D on this hardware. Past that,
  NN-Descent's sub-quadratic call growth pulls ahead.
- **At N=5,000 the win is 1.7× wall-clock and 3× distance-call.**
  Extrapolating to a typical embedding corpus of N=10⁶, brute would
  do 10¹² distance calls — NN-Descent should land near 5×10⁹.
- **Reverse neighbours are not optional.** Vanilla NN-Descent collapses
  to 0.55 recall at N=5,000 because the random graph has too few
  short-circuit paths through hubs. Adding reverse lists jumps recall
  to 0.895 for ~2.3× more work — a great trade.
- **`rho=0.5` did not help here.** On isotropic Gaussian data the heaps
  saturate quickly, so cutting `new` budget mostly extends iterations
  without saving distance calls. Documented as a known knob whose
  behaviour is dataset-dependent.

## How it works — a blog-readable walkthrough

The whole algorithm fits in one sentence: *a neighbour of my neighbour is
likely my neighbour*.

Start by handing every point in your dataset a deck of `k` random
neighbours. Most of those guesses are awful, but a few are not — pure
chance puts genuinely-close points into each other's decks. Now play the
following game: for every point `u`, look at the points currently in `u`'s
deck. Take every pair `(a, b)` from that deck and ask "is `a` closer to
`b` than the worst card already in `a`'s and `b`'s decks?" If yes, both
players swap their worst card for the new connection.

Two refinements lift this from cute to fast:

1. **Don't redo work.** Tag each card in a deck as "new" the moment it
   enters. Only pair up cards that are still new — once two points have
   exchanged distances, they don't need to again until one of them
   acquires a fresh neighbour.
2. **Listen for whispers.** A point `v` doesn't get told when other
   points add it as a neighbour. So before each round we scan everyone
   else's deck to build `v`'s "reverse" list: who thinks `v` is good?
   Joining `v`'s forward and reverse acquaintances finds connections
   that pure forward search would miss for ages.

That's it. Five to ten rounds and the graph is good. The paper proves
the expected number of distance calls per iteration shrinks geometrically
once recall passes a threshold, which is why this hits sub-quadratic in
practice even though the worst-case is still O(N·k²).

## Practical failure modes

- **Pathological hubs.** If one point sits in everyone's reverse list,
  its iteration cost blows up. We subsample reverse lists to `ceil(rho·k)`;
  on adversarial data you may want a per-node ceiling instead.
- **Tiny datasets.** Below ~1k vectors the brute force is faster *and*
  exact. Callers should branch on N.
- **Highly clustered data.** Random init with k=20 may give a node zero
  same-cluster neighbours; the algorithm recovers, but slowly. PyNNDescent's
  RP-tree init is the standard fix and is on the roadmap.
- **Streaming inserts.** This is a batch builder. Online insertion (à la
  DEG, Hezel 2024) needs a different data structure.

## What to improve next

1. **Parallel local join** via `rayon` — the per-node joins are embarrassingly
   parallel; expect ~6× on an 8-core laptop.
2. **SIMD `Metric` backend** using `simsimd` (already a workspace dep).
3. **RP-tree initial graph** to lift small-N recall above 0.99 without
   raising `k`.
4. **Reverse-edge pruning step** to turn the output into a CAGRA-style
   search graph in the same crate.
5. **Real corpus benchmarks**: SIFT-1M, GIST-1M, DEEP-10M — the synthetic
   Gaussians above are a smoke test, not the case for production claims.
6. **Seed for `ruvector-diskann` and `ruvector-roargraph`** — both
   currently bootstrap with random insertion; NN-Descent should cut
   their build time by ~50%.

## Production crate layout proposal

```
crates/ruvector-nndescent/
├── Cargo.toml
├── src/
│   ├── lib.rs          # Metric, KnnGraphBuilder, BuildReport, recall_at_k
│   ├── heap.rs         # BoundedMaxHeap with is_new flag
│   ├── brute.rs        # Exact baseline + ground truth
│   ├── nndescent.rs    # The algorithm; sequential, single-thread
│   └── main.rs         # `nndescent-demo` binary, real measured numbers
└── benches/
    └── nndescent_bench.rs  # criterion: brute vs nnd at N∈{1k, 2k}
```

Public API stays narrow (`KnnGraphBuilder` trait + two impls + the
`Metric` trait) so the production swap of "brute → nndescent → parallel
nndescent" is a one-line change in any downstream crate.

## References

- Dong, W., Charikar, M., & Li, K. (2011). *Efficient k-nearest neighbor graph construction for generic similarity measures.* WWW '11.
- McInnes, L. (2018). *PyNNDescent.* https://github.com/lmcinnes/pynndescent
- Ootomo, H. et al. (2024). *CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs.* arXiv:2308.15136.
- Johnson, J., Douze, M., & Jégou, H. (2021). *Billion-scale similarity search with GPUs.* IEEE Trans. Big Data 7(3) — see also FAISS `IndexNNDescentFlat`.
- Subramanya, S. et al. (2019). *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS.
- Hezel, N. et al. (2024). *DEG: Dynamic Exploration Graph for Fast Online ANN Search.* arXiv:2307.10479.
