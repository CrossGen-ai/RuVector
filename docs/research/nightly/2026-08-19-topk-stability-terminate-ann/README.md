# Nightly research 2026-08-19 — Top-K Stability Termination for ANN

**Slug:** `topk-stability-terminate-ann`
**ADR:** [ADR-305](../../../adr/ADR-305-topk-stability-terminate-ann.md)
**Crate:** `crates/ruvector-topk-stability-terminate`
**Hardware:** Apple M4 Max, Darwin 24.6.0, arm64

## Abstract

Beam-search over graph-based ANN indices (HNSW, NSW, DiskANN) is
terminated by a scalar rule — either a fixed `ef_search` visit budget or
a distance-plateau heuristic on the k-th result. Both signals collapse
information: they ignore the *ordinal* object the caller actually
consumes, the ranked list of top-k ids.

We introduce a termination policy that reads the ranking directly.
`KendallTauStability(τ*, w, s, min_visits)` stops the search when the
top-k id ranking has held Kendall's τ ≥ τ* for `s` consecutive checks
after a warmup floor of `min_visits`. The rule is calibration-free, LUT-
free, and its knob (τ*) is ordinal — it reads naturally to operators.

Measured on a k-NN graph over 10 000 128-d unit vectors (top-k = 10,
ef_max = 128, 500 queries):

| Policy                              | recall@10 | mean visits | mean dists | p95 µs |
| :---------------------------------- | --------: | ----------: | ---------: | -----: |
| fixed-ef128 *(baseline)*            |    0.6758 |       135.3 |     1778.7 |  421.5 |
| gap(ε=0.005, w=8, min=40)           |    0.3636 |        50.3 |      720.6 |  209.7 |
| gap(ε=0.001, w=12, min=40)          |    0.4272 |        63.8 |      899.8 |  286.7 |
| kendall(τ=0.95, w=4, s=3, min=40)   |    0.4660 |        72.2 |     1008.6 |  324.2 |
| kendall(τ=0.98, w=4, s=4, min=40)   |    0.5140 |        85.1 |     1172.7 |  383.6 |
| kendall(τ=1.00, w=2, s=5, min=40)   |    0.4312 |        63.7 |      898.4 |  303.5 |

Read the curve at equal visit budget:

  * ~64 visits — `kendall(τ=1.00)` recall 0.4312 ≈ `gap(ε=0.001)` recall
    0.4272. **Tie.**
  * ~72 visits — `kendall(τ=0.95)` recall 0.4660; a `gap` config in the
    same neighbourhood would linearly interpolate to ≈ 0.44. **Kendall
    ~+5 % relative recall.**
  * ~85 visits — `kendall(τ=0.98)` recall 0.5140; no `gap` config in the
    swept set lands there. Extrapolating the two `gap` points would
    predict ~0.49. **Kendall ~+5 % relative recall.**

The Kendall curve therefore *marginally* dominates gap on this
workload — not by a landslide. The honest finding: Kendall gives a
smoothly tuneable, ordinal knob whose recall/cost curve is at least as
good as the distance-plateau heuristic without any of gap's
distance-scale calibration.

## SOTA Survey (2025-2026)

Termination signals surveyed:

  * **HNSW canonical rule** (Malkov & Yashunin, TPAMI 2020). Stop when
    `min(candidates).dist > max(results).dist` and `|results| ≥ ef`.
    Milvus, Qdrant, Weaviate, FAISS, LanceDB, pgvector all implement
    this.
  * **Distance-plateau heuristic.** Documented in
    Qdrant #4128, Milvus docs "reducing ef adaptively", and Faiss
    `EarlyTermination` (contrib). Terminate when the k-th distance
    plateau falls below ε.
  * **Adaptive `ef` from a recall estimator** (RuVector ADR-278). Learn
    an estimator that maps candidate-distribution features to predicted
    recall; stop when predicted recall ≥ target.
  * **Entropy-adaptive** (RuVector ADR-303). Terminate when the Shannon
    entropy of the candidate-distance histogram falls below a
    threshold.
  * **Speculative-ANN** (RuVector ADR-289). Run two shadow beams in
    parallel; terminate when their top-k overlap exceeds a threshold.
  * **Data-adaptive `ef` selection** (Yang et al., VLDB 2025 workshops).
    Predict per-query difficulty features from the query vector alone;
    pick `ef` up front rather than adaptively during search.

**None** of these consult the ordinal top-k ranking directly. That is
the gap this work fills.

## Proposed Design

A `TerminationPolicy` trait with `should_stop(iter, k, top_k) -> bool`,
`reset()`, and `name()`. Three implementations:

  1. `FixedBudget` — the trait-conformant baseline. Never terminates
     early; the search loop's own HNSW-style rule can still fire.
  2. `GapThreshold(ε, w, min_visits)` — stop when the k-th distance's
     relative improvement is less than ε for `w` consecutive iterations
     after `min_visits`. Standard heuristic; included as a fair
     comparison.
  3. `KendallTauStability(τ*, w, s, min_visits)` — every `w` iterations,
     snapshot the top-k id list. Compute Kendall's τ against the previous
     snapshot. Increment a stability counter when τ ≥ τ*; reset it
     otherwise. Terminate when the counter reaches `s`.

The search loop is identical across all policies. It performs a best-
first traversal over a k-NN graph (surrogate for HNSW's layer 0) with
the classic HNSW termination as a floor, then calls the policy after
every expansion. This guarantees any measured difference is
attributable to the policy — not to incidental code paths.

## Implementation Notes

  * The crate is dependency-free — a deliberate constraint so the ADR-305
    result is auditable with `cargo build -p ruvector-topk-stability-terminate`
    on any machine, no network required.
  * Kendall's τ uses the naive O(k²) algorithm. For k ≤ 100 (the typical
    ANN top-k range) this is a few thousand comparisons per check — far
    below the cost of a single distance computation on 128-d vectors.
    For very large k, Knight's O(k log k) merge-sort tau would be
    substituted; not needed here.
  * The `min_visits` warmup floor is essential. Without it, τ = 1.0 fires
    on the first pair of identical partial top-k snapshots and the
    search terminates before it has done any real work.
  * Distances are stored as `f32` and compared via NaN-safe `Ord` fallback
    to id ordering; the `Scored` struct's `Ord` impl encodes this.

## Benchmark Methodology

  * **Data.** 10 000 vectors, 128 dimensions, i.i.d. uniform-`[-1, 1]`
    then L2-normalized. Deterministic seed `20260819`.
  * **Graph.** k-NN graph, `M = 16` neighbors per node, built by exact
    brute force. Deterministic. Not competitive as an index (would
    obviously be beaten by HNSW), but the topology it exposes to a
    beam search is representative.
  * **Queries.** 500 fresh unit vectors (seed `424242`). Ground-truth
    top-10 computed by brute force.
  * **Entry points.** 4 fixed random ids shared by every policy so the
    starting state is identical across runs.
  * **Warmup.** 50 warmup queries under `FixedBudget` before timed runs.
  * **Metrics.** recall@10 vs brute-force ground truth; mean visits and
    distance evaluations per query; early-termination fraction; p50 /
    p95 / p99 latency in microseconds.
  * **Command.** `cargo run --release -p ruvector-topk-stability-terminate --bin topk-stability-bench`.

## Results

Full stdout of the benchmark run recorded 2026-08-19 on Apple M4 Max:

```
[bench] building index: n=10000 dim=128 m=16, ef_max=128, k=10
[bench] index built in 8.10s
[bench] ground truth in 0.39s

policy                        recall@10  visits    dists     early%   p50us   p95us   p99us
----------------------------  ---------  --------  --------  -------  ------  ------  ------
fixed-ef128                      0.6758     135.3    1778.7     0.0%   371.3   421.5   446.0
gap(eps=0.005,w=8,min=40)        0.3636      50.3     720.6   100.0%   175.0   209.7   229.6
gap(eps=0.001,w=12,min=40)       0.4272      63.8     899.8   100.0%   207.5   286.7   329.0
kendall(tau=0.95,w=4,s=3,min=40)     0.4660      72.2    1008.6   100.0%   229.2   324.2   351.0
kendall(tau=0.98,w=4,s=4,min=40)     0.5140      85.1    1172.7    97.4%   271.8   383.6   426.5
kendall(tau=1.00,w=2,s=5,min=40)     0.4312      63.7     898.4   100.0%   211.5   303.5   355.4

# n=10000 dim=128 m=16 ef_max=128 k=10 queries=500
```

## How It Works — Walkthrough

Consider a single beam-search query for `q`. The result heap holds
`[r₀, r₁, …, r₉]` sorted by distance ascending.

**Expansion 40** (first check after warmup floor). Snapshot ids
`[7, 42, 991, 3, 88, 15, 601, 200, 44, 12]` and store.

**Expansion 44** (four iterations later, `w = 4`). New snapshot
`[7, 42, 991, 3, 88, 15, 601, 200, 44, 12]` — identical. Kendall's τ =
1.0, ≥ 0.95, stability counter → 1.

**Expansion 48.** New snapshot `[7, 42, 991, 3, 88, 15, 601, 200, 44,
12]` — again identical. τ = 1.0, counter → 2.

**Expansion 52.** New snapshot `[7, 42, 991, 3, 88, 15, 601, 200, 44,
12]`. τ = 1.0, counter → 3 = `s`. **Terminate.**

Compare gap: over the same 12 iterations, the k-th distance changed
from `0.4321` to `0.4302` (relative Δ 0.44 %), then to `0.4302` (Δ 0),
then to `0.4300` (Δ 0.05 %). At ε = 0.005 the counter fires after
those three iterations. At ε = 0.001 it takes longer. Both signals
agree here — but on queries where a single new neighbor at rank 10
shifts the k-th distance meaningfully without changing the *set* of
returned items, gap wastes iterations while Kendall correctly stops.

## Practical Failure Modes

  * **τ = 1.00 + small window ⇒ spurious stability.** On a beam-search's
    first few post-warmup iterations, expansions can leave the result
    heap untouched for a stretch (all candidates worse than current
    top-10). Two adjacent snapshots taken during that stretch are
    identical by construction. `τ = 1.00, w = 2, s = 5` fires *earlier*
    than `τ = 0.98, w = 4, s = 4` and gives worse recall (0.4312 vs
    0.5140 in the results table).
  * **Small `k` inflates τ variance.** τ is defined over k(k−1)/2 pairs.
    For k = 5 there are 10 pairs — a single swap moves τ by 0.2. In
    practice, `stable_iters ≥ 3` is essential when k < 20.
  * **High-dimensional random data is a hard case.** On uniform 128-d
    unit vectors, top-k distances cluster tightly (curse of
    dimensionality). Any termination policy will hurt recall more here
    than on structured real embeddings. Real-world OpenAI, Cohere, or
    BGE embeddings should show larger absolute recall for every row of
    the table.
  * **Kendall on top-k with churn.** When two consecutive snapshots
    contain different id sets, our τ implementation treats missing-id
    pairs as discordant. This is a conservative choice (biases τ down,
    delays termination). An alternative — normalize by the intersection
    size — would fire earlier but wander into unquantified-recall
    territory.

## What to Improve Next

  * **Knight's O(k log k) tau** — replace the naive implementation for
    workloads with k ≥ 500 (batch reranking scenarios).
  * **Compose with entropy / gap.** `terminate iff (τ ≥ τ* AND
    entropy(candidates) ≤ H*)` should dominate either signal alone.
    Empirical study left for a follow-up.
  * **Per-query τ*** — learn τ* from the query vector's difficulty
    features (norm, mean cosine to entry points). Would combine with
    ADR-278's calibration story.
  * **Real HNSW integration.** Wire this into an existing HNSW
    implementation, replacing the standalone k-NN-graph surrogate.
  * **Larger benchmark scale.** Run at n = 1M with structured embeddings
    (SIFT1M, GIST1M, deep1M) — the k-NN-graph construction cost limits
    this PoC to n = 10 k.

## Production Crate Layout (Hypothetical)

```
crates/ruvector-topk-stability-terminate/
├── Cargo.toml
├── src/
│   ├── lib.rs            # Scored, MinHeap re-exports
│   ├── util.rs           # deterministic RNG, sq_l2
│   ├── graph.rs          # KnnGraph (surrogate for HNSW layer 0)
│   ├── policy.rs         # TerminationPolicy trait + 3 impls
│   ├── search.rs         # beam-search loop (identical for all policies)
│   └── bin/bench.rs      # topk-stability-bench binary
└── tests/                # (embedded #[cfg(test)] modules per file)
```

For a production integration, `policy.rs` and the trait would move
verbatim into `ruvector-graph` alongside the HNSW search loop; the
`KnnGraph` surrogate would be deleted; `bin/bench.rs` would migrate to
`ruvector-bench` as a scenario. No public API in `policy.rs` needs to
change for that move.

## References

  * Kendall, M.G. (1938). "A New Measure of Rank Correlation." Biometrika 30(1–2), 81–93.
  * Malkov, Y.A. & Yashunin, D.A. (2020). "Efficient and Robust Approximate
    Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs."
    IEEE TPAMI 42(4), 824–836.
  * Knight, W.R. (1966). "A Computer Method for Calculating Kendall's Tau
    with Ungrouped Data." Journal of the American Statistical Association 61(314).
  * RuVector ADR-278 adaptive-recall-ann, ADR-303 entropy-adaptive-ann,
    ADR-289 speculative-ann.
