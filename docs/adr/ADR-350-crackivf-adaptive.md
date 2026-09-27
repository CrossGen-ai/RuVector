# ADR-350: CrackIVF-Adaptive — Query-Density-Driven Online IVF Refinement

## Status

Proposed. Nightly research PoC (`ruvector-crackivf-adaptive`) on
branch `research/nightly/2026-09-27-crackivf-adaptive`. Numbers cited
below come from `cargo bench -p ruvector-crackivf-adaptive` on an
Apple M4 Max running rustc 1.98.1.

## Context

Every IVF-based ANN index in the RuVector tree
(`ruvector-turbovec`, `ruvector-soar-ivf`, `ruvector-norm-bucket-ivf`,
`ruvector-bet4-ivf-bench`, `ruvector-pq-search`) commits to `nlist`
at build time. The chosen constant is a compromise between per-probe
latency (few fat cells is expensive per probe) and memory + build
cost (many thin cells is expensive up front and per centroid rank).
Skewed workloads — the norm for recommendation and retrieval-augmented
generation — waste that budget: 5% of the corpus receives 70% of the
probes.

CrackIVF-Adaptive introduces **online cell refinement**: start with
`nlist_0` small, count probes per cell, and split hot cells via a
balanced 2-means kernel. This mirrors Idreos-style database cracking
(CIDR 2007) applied to Voronoi partitions.

The design surface must be small enough for one nightly and honest
enough that its failure modes are documented rather than hidden.

## Decision

1. Ship a new workspace crate
   `crates/ruvector-crackivf-adaptive`, added to the root workspace
   `members`, with:
   - `IvfIndex<P: SplitPolicy>` — the state machine (vectors,
     cells, RNG, stats).
   - `SplitPolicy` trait, with three built-in implementations:
     `NoSplit` (baseline), `FixedThreshold`, `AdaptiveThreshold`.
   - `search(query, k, nprobe)` returns top-k *and* runs cracking
     inline on probed cells whose policy fires.
   - `brute_topk` + `recall_at_k` helpers for reproducible bench
     recall.
2. Splits use `k-means++` seeding followed by 5 Lloyd iterations
   on the two-cluster restriction of the cell. If one side comes up
   empty the split is aborted and the cell's probe counter is reset.
3. The `SplitPolicy` trait is the sole extension point. Bandit,
   RL, and KL-triggered policies are future work and do not
   perturb the core.
4. The crate is `#![forbid(unsafe_code)]`, depends only on the
   workspace `rand` version, and stays under 500 lines per file.
5. Benchmarks are harness-free (`benches/recall_qps.rs`), emit real
   numbers to stdout, and are reproducible via `cargo bench`.
6. Numbers must be reported both at **fixed nprobe** (shows
   structural latency win and recall cost) and at **scaled nprobe**
   (shows the fair recall/latency Pareto). Any documentation that
   omits either row is considered incomplete.

## Consequences

Positive:

- RuVector gains its first *reactive* IVF, complementing the
  reactive HNSW work (`ruvector-hnsw-batch-shared-frontier`,
  `ruvector-hnsw-delta-neighbors`) and closing an SOTA gap that
  Milvus, Qdrant, LanceDB, and Pinecone all still have.
- Build cost drops (3.6 ms vs 9.0 ms baseline in the 5000×64 case)
  because the initial partition is small — cracking absorbs the
  cost into the query stream.
- Trait-based `SplitPolicy` gives us a clean seat for the
  bandit/RL variants sketched in the nightly roadmap.

Negative / trade-offs:

- Splits are done inline with `search`. Tail-latency-critical
  deployments must gate cracking behind an async worker (roadmap
  item).
- On uniform (non-skewed) workloads cracking has no benefit and a
  small overhead. Runtime instrumentation should suppress cracking
  when the query distribution is flat.
- At fixed `nprobe`, cracking demonstrably hurts recall by ~20
  points on our workload. Deployment guidance is therefore to
  scale `nprobe` with `nlist` (or, better, run cracking on a
  reranker rather than the entry-point IVF).
- Adversarial workloads that cycle hotspots can fragment `nlist`
  without bound; a merge counterpart is deferred.

## Alternatives Considered

- **SPANN-style hierarchical IVF.** Excellent for billion-scale
  static datasets, but rebuild-heavy — offline refinement, not
  online.
- **Offline periodic rebuild.** Zero code cost (Milvus / Qdrant
  already support it), but requires a maintenance window and does
  not exploit intra-window skew.
- **SOAR multi-assignment (already in RuVector).** Fixes centroid
  drift but not the `nlist` tuning problem; complementary, not a
  replacement.
- **Learned partitions (LoRANN, DEANN).** Higher recall ceiling
  but much bigger implementation surface — deferred as its own
  nightly.

## Numbers (Apple M4 Max, rustc 1.98.1, 5000×64, 1000 queries, k=10)

Matched-recall Pareto:

| variant                    | recall@10 | mean lat | QPS    |
|----------------------------|----------:|---------:|-------:|
| baseline (static, scaled)  | 0.407     | 11.4 µs  | 87 663 |
| CrackIVF fixed, scaled     | 0.525     | 17.2 µs  | 58 273 |
| CrackIVF adaptive, scaled  | 0.525     | 17.3 µs  | 57 688 |

Cracking lifts recall by +12 points at ~50% latency cost with
identical `nprobe` scaling and +1.8% memory. Full table in the
nightly's [research README](../research/nightly/2026-09-27-crackivf-adaptive/README.md).
