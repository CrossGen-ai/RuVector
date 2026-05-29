# ADR-194: Bounded Early-Termination IVF (BET-IVF)

**Status:** Proposed (nightly research, 2026-05-29)
**Crate:** `ruvector-betivf`
**Research:** [docs/research/nightly/2026-05-29-bounded-early-termination-ivf](../research/nightly/2026-05-29-bounded-early-termination-ivf/README.md)

## Context

ruvector ships several IVF-family indices (`ruvector-rairs`, `ruvector-cluster`,
the IVF backend of `ruvector-diskann`). All of them probe a fixed `nprobe`
clusters per query. `nprobe` is the only knob and it must be tuned per
dataset: too low and recall collapses on hard queries; too high and easy
queries waste cycles. Recent SOTA (SIGMOD'20, Pinecone'24, CAGRA-Q) either
learns a model or uses heuristic "no-improvement" stops; both add training,
dataset coupling, or unsoundness.

A classical alternative — triangle-inequality lower bounds on unvisited
clusters — is sound, training-free, and adds only one `f32` per cluster.
It is rarely implemented in modern vector DBs.

## Decision

Add **BET-IVF** as a new crate `ruvector-betivf` exposing three swappable
search strategies on a common IVF backbone:

* `FixedNprobe(n)` — baseline.
* `FixedBudget(b)` — candidate-budget baseline.
* `BoundedEarlyTerm { max_nprobe, slack }` — adaptive per-query
  termination using `lb_c = max(0, ‖q − c‖ − r_c)` as a sound lower bound
  on the closest possible distance from any unvisited partition.

`slack = 1.0` is **sound** — provably equivalent to scanning every
partition for the purposes of top-k. `slack > 1.0` widens the stop
condition for aggressive pruning at controlled recall cost.

## Consequences

**Positive.**
* No training, no per-dataset model.
* Sound mode (`slack = 1.0`) cannot regress vs full scan.
* On the 50 k Gaussian-mixture benchmark, BET matches
  `FixedNprobe(16)` recall (0.9988) at 12.95 partitions/query (–19 %)
  and matches `FixedNprobe(32)` recall at 2.8× the QPS.
* Storage overhead: one `f32` per cluster (~1 KB for 256 clusters).
* The same idea is reusable for IVF-PQ, DiskANN, and graph indices.

**Negative.**
* Requires storing one cluster radius — needs index format bump if we
  retrofit existing IVF crates (deferred to a follow-up ADR).
* On thin-shell high-dimensional data the bound is loose and BET
  degrades to `FixedNprobe` performance (no win, no loss).
* `slack > 1.0` is not sound; tooling must surface this clearly.

**Neutral.**
* Adds one more strategy enum variant operators must understand. The
  CLI/server-level default should remain `FixedNprobe` until a follow-up
  integration ADR.

## Alternatives considered

1. **Learned-termination (SIGMOD'20).** Higher peak QPS on benchmarked
   datasets but requires per-dataset training + a model artifact. Rejected
   for a "training-free baseline" first cut. Can layer on top of BET later.
2. **No-improvement heuristic.** Stop after N partitions with no top-k
   change. Unsound; depends on probe order. Rejected.
3. **Bandit per-query budget allocation.** Stateful, harder to reason
   about. Could be added later as a `slack` scheduler.
4. **Inflate `nprobe` globally.** Easy but wastes work on easy queries
   — the exact failure mode BET fixes.

## Follow-ups

* Integrate BET into `ruvector-rairs` (gated by a `Cargo` feature).
* PQ-coupled BET — skip individual PQ table lookups via the same bound.
* HNSW/graph variant — maintain an unvisited-frontier lower bound.
* Bandit-tuned `slack` from centroid-distance ratios.
