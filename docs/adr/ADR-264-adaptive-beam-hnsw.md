---
adr: 264
title: "ruvector-adaptive-beam — Pluggable Per-Query HNSW Beam Termination"
status: accepted
date: 2026-06-21
authors: [ruvnet, claude-flow]
related: [ADR-258, ADR-254, ADR-027]
supersedes: []
tags: [hnsw, ann, anytime, termination, p2-quantile, vector-database, rust]
---

# ADR-264 — ruvector-adaptive-beam: Pluggable Per-Query HNSW Beam Termination

## Status

**Accepted (PoC implemented).** `crates/ruvector-adaptive-beam` ships a
self-contained HNSW and three measured termination strategies. Production
integration into `ruvector-core` behind a `cargo` feature flag is staged
as follow-on work.

---

## Context

HNSW in `ruvector-core` (and every other major library: `hnswlib`,
`usearch`, `pgvector`, Milvus, Qdrant) uses a single global `ef_search`
parameter for layer-0 beam search. Latency is roughly proportional to
`ef_search`. Recall climbs with `ef_search` but with sharply
diminishing returns: in our 20k×64 measurement, the recall delta from
`ef=128` to `ef=256` is < 4 pp while latency doubles.

**The waste is per-query.** Some queries reach near-final recall at
20 expansions; others need 200. A single `ef_search` either over-spends
on easy queries or under-serves hard ones.

### The gap

* `hnsw_rs` (used by ruvector-core): fixed `ef_search`.
* `hnswlib`, `usearch`, `pgvector`, FAISS HNSW: fixed `ef_search`.
* DiskANN/Vamana: fixed beam width L.
* Liu et al., *Anytime ANN* 2024: time-budget-aware but uses fixed
  schedule, not a learned termination signal.

No library exposes a *trait* over the search loop that lets callers
plug in their own termination signal.

---

## Decision

Introduce `crates/ruvector-adaptive-beam` as a research crate with:

1. A self-contained HNSW (no external HNSW dep) so the search loop has
   full access to candidate / result heaps.
2. A `BeamTerminator` trait with three required methods:
   `reset`, `should_stop`, `ef_pool`. `should_stop` receives
   `(expansions, min_unexpanded, worst_topk, improved_this_step)`.
3. Three concrete terminators:
   * `FixedEfTerminator` — classical baseline.
   * `RatioTerminator` — multiplicative gap heuristic.
   * `QuantileTerminator` — online P² quantile estimator over
     improvement deltas.
4. Real benchmark numbers (20 000 × 64-dim, 500 queries, M=16,
   `ef_construction=200`).

The `BeamTerminator` trait and its three implementations are the
production deliverable. The self-contained HNSW in this crate is a
research vehicle; production would inject the trait into
`ruvector-core`'s existing HNSW.

---

## Consequences

**Positive:**

* Per-query latency adapts: easy queries stop early, hard queries use
  the full budget (up to `ef_max`).
* `RatioTerminator(r=1.20)` matches `FixedEfTerminator(ef=128)` recall
  (0.956 vs 0.920) at the cost of ~46% more mean expansions for the
  hard tail but ~3× fewer expansions for the easy head.
* Anytime ANN: a future `DeadlineTerminator` composes naturally with
  this trait — caller's deadline becomes the stop condition.
* Composes with `ruvector-hnsw-repair`: deletion strategy and
  termination strategy are orthogonal.
* `BeamTerminator: Send + Sync` so heterogeneous strategies can run
  across shards.

**Negative / cost:**

* `QuantileTerminator` is over-aggressive on the current PoC: the
  improvement-delta distribution is zero-inflated and the P² estimator
  collapses near zero. Roadmap: non-zero-delta filtering.
* Self-contained HNSW duplicates work that `ruvector-core` already
  does. PoC only; not for production.
* Each call to `should_stop` is ~5–10 ns. On 200 expansions that's
  1–2 µs of overhead per query — acceptable vs the 100s of µs the
  beam itself costs.

---

## Alternatives considered

1. **Per-`ef_search` parameter tuning offline.** Rejected: ignores
   per-query distribution.
2. **Learned termination policy (small NN).** Rejected for this ADR
   as too heavy — `RatioTerminator` is two arithmetic ops and works.
3. **Fix `ef_search` and add a global deadline.** Doesn't reclaim
   the work below the deadline; doesn't help if no deadline is set.
4. **Bake termination into `ruvector-core` as an enum.** Rejected;
   callers will want custom strategies (deadlines, RL policies,
   workload-specific heuristics). Trait wins.

---

## Implementation pointers

* Crate: `crates/ruvector-adaptive-beam/`
* Bench: `cargo run --release -p ruvector-adaptive-beam --bin
  adaptive-beam-bench`
* Tests: `cargo test --release -p ruvector-adaptive-beam`
* Demo: `cargo run --release -p ruvector-adaptive-beam --example demo`
* Research doc: `docs/research/nightly/2026-06-21-adaptive-beam-hnsw/`

---

## Follow-on work

* Non-zero delta filtering for `QuantileTerminator`.
* `DeadlineTerminator` — wall-clock budget.
* Cross-query session-cached P² estimator.
* Integration into `ruvector-core` behind `--features adaptive-beam`.
