---
adr: 264
title: "Anytime HNSW — progressive top-k refinement with monotone guarantees"
status: proposed
date: 2026-06-19
authors: [claude-flow, nightly-research]
related: [ADR-260-coherence-hnsw, ADR-256-hybrid-sparse-dense]
tags: [hnsw, ann, anytime, progressive, streaming, agent-memory, ruvector]
---

# ADR-264 — Anytime HNSW

> **Decision in one line.** Add an opt-in **anytime emission policy** to the
> HNSW beam-search loop that delivers monotonically-improving top-k snapshots
> mid-search, exposed via a `Searcher` trait so existing one-shot call sites
> are unchanged.

## Context

Every ANN backend in ruvector — `ruvector-core` (HNSW), `ruvector-diskann`,
`ruvector-rairs` (IVF), `ruvector-hybrid` — returns a single one-shot top-k
to the caller. This is fine for batch retrieval, but it's a poor fit for:

* **Agent loops** that have a soft deadline (e.g., "I want to start reasoning
  on the best candidate within 5 µs, even if you keep refining for another
  50 µs").
* **Interactive search UIs** that should render the first plausible answer
  immediately and refine as the beam deepens.
* **Latency-bounded RPC** where the server must return *something* by the
  SLA, not a deadline error.
* **Streaming reranking** where downstream stages (cross-encoder, LLM
  reranker) can start work as candidates become available.

All four use cases want the **same primitive**: a stream of progressively
better top-k snapshots, where each snapshot is guaranteed to be no worse than
the previous one.

## Decision

Introduce a new opt-in crate `ruvector-anytime` exposing:

```rust
pub trait Searcher {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
        on_snapshot: &mut dyn FnMut(&AnytimeSnapshot),
    ) -> SearchResult;
}

pub struct AnytimeSnapshot {
    pub neighbors: Vec<(u32, f32)>,
    pub elapsed_ns: u128,
    pub pops: usize,
}
```

with three reference implementations:

* `OneShotSearch` — never emits intermediate snapshots (drop-in for the
  existing one-shot semantics).
* `NaiveAnytime` — emits after every pop. Upper bound on reactivity.
* `BatchedAnytime { initial_batch, growth }` — emits on improvement with an
  exponentially-growing batch threshold. Production default.

The emission decision is decoupled from the search loop via an internal
`EmitPolicy` trait, so new policies (deadline-bounded, confidence-bounded)
can be added without touching the core beam-search.

### Guarantee

**Top-k Monotonicity.** Across any two snapshots `S_i, S_j` with `i ≤ j`:
* `min(S_j.neighbors.dist) ≤ min(S_i.neighbors.dist)` (best-of-k never
  regresses), and
* if both snapshots have `|neighbors| = k`,
  `max(S_j.neighbors.dist) ≤ max(S_i.neighbors.dist)` (farthest never
  regresses).

The guarantee follows from HNSW's top-k max-heap invariant: a node is only
inserted if it strictly improves the heap, and a node is never removed
without a strict improvement.

## Consequences

### Positive

* **Zero impact on existing call sites.** `OneShotSearch` is bit-identical
  to today's HNSW result.
* **Anytime is cheap.** Measured on the PoC: `BatchedAnytime` adds +5%
  latency at p50 while preserving recall exactly. `NaiveAnytime` reaches
  90% of final recall in ~47 µs vs OneShot p50 of 55 µs.
* **Composable** with all other nightly research:
  `coherence-hnsw`, `hnsw-repair`, `hybrid-sparse-dense`, `adaptive-ef`.
  The emission policy wraps the loop and does not interact with the index.
* **Streaming-ready.** The callback shape maps cleanly to `mpsc::channel`,
  `tokio::sync::mpsc`, `futures::Stream`, and gRPC server-streaming.

### Negative / risks

* **Callback allocates.** Each snapshot clones the top-k. Mitigated by
  batched-exponential emission (6× fewer snapshots than naive in measured
  PoC). Future work: emit diffs instead of full snapshots.
* **Re-entrancy hazard.** A blocking `on_snapshot` callback blocks the
  search. Documented in the crate-level docs; callers must push to a
  bounded channel and drop on full.
* **Quantization weakens the guarantee.** With PQ/RaBitQ the distances are
  approximate; the guarantee becomes "monotone in approximate distance".
  Still safe for ranking, but downstream stages should reconstruct exact
  distance if needed.

## Alternatives Considered

1. **Adaptive-ef only.** Adjust `ef_search` per-query (existing nightly).
   Optimises one-shot latency but does not expose intermediate answers.
   *Rejected:* solves a different problem.

2. **Deadline-bounded one-shot.** `search_until(query, k, deadline)` that
   returns the best heap state at the deadline. Doesn't give the caller
   anything mid-flight; only safe if the caller pre-commits a deadline.
   *Will be added as a follow-up*, layered on top of anytime.

3. **Return iterator of candidates.** Caller consumes a `Stream<(id,
   dist)>` of pops directly. Wrong abstraction — the caller wants a top-k,
   not a stream of all visited candidates, most of which will not make
   the final result. *Rejected.*

4. **Server-side only (in `ruvector-server`).** Wire the streaming at the
   RPC layer without touching the search loop. *Rejected:* would require
   simulating progress via polling, losing the monotone guarantee.

## Acceptance Criteria

All measured on `cargo run --release -p ruvector-anytime --bin benchmark`
with a 3,000 × 48-dim clustered dataset, 200 queries, K=10, EF=120:

| Criterion                                                        | Status |
|------------------------------------------------------------------|:------:|
| `OneShot` recall@10 ≥ 0.80                                       |  PASS  |
| `NaiveAnytime` final recall == `OneShot` (exact)                 |  PASS  |
| `BatchedAnytime` final recall == `OneShot` (exact)               |  PASS  |
| Top-k monotonicity (`NaiveAnytime`)                              |  PASS  |
| Top-k monotonicity (`BatchedAnytime`)                            |  PASS  |
| `BatchedAnytime` emits ≥ 4× fewer snapshots than `NaiveAnytime`  |  PASS (5.93×) |

See `docs/research/nightly/2026-06-19-anytime-hnsw/README.md` for full
numbers, methodology, host details, and the SOTA survey.
