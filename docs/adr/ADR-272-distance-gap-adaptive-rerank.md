---
adr: 272
title: "Distance-Gap Adaptive Rerank (DGAR) — Query-Adaptive K' Selection at the PQ → Exact Boundary"
status: accepted
date: 2026-07-02
authors: [claude-code, nightly-researcher]
related: [ADR-268]
tags: [ann, rerank, pq, adaptive, ruvector-dgar, nightly-research]
---

# ADR-272 — Distance-Gap Adaptive Rerank

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-07-02-distance-gap-adaptive-rerank` as
`crates/ruvector-dgar`. `cargo build --release`, `cargo test --release`
(3/3 pass), and `cargo run --release --bin dgar-bench` all succeed with
real numbers captured in the research document.

```
cargo run --release -p ruvector-dgar --bin dgar-bench
```

---

## Context

Every RuVector two-stage index (`ruvector-rabitq`, `ruvector-pq-search`,
`ruvector-matryoshka`, `ruvector-diskann`, `ruvector-spann`, …) shares
the same terminal step: pull `K' = C · K` candidates from an approximate
distance oracle, then compute exact distances for all of them and
return the top-K. The multiplier `C` is a per-index constant baked into
serving code, usually tuned empirically once during index build.

This is a global average — pessimal for easy queries (wastes exact
evaluations) and optimistic for hard queries (silently drops recall).
No RuVector index today makes `K'` a query-time decision. No shipping
vector database we surveyed (Milvus, Qdrant, Weaviate, Pinecone,
LanceDB, FAISS) does either.

We want a policy that:

1. Requires **zero training data** (unlike a learned selector).
2. Exposes a **single tunable scalar** that a runtime control loop can
   drive from recall telemetry.
3. Slots behind a **trait**, so every existing two-stage index can
   adopt it without a rewrite.
4. Has **provable behaviour bounds** (floor, ceiling) so we can never
   regress worse than a well-tuned FixedK.

## Decision

Adopt **Distance-Gap Adaptive Rerank (DGAR)** and standardise the
`Reranker` trait as the integration point for all future two-stage
RuVector indexes.

### The rule

Given approximate-distance-sorted candidates `d̃[0] ≤ d̃[1] ≤ …`:

```
anchor    = d̃[k-1]
threshold = (1 + γ) · anchor
take      = min_k · k
for i in min_k*k .. max_k*k:
    if d̃[i] > threshold: break
    take = i + 1
rerank exact distances for candidates[..take]
```

* γ is the single tunable scalar (typical range 0.02..0.30).
* `min_k` (default 2) is a lower floor — we never rerank fewer than
  `min_k · K`. Prevents collapse when approximate distances are noisy.
* `max_k` (default 32) is an upper ceiling — DGAR degenerates
  gracefully to FixedK(`C = max_k`) if γ is too loose.

### Crate boundary

```
crates/ruvector-dgar/
  src/lib.rs        # DgarError + public re-exports
  src/policies.rs   # Reranker trait; FixedK; AdaptiveGap; OracleUpperBound
  src/pq.rs         # honest reference PQ used to drive the benchmark
  src/main.rs       # dgar-bench binary
  tests/smoke.rs    # 3 invariant tests
```

### Acceptance gates (all met on 2026-07-02)

* `cargo build --release -p ruvector-dgar` — clean.
* `cargo test  --release -p ruvector-dgar` — 3/3 pass.
* Benchmark produces real numbers, not mocked.
* At γ=0.02, DGAR strictly Pareto-dominates the nearest FixedK
  operating point (0.596 vs. 0.576 recall@10 at ~90 exact evals).
* File-size cap: largest source file is `src/pq.rs` at ~180 lines.

## Consequences

### Positive

* **Query-adaptive spend.** Hard queries get more budget, easy queries
  less, without any index rebuild.
* **Runtime tunable.** γ is a single scalar; a bandit / PID loop
  driven by sampled recall can retune it live.
* **Zero-training.** No labeled queries needed; the gap statistic is
  purely local.
* **Trait-based integration.** Every existing RuVector two-stage
  index (`rabitq`, `pq-search`, `matryoshka`, `diskann`, `spann`) can
  adopt DGAR by swapping the reranker.
* **Safe by construction.** `max_k` ceiling guarantees DGAR never
  spends more than a well-known FixedK budget; `min_k` floor prevents
  under-provisioning in noisy regimes.

### Negative / trade-offs

* **γ needs occasional tuning.** Distribution shift moves the
  optimal γ. Deployments should sample recall periodically.
* **Not a full replacement for a learned selector.** The Oracle
  headroom (10 evals vs. DGAR's 91 at γ=0.02) shows a learned model
  could recover ~9× more efficiency. DGAR is the zero-cost baseline
  that a future learned selector must beat.
* **Depends on approximate-distance ordering quality.** If the coarse
  index is nearly-random, DGAR collapses to `min_k · K`.

### Neutral

* Adds one crate (`ruvector-dgar`) — no changes to existing crates
  in this ADR; downstream wiring lands in a follow-up.

## Alternatives considered

1. **Keep FixedK per-index.** Rejected: leaves query-adaptive gains
   permanently on the table and offers no runtime tuning surface.
2. **Learned selector (small MLP).** Rejected *for now*: requires
   labeled query data and per-dataset training. DGAR is the
   zero-cost baseline; a learned selector belongs on the roadmap as
   a strict improvement over DGAR.
3. **Confidence-interval truncation (Chernoff bound on distance
   error).** Rejected *for now*: requires estimating the
   approximate-distance error distribution per index. Interesting as
   a follow-up on top of DGAR (see research doc roadmap).
4. **Anytime / streaming rerank.** Complementary, not alternative;
   can be layered on top of DGAR.

## References

* Research document: `docs/research/nightly/2026-07-02-distance-gap-adaptive-rerank/README.md`
* Implementation: `crates/ruvector-dgar/`
* Benchmark reproduction: `cargo run --release -p ruvector-dgar --bin dgar-bench`
