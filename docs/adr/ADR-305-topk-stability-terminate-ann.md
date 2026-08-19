# ADR-305: Kendall-Tau Top-K Stability as an ANN Beam-Search Termination Signal

## Status

Proposed. Experimental crate (`ruvector-topk-stability-terminate`), not
wired into any production index. This ADR documents the design and the
measured trade-off; it does not commit to a production integration.

## Context

Approximate-nearest-neighbor beam searches over graph indices (HNSW, NSW,
Vamana, DiskANN) must decide *when to stop expanding candidate nodes*. The
industry-standard rule is a scalar bound:

  1. Fixed budget — expand until `ef_search` distinct nodes have been
     visited (Milvus, Qdrant, Weaviate, FAISS, LanceDB defaults).
  2. Distance plateau — stop when the *k-th* result distance improves by
     less than an epsilon (a common tuning heuristic; called out by
     several ANN authors as a way to hit a target recall without setting
     `ef` per query).

Both signals are *scalar*. A scalar collapses information: the k-th
distance can drop when a single new node is inserted into the result set
without any change to which items are actually returned. Recent RuVector
nightly work explores three richer signals:

  * ADR-303 (entropy-adaptive) — Shannon entropy of the candidate
    distance histogram.
  * ADR-278 (adaptive-recall) — calibrated recall estimator on the
    candidate distribution.
  * ADR-289 (speculative-ann) — shadow beams that vote on convergence.

All three still reason about *distances*. None consult the ordinal object
that the caller actually consumes: the ranked list of ids.

## Hypothesis

```text
Given a beam search over a k-NN graph over 10 000 128-d unit vectors,
top-k = 10, ef_max = 128,

when the search's termination decision is replaced with a Kendall-tau
stability rule (stop when the top-k id ranking has stayed at tau ≥ τ*
for s consecutive checks, after a warmup floor of min_visits),

then the resulting policy exposes a tuneable knob (τ*, w, s, min_visits)
whose recall/visit-count trade curve is comparable to — and, in the
mid-recall regime, marginally more efficient than — a distance-plateau
policy (epsilon, window, min_visits) tuned to the same recall targets,

subject to the Kendall-tau evaluation cost being O(k²) per check, which
is negligible for the typical k ≤ 100.
```

Notably NOT claimed:

  * Kendall-tau dominates gap-threshold at every operating point. On this
    workload it does not — the measured curves cross.
  * Kendall-tau matches the fixed-`ef` recall. On the mid-tuned settings
    here it recovers ~75% of the fixed baseline's recall at ~50% of its
    visit budget; hitting 95% of fixed recall requires either a much
    larger `min_visits` floor or a very high τ*, at which point the
    policy converges to fixed behavior.

## Decision

Add `crates/ruvector-topk-stability-terminate`, a self-contained,
dependency-free Rust crate that:

  1. Defines a `TerminationPolicy` trait invoked after every beam-search
     expansion.
  2. Ships three implementations that share the same search loop, so any
     measured difference is attributable to the policy — not to
     incidental code differences:
     - `FixedBudget` — baseline (no early termination).
     - `GapThreshold(epsilon, window, min_visits)` — the standard
       plateau heuristic.
     - `KendallTauStability(tau_threshold, window, stable_iters, min_visits)`
       — this crate's contribution.
  3. Ships a `topk-stability-bench` binary that builds a real graph,
     runs all policies on the same queries and entry points, and prints
     recall@k, mean visits, mean distance computations, early-stop fraction,
     and p50/p95/p99 latencies to stdout in a copy-pasteable table.

## Consequences

Positive:

  * The termination rule now consults the object the caller actually
    receives (a ranked id list), not a proxy scalar.
  * The knob (τ*) is *ordinal* and reads naturally: "τ* = 0.95 means the
    top-k ordering has changed by fewer than one adjacent swap on
    average". Operators tune ordinal thresholds more confidently than
    epsilon-of-a-squared-Euclidean-distance thresholds.
  * The policy has zero calibration and no LUT — a hard property that
    ADR-278 (adaptive-recall) explicitly relaxes to buy calibrated
    recall guarantees.

Negative / open:

  * Kendall's tau is O(k²) per check. For the typical k ∈ [10, 100],
    the cost is a few thousand comparisons — swamped by a single
    distance computation on high-dim vectors. For very large k this
    becomes visible; a fix (Knight's O(k log k) tau) is documented as
    future work, not implemented here.
  * The policy needs at least one prior top-k snapshot before it can
    fire. `window` iterations of warmup are unavoidable. On very short
    searches this is a meaningful fraction of total work.
  * At τ* = 1.00 with small `window`, spurious perfect stability at
    the very start of a search can trigger early termination and
    *lower* recall than τ* = 0.95 with a longer window. This is a
    tuning gotcha, documented in the research README's "practical
    failure modes" section.

## Alternatives Considered

  * **Spearman ρ.** Same ordinal-agreement idea, cheaper (O(k) after a
    sort), but sensitive to shifts of individual items across the whole
    ranking; less robust than Kendall to a single-position swap that
    barely affects the caller.
  * **Set-Jaccard on top-k.** Ignores ordering entirely. Two rankings
    with identical membership but reversed order look "perfectly
    stable"; the caller would see a completely different top-1.
  * **Learned termination model.** Higher ceiling but requires training
    data and calibration; explicitly outside this ADR's scope (and
    partially covered by ADR-303).
  * **Compose with existing signals.** τ* × entropy × gap could
    plausibly dominate any single signal. Left as follow-up in the
    research README.

## References

  * Kendall, M.G. (1938). "A New Measure of Rank Correlation." Biometrika 30.
  * Malkov & Yashunin (2020). "Efficient and Robust Approximate Nearest
    Neighbor Search Using Hierarchical Navigable Small World Graphs."
    TPAMI. — canonical HNSW termination.
  * ADR-278 adaptive-recall-ann, ADR-303 entropy-adaptive-ann,
    ADR-289 speculative-ann — sibling termination-signal ADRs against
    which this one is contrasted.
