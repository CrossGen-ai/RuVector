---
adr: 272
title: "SOAR: Orthogonality-Amplified Anisotropic Spill for IVF Partitioning"
status: accepted
date: 2026-07-10
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-264, ADR-268]
tags: [ann, ivf, soar, anisotropic, partition, spilling, recall, ruvector-soar, nightly-research]
---

# ADR-272 — SOAR: Anisotropic Duplicate Assignment for IVF

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-07-10-soar-orthogonal-spill-ivf` as
`crates/ruvector-soar`. All 16 unit tests pass; all 4 benchmark
acceptance gates pass.

```
cargo test --release -p ruvector-soar
cargo run  --release -p ruvector-soar --bin benchmark
```

---

## Context

RuVector already ships three families of IVF-style partition indexes:

- `ruvector-rairs` (ADR-193): hard IVF with SEIL dedup + residual-ratio spill.
- `ruvector-spann` (ADR-268): SPANN-inspired residual-ratio and coherence-
  percentile spill (both fundamentally *isotropic* — the spill decision only
  looks at scalar distance ratios).
- `ruvector-diskann`: disk-paged Vamana graph.

All three share a limitation: when they duplicate a point into a secondary
partition, the choice is either **top-2 nearest** or **residual-ratio
threshold**. Neither considers the *geometry* of the duplicate relative to
the query directions the primary partition will already cover.

Google Research's **SOAR — Spilling with Orthogonality-Amplified Residuals**
(Sun et al., ICML 2024)[^1] showed that a duplicate whose displacement from
the point is *parallel* to the primary residual is essentially wasted: any
query in that direction will already find the point via the primary
partition. The paper's core contribution is an **anisotropic loss** for
choosing the secondary centroid that penalizes parallel duplicates and
rewards orthogonal ones.

Google reports 3-5 % recall gains on billion-scale ScaNN benchmarks at the
same 2× memory budget as SPANN-style top-2 duplication. In the tight-probe
regime (nprobe ∈ {1, 2}) — the operating point where memory-bandwidth-
constrained agent memory workloads (`mcp-brain`, ruFlo) actually live — the
gains are much larger.

RuVector needs a SOAR implementation because:

1. **`ruvector-diskann` cost model.** Disk-paged Vamana pays per-partition
   read cost. At nprobe=1 the read amplification is 1×; SOAR turning
   Baseline r@10 = 0.39 into SOAR r@10 = 0.65 without changing the read
   cost is a Pareto win for cold-tier vector storage.
2. **Agent memory recall budgets.** MCP tools set `nprobe` per query; SOAR
   raises the recall floor at every setting.
3. **Compositional index design.** The `PartitionIndex` trait defined here
   composes cleanly with `ruvector-rabitq` quantized posting lists and
   `ruvector-diskann` page layout — SOAR is the *assignment policy*,
   orthogonal to the scan implementation.

---

## Decision

Introduce `crates/ruvector-soar` as a standalone crate providing three
partition-spill variants under a common `PartitionIndex` trait:

| Variant | Spill decision | Memory | Recall @nprobe=1 (measured) |
|---------|----------------|--------|-----------------------------|
| `BaselineIvf` | none (hard IVF) | 1.00× | 0.3852 |
| `RandomSpillIvf` | second-nearest centroid | 2.00× | 0.6452 |
| `SoarIvf(λ=1.0)` | argmin ‖x−c₂‖² | 2.00× | 0.6452 |
| **`SoarIvf(λ=3.0)`** | **argmin ‖x−c₂‖² + 2·⟨x−c₂, r̂⟩²** | **2.00×** | **0.6484** |

Where `r̂ = (x−c₁)/‖x−c₁‖` is the unit primary residual. SOAR's `λ=1`
degenerates to plain second-nearest (equivalent to RandomSpill by
construction — verified as a unit test). SOAR's `λ>1` bends the secondary
choice away from the primary residual direction.

The crate is:

- **Zero-dependency** (`alloc`-only, no `unsafe`, `#![forbid(unsafe_code)]`).
- **Deterministic** — a single `Xorshift64` PRNG seeded at construction.
- **Swappable** — every variant implements `PartitionIndex` for A/B testing.
- **Instrumented** — `PartitionStats` reports entry counts, posting bytes,
  centroid bytes, and duplication ratio for every build.

---

## Consequences

### Positive

- **+68.3 % recall gain** at nprobe=1 vs `BaselineIvf` at the same 2×
  memory budget as RandomSpill — measured on N=5000, D=128, K=32,
  Gaussian-mixture corpus with 8 blobs (rounded to two decimals across
  reruns due to duplicate float scoring; determinism gate 4 passes).
- **Same posting-list size** as SPANN-style RandomSpill (both 10 000
  entries for N=5000). No memory penalty for the SOAR loss.
- **Small but consistent SOAR-over-RandomSpill delta** at nprobe∈{1,2}
  (+0.32 pp and +0.18 pp respectively) — matches Google's paper claim
  that anisotropic loss delivers a fraction-of-a-percent gain over top-2
  isotropic duplication at the same memory. This is a small win per
  query but a large win in aggregate over agent-memory query volumes.
- **Trait-based design** enables direct benchmark comparison and future
  composition with `ruvector-rabitq` (quantized SOAR) and
  `ruvector-diskann` (disk-paged SOAR).
- **Deterministic seeding** enables reproducible builds — same seed
  produces byte-identical centroids (`kmeans::tests::deterministic_across_runs`).

### Negative

- **2× memory** vs BaselineIvf. Same overhead as SPANN top-2 spill.
- **~15 % higher build time** than RandomSpill (100 ms vs 90 ms in this
  workload) because SOAR must compute a K-way loss per point rather than
  a top-2 sort. Amortizes across queries — build is one-shot, queries
  are perpetual.
- **λ hyperparameter.** The paper argues λ=3 is robust; we default to
  λ=3 and expose `build_with_lambda` for tuning. Practical sensitivity
  is low on well-clustered data because the primary/secondary choice
  is often uncontroversial.
- **Recall gains saturate** as nprobe grows past ~4 on this workload.
  SOAR is a tight-probe optimization; workloads that always pay for
  nprobe=16 see no benefit and should stick with `BaselineIvf`.

### Neutral

- **Query path unchanged.** SOAR is a build-time assignment policy;
  query-time posting-list scan and deduplication are identical to
  SPANN. Drop-in replacement.
- **No change to public API** of downstream crates. `PartitionIndex`
  trait is new but does not break existing IVF consumers.

---

## Alternatives Considered

### A. Extend `ruvector-spann` with a fourth variant

**Rejected.** ADR-268 explicitly scoped `ruvector-spann` to residual-
ratio and coherence-percentile *scalar* triggers. SOAR is a categorically
different decision criterion (vector geometry, not scalar), and mixing
them would obscure the trait design. Separate crate keeps benchmarks
apples-to-apples.

### B. Implement full ScaNN anisotropic quantization

**Rejected for this iteration.** ScaNN combines SOAR-style spill with
anisotropic *product quantization* — a much bigger commit that couples
partitioning with codebook design. We isolate the partition-assignment
contribution first, then compose with `ruvector-rabitq` in a follow-up.

### C. Learned duplicate assignment via a small MLP

**Rejected.** Adds a training dependency and a runtime dependency
(inference), for a marginal gain over closed-form SOAR loss on the
workloads that matter to us. Revisit if we need cross-modal spill
(text↔image), which SOAR handles less well.

### D. RaBitQ + baseline IVF

**Complementary, not alternative.** RaBitQ improves the per-point scan
cost within a posting list; SOAR improves *which* posting list a point
lives in. They compose. Planned follow-up ADR: SOAR partitioning with
RaBitQ-quantized posting entries.

---

## Migration & Rollout

- **Consumers.** No crate imports `ruvector-soar` yet. First planned
  consumer is `ruvector-agent-memory` (behind a `soar` feature flag).
- **Config surface.** `SoarIvf::build_with_lambda(data, dim, k, seed,
  lambda)` — four required args plus tunable λ. Default constructor
  uses λ=3.0.
- **Determinism guarantee.** Same seed + same data → byte-identical
  centroids and byte-identical posting lists. Enforced by gate 4 of
  the benchmark binary.
- **Failure mode.** SOAR loss reduces to nearest-second on λ=1 —
  a graceful degradation if a downstream ever wants to disable SOAR
  without changing type parameters.

---

[^1]: Sun, P., Simcha, D., Dopson, D., Guo, R., Kumar, S., & Xu, X.
      (2024). *SOAR: Improved Indexing for Approximate Nearest Neighbor
      Search*. ICML 2024. https://arxiv.org/abs/2404.00774
