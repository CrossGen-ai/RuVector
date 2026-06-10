# ADR-199 — Adaptive nprobe IVF via the `ProbeStrategy` trait

- Status: Proposed
- Date: 2026-06-10
- Owner: Nightly research (auto-generated)
- Related: ADR-193 (rairs-ivf), `crates/ruvector-rairs`

## Context

ruvector ships several IVF-derived indexes (`ruvector-rairs`,
`ruvector-betivf`, `ruvector-anisotropic-pq`). They all share the same
static-`nprobe` interface: the operator picks a single integer at index-load
time, sized for the recall target on the worst expected query in the
workload. Empirically, this over-budgets the average query by 2–4×.

Two literatures address this:

1. **Plateau / early-termination heuristics** (AdaIVF SIGMOD 2023,
   Milvus `search_iterator`, Qdrant adaptive `ef`). Cheap, no model, no
   per-query features.
2. **Learned per-query budget prediction** (AdaptNN VLDB 2020). More
   accurate but introduces a model artifact and a training loop.

We need a path that lets ruvector adopt (1) immediately without locking
the door on (2) and without disturbing the public API of any existing
crate.

## Decision

Add a standalone crate `crates/ruvector-adaptive-probe` that defines a
small `ProbeStrategy` trait and ships three concrete implementations:

- `FixedNprobe` — baseline, behaviour-compatible with current IVF crates.
- `PlateauProbe` — terminates on a `patience`-step stagnation of the
  running best score.
- `MarginBudget` — terminates when the next centroid is farther from the
  query than the current k-th score plus a configurable margin.

The trait is the integration surface; future learned strategies (e.g. an
AdaptNN-style regressor) implement the same trait and slot in without
changes elsewhere. Existing IVF crates remain unchanged; adoption is opt-in.

## Consequences

**Positive**

- Real measured savings on a 20k / D=64 / k=10 workload: 56% of the probe
  budget at 1.3 pp recall loss versus the closest fixed setting, +14% QPS.
- Net-new public surface is one trait, three structs, ≈30 lines of API.
- The trait is narrow enough (only quantities already computed by IVF
  appear in the signature) that the per-decision overhead is one branch.
- No dependencies on workspace `ruvector-core` — the crate can be
  vendored or backported standalone.

**Negative**

- A new crate joins the workspace; CI build time goes up by the cost of
  one small Rust crate (≈4 s release on M4 Max).
- `MarginBudget` is provably correct but, on high-dim Gaussian clusters,
  collapses to "stop at warmup" — operators must understand the bound
  before tuning the `margin` knob.
- The k-means used at build time is a tiny Lloyd loop and will not scale
  past ~100k points; a real integration must replace it with the
  production k-means already used by `ruvector-rairs`.

## Alternatives considered

- **Patch each IVF crate in place.** Rejected: locks every IVF index to a
  single strategy, breaks the existing public API, and forces all callers
  to migrate in one PR.
- **Build it as a feature flag in `ruvector-rairs`.** Rejected for the
  first cut: keeping the algorithm in its own crate makes the
  trait+strategies easier to fuzz, bench, and reason about in isolation.
  A feature-flagged re-export from `ruvector-rairs` is the natural next
  step once the API settles.
- **Adopt AdaptNN-style learned prediction directly.** Rejected for now:
  introduces a model artifact, a feature pipeline, and a training loop
  before we have evidence that the heuristic strategy is insufficient.
  The trait keeps this door open for a later ADR.

## Verification

- `cargo build --release -p ruvector-adaptive-probe` — passes.
- `cargo test --release -p ruvector-adaptive-probe` — 9 tests passing,
  zero ignored, no mocks.
- `cargo run --release -p ruvector-adaptive-probe --bin adaptive-probe-demo`
  prints the measured table reproduced in
  `docs/research/nightly/2026-06-10-adaptive-probe-ivf/README.md`.
- `cargo bench -p ruvector-adaptive-probe` runs the criterion harness.
