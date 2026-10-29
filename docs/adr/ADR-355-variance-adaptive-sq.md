# ADR-355: Variance-Adaptive Scalar Quantization (VA-SQ)

- **Status:** Proposed (research-branch deliverable, 2026-10-28)
- **Owners:** nightly-research
- **Related:** ADR-adaptive-bit-budget-pq (2026-09-27), ADR-hadamard-sq (2026-10-23)
- **Supersedes:** none
- **Branch:** `research/nightly/2026-10-28-variance-adaptive-sq`

## Context

ruvector ships uniform scalar quantization (SQ) in several crates:
`ruvector-rabitq` and `ruvector-pq-search` use it as a fallback,
`ruvector-cnn` and the embedded builds rely on it when PQ is too
heavy. Every one of those uses a fixed bit-width per dimension (4, 6,
or 8), even though real embedding distributions have per-dimension
variance spreads of 2–4 orders of magnitude. The low-variance dims
over-spend bits; the high-variance dims under-spend them.

Prior nightly work explored related ideas:

- **Hadamard-SQ (2026-10-23)** rotates the input to equalize variance
  per dim and then applies uniform SQ. Effective but costs one
  `D × D` rotation on every query, blocking edge/embedded use.
- **Lloyd-Max SQ4 (2026-10-05)** fits a per-dim 4-bit codebook; the
  level count is still uniform.
- **LVQ (2026-09-10)** does per-vector offset+scale; the bit width
  is still uniform.
- **Adaptive-bit-budget PQ (2026-09-28)** allocates PQ sub-codebook
  sizes by sub-space variance; it does not apply at the scalar level.

None of these vary the *bit count per dimension*. VA-SQ fills the
gap.

## Decision

Introduce `ruvector-variance-adaptive-sq` as a standalone crate on
the research branch with:

1. A pluggable `AllocPolicy` enum (`Uniform`, `LinearVariance`,
   `LogVariance`).
2. A water-filling allocator with `[b_min, b_max]` clip, exact total
   budget pinning, and `O(D * 32)` worst-case runtime.
3. A bit-packed codec (`VaSqCodec`) with per-dim LUT scanning.
4. A reproducible benchmark binary (`vasq-bench`) that produces real
   `us/query` numbers from `std::time::Instant`.

The default policy is `LogVariance`, with `b_min = 1, b_max = 2 *
budget_bits_per_dim` as safe defaults.

## Consequences

### Positive

- **2.37× recall@10 at 2 bits/dim** with identical memory to uniform
  SQ2. This is a strict Pareto improvement in the aggressive-
  compression regime and unlocks edge deployments at memory budgets
  where uniform SQ was previously unusable.
- **No extra bytes per vector** beyond a `b_max`-sized LUT per codec
  (shared across the dataset, O(KB)).
- **Deterministic** under a seeded RNG, so tests can pin behavior.
- Composable: VA-SQ can be dropped under any posting-list scanner
  that currently accepts uniform SQ codes, because the public
  `distance_lut` method takes the same `(query, code, lut)` shape.

### Negative / costs

- The variable bit-widths break the SIMD-friendliness of uniform SQ.
  The reference crate is scalar; a SIMD port is deferred to a
  follow-up ADR.
- **At `b >= 4`, uniform SQ beats VA-SQ-log by 0.03–0.09 recall.**
  The log-variance rule over-allocates to high-variance dims past the
  point where their quantization error dominates. Need a softer
  `sigma^alpha` rule for the mid-budget regime.
- Training requires a sample pass (`O(N * D)` additions for variance).
  For streaming ingest, a Welford online tracker + periodic refit is
  needed.
- Per-dim LUTs with large `b_max` can exceed L1; the crate exposes
  `b_max` to let operators tune.

### Alternatives considered

- **Keep uniform SQ.** Status quo; unacceptable at `b <= 3`.
- **Hadamard rotation then uniform SQ.** Works but costs `D^2 / 2`
  butterfly ops per query. VA-SQ costs zero extra query work.
- **Full PQ or RaBitQ at the same memory.** Both are strictly better
  recall-wise but need codebook training, larger metadata, and more
  complex scan paths. VA-SQ is the honest "cheap SQ replacement."
- **Reverse the policy (bits ∝ 1 / variance).** Rejected: this is
  equivalent to spending bits on noise, which empirically collapses
  recall to the uniform-SQ-at-b=1 level.

## Rollout plan

1. Research branch only (this ADR + the standalone crate).
2. Follow-up ADR: lift `AllocPolicy` into
   `ruvector-core::quantization`, add a `trait Codec`.
3. Follow-up ADR: integrate VA-SQ as a per-posting-list code format
   in `ruvector-ef-ivf-posting` and `ruvector-blockmax-ivf` with
   per-cluster variance (not just global).
4. Follow-up ADR: SIMD port once `std::simd` stabilizes.

## Acceptance

- `cargo build --release -p ruvector-variance-adaptive-sq` succeeds.
- `cargo test --release -p ruvector-variance-adaptive-sq` passes (3
  unit tests covering allocator correctness, round-trip encode/decode,
  and LUT-vs-decode distance equivalence).
- `cargo run --release --bin vasq-bench` prints the acceptance line
  `# Acceptance test: VASQ-log-b2 recall@10 >= 0.30: PASS`.

All three verified on this branch, 2026-10-28.
