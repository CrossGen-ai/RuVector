# ADR-353: Rotation-Ensemble RaBitQ

## Status

Proposed. New opt-in crate (`crates/ruvector-rabitq-ensemble`). Nightly research experiment, not yet promoted to a production dependency of any `ruvector-*` index crate.

## Context

RaBitQ (SIGMOD 2024, Gao & Long; ADR-154 in this tree) is our 1-bit vector-quantisation primitive: rotate a vector by a random orthogonal `R`, store `sign(R x)` as `D` bits, and estimate inner products with `(1/√D) · ‖x‖ · Σᵢ sign(Rx)ᵢ · (Rq)ᵢ`. Variance scales as `O(‖x‖²‖q‖²/D)` under Haar-uniform `R`. The 2024 paper gives this bound; the TurboQuant preprint (arXiv:2504.19874 §3.2) shows the randomised Hadamard construction `D₁·H·D₂·H·D₃` hits the same asymptotic regime at `O(D log D)` apply cost and `O(D)` rotation storage.

Two independent axes let us spend more memory to shrink the per-row estimator variance:

1. **More bits per rotation** — multibit RaBitQ (b bits per coordinate under one rotation). Variance shrinks `~2^(−b)` per coordinate. Covered by our nightly `2026-10-11-extended-rabitq-multibit`.
2. **More independent rotations** — K rotations, average K unbiased estimators. Variance shrinks `~1/K`.

Both pay `K` or `b` bits per coordinate. No experiment in `ruvector/` yet compares the two axes at matched bit budget.

## Decision

Ship `ruvector-rabitq-ensemble` as a self-contained nightly crate implementing axis (2) and the measurement harness for it. Interface:

* `trait Index { fn dim; fn len; fn aux_bytes; fn search(q, k) -> Vec<SearchResult>; }`
* `FlatF32Index` — exact oracle.
* `RabitqSingle` — single-rotation 1-bit baseline. Stores `3·D_pad` floats of rotation state (Hadamard-signed) or `D²` floats (Haar-dense), plus `D_pad/8` bytes of signs and one f32 norm per row.
* `RabitqEnsemble` — K independent `RabitqSingle` members built with seeds `seed, seed+1, …, seed+K-1`. Query score is `(1/K) · Σⱼ est_ipⱼ(q, x)`.
* `Rotation { HaarDense | HadamardSigned | HadamardSingle }` with deterministic seeding, O(D²) or O(D log D) apply, exposed via `apply_into`.

Non-dyadic `d` zero-pads to `D_pad = next_pow2(d)` for the FWHT backends; the Haar backend stays at `D`.

Independence between ensemble members is enforced through independent seeds for the three sign vectors (HadamardSigned), the one sign vector (HadamardSingle), or the Gaussian block (HaarDense). All members share the same `RotationKind` and are built in a deterministic order so `(dim, kind, seed, data, K)` fully determines the index.

### What this ADR explicitly does *not* decide

* Whether to replace single-rotation RaBitQ in any existing index. We treat this as a research artefact until a head-to-head against multibit RaBitQ at matched storage settles the question.
* The on-disk serialisation. The current crate has none; adding `serde` + mmap is deferred to a follow-up ADR if we promote.
* SIMD. The current scan loop is scalar; we document the fused K-code SIMD rewrite as a next step but do not block the ADR on it.

## Consequences

### Positive

* Clear factored comparison between "more bits" and "more rotations" at matched memory, with runnable code.
* Opt-in and isolated: no change to existing crates; no new workspace-level dependencies beyond `rand`, `rand_distr`, `thiserror` (all already present).
* `#![forbid(unsafe_code)]` throughout; deterministic given `(dim, kind, seed, data, K)`; bit-identical rebuild under test.
* K=1 ensemble is score-equivalent to `RabitqSingle` (bit-identical for `HadamardSingle`), so the ensemble path cleanly subsumes the baseline without being wasteful at K=1.
* Three rotation backends let us empirically confirm the TurboQuant claim that one-sign HD is "close to Haar" at d ≥ 64 (we see <1% recall gap on random L2-normalised data at d=64).

### Negative

* **Linear memory in K.** At K=8 we pay 8× the aux bytes of the baseline. For comparison, 8-bit multibit RaBitQ also pays 8× — but you then get a non-uniform codebook, not independent re-rolls of the same 1-bit codebook. The crossover is empirical.
* **Linear scan latency in K.** The current scalar scan reads K rows per candidate. A fused SIMD scan that packs all K bits into one register per coordinate is the obvious fix and is not yet implemented. On uniform random L2-normalised data at n=100k, d=128, K=4 is ~5× slower per query than K=1; K=8 is ~6×. Not production-ready.
* **Variance reduction is only `1/K` in the Haar limit.** The HadamardSigned and HadamardSingle variants give slightly worse-than-independent per-member errors because the construction leaves residual correlations between members. Empirically at d ≥ 64 we see recall gains tracking the Haar prediction closely, but the inequality is not tight.
* **Recall ceiling is modest.** On random L2-normalised data at d=128, K=8 reaches ~0.60 recall@10. Any production use must pair the ensemble with an exact-f32 rerank over top-M candidates.

### Risk

* **Research-only numbers.** All measurements use uniform random L2-normalised synthetic vectors. Natural embeddings (clustered, anisotropic) typically yield much higher single-rotation RaBitQ recall, which compresses the marginal gain from K>1. Before promoting, re-measure on a real embedding set (OpenAI/E5/BGE at d=768 or d=1024, SIFT-1M, GIST-1M).
* **Scope creep risk.** If an SIMD scan rewrite plus serialisation plus residual-rerank integration land in the same crate they add up to a non-trivial new surface; keep each in its own PR under its own ADR.

## Alternatives considered

1. **Multibit RaBitQ only (status quo + ADR-draft for `extended-rabitq`).** Pros: single rotation per row, variance drops geometrically with b. Cons: non-uniform codebook adds code complexity; we miss the direct ensemble baseline needed to prove multibit is actually better at matched memory. The right move is to ship ensemble first *as the comparison baseline for multibit*, then land multibit against it. This ADR does the first half.

2. **Correlated rotations (orthogonal signs).** Draw sign vectors with provable low-correlation structure across members (e.g., signs from independent cosets of a Walsh code). Expected to give strict variance improvement over free-independent at the same K. Rejected for v1 because the theory is not nailed down; keep as a documented follow-up.

3. **Random projections ensemble without the sign step.** Classical: average IPs of f32-projected vectors across K independent projections. Pros: no quantisation error. Cons: no bit compression — gives up the entire reason to use RaBitQ. Out of scope.

4. **Residual rerank only (status quo of `ruvector-fused-rabitq-residual`).** Pros: already lands most of the recall benefit at the top-M level. Cons: adds f32 storage proportional to the residual; does not improve the pre-rerank candidate quality. The ensemble improves candidate quality directly, so the two are complementary rather than mutually exclusive.

## Related

* ADR-154 — original RaBitQ rotation / binary quantization.
* ADR-161 — RaBitQ WASM package.
* Nightly `2026-10-11-extended-rabitq-multibit` — the multibit axis.
* Nightly `2026-09-04-fused-rabitq-residual` — residual rerank over RaBitQ.
* Nightly `2026-09-22-opq-rotation` — OPQ-style rotation for PQ.
* TurboQuant preprint — RHT `D·H·D·H·D` construction (arXiv:2504.19874).
