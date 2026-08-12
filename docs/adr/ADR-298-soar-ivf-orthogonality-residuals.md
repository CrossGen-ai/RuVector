# ADR-298: SOAR-IVF — Orthogonality-Amplified Residuals for IVF Spilling

- **Status**: Proposed (research spike)
- **Date**: 2026-08-12
- **Related crates**: `ruvector-soar-ivf` (new), `ruvector-core`, `ruvector-turboquant`
- **Related ADRs**: ADR-297 (adaptive compression & retrieval plane), ADR-296 (Turbo4)
- **Research doc**: `docs/research/nightly/2026-08-12-soar-ivf-orthogonality-residuals/`

## Context

IVF (Inverted File) partitioning assigns each vector to its single nearest
centroid. At query time we probe the top-`n_probe` centroids and scan their
posting lists. Recall at low `n_probe` is bounded by the fraction of true
neighbors that live in the probed cells — vectors near cell boundaries are
routinely missed.

The classical fix is **spilling**: assign each vector to its top-`s` nearest
centroids so a boundary vector shows up in every posting list that plausibly
covers it. Storage grows `s×`, but recall at low `n_probe` improves.

The open question is *which* secondary centroid a spilled vector should land
in. "Top-`s` nearest" is a greedy answer that ignores the geometry of the
primary residual `r_p = x − c_p`. If two candidate secondaries are equidistant
from `x`, one may sit *along* `r_p` (redundant — covers the same error
directions the primary already covers) and one may sit *orthogonal* to `r_p`
(complementary — covers a new direction). Top-`s` picks either with equal
probability.

Google Research's **SOAR** (Sun et al., *Improved Indexing for Approximate
Nearest Neighbor Search*, NeurIPS 2024, used in ScaNN) formalises this: pick
the secondary centroid that minimises

```
cost_secondary(c_s) = ‖r_s‖² + λ · ((r_p · r_s)² / ‖r_p‖²)
```

The second term is the squared projection of the secondary residual onto the
primary residual, normalised by primary residual magnitude. It penalises
secondaries that duplicate the primary's error directions — hence
"**orthogonality-amplified residuals**".

We need to know, on our own data and with our own IVF machinery, whether this
is a real, honest, non-cherry-picked gain over ordinary top-`s` spilling — and
under which scoring regime the win holds.

## Decision

Ship `ruvector-soar-ivf` as a **research crate** (not a production index)
implementing three variants under one `IvfVariant` trait so recall differences
are attributable to assignment strategy alone:

1. `IvfSingle`     — classic IVF-Flat, one centroid per vector (control).
2. `IvfSpillTopK`  — naive spilling to top-`s` nearest centroids (baseline).
3. `IvfSoar`       — SOAR assignment with the orthogonality-amplified cost.

All three share:

- The same k-means-lite centroid trainer (fixed seed, deterministic).
- The same IVF-Flat probe-and-scan retrieval.
- The same benchmark harness with two scoring modes:
  - **EXACT** — probed candidates are re-scored against the actual query with
    squared-L2 (i.e. IVF is used only to prune the corpus; distances are true).
  - **APPROX** — probed candidates are scored with a centroid-anchored lower
    bound (no corpus rescan). This isolates the value of the *cell assignment*
    itself, independent of a rescore that would smooth over assignment errors.

Only the assignment step differs between variants. This makes any recall
delta attributable to SOAR alone.

The crate is `no-external-deps`, `f32`, squared-L2, and lives at
`crates/ruvector-soar-ivf/`. It exposes:

- A `benchmark` bin producing the numbers reported in the research doc.
- Unit tests including `soar_cost_penalises_parallel_residuals` that pins
  the SOAR scoring identity on a hand-built configuration.

## Consequences

**Positive**

- We have a fair, in-tree, reproducible A/B between top-`s` spilling and SOAR
  spilling, so ADR-297's compression-and-retrieval plane can be extended with
  data instead of paper claims.
- The `IvfVariant` trait becomes a shared template for future IVF assignment
  strategies (learned assignment, RaBitQ-aware assignment, etc.).
- Real benchmark output is captured in
  `docs/research/nightly/2026-08-12-soar-ivf-orthogonality-residuals/benchmark-output.txt`
  and referenced verbatim in the research README, giving us a durable receipt.

**Negative / risks**

- SOAR does **not** universally win. On our 20k×128 synthetic, in **EXACT**
  scoring mode SOAR *loses* modestly to naive top-`s` (Δ recall@10 up to
  −0.05 at low `n_probe`). Any messaging that says "SOAR always wins" would
  be false.
- SOAR *does* win, marginally, in **APPROX** scoring mode at `spill=3` and
  higher `n_probe` (Δ up to +0.0016) — its designed regime. Small deltas at
  our scale mean we cannot yet claim SOAR is a slam-dunk upgrade for a
  production plane.
- The current implementation is `f32` scalar. It is a correctness reference,
  not a perf reference — do not benchmark it against Turbo4 / RaBitQ code
  paths.

**Follow-ups**

- Larger corpora (`n ≥ 1M`) and non-uniform (real-embedding) distributions
  before promoting SOAR into the retrieval plane.
- Compose SOAR assignment with **RaBitQ** candidate scoring (ADR-297 role
  split) — the SOAR paper's headline gains show up with 1-bit / low-bit
  scoring, which is exactly where APPROX-mode wins on our data hint.
- Sweep `λ` (currently fixed at 1.0). The paper suggests tuning per corpus.

## Alternatives Considered

- **Do nothing / keep only top-`s` spilling.** Simple, but we can't tell
  whether SOAR is worth wiring into ADR-297. Rejected — this ADR is precisely
  about earning that answer.
- **Import ScaNN.** ScaNN is C++/BLAS-heavy and not WASM-safe; would violate
  the "no external deps" property that makes the research crate a clean
  reference. Rejected.
- **Skip the EXACT-mode measurement.** Tempting because SOAR loses there, but
  that's the honest signal that SOAR is a *cell-assignment* improvement, not
  a *distance-computation* improvement — worth publishing.

## References

- Sun, P., Simcha, D., Dopson, D., Guo, R., Kumar, S. *SOAR: Improved
  Indexing for Approximate Nearest Neighbor Search.* NeurIPS 2024. Google
  Research / ScaNN.
- Chen, Q. et al. *SPANN: Highly-efficient Billion-scale Approximate
  Nearest Neighbor Search.* NeurIPS 2021. (Contrast: SPANN spills to reduce
  boundary loss; SOAR spills *smarter*.)
- Guo, R. et al. *Accelerating Large-Scale Inference with Anisotropic
  Vector Quantization.* ICML 2020. (ScaNN's anisotropic quantization; SOAR
  extends the same "penalise error along query direction" idea from
  quantization to spilling.)
- Jégou, H., Douze, M., Schmid, C. *Product Quantization for Nearest
  Neighbor Search.* PAMI 2011. (IVF-PQ; the base architecture SOAR augments.)
