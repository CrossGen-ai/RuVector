# ADR-361: Residual Vector Quantization with Query-Adaptive Early Exit

- **Status:** Proposed (nightly research PoC; not yet integrated)
- **Date:** 2026-11-15
- **Author:** nightly research agent
- **Supersedes:** —
- **Relates to:** ADR-359 (e-means shared clustering), prior quantization nightlies
  (RaBitQ, PQ-ADC, SPQ, LVQ, cascade-ADC, variance-adaptive SQ).

## Context

The ruvector corpus has shipped nightly explorations of nearly every single
practical quantizer for ANN — PQ with its many fastscan variants, RaBitQ and
its multi-bit and rotation-ensemble cousins, SQ (Lloyd-Max, μ-law, variance-
adaptive), LVQ, LeanVec, SPQ, and more. One classical family was conspicuously
absent: **Residual Vector Quantization** (RVQ) and its modern descendants
(Additive Quantization, LSQ). RVQ is production-deployed (FAISS `IVF_RVQ`,
Milvus `IVF_RQ`, Qdrant) and is the natural building block for learnable
"LLM-style" codebook stacks. We needed an in-tree reference.

Additionally, every quantizer in the corpus decodes **all** codebook stages
for every query. The "early-exit" idea from neural inference literature has
not been applied to residual ADC scanning in ruvector.

## Decision

Introduce a standalone research crate `crates/rvq-early-exit/` implementing:

1. **Residual VQ** with M stages × K codebooks, greedy residual encoding,
   `u8` codes, per-item partial-norm precomputation.
2. **Asymmetric ADC** decomposed as
   `||q - x_i||^2 = ||q||^2 + part_norms[i][M] - 2 Σ_s lut_s[q][i_s]`, which
   admits exact O(1) per-stage refinement.
3. **Deterministic early-exit policy**: shortlist by stage-0 partial dist,
   refine stage-by-stage, stop when top-K identity is stable for
   `stable_streak` consecutive stages.
4. **Benchmark binary** emitting real numbers (not mocks) for RVQ-full vs.
   RVQ-early-exit vs. exact brute force.
5. **Four correctness tests** covering reconstruction consistency,
   partial-norm consistency, full-vs-early overlap, and recall vs. exact.

The crate is a **standalone `[workspace]`**, zero external dependencies, so
it compiles without touching the main ruvector workspace manifest.

## Consequences

**Positive**
- Fills the RVQ gap in the quantizer corpus.
- Introduces a composable "early-exit" idea that applies verbatim to any
  staged-decoding ADC (PQ with multiple subcodebooks, AQ, cascade quantizers).
- Zero deps + standalone workspace keeps review scope tight.
- Real benchmarks: 3.4–4.5× speedup over full RVQ-ADC, 9–14× over exact brute
  force on a 20 k × 128 dataset, single-threaded, scalar code.
- Deterministic PRNG and fixed seeds make results reproducible.

**Negative / Trade-offs**
- Greedy residual encoding leaves 10–15 recall points on the table vs. joint-
  optimized AQ/LSQ. Production path requires beam-search encoding.
- On very noisy query distributions the stability predicate rarely fires;
  observed savings come from shortlist refinement rather than true early exit.
  Margin-based exit is the next-step fix (see research doc §roadmap).
- No SIMD/fastscan path yet — scalar only. Production integration will need
  a 4-bit-packed, SIMD-shuffle inner loop.
- Early-exit is approximate relative to full-depth RVQ-ADC, bounded by
  stage-0 shortlist recall; must be documented as "approximate ADC" and not
  "exact ADC with speedup".

## Alternatives considered

1. **Full-depth RVQ only, no early exit.** Simpler but misses the novelty
   and leaves known-free runtime on the table.
2. **Learned exit predictor.** More recall at equal latency but adds a
   training step and a model dependency; rejected for a nightly PoC.
3. **Build on top of existing `ruvector-rabitq` crate.** Natural, but would
   couple the PoC to that crate's build graph. Standalone is cleaner for a
   research deliverable; integration lives in the roadmap.
4. **Skip early-exit; just contribute RVQ.** Less interesting; RVQ in
   isolation is a 40-year-old technique already documented everywhere.

## Follow-ups / upgrade path

1. Port encoder to `beam-search` (AQ) under a feature flag.
2. Replace identity-stability exit with margin-based bound using per-stage
   LUT max as a remaining-contribution upper bound.
3. Add `fastscan` SIMD backend (4-bit packed codes + `pshufb`-equivalent).
4. Integrate as a new `IndexBackend` variant in `ruvector-core` once the
   crate graduates from `crates/rvq-early-exit/` to `crates/ruvector-rvq/`.
