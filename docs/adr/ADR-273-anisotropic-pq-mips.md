# ADR-273: Anisotropic Product Quantization for MIPS

**Status**: Proposed
**Date**: 2026-08-01
**Author**: Nightly Research Agent
**Branch**: `research/nightly/2026-08-01-anisotropic-pq-mips`
**Crate**: `crates/ruvector-anisotropic-pq`
**Related**: ADR-264 (PQ-ADC), ADR-141 (RaBitQ), ADR-272 (Speculative ANN), ADR-193 (RAIRS IVF)

---

## Context

Every product-quantization variant currently in the RuVector stack
(`ruvector-pq-search`, ADR-264; the PQ path of `ruvector-rabitq`,
ADR-141) trains its sub-codebooks by minimising unweighted squared
reconstruction error `‖x - x̂‖²` — the classical Lloyd k-means loss
introduced by Jégou, Douze, and Schmid (TPAMI 2011). This loss is a
principled objective when the downstream task is L2 nearest-neighbour
search, because L2 error and reconstruction error coincide.

For **Maximum Inner Product Search (MIPS)** — the retrieval regime for
LLM RAG, recommendation, and any embedding whose relevance signal is a
dot product — the classical loss is provably wrong. The MIPS score error
is

```
⟨q, x⟩ - ⟨q, x̂⟩ = ⟨q, r⟩,   r = x - x̂
```

which depends only on the component of `r` **parallel to `q`**. The
component orthogonal to `q` contributes zero score error and therefore
zero recall error under a random or `x`-correlated query distribution.
Bits spent minimising the orthogonal component are wasted for MIPS.

Guo et al. (ICML 2020, ScaNN) resolved this with an **anisotropic
quantization loss** that penalises the parallel component of the
residual `η ≥ 1` times more heavily than the orthogonal component. ScaNN
is the reigning open-source MIPS system on billion-scale benchmarks;
its anisotropic loss is the single largest algorithmic win over vanilla
PQ, larger than the OPQ rotation or the reordering tricks that came
before.

RuVector has no anisotropic-loss variant. Every RAG-facing consumer of
our PQ index is silently paying a recall tax whose magnitude the ScaNN
paper reports as +5–15% at fixed memory.

---

## Decision

Introduce `crates/ruvector-anisotropic-pq` as a standalone,
zero-runtime-dependency crate implementing three swappable
`Pq`-trait variants:

| Variant                | η    | Loss                                             |
|------------------------|-----:|--------------------------------------------------|
| `StandardPq`           |    1 | `‖r‖²` (Lloyd k-means baseline)                  |
| `AnisotropicPq { η=4 }`|    4 | `h_orth ‖r‖² + (h_par − h_orth)(r · û)²`         |
| `AnisotropicPq { η=16 }`|  16 | same, heavier parallel weight                    |

Assignment and centroid update both use the anisotropic loss, so
training is monotone in the objective. The centroid update is a per-cluster
symmetric positive-definite linear system solved by in-place Cholesky
(`src/math.rs`) — no external linear-algebra dependency. At `d_sub = 16`
the solve is 4 096 FLOPs per cluster per iteration, negligible relative
to O(n·k·d_sub) assignment.

The query path is unchanged: build an inner-product LUT per subspace
once per query, sum `m` lookups per database vector. Encoded bytes and
memory footprint are identical across variants; any recall difference
is attributable purely to codebook geometry, which makes the ablation
clean.

Public API:

```rust
pub trait Pq: Send + Sync {
    fn train(&mut self, data: &[Vec<f32>]);
    fn encode(&self, x: &[f32]) -> Vec<u8>;
    fn search(&self, query: &[f32], codes: &[Vec<u8>], k: usize) -> Vec<Hit>;
    fn name(&self) -> &str;
    fn memory_bytes(&self, n_codes: usize) -> usize;
}
```

Benchmark harness lives at `src/bin/benchmark.rs` and reports Recall@10,
per-top-k score MSE, mean and P95 latency, QPS, codebook memory, and
L2 reconstruction error against an exact f32 brute-force baseline.

---

## Consequences

### Positive

- **+4.3% relative Recall@10** on our synthetic MIPS bench
  (10 000 × 128, m = 8, k = 256) at η = 4, at zero query-time cost, zero
  extra memory, and no change to the search kernel.
- **Order-of-magnitude tighter score reconstruction** on the
  ground-truth top-10 (verified by unit test
  `higher_eta_tightens_score_on_top_relevant_vectors`).
- Codebook footprint identical to standard PQ (128 KB vs 5 MB for the
  f32 index: 39× compression).
- Query throughput ≈ 6 900 QPS on a single Apple M4 Max core, faster
  than the 2 510 QPS exact scan.
- Zero external runtime dependencies. Matches the discipline of
  `ruvector-speculative-ann` and the workspace-wide "no gratuitous
  crates" convention.
- Orthogonal to speculative verification (ADR-272) and to bit-level
  quantization (ADR-141), so this crate composes rather than competes.

### Negative

- **Training cost 2.1× standard PQ** on this size (1 339 ms vs 627 ms).
  For n = 10⁶ we expect this ratio to hold, so a full train is
  ≈ 2 minutes single-threaded — acceptable but noticeable.
- **η is a hyperparameter.** The sweet spot depends on n, d, and query
  distribution. On our bench η = 16 *hurts* recall (0.227 vs 0.257
  baseline) because the centroids collapse toward per-vector rays. A
  sweep in {1, 2, 4, 8, 16} is required per dataset.
- **Assumes query correlates with database directions.** Purely
  adversarial queries orthogonal to every `x_i` see no benefit.
- **Small-cluster edge cases.** Cholesky can refuse on ill-conditioned
  or single-point clusters; the code falls back to unweighted mean,
  which is safe but slightly degrades training quality.

### Neutral

- The Recall@10 absolute (0.268) is low because we deliberately picked
  a small m = 8 to isolate the anisotropic-vs-isotropic ablation.
  Increasing m or coupling with IVF (ADR-193) or with speculative
  verification (ADR-272) will move the absolute recall into
  production-relevant territory.

---

## Alternatives considered

1. **OPQ (Optimized PQ).** Learns a rotation `R` before quantization to
   equalise subspace variances. Complementary rather than competitive
   with anisotropic loss — the natural next step is `AnisotropicOPQ`.
   Rejected as a first step because rotation alone does not address the
   parallel-vs-orthogonal MIPS asymmetry.

2. **Additive / Composite Quantization (AQ, CQ).** Higher codebook
   expressivity, but much heavier training and no MIPS-specific loss.
   Deferred: promising for a follow-up ADR but too much scope for one
   nightly.

3. **RaBitQ / RaBitQ+.** Already in the workspace as ADR-141. Extreme
   compression via 1-bit sign quantization. Excellent for L2 and for
   scan-heavy MIPS but its per-vector code is a fixed sign pattern; the
   anisotropic idea does not directly apply. Complementary.

4. **LeanVec** (SIGMOD 2024). Learns a query-conditional low-rank
   projection. Compelling but requires a separate training pipeline
   and a per-query cost that would need its own benchmark. Deferred.

5. **Skip PQ entirely, use HNSW-only.** Rejected: PQ is required for
   memory-constrained deployments (agent laptops, on-device RAG). A
   +4 % free recall win at fixed memory is worth banking regardless of
   the graph-index track.

6. **Do nothing.** Rejected: current MIPS callers silently overpay on
   recall, and the fix is a self-contained ≈ 1 000-line crate.

---

## Follow-up work

- **SIMD LUT scoring.** Portable `simsimd` path to drop mean latency
  from ~144 µs to single-digit µs at this n.
- **Compose with speculative-ANN (ADR-272).** Use anisotropic PQ codes
  as the draft, brute-force verify top-k'. Expected Recall@10 > 0.99.
- **Rotation warm-up.** Add OPQ rotation before anisotropic training
  (`AnisotropicOPQ`). +2–4 % recall at zero query-time cost.
- **IVF coarse quantizer.** Pair with ADR-193 RAIRS IVF for n ≥ 10⁶.
- **Query-conditional η.** Learn η per query from a small regression on
  `‖x_i‖`, `‖q‖`, cluster occupancy.

---

## References

Full bibliography in
[`docs/research/nightly/2026-08-01-anisotropic-pq-mips/README.md`](../research/nightly/2026-08-01-anisotropic-pq-mips/README.md).
Primary source: Guo et al., "Accelerating Large-Scale Inference with
Anisotropic Vector Quantization", ICML 2020,
<https://arxiv.org/abs/1908.10396>.
