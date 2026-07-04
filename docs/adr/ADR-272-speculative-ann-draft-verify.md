# ADR-272: Speculative ANN Search — Draft-and-Verify Vector Retrieval

- **Status**: Proposed (PoC lands in `crates/ruvector-specann` on branch
  `research/nightly/2026-07-04-speculative-ann-draft-verify`)
- **Date**: 2026-07-04
- **External anchor**: Leviathan et al., *Fast Inference from Transformers via
  Speculative Decoding* (ICML 2023); Chen et al., *Accelerating LLM Decoding
  with Speculative Sampling* (DeepMind 2023).
- **Related crates**: `ruvector-rabitq`, `ruvector-lvq`, `ruvector-pq-search`,
  `ruvector-diskann`, `ruvector-core` (HNSW).

---

## Context

Every quantized ANN index in the repo (`ruvector-rabitq`, `ruvector-lvq`,
`ruvector-pq-search`, `ruvector-anisotropic-pq`, `ruvector-avq`, ...) already
implements a de-facto two-stage pipeline: quantized approximate ranking followed
by float32 rescoring on a top-K subset. That pattern is baked into each crate
opaquely — the K, the rescore batch, and the "when to rescore more" decision
live inside each index type. There is **no shared abstraction** for:

1. The **draft/verifier split** as a first-class trait pair.
2. The **escalation policy** that decides when the draft is trustworthy vs
   when to widen the candidate set (or fall back to exact).
3. **Observable stats** per query (candidates drafted, candidates verified,
   escalations fired, full-verify fallbacks).

Meanwhile, LLM inference has converged on *speculative decoding* as a decisive
throughput technique. The structural analogy is exact:

| LLM speculative decoding      | ANN speculative search              |
|-------------------------------|-------------------------------------|
| Draft model proposes N tokens | Draft index proposes k_draft candidates |
| Verifier does 1 fwd pass      | Verifier does k_draft float32 rescores  |
| Accept prefix, reject suffix  | Accept top-k, escalate on thin gap  |
| Speedup ~2–3× at same quality | Speedup depends on draft speed & recall |

Making this pattern explicit lets us swap drafts (int8, 1-bit, PQ, RaBitQ,
HNSW-partial-walk) without touching downstream code and lets us tune the
escalation policy per workload.

## Decision

Introduce `crates/ruvector-specann` with a trait-based draft-and-verify
architecture:

```rust
trait DraftIndex : Send + Sync {
    fn draft(&self, q: &[f32], k_draft: usize) -> Result<Vec<Neighbor>>;
    fn len(&self) -> usize;
}
trait Verifier : Send + Sync {
    fn verify(&self, q: &[f32], ids: &[u32]) -> Result<Vec<Neighbor>>;
    fn dim(&self) -> usize;
    fn len(&self) -> usize;
}
struct SpecAnnIndex<D: DraftIndex, V: Verifier> { draft: D, verifier: V, policy: EscalationPolicy }
struct EscalationPolicy { alpha, gap_threshold, max_escalations, escalate_multiplier }
struct SpecStats { escalations, draft_candidates, verified, full_verify_fallback }
```

Ship three implementations in the PoC:

1. `F32BruteForce` — exact float32 baseline. Implements both traits.
2. `Int8BruteForce` — symmetric per-vector int8 draft.
3. `Sign1BitDraft` — 1-bit sign packed into u64s (Hamming proxy for L2).

Adapters for `ruvector-rabitq`, `ruvector-lvq`, and HNSW graph walks land in a
follow-up (roadmap §1-2 of the research doc).

## Consequences

**Positive.**

- **One clear abstraction** for a pattern already implemented five different
  ways across the tree. New index types add value by picking a
  `DraftIndex` / `Verifier` combination, not by re-implementing rescore
  bookkeeping.
- **Measured verification savings.** On 10 000 × 128 gaussian, int8 SpecANN
  verifies **255 / 10 000 vectors on average at 100 % recall@10**. When the
  draft becomes sub-linear (HNSW), this savings *composes* with the graph
  speedup instead of duplicating it.
- **Escalation policy is observable and swappable.** `SpecStats` per query
  makes it possible to train a learned escalation policy offline (roadmap §3).
- **Composability with existing crates.** `ruvector-rabitq` becomes a plug-in
  `DraftIndex`; `ruvector-core` HNSW becomes a plug-in `DraftIndex`;
  `ruvector-postgres` becomes a plug-in `Verifier` backed by disk.

**Negative / risks.**

- **Naive brute-force draft is slower than raw f32.** The PoC's `Int8BruteForce`
  is O(n) with per-vector scale multiplies — it costs *more* per compare than
  a flat f32 dot on M4 Max. The QPS win only appears when the draft is
  genuinely sub-linear or SIMD-packed. This is noted honestly in the research
  doc's results table.
- **1-bit draft has low recall on isotropic-gaussian data without rotation.**
  Variant D hits only 0.699 recall@10; wiring in RaBitQ's Hadamard rotation
  is required for real-world text/image embeddings and is roadmap §1.
- **Trait-object dispatch cost** if users need runtime backend selection. We
  keep the generic `SpecAnnIndex<D, V>` form; a boxed variant can land later
  behind a feature flag.

## Alternatives

1. **Bake speculation into each index type (status quo).** Rejected:
   duplicates code, prevents policy tuning across index types, and hides the
   verification-savings metric that motivates the whole pattern.
2. **Adaptive query-time recall calibration (single-index).** Related but
   different: adjusts `ef_search`-style parameters inside one index. Doesn't
   exploit draft/verifier asymmetry across index *types* the way SpecANN does.
3. **Cascading rerank pipeline (Multi-stage IR).** Similar in spirit, but IR
   pipelines are typically hand-composed and lack the confidence-gap
   escalation loop. SpecANN's gap heuristic is closer to speculative
   decoding's reject-and-roll-back than to a fixed cascade.

## Acceptance criteria (met by the PoC)

- [x] `cargo build --release -p ruvector-specann` succeeds.
- [x] `cargo test --release -p ruvector-specann` — 4 unit + 2 integration
      tests pass. Int8 SpecANN ≥ 0.98 recall@10, 1-bit SpecANN ≥ 0.90
      recall@10 with tuned policy.
- [x] `cargo run --release -p ruvector-specann --bin specann-bench` emits
      real JSON results (see `docs/research/nightly/2026-07-04-*/bench_results.json`).
- [x] All files under 500 lines.
- [x] No mocks, no TODO stubs. Rust only.
