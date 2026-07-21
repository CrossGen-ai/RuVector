# ADR-272: Elastic Bit Allocation for Product Quantization (EBA-PQ)

- **Status**: Proposed (working PoC, benchmarks captured — `crates/ruvector-elastic-pq`)
- **Date**: 2026-07-21
- **Related**: ADR-267 (SOTA validation protocol); prior PQ-family research
  in `docs/research/nightly/` (RaBitQ 2026-04-23, anisotropic-VQ 2026-05-09,
  LVQ 2026-05-08 / 2026-05-19, LeanVec 2026-05-12, OPQ 2026-05-24, PQ-Fast-Scan
  2026-05-20, PQ-ADC-Search 2026-06-20, anisotropic-PQ 2026-05-21 / 2026-05-26 /
  2026-06-04, SymphonyQG 2026-05-17 / 2026-05-22 / 2026-06-02).

---

## Context

Every PQ variant explored in the RuVector nightly line so far attacks one
of three levers:

1. **The codebook** — RaBitQ (random projection + binary), LVQ / LeanVec
   (local scaling), SymphonyQG (jointly-learned graph+codebook).
2. **The rotation** — OPQ (learn a global rotation), Anisotropic-VQ /
   Anisotropic-PQ (loss reweighting to align with query direction).
3. **The scan** — PQ-Fast-Scan (SIMD LUT), PQ-ADC-Search (asymmetric
   distance computation).

Nobody has touched the fourth lever: **how many bits each subspace gets.**
Standard PQ hands every subspace an identical `b`-bit budget. Real
embedding datasets have wildly non-uniform per-subspace variance (up to
4× on typical CLIP and sentence-encoder tails), so a uniform budget over-
serves the quiet subspaces and starves the noisy ones. The result is
wasted distortion (and therefore recall) at the same storage cost.

## Decision

Ship `crates/ruvector-elastic-pq` — a PQ implementation that keeps the
*total* bit budget across all subspaces fixed and lets an **allocator**
decide how to split it. Three allocators, one trait:

1. `Allocator::Uniform { bits }` — classical baseline.
2. `Allocator::VarianceProportional { total_bits, min_bits, max_bits }` —
   PCA-style prior: `bits[s] ∝ log₂(σ²[s])`, clamped and rounded so the
   integer budget matches exactly.
3. `Allocator::DistortionIterative { start_bits, min_bits, max_bits,
   max_swaps }` — the novel one. Start uniform, retrain, then greedily
   move one bit at a time from the lowest expected marginal cost to the
   highest expected marginal gain, actually retraining the two touched
   codebooks each swap and rejecting the swap if it doesn't lower total
   distortion.

Per-subspace bit width is clamped to `1..=8` so each code index still
fits in one `u8` — storage, in-memory layout, and the fast-scan LUT
plumbing (see `ruvector-pq-search`) drop in unchanged.

## Consequences

**Positive**

- Measured on 5 000 × 32 anisotropic synthetic data at a matched 32-bit
  budget:
  - Uniform 4-bit PQ: training distortion 4531.8, recall@10 0.302
  - VarianceProportional (32 bits): 4064.1 (-10.3 %), recall 0.346 (+14.6 %)
  - DistortionIterative (32 bits): 3867.1 (-14.7 %), recall 0.338 (+12.1 %)
- Byte-aligned storage means every downstream scan crate keeps working.
- Trait-based (`Quantizer`) so the allocator is a swap-in: future
  learned allocators or entropy-coded codes plug in without disturbing
  search-side code.
- Composable with OPQ (rotate first, allocate bits second) and with
  RaBitQ (hybrid: RaBitQ high-variance subspaces, EBA-PQ low-variance).

**Negative / risks**

- **Distortion is a proxy for recall, not recall itself.** The results
  make this visible: `VarianceProportional` slightly out-recalls
  `DistortionIterative` even though the elastic loop wins on training
  distortion. The elastic swap-step's objective needs to move to
  *held-out ADC recall* on a validation slice before we can promote
  the allocator to production (see "Improvements" in the research doc).
- Training cost is 2.3× uniform on the benchmark (81 ms vs 35 ms) —
  linear in `max_swaps`, and every swap is a real k-means on two
  subspaces. Not a concern for offline index builds, but rules out
  streaming re-quantisation without a cheaper swap surrogate.
- The information-theoretic saving (fractional bits) is left on the
  floor by byte-aligned storage. Fine for a research artifact — anyone
  chasing bytes-per-vector should either compose EBA-PQ with an entropy
  coder or accept the constraint.

## Alternatives

- **Entropy-coded PQ (Xu et al. 2025).** Realises the fractional-bit
  saving with ANS or Rice codes. Faster on the wire; slower on the scan
  loop (no random access); much more code. Rejected as scope-inflation
  for a nightly PoC.
- **Learn a rotation instead (OPQ).** Solves a different problem (axis
  alignment) and composes with EBA-PQ. Keep as a future compose step,
  don't substitute.
- **Locally-adaptive (LVQ / LeanVec).** Per-vector scaling adds one
  scalar per vector; EBA-PQ makes structural changes to the codebook
  layout at zero per-vector overhead. Complementary, not competitive.
- **Fixed 8-bit PQ.** The trivial upper bound — always fits — but
  doubles storage vs 4-bit for a recall gain that EBA-PQ can partially
  claim without the storage tax.

## Follow-ups (tracked in the research doc)

1. Move elastic swap objective to held-out recall.
2. OPQ ∘ EBA-PQ composition.
3. Tiny MLP allocator (subspace stats → bit vector).
4. Wire EBA-PQ into `crates/ruvector-pq-search` for SIMD-16 LUT scan.
5. RaBitQ ∪ EBA-PQ hybrid encoder.
