# ADR-361: Gray Code Scalar Quantization for Short-Code ANN Reranking

- **Status**: Proposed (nightly research, 2026-11-22)
- **Context branch**: `research/nightly/2026-11-22-gray-code-scalar-quantization`
- **Crate**: [`crates/ruvector-gcsq`](../../crates/ruvector-gcsq)
- **Research report**: [`docs/research/nightly/2026-11-22-gray-code-scalar-quantization/README.md`](../research/nightly/2026-11-22-gray-code-scalar-quantization/README.md)

## Context

ruvector already ships a wide family of short-code rerankers over packed
binary codes: SimHash, ITQ, polysemous PQ-Hamming, RaBitQ, and plain
scalar-quantization variants (LVQ, Lloyd-Max SQ, variance-adaptive SQ).
All of them optimise some combination of **encode cost**, **code size**,
and **score fidelity vs. L2**. None of them exploit a 70-year-old fact
from signal theory: *the reflected binary Gray code is the integer
labelling that minimises bit-flip distance between adjacent integers.*

A short-code reranker that scores by Hamming distance implicitly asks
the encoding to make bit-flip count ≈ metric distance. Plain-binary
scalar quantisation fails this: adjacent bins `3=011` and `4=100` are
three Hamming bits apart, injecting non-monotone noise into the
Hamming-vs-L2 relationship. Gray coding fixes this by construction.

Prior nightly work has explored every neighbouring point in the design
space (binary SQ, ITQ, SimHash, PQ-Hamming) but never flipped the one
cheap bit this proposal flips.

## Decision

Add `ruvector-gcsq`, a self-contained crate that:

1. Fits a per-dimension uniform scalar quantiser from the training
   corpus (`ScalarRange::fit`).
2. Encodes each vector's per-dim bin index via the reflected binary
   Gray code (`gray_encode(n) = n ^ (n >> 1)`).
3. Bitpacks the Gray codewords into a compact byte stream.
4. Scores pairs with `popcount(a XOR b)` over `u64` lanes.

The crate exposes a `Reranker` trait so GCSQ, Binary SQ, and SimHash
(and future RaBitQ/ITQ implementations) live behind one interface.
Index-level crates (`ruvector-ivf`, `ruvector-hnsw`) are not modified
on this branch — that integration is a follow-up when the trait
migrates to a shared façade crate.

### Numeric rationale

On gaussian `n=5000, d=128` with matched 32-byte codes:

- **GCSQ (2 bits/dim)**: Spearman ρ = **0.630**, recall@10 = **0.132**,
  encode 215 ns/vec.
- **Binary SQ (2 bits/dim)**: ρ = 0.324, recall@10 = 0.044.
- **SimHash (32 bytes)**: ρ = 0.462, recall@10 = 0.088, encode
  13230 ns/vec (~60× slower).

GCSQ is a strict Pareto win at the Hamming-reranker operating point.

## Consequences

### Positive

- Highest Hamming-vs-L2 Spearman correlation we've measured at matched
  bitrate on random gaussian data; expected to improve further on real
  embeddings where per-dim ranges vary.
- Encode is matmul-free and cache-friendly (~60× faster than SimHash).
- Score is one `XOR` + `popcnt` per `u64` lane — same as every other
  binary reranker; no new hot path.
- Trivially layers on top of any existing SQ-fitted quantiser (ITQ,
  Lloyd-Max, variance-adaptive) by Gray-encoding the bin index instead
  of bitpacking directly.

### Negative

- At 1 bit/dim GCSQ is identical to plain binary SQ — no wins for users
  who already use 1-bit codes. The crate makes this explicit.
- Hamming stops being a good L2 proxy for `bits_per_dim ≥ 5` — the
  Lipschitz bound widens beyond usefulness. Use Lloyd-Max + float
  reconstruct for SQ-8.
- Not metric: Hamming on Gray codes is only an ordering proxy. Not a
  problem for shortlist reranking but makes GCSQ unsuitable as the
  index-level scoring function for metric-tree ANN.

### Risks

- Real embedding distributions are heavy-tailed; naive `(min, max)`
  fitting wastes bins. Follow-up should percentile-fit (1/99) by
  default.
- The `Reranker` trait sketched here will need to live in a shared
  crate before `ruvector-ivf` can swap backends — introduces a small
  refactor debt.

## Alternatives considered

1. **Lloyd-Max SQ (ADR-nn)**: better L2 reconstruction, but scored on
   decoded floats, not Hamming. Different operating regime.
2. **ITQ-rotated binary** (`ruvector-itq-binary`): adds a learned
   rotation; higher recall at 1 bit, but encode is `O(dim²)` and does
   not stack with multi-bit SQ.
3. **Polysemous PQ-Hamming** (`ruvector-polysemous-pq-hamming-prefilter`,
   2026-11-21): learns a PQ atom permutation for Hamming-ADC
   correlation. Addresses the same objective but for PQ, not SQ, and
   requires k-means training.
4. **Do nothing**: keep binary SQ for the "cheap Hamming score over
   scalar codes" slot, accept ρ ≈ 0.32. Rejected — GCSQ roughly doubles
   it at no build-time or runtime cost.

## Follow-ups

- Pull GCSQ encode into `ruvector-sota-bench` for cross-method shootout.
- Prototype PCA-rotated GCSQ.
- Variance-weighted bit allocation.
- AVX-512 VPOPCNTDQ fast path for the scorer.
