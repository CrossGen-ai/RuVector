# ADR-353: Extended RaBitQ (eRaBitQ) multi-bit vector quantization

- Status: Accepted
- Date: 2026-10-11
- Supersedes: —
- Related: ADR-343 (RaBitQ 1-bit), nightly `fused-rabitq-residual`,
  nightly `anisotropic-pq`
- Crate: `crates/ruvector-extended-rabitq`

## Context

RaBitQ gives 1-bit codes with a tight 2σ error bound of
`O(1/√D)` on inner products, which is state-of-the-art for storage.
Its weakness is that **the bit budget is fixed at 1**: workloads
that can spare 2–8 bits per coordinate for a step-change in recall
(embedding search over long-tail content, cross-encoder reranking
candidates, agent-memory retrieval where precision matters more than
footprint) have had to fall back to PQ / OPQ, which train a per-dim
codebook on data and have no closed-form variance guarantee.

We want a quantizer with:
1. A single bit knob (B ∈ {1,2,3,4,6,8}).
2. A per-coordinate error bound that scales as `σ²_B / D`.
3. Zero training data: codebook derived once from the standard
   normal.
4. Pure Rust, no `unsafe`, composable with existing ruvector indexes
   (IVF, HNSW, cascade reranking).

## Decision

Add `crates/ruvector-extended-rabitq` implementing **Extended RaBitQ
(eRaBitQ)**:

- Rotate each input with a sign-flipped Fast Walsh-Hadamard transform
  (`FastRotation`).
- Normalise the rotated vector and rescale by √D so each coordinate
  is a standard-normal sample.
- Encode each coordinate with a **Lloyd-Max optimal B-level codebook
  on N(0, 1)**, computed once at startup over a 20 001-point grid.
- Store packed B-bit codes plus the pre-rotation L2 norm.
- Score with an **asymmetric estimator**: rotate the query (no
  quantization), accumulate `q_rot_i · levels[code_i]`, scale by
  `‖x‖ / √D`. Convert IP to L2² when needed via polarisation.

The variance bound
`Var(⟨q,x⟩_hat − ⟨q,x⟩) ≤ ‖q‖² · ‖x‖² · σ²_B / D` is exposed
through `VarianceBound::ip_variance_bound` so callers can skip
reranking any candidate whose upper bound is worse than the current
k-th best.

## Consequences

Positive:
- Tunable bits/recall knob with closed-form error bound.
- One codebook for all datasets — no training step, no drift.
- Composes with existing ruvector pieces: IVF postings can store
  eRaBitQ codes; `dyn-rerank-budget` can mix 2/4/8-bit tiers.
- Measured: 95.3 % recall@10 at 5.1× compression (6-bit, D=128);
  98 % recall at 3.9× compression (8-bit); scan throughput ~5.5 Mvec/s.
- Pure-Rust implementation, 7 files, all under 300 LOC.

Negative:
- Padded dim: FWHT needs power-of-two length. D just above a power of
  two wastes code bytes on zero coordinates (preserves the bound).
- Encode throughput at B=8 is 5× slower than B=1 because
  nearest-level search is a linear scan over 256 levels. A balanced
  BST or branchless binary search is a follow-up.
- The Gaussian-marginal assumption weakens for heavy-tailed input
  distributions; recall still works but σ²_B is a looser bound.

## Alternatives

- **Stay with 1-bit RaBitQ**. Rejected: no bits/recall knob.
- **PQ / OPQ**. Rejected: data-trained codebooks have no closed-form
  variance bound and need re-training as the base distribution drifts.
- **LVQ-style per-vector scalar quantization**. Rejected: needs per-
  vector constants, more memory for the metadata, no rotation step so
  tails are harder to bound.
- **Fused RaBitQ + residual (float Δ)**. Rejected for primary codes
  (eats the compression gain) but useful as a reranking tier on top
  of eRaBitQ codes — captured in follow-up roadmap.

## Verification

- `cargo build --release -p ruvector-extended-rabitq` → green.
- `cargo test -p ruvector-extended-rabitq --release` → 9 passed.
- `cargo run -p ruvector-extended-rabitq --release --bin erabitq-demo`
  reproduces the recall / compression table.
- `cargo bench -p ruvector-extended-rabitq --bench erabitq_bench`
  reproduces scan throughput.

Repro commit: see branch `research/nightly/2026-10-11-extended-rabitq-multibit`.
