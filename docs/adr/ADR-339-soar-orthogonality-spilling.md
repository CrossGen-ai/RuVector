# ADR-339: SOAR (Spilling with Orthogonality-Amplified Residuals) for IVF partition assignment

- **Status**: Proposed (implementation landed as reference crate `ruvector-soar`; adoption in ivf-pq / SPANN deferred to a follow-up ADR)
- **Date**: 2026-08-23
- **Deciders**: Nightly research agent (sean@crossgen-ai.com)
- **Related**: ADR-241 (SPANN partition spilling; `crates/ruvector-spann`), the IVF-PQ ADC work under `crates/ruvector-ivfpq`, nightly research `docs/research/nightly/2026-06-24-spann-partition-spill/`
- **Tags**: ann, ivf, spilling, quantization, nightly-research, recall

## Context

IVF-family ANN indexes (SPANN, IVF-PQ, IVF-RaBitQ — all present in the
repository) partition the dataset into Voronoi cells at build time and probe
the top-`nprobe` cells at query time. Recall at low `nprobe` is bounded by
how often the true nearest neighbour sits in one of the probed cells.
Boundary vectors are the failure case: their nearest neighbour lands in an
adjacent cell that never gets probed.

The standard mitigation is **spilling**: at build time, copy each vector
into its top-2 (or top-k) nearest partitions. This is what
`crates/ruvector-spann` currently implements. But naive top-k spilling has
a problem — the second-nearest centroid often lies in nearly the same
direction as the first, so the redundant copy adds little new coverage.

Google's NeurIPS 2024 paper *"SOAR: Improved Indexing for Approximate
Nearest Neighbor Search"* (Sun, Simcha, Dopson, Guo, Kumar; arXiv:2404.00774)
proposes an alternative: pick the secondary centroid to minimise an
**orthogonality-amplified** loss that prefers partitions whose residual is
orthogonal to the primary residual. ScaNN adopted this in 2024. Milvus and
Qdrant have not.

We had never prototyped SOAR in this repo. Prior nightly work covered
naive SPANN spilling, IVF-PQ ADC, and RaBitQ but not the assignment-rule
axis specifically.

## Decision

Land a reference-quality Rust implementation of SOAR as
`crates/ruvector-soar`, exposing a `PartitionIndex` trait with three
concrete backends — `IvfTop1` (baseline), `IvfNaiveSpill` (top-k), and
`IvfSoar` (orthogonality-amplified). Ship a deterministic benchmark binary
that reports real recall@10 and query latency across an `nprobe` sweep so
that the numbers are captured verbatim in the nightly research doc, not
paraphrased.

The SOAR loss for choosing the secondary centroid `c'`, given primary
residual `r₁ = x − c₁`, is:

```
L_SOAR(c') = ||x − c'||²  +  λ · ⟨r₁, x − c'⟩² / ‖r₁‖²
```

with `λ = 1.5` as default (paper's recommendation).

Adoption in production crates (`ruvector-spann` and `ruvector-ivfpq`) is
**not** part of this ADR. That change is breaking to their public API and
deserves its own decision. This ADR is scoped to landing the trait, the
reference implementations, the benchmark, and the recall evidence.

## Consequences

**Positive**

- +1.77 pp recall@10 at nprobe=1 vs naive spill, for identical storage
  (bench, 8000×64 mixture — see nightly research doc).
- Perfect recall reached 2 probes earlier (nprobe=16 vs 32).
- `PartitionIndex` trait creates a shared seam for future assignment-rule
  experiments (multi-secondary, anisotropic-λ, learned assignment).
- Zero-dep crate — never blocks workspace builds.
- All tests pass with real numbers, no mocks.

**Neutral**

- Query-time latency parity with naive spill (assignment rule only affects
  build).
- Query API unchanged — anything that already uses a spill-style IVF can
  swap SOAR in.

**Negative / cost**

- Build time ~5.8× naive spill (111 ms vs 19 ms at N=8000, nlist=128). The
  extra work is `O(N · nlist · D)` for the secondary scan; parallelisable
  with rayon, and reducible with SOAR-over-shortlist.
- Adds one more IVF flavour to a codebase that already has three. Mitigated
  by making it a *reference* crate under `ruvector-soar` rather than a fork
  of existing backends.

## Alternatives

1. **Do nothing.** Keep naive top-2 spilling. Rejected — leaves ~1.7 pp
   recall on the table at the operating point where IVF matters.
2. **Adopt SOAR directly inside `ruvector-spann`.** Rejected as premature —
   SPANN's disk layout, posting-list format, and DiskANN-style graph
   companion make an in-place swap a breaking API change deserving its own
   ADR and benchmark. This ADR clears the way by proving the assignment
   rule works.
3. **Learned assignment (small MLP trained on residuals).** Ruled out on
   cost/benefit grounds — SOAR is closed-form, zero-training-data, and
   captures most of the gain. Learned assignment is on the roadmap
   (nightly-research future) but not in scope here.
4. **Anisotropic vector quantization (ScaNN's core loss)** instead of
   SOAR. Complementary, not competing — anisotropic VQ improves centroid
   *training* while SOAR improves *assignment*; they compose.
5. **More spilling (top-3, top-4)** with naive assignment. Rejected — same
   recall-per-byte curve; SOAR strictly dominates at the same storage
   budget in our sweeps.
