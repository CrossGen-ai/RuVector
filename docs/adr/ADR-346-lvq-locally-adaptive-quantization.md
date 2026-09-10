# ADR-346: Locally-adaptive Vector Quantization (LVQ) — Per-vector Adaptive Scalar Quantization for Compressed ANN

## Status

Accepted (experimental crate, `ruvector-lvq`, off any hot path by default).
Correctness-verified reference implementation without a SIMD kernel; recall
and memory results are architecture-neutral, latency picture is not yet
competitive on Apple silicon and is documented as a known limitation with a
clear next-step roadmap.

## Context

`ruvector` already ships several code-side compression schemes:
`ruvector-rabitq`, `ruvector-fused-rabitq-residual`,
`ruvector-pq-search`, `ruvector-cascade-adc`, `ruvector-turboquant`,
`ruvector-matryoshka`. None of them is a **per-vector adaptive scalar
quantizer** — the current gap in the compression menu.

The distinction matters:

- Product quantization (PQ, IVF-PQ, OPQ) partitions dimensions into
  subvectors and quantizes each subvector against a shared codebook.
  Recall is capped by the codebook size × subvector count and by how well
  a single codebook fits every vector.
- Global scalar quantization (INT8/INT4 with corpus-wide `lo`/`hi`)
  wastes precision whenever a single vector's active range is narrower
  than the corpus range. Modern embeddings exhibit exactly this pattern
  (per-vector active mass ≪ global range).
- **Per-vector** scalar quantization (LVQ) pays 8 bytes/vector for
  `(lo, scale)` and recovers most of the recall a global INT8 loses,
  because the code space is spent on the range this vector actually
  occupies.

Intel's Scalable Vector Search (SVS) demonstrated LVQ-8 and Turbo-LVQ
achieve 3-4× the QPS of FAISS-IVF-PQ at matched Recall@10 on DEEP-1B
and SIFT-1B (Aguerrebere et al., VLDB 2023). Competitor coverage as of
Sept 2026: Milvus 2.4+ ships `IVF_LVQ8` / `HNSW_LVQ8`; Qdrant, Weaviate,
LanceDB, FAISS remain on global scalar quantization. This ADR closes
the gap in the Rust ecosystem side.

## Hypothesis

```text
Given a synthetic corpus of N=10 000 vectors, dim=128, 32 Gaussian
clusters (sigma=0.1), and 100 seeded queries, three LVQ variants
(LVQ-8, LVQ-4, LVQ-4x8) implemented against a shared Quantizer trait
in `crates/ruvector-lvq`,

when Recall@10 (approx top-1 within exact top-10) and per-vector code
size are measured against an fp32 baseline,

then:
  (a) LVQ-8   achieves >= 0.99 Recall@10 at <= 30% of fp32 memory,
  (b) LVQ-4x8 achieves >= 0.99 Recall@10 at <= 45% of fp32 memory,
  (c) LVQ-4   achieves >= 0.85 Recall@10 at <= 20% of fp32 memory,
  (d) every asymmetric distance function agrees exactly with
      l2_sq(query, decode(code)) up to floating-point precision,
  (e) cargo test -p ruvector-lvq is 100% green.

Latency vs fp32 asymmetric L2 is reported as measured; it is not a
pass/fail gate for this ADR because the scalar kernel is a reference
implementation and the SIMD kernel is next-research.
```

## Decision

1. Add crate `crates/ruvector-lvq`, added to the top-level workspace
   `members` list. No dependency on any other `ruvector-*` crate; the
   two dependencies are `rand` and `rand_distr` (both workspace
   dependencies already).
2. Ship three variants — `Lvq8`, `Lvq4`, `Lvq4x8` — behind a common
   `Quantizer` trait, each with its own `Code` type. Trait-based
   design deliberately kept minimal to make later backends (SIMD,
   anisotropic, learned codebook) drop-in.
3. Ship a runnable `examples/bench.rs` producing the numbers cited in
   this ADR and the nightly README. No mocks; no aspirational
   figures.
4. Do **not** wire LVQ into any HNSW/IVF crate in this pass. HNSW
   integration is a separate PR whose payoff dominates only after the
   SIMD kernel lands.
5. Do not promote LVQ to a default. Off any hot path until (a) SIMD
   kernel and (b) large-scale (N >= 1 M) benchmark on a real embedding
   corpus land.

## Evidence

Measured on Apple M4 Max, arm64, rustc 1.89, `--release`, `seed=42`:

| Variant | Bytes/vec | Ratio  | Encode (ns) | Asym L2 (ns) | Recall@10 | Gate      |
|---------|-----------|--------|-------------|--------------|-----------|-----------|
| fp32    | 512       | 1.000x | —           | 35.1         | 1.0000    | (ref)     |
| LVQ-8   | 136       | 0.266x | 178.7       | 47.7         | 1.0000    | (a) PASS  |
| LVQ-4x8 | 208       | 0.406x | 547.1       | 99.2         | 1.0000    | (b) PASS  |
| LVQ-4   | 72        | 0.141x | 192.9       | 78.6         | 0.9500    | (c) PASS  |

Gates (d), (e) also PASS:

- (d) `Lvq{8,4,4x8}::asymmetric_matches_decoded_l2*` tests verify
  every variant's asymmetric distance matches `l2_sq(query, decode(code))`
  to `< 1e-5` relative error over 8 random pairs each.
- (e) 14/14 tests pass: `cargo test --release -p ruvector-lvq`.

Full raw output and hardware/methodology in the nightly README:
`docs/research/nightly/2026-09-10-lvq-locally-adaptive-quantization/README.md`.

## Consequences

- ruvector now has a per-vector adaptive scalar quantizer, closing the
  gap against Milvus's `IVF_LVQ8`/`HNSW_LVQ8`.
- LVQ-4x8 provides a natural two-stage re-rank code path (4-bit for
  candidate generation, 8-bit residual for refinement) that maps
  cleanly onto HNSW beam search with no changes to the graph layer.
- Recall/memory results are architecture-neutral. Latency vs fp32 on
  Apple silicon is worse than fp32 in this scalar reference; this is
  documented and the fix (SIMD kernel + Turbo-LVQ layout) is scoped
  as next-research.
- No existing crate is touched. Workspace's `cargo build --workspace`
  gains one new small crate (~450 LOC, 14 tests, no non-workspace
  deps).

## Alternatives Considered

- **Global INT8 scalar quantization.** Rejected: known recall gap vs
  LVQ; would duplicate machinery already used inside PQ codebook
  training without solving the per-vector-range problem.
- **Extend `ruvector-rabitq` with an INT8 residual.** Rejected for
  this pass: RaBitQ's error bound is tied to its randomized rotation
  and 1-bit code; adding a residual layer would change its theoretical
  guarantees and force us to re-verify the bound. LVQ's design is
  simpler and orthogonal.
- **Wire LVQ into `ruvector-hnsw-*` in this PR.** Rejected: HNSW
  integration is where the production payoff is, but only after the
  SIMD kernel lands. Doing it now would ship a slower HNSW path.
- **Skip LVQ-4.** Rejected: LVQ-4 alone is not the target for
  production use, but including it makes the "3 measured variants"
  requirement genuine and lets downstream users trade the extra 5pp
  recall for 2× additional memory savings on workloads that tolerate
  it.

## Implementation Plan

Already implemented in this PR:

- `crates/ruvector-lvq/Cargo.toml`
- `crates/ruvector-lvq/src/lib.rs` — trait + `l2_sq_f32` + `recall_at_k`
- `crates/ruvector-lvq/src/lvq8.rs` — 8-bit variant + 4 unit tests
- `crates/ruvector-lvq/src/lvq4.rs` — 4-bit variant + 4 unit tests
- `crates/ruvector-lvq/src/residual.rs` — 4x8 residual + 2 unit tests
- `crates/ruvector-lvq/examples/bench.rs`
- `crates/ruvector-lvq/tests/roundtrip.rs` — 3 integration tests
- `Cargo.toml` workspace membership entry

Every file under the 500-line project cap.

## API Shape

```rust
pub trait Quantizer {
    type Code;
    fn dim(&self) -> usize;
    fn encode(&self, v: &[f32]) -> Self::Code;
    fn decode(&self, c: &Self::Code) -> Vec<f32>;
    fn asymmetric_l2_sq(&self, q: &[f32], c: &Self::Code) -> f32;
    fn bytes_per_code(&self) -> usize;
}

pub struct Lvq8    { /* dim */ }
pub struct Lvq4    { /* dim */ }
pub struct Lvq4x8  { /* dim, inner Lvq4 + Lvq8 */ }

pub struct Lvq8Code   { pub lo: f32, pub scale: f32, pub codes: Vec<u8> }
pub struct Lvq4Code   { pub lo: f32, pub scale: f32, pub packed: Vec<u8>, pub dim: usize }
pub struct Lvq4x8Code { pub primary: Lvq4Code, pub residual: Lvq8Code }
```

## Feature Flags

None in this pass. Follow-up: `simd-neon`, `simd-avx512`, `serde`
(all off by default), scoped to the SIMD next-research PR.

## Benchmark Evidence

See "Evidence" above and the nightly README for full raw output,
methodology, and hardware.

## Security

No cryptographic primitives. No new attack surface: the quantizers are
pure functions over `&[f32]`; asymmetric distance is bounds-checked
against `dim` via `assert_eq!` and `debug_assert_eq!`.

## Governance

None. Off-hot-path experimental crate. Nothing in the workspace's
default build path calls into `ruvector-lvq`.

## Failure Modes

See the nightly README's "Practical failure modes" section for the
full account of the M4-Max latency inversion, the LVQ-4 recall loss on
tight clusters, the metadata-overhead-at-tiny-dim degenerate case, and
the fact that queries stay fp32.

## Migration

None: new crate, additive to workspace.

## Rollback

Remove the workspace membership line and delete `crates/ruvector-lvq/`
entirely. No existing crate depends on it.

## Rejection Criteria

Not rejected. Would be rejected if:

1. Recall@10 for LVQ-8 or LVQ-4x8 dropped below 0.99 on the seeded
   bench (measured 1.0000 for both).
2. `cargo test -p ruvector-lvq` failed (measured 14/14 green).
3. `cargo build --release -p ruvector-lvq` failed (measured green).

## Open Questions

1. How much of the M4-Max latency inversion is compiler-defeatable in
   the scalar loop (via explicit chunk-of-4 unrolling), and how much
   requires the NEON kernel? Next-research item 1.
2. Do the Recall@10 = 1.0000 figures for LVQ-8 and LVQ-4x8 hold on
   real embeddings (e.g. text-embedding-3-small@1536 dim) or is this
   an artifact of the Gaussian-cluster generator? Next-research item
   before HNSW integration.
3. Does Turbo-LVQ's interleaved layout (SVS §5.2) survive a NEON port
   or is it AVX-512-VBMI2-specific? Open, unmeasured.
