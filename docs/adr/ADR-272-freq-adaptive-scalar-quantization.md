# ADR-272 — Frequency-Adaptive Scalar Quantization (FASQ)

* **Status**: Proposed (nightly research, 2026-07-01)
* **Deciders**: nightly-research
* **Related**: ADR-264 (PQ-ADC), RaBitQ crate, LVQ crate

## Context

ruvector already ships several vector-compression crates (`ruvector-rabitq`,
`ruvector-lvq`, `ruvector-avq`, `ruvector-opq`, `ruvector-pq-search`,
`ruvector-matryoshka`). None of them address the specific case of **flat
scalar quantization under an integer per-dim bit budget**. In practice this
matters because:

1. Uniform SQ8 is the default fall-back in Milvus, Qdrant, and Weaviate. It
   is cheap, deterministic, and shippable — but wastes bits on low-variance
   dimensions.
2. Uniform SQ4 halves storage but tanks recall (measured: 0.708 vs 0.976 on
   an anisotropic corpus at dim=64).
3. Most production embedding distributions have highly anisotropic
   per-dimension variance — a small number of principal dims carry most of
   the signal.

A per-dimension bit allocator that spends more bits on higher-variance dims
under a fixed total budget is a well-known information-theoretic result
(Cover & Thomas ch. 13) but has not been packaged as a first-class ruvector
backend.

## Decision

Introduce a new crate `ruvector-fasq` that provides:

1. A shared `Quantizer` trait (`encode`, `decode`, `distance_sq`,
   `bits_per_vector`) implemented by `UniformSq8`, `UniformSq4`, and `Fasq`.
2. A discrete water-filling `allocator::allocate(σ², B_avg, b_lo, b_hi)`
   that returns integer bit counts per dim.
3. Per-dim uniform scalar quantizer with MSB-first bit packing.
4. An end-to-end demo (`fasq-demo`) and Criterion benchmarks.

FASQ ships as a standalone crate to keep it decoupled from HNSW/IVF backends
and to let existing crates (`ruvector-lsm-ann`, `ruvector-spann`) pick it
up as a `Box<dyn Quantizer>` when ready.

## Consequences

**Positive**

- **Same storage as SQ4, near-SQ8 recall** on anisotropic inputs
  (0.9745 vs 0.9765 at 32 bytes/vec, measured on 4 000×d=64).
- **28.72× lower MSE** than uniform SQ4 at the same 4 bits/dim average.
- **Graceful degradation**: on isotropic inputs FASQ matches SQ4 exactly
  — no regression risk when the input distribution shifts.
- **Composable**: the `Quantizer` trait lets HNSW / IVF crates hold a
  `Box<dyn Quantizer>` and pick a backend at runtime.
- **Simple**: no rotation, no learned codebook, no query-time state.

**Negative**

- Distance path currently reconstructs full f32 vectors — slower than
  ADC/LUT-based PQ at scan time (220 µs vs 56 µs for SQ8 direct decode
  at d=128, 1 000 vectors). SIMD kernels will close this gap.
- Requires training data at index-build time to estimate variances.
- On distribution drift the allocator becomes suboptimal — needs periodic
  recalibration.

**Neutral**

- Files remain under 500 lines each per project conventions.
- No new workspace dependencies (uses existing `rand`, `rand_distr`,
  `thiserror`, `serde`, `criterion`).

## Alternatives considered

1. **Ship as feature-flag on `ruvector-lvq`.** Rejected — LVQ is a
   *learned* per-dim quantizer with Lloyd-Max levels; FASQ is a
   *allocation* layer on top of a uniform quantizer. Different concern.
2. **Bake FASQ into `ruvector-rabitq`.** Rejected — RaBitQ is rotation
   + 1-bit; combining loses the didactic value and complicates the API.
3. **Do nothing; keep SQ8/SQ4 baselines.** Rejected — measurements show
   a 28.72× MSE improvement is on the table for anisotropic data at
   zero additional storage cost.
4. **Rotation-first (à la RaBitQ) + FASQ.** Deferred — meaningful as a
   follow-up but doubles the moving parts for a first PoC.

## Validation

- `cargo build --release -p ruvector-fasq` succeeds.
- `cargo test -p ruvector-fasq --release` — 10 tests, all pass.
- `cargo run --release --bin fasq-demo` — reproducible numbers logged
  in `docs/research/nightly/2026-07-01-freq-adaptive-sq/README.md`.
- `cargo bench -p ruvector-fasq` — encode & distance micro-benchmarks.
