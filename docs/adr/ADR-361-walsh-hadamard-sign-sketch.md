# ADR-361: Walsh-Hadamard Sign Sketch (WHSS) prefilter for ANN

## Status

**Proposed.** 2026-10-09. PoC crate `ruvector-whss` landed with unit tests
and a reproducible benchmark binary; integration into HNSW / IVF
pipelines is scoped under "What to improve next" and deferred to a
follow-up ADR.

## Context

ruvector's sign-sketch prefilter path (used by `ruvector-bit-sliced-hamming`,
`ruvector-cascade-adc`, and the Hamming reranker in `ruvector-hubness-
rerank`) relies on classic Gaussian **sign-random-projection** (SRP /
SimHash): a dense m×d Gaussian matrix projects every corpus vector before
the sign step is applied. The costs are:

* **Build time** O(d·m) per vector — a measurable fraction of index
  construction at d ≥ 512.
* **Projector memory** m·d floats — 1 MB at d=512, m=512, resident per
  shard.
* No reuse of the structure in the transform; cache behaviour on wide d
  is poor.

A structured alternative — H·D with a Fast Walsh-Hadamard Transform —
has been known since FastFood (Le et al., 2013) for kernel approximation
and JL embeddings, but has not been wired up as the standalone sign-sketch
prefilter inside this project (nor, from survey of 2026 Milvus / Qdrant /
Weaviate releases, in competing systems; FALCONN uses FWHT inside its own
cross-polytope hash family, not as a drop-in SimHash replacement).

## Decision

Introduce **ruvector-whss**, a Rust crate that provides a trait-based
sign-sketch prefilter with two backends:

1. **`Whss`** — H·D random-sign + FWHT + bit-pack. O(d log d) build /
   query, O(d) projector memory, scalar reference implementation plus
   SIMD-ready layout.
2. **`Srp`** — classic Gaussian baseline. Kept as the canonical control
   and continues to be the default for small-d / high-m configurations
   where it provides higher recall ceilings.

The PoC crate is standalone (its Cargo.toml declares its own
`[workspace]`), so it does not interact with the main workspace's build
graph until a dedicated integration PR lands.

## Consequences

### Positive

* **36× faster index build** at d=512 (35.6 ms vs 1 301.1 ms per 10 000
  vectors on Apple M-series, scalar Rust).
* **512× smaller projector** (2 KB WHSS vs 1 MB SRP at d=512, m=512) —
  important for the ruvector-edge DO memory budget (14 MB/shard).
* **Comparable or better recall** at d ≥ 512: WHSS at m=512, cand=1024
  reaches recall@10 = 0.80 vs SRP's 0.68 on synthetic uniform data.
* **No `unsafe`, no external BLAS**, no SIMD-portability tax at the
  reference level.

### Negative

* **Rank cap.** WHSS sketch has effective rank ≤ d regardless of m. At
  d=128, recall saturates at ~0.74 past m=128; SRP keeps climbing to
  0.93 at m=512. This forces a backend choice: use WHSS when d ≥ 2m
  (common for 384-/768-/1536-d embeddings), fall back to SRP (or stacked
  HD-blocks) otherwise.
* **One more dependency surface** in the sign-sketch layer. The trait
  (planned in follow-up) must stay narrow or crates downstream will
  bleed into both branches.
* **Scalar FWHT is on the hot path** until SIMD intrinsics land; this is
  a known debt, not a design flaw.

### Neutral

* The PoC is deliberately standalone and does not alter workspace
  membership. Integration into `ruvector-bit-sliced-hamming` and the
  cascade-ADC prefilter requires a separate ADR because it changes a
  default backend.

## Alternatives considered

1. **Keep SRP as the only backend.** Rejected: 36× slower build at
   d=512 is a real cost on index rebuild paths, and the 1 MB projector
   does not fit comfortably alongside the 14 MB VectorShard cap in the
   edge DO (ADR-351).
2. **Switch to FALCONN-style cross-polytope LSH.** Rejected for now:
   wider implementation surface (hash tables + rotations + ranking),
   worse integration with existing bit-sliced Hamming scan, and no
   obvious wall-clock win over structured SimHash for recall@10.
3. **PQ-coded prefilter (ADC) only.** Already present via
   `ruvector-cascade-adc`. Complementary rather than a substitute: WHSS
   is a cheaper first stage with much smaller sketches.
4. **Stacked HD-blocks (HD₂HD₁) as the only backend.** Deferred: a single
   HD block already beats SRP for d ≥ 2m, and the stacked variant adds
   build cost that erases some of the structured advantage. Will revisit
   if production workloads need higher recall at d < m.

## Measurement

The PoC bench (`whss-bench`) runs reproducibly from
`crates/ruvector-whss/`:

```
cargo run --release --bin whss-bench
WHSS_D=512 WHSS_N=10000 cargo run --release --bin whss-bench
```

Results are captured in
`docs/research/nightly/2026-10-09-walsh-hadamard-sign-sketch/README.md`
("Results" section).

## References

* `crates/ruvector-whss/src/lib.rs`
* `crates/ruvector-whss/benches/bench.rs`
* `docs/research/nightly/2026-10-09-walsh-hadamard-sign-sketch/README.md`
* ADR-351 (edge DO memory budget — sets the 14 MB/shard cap WHSS fits
  under; SRP does not at d ≥ 1024, m ≥ 512).
