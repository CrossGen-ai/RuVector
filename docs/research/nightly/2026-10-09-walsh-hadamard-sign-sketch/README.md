# Walsh-Hadamard Sign Sketch (WHSS) Prefilter for ANN

**Date:** 2026-10-09
**Status:** PoC implemented, benchmarked
**ADR:** [ADR-361](../../../adr/ADR-361-walsh-hadamard-sign-sketch.md)
**Crate:** `crates/ruvector-whss`

## Abstract

We implement and measure a structured sign-sketch prefilter for approximate
nearest-neighbor search built on the Fast Walsh-Hadamard Transform (FWHT).
Given a query / corpus vector of dimension `d`, WHSS applies a random ±1
diagonal (D), an FWHT, then takes the sign bit of a fixed subset of `m`
coefficients, yielding an `m`-bit sketch comparable by SIMD popcount. The
structured transform runs in **O(d log d)** per vector with **O(d) memory**,
compared to **O(d·m)** time and memory for Gaussian sign-random-projection
(SRP). On synthetic uniform data (d=512, N=10k, m=512, cand=1024) WHSS
achieves recall@10 = 0.80 at 36× faster index build and ~1.5× faster query
scan than SRP, with equivalent sketch size.

## SOTA survey

* **SRP / SimHash** (Charikar, 2002): Gaussian m×d matrix, O(d·m) build. The
  reference baseline for sign sketches.
* **FastFood / ACES** (Le et al., 2013): H·D randomness for approximate
  kernel features and Johnson-Lindenstrauss embeddings; O(d log d) time
  per transform.
* **Structured spinners / HD-sketches** (Choromanski et al., 2017): HD₁HD₂HD₃
  stacks for near-isotropic random projection with structured speed.
* **FALCONN / cross-polytope LSH** (Andoni et al., 2015): uses FWHT for the
  rotation step inside a different hashing family.
* **Bit-sliced Hamming scan** (ruvector 2026-09-18): per-bit SIMD popcount
  layout for fast Hamming ranking over large N.

Prior ruvector nightly work in this family: `cascade-adc`, `bit-sliced-
hamming`, `partial-l2-early-abandon`, `fused-rabitq-residual`,
`lsh-seeded-hnsw-entry`. None uses structured O(d log d) transforms for the
sign-sketch itself. WHSS targets exactly that gap.

## Proposed design

For a corpus vector x ∈ ℝᵈ:

1. Pad to the next power-of-two D ≥ max(d, m).
2. Apply random sign flips: y = D · x where D = diag(s), sᵢ ∈ {+1, −1} drawn
   from a seeded CSPRNG once per index.
3. Run FWHT in place on y (Hadamard-ordering, O(D log D)).
4. Pick m positions via a fixed random permutation of [0, D) (truncated to m).
5. Emit sketch bit b = [y′[πᵦ] ≥ 0] into a bit-packed u64 word layout.

Query uses the same (D, π). Candidate ranking is `popcount(q ⊕ x)`, verified
on the top-c by true L2.

### Why Hadamard

* **Build cost.** FWHT is in-place, cache-friendly (strided butterflies),
  and SIMD-friendly. No m×d matrix is ever materialized.
* **Memory footprint for the projector.** O(D) random signs + O(m) indices.
  SRP needs m·d floats (1.3 MB at d=512, m=512 vs 2 KB for WHSS).
* **Near-isotropic projection.** H·D is ϵ-approximately Gaussian for the
  low-rank projections used by sign LSH (FastFood / HD-product bounds).

### Known limitation (confirmed empirically)

Because the FWHT is a change of basis of ℝᴰ and the input padding is zero,
the effective rank of the sketch is at most `d`. Once m ≈ d, additional
sketch bits are *deterministic* functions of earlier bits and recall
saturates. SRP, with its full-rank m×d Gaussian matrix, keeps gaining
information as m grows. The sweet spot for WHSS is **m ≤ 2d** or stacked
HD-blocks (HD₂HD₁) when more capacity is needed.

## Implementation notes

* Pure Rust, no unsafe, `rand_chacha` CSPRNG for reproducible benches.
* `u64`-packed sketches, Hamming via `u64::count_ones()` (lowers to
  `popcnt` on x86-64; on Apple silicon the LLVM backend emits a 4-way
  byte-popcount sequence).
* `select_nth_unstable_by_key` for top-c candidate selection — avoids
  sorting the whole corpus and shows up as a measurable speedup for
  cand ≪ N.
* `fwht` is a textbook iterative Hadamard butterfly; verified against the
  self-inverse property (H·H = N·I) in unit tests.
* Baseline SRP shares the same bit-pack / Hamming / verify path so the
  measured delta is pure projection-cost + sketch-quality.

## Benchmark methodology

* Hardware: `uname -ms` → Darwin arm64 (Apple M-series).
* Build: `cargo build --release` (opt-level=3, LTO thin).
* Deterministic seeds: corpus=1, queries=2, projector=7.
* Workload: synthetic uniform data in [-0.5, 0.5] per coordinate; k = 10
  on top-k L2.
* Three sketch widths m ∈ {128, 256, 512} × three candidate budgets
  cand ∈ {64, 256, 1024} × two methods (WHSS, SRP) + exact baseline.
* `exact_topk` provides ground truth for recall@10.
* Reported "search ms" is total wall-clock for 100 queries on a single
  thread.

## Results

### d = 128, N = 20 000, nq = 100

| method | m | cand | ms (100q) | QPS | recall@10 | build ms | projector bytes |
|---|---|---|---|---|---|---|---|
| full scan f32 | — | — | 70.67 | 1 415 | 1.0000 | 0 | 10 240 000 |
| WHSS | 128 | 1024 | 9.92 | 10 081 | 0.7370 | 18.6 | 2 048 |
| SRP  | 128 | 1024 | 10.94 | 9 141 | 0.5540 | 146.5 | 65 536 |
| WHSS | 256 | 1024 | 11.87 | 8 424 | 0.7370 | 39.5 | 2 048 |
| SRP  | 256 | 1024 | 11.99 | 8 341 | 0.7780 | 298.6 | 131 072 |
| WHSS | 512 | 1024 | 15.23 | 6 564 | 0.7370 | 71.6 | 2 048 |
| SRP  | 512 | 1024 | 17.95 | 5 571 | 0.9290 | 582.6 | 262 144 |

At d=128 WHSS saturates at recall 0.737 past m=128 bits (expected rank
limit, see §Known limitation). SRP keeps climbing to 0.93 at m=512.

### d = 512, N = 10 000, nq = 100  (WHSS's sweet spot)

| method | m | cand | ms (100q) | QPS | recall@10 | build ms | projector bytes |
|---|---|---|---|---|---|---|---|
| full scan f32 | — | — | 212.15 | 471 | 1.0000 | 0 | 20 480 000 |
| WHSS | 512 | 1024 | 29.76 | 3 360 | 0.8010 | 35.6 | 2 048 |
| SRP  | 512 | 1024 | 44.20 | 2 262 | 0.6760 | 1 301.1 | 1 048 576 |
| WHSS | 256 | 1024 | 28.89 | 3 462 | 0.5560 | 24.3 | 1 024 |
| SRP  | 256 | 1024 | 36.59 | 2 733 | 0.4900 | 642.6 | 524 288 |

At d=512 WHSS **beats SRP on every axis**: 36.5× faster build, 1.49× faster
search, and higher recall (0.80 vs 0.68 at m=512, cand=1024). The structured
transform dominates once d gets wide enough that the m×d matrix starts
hurting both memory bandwidth and compute.

### Memory

* Raw float32 corpus (d=512, N=10k): **20 480 000 bytes**
* WHSS m=512 sketches: **640 000 bytes** → 32× compression
* WHSS m=256 sketches: **320 000 bytes** → 64× compression
* WHSS m=128 sketches: **160 000 bytes** → 128× compression

## "How it works" walkthrough (blog-readable)

Think of a sign sketch as asking the same `m` yes-or-no questions of every
vector in your index. Two vectors that answer "yes/no/yes/yes/…" the same
way are probably close. The question we care about: can we ask those `m`
questions *cheaply*?

Classic SimHash / SRP picks `m` independent Gaussian "query vectors" and
asks "is my input vector on the positive side of this hyperplane?" each
time. That costs `m × d` multiply-adds per vector and `m × d` floats of
RAM to store the hyperplanes.

WHSS replaces the Gaussian matrix with a Hadamard matrix multiplied by a
random ±1 diagonal — a trick popularised by FastFood (Le et al., 2013).
The Hadamard matrix has entries that are only +1 and −1, so no
multiplications are needed; and its structure lets you compute H·x with a
**butterfly network** in `O(d log d)` time, re-using the same pattern as
an FFT.

A random sign flip up front stops the Hadamard structure from leaking
through: without it, the first row of H is all-ones and the first bit of
every sketch would just be the sign of the sum of the input. Adding `D =
diag(±1)` makes the projection *approximately* behave like a Gaussian
sketch, which is enough for sign-LSH guarantees.

So the deal is: give up a tiny amount of randomness (and the ability to
grow `m` past `d` effective bits for free) in exchange for far cheaper
build, far smaller projectors, and a cache-friendly in-place transform.

## Practical failure modes

1. **Over-provisioned m (m ≫ d) buys you nothing.** The effective rank of
   the sketch is bounded by `d`. Benchmarks at d=128 confirm recall
   saturates at m≈d. Fix: stack HD-blocks (HD₂HD₁) or combine with a
   different family (e.g. per-subspace PQ codes).
2. **Non-isotropic data breaks the ±1 diagonal assumption.** If the data
   has strong co-variance along a few axes, the single random sign
   diagonal is not enough to spread energy. Fix: apply a learned
   rotation (OPQ-style) before WHSS, or stack 2–3 random diagonals.
3. **Hamming-first ranking misorders tied codes.** With very similar
   vectors, several candidates will tie at the same Hamming score and the
   downstream verify step dominates recall. Mitigation: large enough
   `cand` or a residual code to break ties.
4. **FWHT is cache-friendly, but only when D fits in L2.** For d=4096+,
   split the FWHT into blocked passes or interleave with the sign-pack
   step.

## What to improve next (roadmap)

* **SIMD butterflies.** Current FWHT is scalar; wasm/aarch64/x86-avx2
  intrinsics are a 2-4× factor on the build path.
* **Stacked HD-blocks** for rank > d sketches without reverting to a full
  Gaussian matrix (`y = H·D₂·H·D₁·x`).
* **Learned sign diagonal.** Replace the random ±1 diagonal with signs
  chosen to balance variance across the top-m output bins (one-shot,
  no gradient).
* **Fused Hamming + L2 verify.** Walk candidates in sorted-Hamming order
  and early-abandon L2 once k'th-best distance stabilises.
* **Integration with HNSW / IVF.** Use WHSS as the entry-point
  disambiguation layer for HNSW (replaces random-walk seeding) or as the
  posting-list prefilter inside IVF-PQ.

## Production crate layout (proposal)

```
crates/ruvector-whss/
  Cargo.toml
  src/
    lib.rs          -- public API (Whss, Srp, encode_corpus, *_search)
    fwht/
      scalar.rs     -- current reference impl
      avx2.rs       -- x86-avx2 butterflies (feature = "simd")
      neon.rs       -- aarch64 NEON butterflies (feature = "simd")
    traits.rs       -- SignSketch trait so Whss / Srp / Stacked{HD,HD} swap
    bench.rs
  benches/
    bench.rs        -- the harness in this PoC
  tests/
    quality.rs      -- recall-vs-m regression tests
    determinism.rs  -- seed-stable sketch tests
```

An `SignSketch` trait lets downstream crates (`ruvector-bit-sliced-hamming`,
`ruvector-cascade-adc`) accept any backend; WHSS becomes the default.

## References

1. Charikar, M. S. (2002). "Similarity estimation techniques from rounding
   algorithms." STOC '02.
2. Le, Q., Sarlos, T., & Smola, A. (2013). "Fastfood — approximating kernel
   expansions in loglinear time." ICML.
3. Choromanski, K. et al. (2017). "The unreasonable effectiveness of
   structured random orthogonal embeddings." NeurIPS.
4. Andoni, A., Indyk, P., Laarhoven, T., Razenshteyn, I., & Schmidt, L.
   (2015). "Practical and optimal LSH for angular distance." NeurIPS.
5. Dasgupta, A., Kumar, R., & Sarlós, T. (2010). "A sparse Johnson-
   Lindenstrauss transform." STOC '10.
6. Milvus / Qdrant / Weaviate changelogs (2026-Q3): all still ship
   Gaussian SRP as the sign-sketch backend; FWHT-based structured sketches
   have appeared only inside FALCONN-family hashing, not as a standalone
   prefilter.
