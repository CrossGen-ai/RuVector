# ADR-346: Partial L2 with Early Abandonment as a First-Class Rerank Kernel

## Status

Accepted (experimental crate `ruvector-partial-l2`, opt-in from callers).
Companion nightly research:
`docs/research/nightly/2026-09-20-partial-l2-early-abandon/`.

## Context

The rerank stage of every ANN pipeline in this workspace
(`ruvector-pq-search`, `ruvector-rabitq`, `ruvector-hybrid`,
`ruvector-adaptive-ann`, `ruvector-diskann`, `ruvector-hnsw-repair`)
ends the same way: score a candidate set of size `N_cand` (typically
`10⋅k` to `100⋅k`) against a query with full-precision L2, keep the
top `k`. Every such crate today does this with a fused SIMD full-L2
loop, no early exit. On synthetic uniform data with `dim=768`,
`k=10`, `N_cand=100 000`, this costs ~230 ns per candidate on Apple
M4 Max — around 76 M FMAs per query.

The classic LAESA optimization (Vidal 1986, Micó 1994) — abandon the
per-vector distance the moment a partial sum exceeds the current
worst-in-top-k `τ` — is textbook. It appears in FAISS's `IndexRefine`
path and ScaNN's rerank stage. It is *not* present in `hnsw_rs`,
`instant-distance`, `simsimd`, or any of ruvector's own crates.

The common objection ("branch overhead kills SIMD") is only true
conditionally. Under a const-generic block stride, the compiler
unrolls each block, keeps four independent FP accumulators live in
vector registers, and inserts the abandonment check between blocks
where the reduction has to happen anyway. Under a *runtime* stride,
the objection holds — bounds checks and non-unrolled inner loops
dominate.

## Decision

Ship `crates/ruvector-partial-l2` as an experimental, opt-in kernel
crate that exposes:

* `l2_full(a, b) -> f32` — reference full L2.
* `l2_early_abandon::<STRIDE>(a, b, τ²) -> AbandonResult` — const-generic
  block early-abandon kernel. **Primary API.**
* `l2_early_abandon_dyn(a, b, stride, τ²) -> AbandonResult` — runtime
  variant. Provided for completeness, documented as slower.
* `TopK` — small max-heap with `threshold()` accessor returning `+∞`
  before `k` items and the current worst distance thereafter.
* `rerank_full` and `rerank_early_abandon::<STRIDE>` — end-to-end
  helpers that wrap the kernels + heap.

The crate takes no dependency beyond `rand` (dev-only for
determinism). It has no `unsafe`, no `std::simd`, no target-specific
intrinsics — autovectorization is driven purely by four-way accumulator
unrolling.

Downstream crates opt in by adding a path-dep on
`ruvector-partial-l2` and replacing their inner full-L2 loop with a
`rerank_early_abandon::<32>` call. `k=32` is the recommended stride
based on the M4 benchmark; each crate should re-measure on its target
hardware before promoting.

The crate joins the workspace at
`crates/ruvector-partial-l2`; it is added to `members`, not to
`exclude`. `cargo test -p ruvector-partial-l2` runs 6 unit tests
including a recall-parity check across 4 strides and 500 vectors.

## Consequences

**Positive**

* Rerank paths that adopt the const-generic form get a **1.32×**
  wall-clock speedup at **100 % recall parity** on synthetic uniform
  `dim=768` data (measured; see companion research doc).
* The optimization composes: it can sit downstream of PQ-ADC
  (`ruvector-pq-search`), cascade-ADC (`ruvector-cascade-adc`), or a
  full-precision rerank without touching the prefilter.
* Recall parity is exact by construction, so there is no accuracy
  regression to police. Tests enforce this.
* The public API is 5 functions plus one `enum`; the crate is under
  500 lines and easy to review.

**Negative / risks**

* On small candidate sets (`N_cand < ~10⋅k`) or highly clustered data
  the abandonment rate is too low to overcome the check overhead;
  benchmarks on real data (SIFT / GIST) are still to come.
* Two callsite forms (const-generic vs runtime) risk API confusion.
  Documented, but a lint could catch runtime-stride usage in hot paths.
* Speedup is dominated by data distribution and by how tightly the
  first few candidates set `τ`. Adversarial candidate orderings can
  produce ~1.0× wins.
* No AVX-512 gather-scatter batch-of-4 backend yet; on x86_64 servers
  the win may be smaller than on M4 Max.

**Neutral**

* The crate leaves the workspace's existing rerank paths untouched.
  Adoption is per-crate, gated on each crate's own benchmark
  confirming a win on its target hardware.

## Alternatives Considered

1. **Do nothing.** The `simsimd`-backed full-L2 loops are already fast
   and every candidate crate could argue "our data is not uniform".
   Rejected because the measured 1.32× is real, is free at the callsite,
   and the optimization is a well-known omission.
2. **Add a `simsimd`-hosted variant.** `simsimd` is an upstream C
   dependency and does not currently expose an abandonment-aware L2.
   Landing this in `simsimd` upstream is possible but slow; a small
   pure-Rust crate ships now and can be replaced later.
3. **Portable-SIMD from day one.** `std::simd` is unstable; we would
   have to gate the whole crate on nightly Rust. Deferred to a future
   `--features portable-simd` toggle once the API stabilizes.
4. **Bake the kernel into `ruvector-core`.** Would force every consumer
   to rebuild `core` for a change here. Kept out-of-tree so the kernel
   can evolve independently.
5. **Batch-4 vectorized-across-candidates form.** Process 4 candidates
   per SIMD lane with per-lane abandonment masks. More complex, needs
   architecture-specific intrinsics, and only pays off when
   `abandonment_rate` is moderate (else lanes desynchronize). Deferred.

## References

* Companion research:
  `docs/research/nightly/2026-09-20-partial-l2-early-abandon/README.md`.
* Prior nightly on rerank / kernel work:
  `docs/research/nightly/2026-09-05-cascade-adc-scan/`,
  `docs/research/nightly/2026-09-14-pdx-vertical/`.
* FAISS `IndexRefine`: <https://github.com/facebookresearch/faiss/blob/main/faiss/IndexRefine.h>.
* ScaNN rerank: Guo et al. 2020, <https://arxiv.org/abs/1908.10396>.
* Vidal, E. (1986). LAESA.
* Micó, L. et al. (1994). AESA revision.
