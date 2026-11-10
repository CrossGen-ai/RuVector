# ADR-357: Permutation-Prefix Reranker (PPR) as a lightweight ANN rerank sidecar

- **Status**: Proposed
- **Date**: 2026-11-10
- **Deciders**: ruvector research (nightly)
- **Related**: ADR-302 (streaming QNG), prior PQ / RaBitQ / SQ cascade nightlies
- **Tags**: ann, rerank, quantization, permutation, pivot-based-indexing

## Context

ruvector already ships many rerank backends: Product Quantization ADC,
RaBitQ, Scalar-Quantized cascades, bit-sliced Hamming, learned coarse
rerank. All of them assume floating-point arithmetic is cheap and that a
small codeword can be decoded into an approximate distance.

That assumption breaks in three places:

1. **No-FPU targets** (embedded Hailo, some microcontrollers,
   WASM-no-SIMD): every float multiply is expensive or unavailable.
2. **Progressive / anytime rerank**: current rerankers produce one scalar
   per vector and cannot refine it incrementally when the latency budget
   stretches.
3. **Rerank sidecar for exotic first-stage indexes** (hyperbolic HNSW,
   graph-transformer traversal, SPANN postings) where the first-stage
   distance is non-Euclidean and PQ's L2-centric ADC is a mismatch.

A thirty-year-old idea solves all three at once: **permutation-based
indexing** (Chavez, Figueroa, Navarro, *"Effective Proximity Retrieval
by Ordering Permutations"*, TPAMI 2008). It stores, for each vector, the
*ordered* ids of its top-m nearest reference points ("pivots"). Query
rerank is the Spearman-footrule distance between two short permutations
— a handful of integer additions. Recall order correlates with true
distance under very weak assumptions (ranks are more stable than exact
distances under high-dimensional noise).

The idea has been neglected in the modern ANN literature, which has
chased ever-shorter PQ codes and SIMD ADC kernels. No crate in ruvector
implements it. This ADR proposes `ruvector-permutation-prefix` as a
complementary reranker — not a replacement for PQ, but a drop-in for the
three gap cases above.

## Decision

Introduce crate `crates/ruvector-permutation-prefix` with the public
`Reranker` trait:

```rust
pub trait Reranker {
    fn name(&self) -> &'static str;
    fn bytes_per_vector(&self) -> usize;
    fn rerank(&self, query: &[f32], candidates: &[u32], k: usize) -> Vec<u32>;
}
```

Implementations:

- `l2::L2Reranker` — ground-truth full-precision L2, 4·dim bytes/vec.
- `pq::PqReranker` — classical PQ-ADC with Lloyd codebook training,
  `n_sub` bytes/vec at `k ≤ 256`.
- `ppr::PermutationPrefixReranker` — stores per vector the sorted ids
  of its top-`prefix_m` pivots out of `n_pivots` total pivots chosen by
  farthest-point sampling. Rerank cost per candidate is one pass over
  `prefix_m` u16 ids, accumulating an integer footrule score against a
  query rank map built once per query.

Default recommended settings:

- `n_pivots = 128`, `prefix_m = 8` → **16 B/vec**, all integer, FP-free
  rerank (one float sq-L2 per pivot at query time only).
- `n_pivots = 256`, `prefix_m = 16` → **32 B/vec**, recall ceiling
  roughly doubles vs tiny config.

Farthest-point sampling (FPS) is chosen over k-means for pivot selection
because: (a) it has no training iterations, so build time is linear in
`n_pivots × n`; (b) it inherently maximises pivot diversity, which is
the property the footrule distance depends on; (c) it is deterministic
given a seed.

## Consequences

**Positive**

- A new rerank backend that works with zero floating-point multiplies
  in the hot loop, suitable for Hailo, no-SIMD WASM, and any no-FPU
  target. Build time still needs f32 to pick pivots and encode, but
  the query path is additions over u16 and i32 only.
- First reranker in ruvector that is natively *progressive* — stopping
  after reading only the first `p < prefix_m` pivot ids still yields a
  valid-but-coarser ranking. Future ADRs can add anytime/early-abandon
  wrappers.
- Independent of the first-stage index metric. Works unchanged for
  cosine, inner-product, hyperbolic, Manhattan — pivots just change.
- Tiny, auditable crate (~600 LOC under 500-per-file limit) with real
  tests and a benchmark that ships cargo-run numbers.

**Negative**

- At equal byte budget PPR's recall@10 trails PQ-ADC by roughly half on
  dense Gaussian-mixture data (see Benchmark). Workloads that *can*
  afford FP ADC and whose metric is L2 should keep PQ.
- Build is `O(n_pivots × n × dim)` — for `n=10⁶, dim=768, n_pivots=256`
  this is ~200 GFLOPs, comparable to k-means PQ training; not a
  deal-breaker but not free.
- The footrule distance is a coarse Kendall-tau surrogate; it ties for
  any two candidates with identical prefix membership but different
  internal order *inside* the prefix. Mitigated by increasing
  `prefix_m`, but at linear cost.

## Alternatives

- **Pure PQ-ADC** — status quo. Rejected only for the three gap cases;
  stays the default for general L2 ANN.
- **Hamming over LSH signatures** — covered by `ruvector-bit-sliced-hamming`
  and the `hamming-weight-prebucket` nightly. Also integer-only but
  depends on a locality-sensitive hash; permutation prefixes work on
  any metric a pivot distance function is defined for.
- **Cosine-signature rerank** (binary sign bits of random projections) —
  simpler than PPR but gives one bit per projection; no progressive
  refinement and identical ties on all vectors in the same orthant.
- **Learned rerank** (small MLP on first-stage features) — much higher
  recall ceiling but non-portable, requires training pipeline, and is
  not FP-free.

## Rollout

- Crate lives behind no feature flag; it is a standalone library.
- The `Reranker` trait name is generic enough to be hoisted to a shared
  crate in a follow-up ADR if other rerankers adopt it.
- Benchmark binary (`cargo run -p ruvector-permutation-prefix --bin benchmark`)
  prints a canonical table; numbers in the research README are from a
  release build on the author's machine, not aspirational.
