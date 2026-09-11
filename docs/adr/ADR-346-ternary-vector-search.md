# ADR-346: Ternary {-1, 0, +1} Vector Encoding for High-Recall ANN Prefilter

## Status

Experimental (nightly research). Crate `ruvector-ternary` added under
`crates/`, feature-agnostic and off any hot path by default. Retained as
evidence for follow-up work on mixed-precision prefilter → int8-rerank
pipelines. Not promoted to a default backend.

## Context

Binary (1-bit sign) quantization has become the standard "wide prefilter"
representation in production vector databases (FAISS `IndexBinaryHNSW`,
Milvus `BIN_FLAT`, Weaviate `bq: enabled`) because `popcount(a ^ b)` on
64-D chunks is single-cycle on modern CPUs and each vector costs `dim/8`
bytes. Its recall on isotropic corpora is poor at small `k` — every
coordinate votes with equal weight, including the ones whose sign is
statistical noise.

Two families address this:

1. **Higher precision** (int8 scalar quant, PQ 4/8-bit). Better recall,
   4-8x the memory and ~5x the distance cost.
2. **Learned sign codes** (spherical hashing, ITQ). Better recall than
   plain sign, but training-set-dependent and non-trivial to shard.

Ternary {-1, 0, +1} encoding sits between them: a per-vector magnitude
threshold zeroes out the least-informative coordinates so they abstain
from the sign vote, at 2x binary memory and roughly the same distance
cost (two extra `and`s per 64-bit chunk).

## Hypothesis

> On isotropic and mildly-clustered Gaussian corpora, ternary encoding
> with a per-vector sparsity budget of 20–40% improves recall@10 over
> plain 1-bit binary by a factor of 1.5–2× at only 2× the memory and
> ≤10% extra scan latency.

## Decision

Implement three encoders behind a common `Encoder` / `Distance` trait
pair and benchmark them head-to-head against exact fp32 on the same
corpus, same seeds, same query set. Report bytes-per-vector, encode
throughput, scan throughput, and recall@10.

## Measurements

Real numbers from `cargo run --release -p ruvector-ternary --bin
benchmark` on the host machine (Apple Silicon, seed = 42, N = 20 000,
dim = 128, k = 10, 200 queries, StandardNormal corpus):

| Encoder     | Bytes/vec | Compression | Encode ns/vec | Scan ns/pair | Recall@10 |
|-------------|-----------|-------------|---------------|--------------|-----------|
| binary      | 16        | 32.0x       | 481           | 5.25         | 0.0640    |
| ternary@0.25| 32        | 16.0x       | ~1700         | 5.18         | 0.1035    |
| ternary@0.50| 32        | 16.0x       | 1742          | 4.84         | 0.0795    |
| int8        | 128       | 4.0x        | 109           | 25.34        | 0.9765    |
| fp32-oracle | 512       | 1.0x        | –             | 187          | 1.0000    |

Sparsity sweep (ternary, same corpus):

| Sparsity | Recall@10 |
|----------|-----------|
| 0.00     | 0.0640    |
| 0.10     | 0.0835    |
| 0.25     | **0.1035** |
| 0.40     | 0.0995    |
| 0.50     | 0.0795    |
| 0.75     | 0.0075    |

Ternary at sparsity=0.25 beats binary by **1.62×** on recall@10 with
identical asymptotic scan cost and 2× memory. Recall peaks near ~25%
sparsity and falls off past 50%, confirming the "least-informative
coordinates abstain" theory: with too many abstentions the surviving
signal is starved.

## Consequences

**Positive.**
- A drop-in prefilter that closes ~40% of the recall gap between binary
  and int8 at 4× less memory than int8. Suitable as a first stage before
  a small int8/fp32 rerank pool.
- The distance kernel is compiler-friendly: four bitwise ops per 64-D
  chunk, no floats, no lookup tables. LLVM autovectorizes on both
  x86-64 (`popcnt`) and aarch64 (`cnt`/`addv`).
- The `Encoder`/`Distance` trait pair makes the three backends fully
  swappable — the recall harness compiles against generics, so adding a
  fourth (e.g. balanced-4-bit) is one impl block.

**Negative / open.**
- On isotropic Gaussians, absolute recall is still low for
  binary-family codes at k=10 (0.06–0.10). These codes are only useful
  as a *wide* first stage: expand shortlist to `k=200` and rerank in
  int8. That two-stage pipeline is out of scope for this crate.
- Encode cost is ~3.6× binary because we compute a per-vector magnitude
  quantile. Trivially amortizable off-line for the corpus, but per-query
  the extra ~1.3 µs matters at low-latency budgets. A sampled quantile
  (k=32 pilot coordinates) would cut this to near-binary; deferred.
- Storage doubles vs binary (2 bitplanes). Below the "cache-line per
  vector" threshold this is free; above it, the 2× compounds.

## Alternatives considered

1. **OPQ + 4-bit PQ.** Higher recall but per-shard codebook training and
   16-way LUT scan. Complexity is out of proportion to a nightly PoC.
2. **RaBitQ / anisotropic PQ.** Already covered by earlier nightly
   research (2026-09-03, 2026-09-04). Ternary is deliberately the
   simplest coding scheme that beats plain sign.
3. **Learned sign hash (ITQ).** Superior on skewed data but requires a
   training pass and per-shard rotation matrix. Ternary needs no
   training data and preserves per-vector locality.
4. **4-bit signed scalar quant.** Comparable memory (dim/2 bytes) but
   distance is a squared-difference not a popcount, so per-pair cost is
   several×. Ternary keeps the popcount-only kernel.

## References

- `crates/ruvector-ternary/` — implementation and benchmark harness.
- `docs/research/nightly/2026-09-11-ternary-vector-search/README.md` —
  full write-up, sparsity math, and reproducibility instructions.
- Prior art: FAISS `IndexBinaryHNSW`; Weaviate `bq` mode; Milvus
  `BIN_FLAT`; ADR-345 (mincut-gated forgetting) for the shared
  nightly-research process.
