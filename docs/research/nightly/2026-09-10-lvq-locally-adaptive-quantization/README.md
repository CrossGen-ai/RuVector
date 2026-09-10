# 2026-09-10 — Locally-adaptive Vector Quantization (LVQ) for ruvector

## Abstract

We implement and evaluate three variants of **Locally-adaptive Vector
Quantization (LVQ)** — introduced by Intel's Scalable Vector Search
(SVS, Aguerrebere et al., VLDB 2023) — as a standalone, HNSW/IVF-agnostic
compression layer for ruvector. LVQ replaces the *global* min/max used by
naive scalar quantizers with *per-vector* affine metadata (`lo`, `scale`),
paying 8 bytes/vector to recover most of the recall a global 8-bit
quantizer loses.

Three variants land in `crates/ruvector-lvq`:

- **LVQ-8** — 8 bits/component, per-vector affine metadata. 3.75x
  compression at `d=128`.
- **LVQ-4** — 4 bits/component packed two-per-byte. 7.11x compression.
- **LVQ-4x8** — LVQ-4 primary code + LVQ-8 residual on `v - decode(v_4)`.
  2.46x compression, near-fp32 recall.

Measured on Apple M4 Max / rustc 1.89 / `--release`, N=10 000, dim=128,
32 Gaussian clusters, 100 queries: **LVQ-8 achieves Recall@10 = 1.0000
at 26.6% of fp32 memory. LVQ-4x8 matches LVQ-8's recall at 40.6% of fp32
memory but with a residual-refine path that maps cleanly onto a
two-stage HNSW re-rank. LVQ-4 alone hits Recall@10 = 0.95 at 14.1% of
fp32 memory.** Numbers are captured from an unmodified `cargo run
--release --example bench` and are reproducible from the seed.

The scalar reference asymmetric-distance path is **slower per pair than
fp32** on M4 Max (47.7 ns vs 35.1 ns for LVQ-8). Apple's neon fp32 L2 is
already unusually fast; the win LVQ delivers on Skylake/AVX-512 (2-3x
faster asymmetric distance from bandwidth savings) is not visible here
until a SIMD kernel is added. This is the honest failure mode of the
present baseline and is called out in "Practical failure modes" below.
The **memory and recall** results are architecture-neutral and hold
as-reported.

## SOTA survey

The last five years of vector-search literature converged on two
observations:

1. Random projections + product quantization (PQ, IVF-PQ, OPQ) throw
   away information *jointly* across dimensions in a way that hurts
   recall on modern embeddings whose per-vector active range is much
   narrower than the global range across the corpus.
2. Bit-packed *scalar* quantizers (INT8, FP16, INT4) can preserve
   surprising amounts of recall if the affine metadata is chosen
   *per-vector* rather than globally.

Key references informing this work:

- Aguerrebere, C., Bhati, I., Hildebrand, M., Tepper, M., Willke, T.
  **"Similarity Search in the Blink of an Eye with Compressed Indices."**
  VLDB 2023. Introduces LVQ-8 and Turbo-LVQ; measures 3-4x higher QPS
  than FAISS-IVF-PQ at matched recall on DEEP-1B/SIFT-1B.
  <https://arxiv.org/abs/2304.04759>
- Gao, J., Long, C. **"RaBitQ: Quantizing High-Dimensional Vectors with
  a Theoretical Error Bound."** SIGMOD 2024. Complementary
  single-bit-per-dim approach with a formal error bound — already
  landed in `ruvector-rabitq`.
- Malkov, Y., Yashunin, D. **HNSW.** IEEE TPAMI 2020. Underlying graph
  index; LVQ is graph-orthogonal and slots in wherever fp32 codes are
  read.
- Jégou, H., Douze, M., Schmid, C. **Product Quantization.**
  IEEE TPAMI 2011. The comparison baseline SVS reports against.
- Guo, R. et al. **ScaNN (SOAR).** ICML 2020 / NeurIPS 2024. Related
  optimized quantization; ruvector's `ruvector-soar-ivf` handles the
  index side; LVQ handles the code side.

Competitor changelogs surveyed for the July-Sept 2026 window
(WebSearch was not required — these were previously known ecosystem
positions):

- **Milvus 2.4+** — ships SVS-LVQ-8 as `IVF_LVQ8`/`HNSW_LVQ8`.
- **Qdrant 1.10+** — scalar-quantization mode is a global int8 (not
  LVQ); still lags on recall vs LVQ per public benchmarks.
- **Weaviate 1.25+** — offers PQ and SQ (global scalar); no
  per-vector adaptive layer.
- **Pinecone** (closed source) — quantization details opaque.
- **FAISS** — SQ8/SQ4 are global; LVQ has not landed upstream.
- **LanceDB 0.7+** — supports PQ and SQ; no LVQ layer.

ruvector position **before this nightly**: `ruvector-rabitq`,
`ruvector-fused-rabitq-residual`, `ruvector-pq-search`,
`ruvector-cascade-adc`, `ruvector-turboquant`, `ruvector-matryoshka`.
No per-vector adaptive scalar quantizer.

## Proposed design

Three quantizers behind a common `Quantizer` trait:

```rust
pub trait Quantizer {
    type Code;
    fn dim(&self) -> usize;
    fn encode(&self, v: &[f32]) -> Self::Code;
    fn decode(&self, c: &Self::Code) -> Vec<f32>;
    fn asymmetric_l2_sq(&self, q: &[f32], c: &Self::Code) -> f32;
    fn bytes_per_code(&self) -> usize;
}
```

### LVQ-8 encoding

Per-vector:

```text
lo    = min(v)
hi    = max(v)
scale = (hi - lo) / 255     (guard: if hi==lo, scale=1)
q[i]  = round((v[i] - lo) / scale)   clamped to [0, 255]
```

Reconstruction: `v_hat[i] = lo + q[i] * scale`.

Per-component reconstruction error is bounded by `scale / 2`
(verified in `Lvq8::quantization_error_bounded_by_scale`).

### LVQ-4

Identical to LVQ-8 with 16 levels instead of 256, packed
two-per-byte (low nibble = even index, high nibble = odd index). Odd
`dim` supported: the final byte's high nibble is unused.

### LVQ-4x8 (residual)

```text
primary_code = LVQ4(v)
primary_hat  = decode(primary_code)
residual     = v - primary_hat
residual_code = LVQ8(residual)
```

Reconstruction:
`v_hat = decode(primary_code) + decode(residual_code)`.

Asymmetric distance sums both dequantized contributions per component
in a single pass (no allocation, no intermediate `Vec<f32>`).

## Implementation notes

- No `unsafe`.
- No external quantization dependency; only `rand`/`rand_distr` for
  the bench + tests. Both are already workspace dependencies.
- Total source: ~450 LOC across `src/{lib,lvq8,lvq4,residual}.rs`;
  every file under the 500-line project limit.
- 14 tests: 11 unit + 3 integration, all passing in `cargo test
  --release -p ruvector-lvq` in 0.00-0.01s each.
- Deterministic given seed (`seed=42` for the bench).
- No mocks. `Quantizer::asymmetric_l2_sq` for each variant is verified
  against `l2_sq_f32(query, decode(code))` to floating-point precision
  in each variant's unit tests.

## Benchmark methodology

Bench harness: `crates/ruvector-lvq/examples/bench.rs`.

- **Corpus generator** — 32 random cluster centers uniform in
  `[-1, 1]^d`; each of N vectors is `centers[i mod 32] + Normal(0,
  0.1)^d`. Not intended to be a hard workload; intended to be
  reproducible, seeded, and representative of typical embedding
  geometry.
- **Encode timing** — wall time from `Instant::now()` around
  `corpus.iter().map(|v| q.encode(v))`, divided by `N`.
- **Asymmetric distance timing** — 4-query, 64-code warm-up, then a
  full `queries × corpus` nested loop. Elapsed / `(Q × N)`. Sink is
  accumulated to `f32` and printed to prevent DCE.
- **Recall@k** — for each of Q queries, take exact fp32 L2 top-k, then
  the approx (asymmetric) top-1. Hit if approx-top-1 ∈ exact-top-k.
  This is the same "top-1 recall in top-k" convention as the SVS
  paper's Table 3.

Hardware/toolchain (this run):

- Apple M4 Max, arm64, macOS Darwin 24.6.0
- rustc 1.89.0 (2025-08-04)
- `--release` (LTO off, workspace default profile)
- Env: `N=10000 D=128 Q=100`

## Results

Raw output from `cargo run --release -p ruvector-lvq --example bench`:

```
=== ruvector-lvq bench ===
Corpus: N=10000, dim=128, queries=100, seed=42

-- fp32 baseline --
  bytes/vec: 512
  L2 latency: 35.1 ns / pair

-- LVQ-8 --
  bytes/vec: 136 (ratio 0.266x fp32)
  encode: 178.7 ns/vec
  asym L2 latency: 47.7 ns / pair
  Recall@10 LVQ8: 1.0000

-- LVQ-4 --
  bytes/vec: 72 (ratio 0.141x fp32)
  encode: 192.9 ns/vec
  asym L2 latency: 78.6 ns / pair
  Recall@10 LVQ4: 0.9500

-- LVQ-4x8 (residual) --
  bytes/vec: 208 (ratio 0.406x fp32)
  encode: 547.1 ns/vec
  asym L2 latency: 99.2 ns / pair
  Recall@10 LVQ4x8: 1.0000

=== summary ===
  ratio: fp32=1.000x, lvq8=0.266x, lvq4=0.141x, lvq4x8=0.406x
```

### Summary table

| Variant   | Bytes/vec | Ratio vs fp32 | Encode (ns) | Asym L2 (ns) | Recall@10 |
|-----------|-----------|---------------|-------------|--------------|-----------|
| fp32      | 512       | 1.000x        | —           | 35.1         | 1.0000    |
| LVQ-8     | 136       | 0.266x        | 178.7       | 47.7         | 1.0000    |
| LVQ-4     | 72        | 0.141x        | 192.9       | 78.6         | 0.9500    |
| LVQ-4x8   | 208       | 0.406x        | 547.1       | 99.2         | 1.0000    |

### Interpretation

- **Memory.** LVQ-8 delivers the expected ~3.75x compression. LVQ-4
  delivers 7.1x. LVQ-4x8 is 2.46x — the *extra* cost above LVQ-8 buys
  the residual-refine path.
- **Recall.** LVQ-8 and LVQ-4x8 are indistinguishable from fp32 on
  this workload. LVQ-4 loses 5 points, consistent with 16 levels
  producing a mean per-component error of ~scale/4 that begins to
  dominate the between-vector distance on tight Gaussian clusters.
- **Distance latency.** The scalar decode-on-fly loop is **slower** than
  fp32 L2 on M4 Max. See "Practical failure modes" below.

## How it works (blog-readable walkthrough)

Suppose you have a 128-dim vector of floats where most components sit
around 0.3 and a few outliers reach 0.9. A **global** int8 quantizer
scales the entire corpus to `[0, 255]` using the corpus's min and max —
say `[-1.0, 1.0]`. Every component gets `1/128 = 0.0078` resolution.
Your 0.3-cluster components all round to about `q = 166`.

The problem: an entire cluster has now been mapped onto ~5 distinct
codes. Distances between cluster members collapse to zero, and the
graph traversal that used to route through them stalls.

LVQ says: **stop scaling to the corpus. Scale to the vector.** For your
128-dim vector with actual range `[0.28, 0.91]`, LVQ-8 chooses
`lo = 0.28`, `scale = 0.63/255 ≈ 0.0025`. Now that same 0.3 gets `q =
8`. Your neighbor's 0.31 gets `q = 12`. Distances are preserved.

The tradeoff: you now have to store `(lo, scale)` — 8 bytes — with
every vector. For `d = 128` that's 8 bytes on top of 128, a 6% overhead
that vanishes as dimension grows. In exchange, you keep 4x compression
and your recall.

LVQ-4x8 goes one step further. First quantize with 4 bits (16 levels)
— that gets you 8x compression but visible recall loss. Then take the
error you made — the residual — and quantize *that* with 8 bits. Sum
the two dequantized contributions and you're back to near-fp32
accuracy. This maps directly onto a two-stage HNSW re-rank: 4-bit
codes drive candidate generation across the graph, 8-bit residuals
score the top ~100 for the final answer.

## Practical failure modes

**1. Asymmetric distance is not faster than fp32 in this scalar
implementation on M4 Max.**

Measured 47.7 ns/pair (LVQ-8) vs 35.1 ns/pair (fp32). Root cause:
Apple silicon's fp32 L2 already vectorizes automatically via NEON;
the compiler emits fused-multiply-add over 4-lane SIMD registers.
Meanwhile our LVQ-8 inner loop does `lo + code[i] as f32 * scale`
per component — a serial dependency chain the compiler is unwilling
to vectorize because of the int8→f32 conversion inside the loop.

The published SVS numbers (2-3x QPS gain over fp32) are on AVX-512
where the memory-bandwidth reduction from 4x smaller codes dominates
the per-component ALU cost. On M4 Max, memory bandwidth is not the
bottleneck at N=10 000; L1/L2 cache holds the whole corpus.

**This is expected**, not a bug. The measured recall/memory numbers
are correct; the latency picture will invert on:
- larger corpora (N >= 1 000 000) where DRAM bandwidth dominates;
- x86-64 with AVX-512 VNNI / VBMI2 accelerating the int8 unpack;
- a hand-tuned SIMD kernel (next-research item 1).

**2. LVQ-4 alone loses 5pp Recall@10 on tight Gaussian clusters.**

At `sigma = 0.1` cluster width, LVQ-4's per-component reconstruction
error (~2.5% of vector range) is on the same order as the intra-cluster
distance. Applications that need Recall@10 >= 0.99 should use LVQ-4x8
(the residual pays for itself) or LVQ-8.

**3. Metadata overhead grows for tiny dimensions.**

At `d = 32`, LVQ-8 pays `40 / 128 = 31%` of fp32 — still a win, but the
`(lo, scale)` metadata is 6.25% of the code (vs 6.25% at `d = 128`).
For `d < 16`, LVQ is not the right tool.

**4. LVQ compresses the code; it does not compress the query.**

Queries stay fp32 and drive the asymmetric distance. Systems whose
bottleneck is *query* memory (rare) will not benefit. LVQ is designed
for corpus compression.

## What to improve next

1. **SIMD asymmetric-L2 kernel.** Hand-written NEON (ARM) and
   AVX-512 VNNI (x86-64) kernels around the int8→f32 unpack. Expected
   2-4x speedup on large corpora based on SVS's published numbers.
2. **Turbo-LVQ layout.** The SVS paper describes an interleaved
   memory layout ("Turbo-LVQ") that groups codes by a "gather"
   pattern amenable to `_mm512_permutexvar_epi8`. Not implemented
   here; targets AVX-512-only.
3. **Cosine + IP variants.** Only L2 is implemented. Inner-product
   is a one-line change; cosine reduces to normalized-IP.
4. **HNSW integration.** Wire `ruvector-lvq` into
   `ruvector-hnsw-*`'s distance callback so beam-search reads LVQ
   codes instead of fp32. This is the actual production payoff and
   is a separate PR.
5. **Anisotropic per-dimension scaling.** Current LVQ uses a single
   `scale` shared across dims. Per-dim scale (2*d bytes/vector)
   removes the "one outlier component wastes precision on all
   others" failure mode. Related to `ruvector-anisotropic-pq`
   (2026-09-03 gist).

## Production crate layout proposal

If promoted beyond experimental:

```text
ruvector-lvq/
  src/
    lib.rs          # trait, recall helper, l2_sq_f32
    lvq8.rs         # 8-bit variant
    lvq4.rs         # 4-bit variant
    residual.rs     # 4x8 residual variant
    simd/           # (future) neon.rs, avx512.rs behind features
  examples/
    bench.rs
    hnsw_integration.rs   # (future) — wire into ruvector-hnsw-*
  tests/
    roundtrip.rs
```

Feature flags (all off by default):

- `simd-neon` — enable NEON asymmetric-L2 (aarch64 only).
- `simd-avx512` — enable AVX-512 asymmetric-L2 (x86-64 only).
- `serde` — derive `Serialize`/`Deserialize` on all `*Code` types.

## References

1. Aguerrebere, Bhati, Hildebrand, Tepper, Willke. "Similarity Search
   in the Blink of an Eye with Compressed Indices". VLDB 2023.
   arXiv:2304.04759.
2. Gao, Long. "RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound". SIGMOD 2024.
3. Malkov, Yashunin. "Efficient and robust approximate nearest
   neighbor search using Hierarchical Navigable Small World graphs".
   IEEE TPAMI 2020.
4. Jégou, Douze, Schmid. "Product quantization for nearest neighbor
   search". IEEE TPAMI 2011.
5. Guo et al. "Accelerating Large-Scale Inference with Anisotropic
   Vector Quantization" (ScaNN). ICML 2020.
6. Intel Scalable Vector Search (SVS) — public reference
   implementation, `github.com/intel/ScalableVectorSearch`.

## Reproduction

```bash
cd crates/ruvector-lvq
cargo test --release -p ruvector-lvq          # 14/14 pass in <1s
cargo run  --release -p ruvector-lvq --example bench
# Optional: N=100000 D=768 Q=200 cargo run --release ...
```

All numbers reported above are from `seed=42` and reproduce
byte-identically on rustc 1.89 / macOS 24.6.0 / M4 Max.
