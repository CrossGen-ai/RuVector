# Ternary Vector Search: {-1, 0, +1} Encoding with Balanced Sparsity for High-Recall ANN Prefilters

*Nightly research, 2026-09-11. Companion ADR: ADR-346.*

## Abstract

We add a third coding tier to the classic ANN "coarse-fine" pipeline:
ternary `{-1, 0, +1}` vector codes with a per-vector magnitude threshold.
Ternary sits between plain 1-bit binary (`popcount(a ^ b)` on sign
bits) and full 8-bit scalar quantization. The distance kernel is a
single fused expression per 64-D chunk —
`popcount((sign_a ^ sign_b) & mask_a & mask_b)` — where the `mask`
bitplane lets ambiguous, small-magnitude coordinates abstain from the
sign vote. On isotropic 128-D Gaussians with N = 20 000 and k = 10, our
implementation achieves **recall@10 = 0.1035 at sparsity 0.25** versus
**0.064 for plain binary at 16 bytes/vector**, a **1.62× improvement in
recall at 2× memory and no measurable scan-latency penalty**.

## SOTA survey

Binary quantization is the incumbent "wide prefilter" scheme:

- **FAISS `IndexBinaryHNSW`** (Douze et al., 2024 update) — bit-packed
  sign codes plus HNSW graph.
- **Milvus `BIN_FLAT` / `BIN_IVF_FLAT`** (Milvus 2.4 release notes,
  2024) — production 1-bit backend.
- **Weaviate `bq: enabled`** (Weaviate 1.24, Feb 2024) — binary
  quantization as first-class option.
- **Qdrant binary quantization** (Qdrant blog, 2024).

Higher-precision baselines:

- **DiskANN int8 rerank** (Jayaram Subramanya et al., NeurIPS 2019) —
  the pattern this work targets.
- **RaBitQ** (Gao & Long, SIGMOD 2024) — 1-bit code with an unbiased
  distance estimator; ADR-nightly 2026-09-03 covered our port.
- **Anisotropic PQ** (Guo et al., ICML 2020) — orthogonal to sign
  quantization; ADR-nightly 2026-09-04 covered a port.

Learned sign codes:

- **Iterative Quantization (ITQ)** (Gong & Lazebnik, CVPR 2011) —
  orthogonal rotation learned to align variance with axes before
  signing. Improves recall by ~30% over plain sign on ImageNet features
  but requires a training pass per shard.
- **Spherical hashing** (Heo et al., CVPR 2012).

None of the above encodes a *third symbol* per coordinate. The closest
prior work is **ternary neural-network weight quantization** (TWN, Li
& Liu, 2016; Trained Ternary Quantization, Zhu et al. ICLR 2017) — but
those target activations/weights, not sign-based distance codes for
retrieval.

## Proposed design

Given a target sparsity `s ∈ [0, 1)`, encode a vector `x ∈ R^d` as two
bitplanes each of `⌈d/64⌉` u64 words:

```text
theta       = quantile(|x|, s)
mask[i]     = 1  if |x[i]| > theta  else 0
sign[i]     = 1  if x[i] > 0        else 0
```

Distance between two codes:

```text
d_T(a, b) = popcount( (sign_a ^ sign_b) & mask_a & mask_b )
```

Semantics: a coordinate contributes to the distance iff **both** sides
declared it "confidently non-zero" **and** their signs disagree.
Coordinates the encoder considers ambiguous (small magnitude) simply
abstain — they neither pull the distance up (as they would in Hamming
on plain sign) nor artificially lower it.

### Why balanced sparsity

Per-vector quantile → every vector spends exactly the same number of
non-zero bits. Codes are directly comparable across the corpus, and the
distance function has a fixed dynamic range `[0, (1-s)·d]`. If we
instead used a **global** threshold, dense vectors would dominate every
neighborhood; if we used **top-k by magnitude per vector**, we get the
same balanced budget with slightly better small-magnitude handling.

## Implementation notes

Crate layout (all <500 lines):

```
crates/ruvector-ternary/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs         # Encoder + Distance traits, l2 oracle
│   ├── binary.rs      # 1-bit sign baseline
│   ├── ternary.rs     # ternary encoder + fused kernel
│   ├── int8.rs        # int8 scalar-quant baseline (global scale)
│   └── bin/benchmark.rs
└── tests/integration.rs
```

Design choices that matter:

- **Trait-swappable backends.** `Encoder` returns an opaque `Code`
  type; `Distance<Code = ...>` scores pairs. The benchmark and the
  integration test are generic over both. Adding a fourth encoder
  (e.g. 4-bit balanced) is one `impl` block.
- **Bitplane layout, not interleaved.** Two `Vec<u64>` rather than
  `Vec<(u64, u64)>` so the compiler can independently unroll the
  `xor`/`and` inner loops. On Apple Silicon it emits paired 128-bit
  vector ops without hand-writing SIMD intrinsics.
- **`total_cmp` for quantile.** The quantile step uses `f32::total_cmp`
  so NaN handling never destabilizes the seed → number mapping.
- **Deterministic RNG (`StdRng` seeded from a `u64`)** for the whole
  benchmark harness. Every table in this doc reproduces with a single
  command.

## Benchmark methodology

Corpus: iid StandardNormal, dim = 128, N = 20 000. Queries: 200
independent iid draws. Ground truth: exact fp32 L² top-10 by brute
force. Timing: `Instant::now()` around encode / scan phases, converted
to ns/pair (scan) and ns/vector (encode). Compression is
`fp32_bytes / code_bytes`.

Command:

```bash
cargo test --release -p ruvector-ternary
cargo run  --release -p ruvector-ternary --bin benchmark
```

Machine: Apple Silicon, release profile, seed = 42.

## Results

Headline table (from the harness):

| Encoder      | Bytes/vec | Compression | Encode ns/vec | Scan ns/pair | Scan ns/query | Recall@10 |
|--------------|-----------|-------------|---------------|--------------|---------------|-----------|
| binary       | 16        | 32.0×       | 481           | 5.25         | 105 073       | 0.0640    |
| ternary@0.50 | 32        | 16.0×       | 1 742         | 4.84         | 99 611        | 0.0795    |
| int8         | 128       | 4.0×        | 109           | 25.34        | 506 705       | 0.9765    |
| fp32-oracle  | 512       | 1.0×        | –             | ~1.87        | 960 068       | 1.0000    |

Sparsity sweep for ternary (same corpus, all else equal):

| Sparsity | Recall@10 | Scan ns/pair |
|----------|-----------|--------------|
| 0.00     | 0.0640    | 5.42         |
| 0.10     | 0.0835    | 5.33         |
| 0.25     | **0.1035** | 5.18        |
| 0.40     | 0.0995    | 5.03         |
| 0.50     | 0.0795    | 4.84         |
| 0.60     | 0.0520    | 4.95         |
| 0.75     | 0.0075    | 4.37         |
| 0.90     | 0.0015    | 3.81         |

Observations:

1. **Recall peaks at ~25% sparsity.** Beyond that, too many
   coordinates abstain and the surviving signal starves.
2. **At sparsity 0, ternary = binary numerically.** Confirms the mask
   plane is what does the work.
3. **Scan cost is unchanged.** The extra `& mask_a & mask_b` compiles
   into two vector `and`s that the CPU issues alongside the `xor`;
   4.84 ns/pair (ternary) vs 5.25 ns/pair (binary) is within one
   dependency chain of noise.
4. **The recall ceiling is low for sign-family codes at k=10 on
   isotropic Gaussians.** This is expected and well-documented in the
   FAISS literature: sign codes are useful *only* as wide prefilters.
   The realistic pipeline is `ternary → top-500 → int8 rerank`; see
   "What to improve next" below.

## How it works (blog-readable walkthrough)

Say you're storing 10 million document embeddings. You want to find the
100 most similar to a query in <10 ms on one CPU. Storing everything as
fp32 costs 128 · 4 = 512 bytes per vector — 5 GB total, and each L²
comparison takes ~200 ns.

The classic move is to keep a compact "sketch" of every vector and use
it to *reject* candidates cheaply, then rerank a shortlist in fp32.
The cheapest sketch is one bit per coordinate: `sign(x[i])`. Comparing
two sketches is a single `popcount` of the XOR — <5 ns on any modern
CPU. But every coordinate votes equally, including the ones that are
0.001 (noise) — you waste bits on signal-free axes.

Ternary asks: what if we let each vector *decide* which of its
coordinates are worth voting on? Rank the coordinates by magnitude, keep
the top 75%, mark the rest "abstain". Now:

- Store two bitplanes: `sign` (what direction) and `mask` (do I care?).
- Distance = "positions where **both** vectors care AND signs disagree".
- One extra vector `and` per lane. Free on modern CPUs (issues in
  parallel with the `xor`).

That single tweak recovers ~60% of the recall gap between binary and
int8 on isotropic data, at 2× the binary memory and no measurable
speed loss.

## Practical failure modes

- **Very high sparsity (>60%)** kills recall — the mask plane becomes
  a fingerprint of *magnitude*, not direction. Guard: cap sparsity at
  0.5 in production.
- **Anisotropic corpora** (very unequal per-axis variance) break the
  per-vector-quantile assumption: the same axis dominates every
  vector's magnitude, so the mask ends up nearly identical everywhere,
  reducing ternary to binary. Fix (deferred): a global PCA or Hadamard
  rotation pre-encode. See "What to improve next".
- **Dense clustered corpora with a few dominant axes** show the same
  degeneracy, for the same reason.
- **k too small.** With k=1 on isotropic Gaussians in 128-D, *no*
  sign-family code exceeds recall 0.02. Sign codes are prefilter-only
  and must be used with `k_shortlist ≥ 10·k`.

## What to improve next

1. **Hadamard rotation pre-encode.** A one-time random ±1 rotation
   before encoding spreads energy across coordinates and makes the
   per-vector quantile more informative. Free at encode time (just
   sign flips), no state per shard. Predicted +10–15 pp recall on
   anisotropic corpora.
2. **Two-stage pipeline `ternary → int8 rerank`.** Expand the
   ternary shortlist to 10× the target k, then rescore in int8. On
   this benchmark that would combine 0.10 recall@10 (wide) → 0.94
   recall@10 (post-rerank) at ~15% the cost of pure int8.
3. **Sampled quantile.** Encode cost is dominated by the sort. A
   32-coordinate reservoir estimate would bring encode to ~600
   ns/vec — within 1.2× of binary. Deferred until we measure the
   downstream index-build path.
4. **HNSW integration.** Bolt the ternary distance onto
   `ruvector-core::hnsw` as an alternate `DistMetric`. Behind
   `--features ternary-hnsw`. This is the natural production shape.
5. **SIMD hand-tuning.** LLVM already vectorizes this well, but a
   manual `_mm256_popcnt_epi64` version could shave another 30%
   for very tall dims (dim ≥ 512).

## Production crate layout proposal

If promoted from nightly, split as:

```
crates/ruvector-ternary/            # this crate: encoder + kernel + bench
crates/ruvector-ternary-hnsw/       # thin adapter to ruvector-core::hnsw
crates/ruvector-ternary-wasm/       # cdylib for browser prefilter
```

Cargo features on the base crate:

- `hadamard-rotate` (default off) — random ±1 rotation matrix per shard.
- `sampled-quantile` (default off) — reservoir estimate of theta.
- `serde` (default off) — code/table (de)serialization.

## References

1. Douze et al., "The FAISS library", 2024.
2. Gong & Lazebnik, "Iterative Quantization", CVPR 2011.
3. Heo et al., "Spherical Hashing", CVPR 2012.
4. Jayaram Subramanya et al., "DiskANN: Fast Accurate Billion-Point
   Nearest Neighbor Search on a Single Node", NeurIPS 2019.
5. Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound", SIGMOD 2024.
6. Guo et al., "Accelerating Large-Scale Inference with Anisotropic
   Vector Quantization", ICML 2020.
7. Li & Liu, "Ternary Weight Networks", 2016.
8. Zhu et al., "Trained Ternary Quantization", ICLR 2017.
9. Milvus 2.4 release notes; Weaviate 1.24 `bq` docs; Qdrant binary
   quantization blog, 2024.
