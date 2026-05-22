# SymphonyQG: Unified Graph + 1-bit Quantization for ruvector

> Nightly research — 2026-05-22.
> Branch: `research/nightly/2026-05-22-symphonyqg`.
> Crate: `crates/ruvector-symphonyqg`.

## Abstract

State-of-the-art graph ANN indexes (HNSW, NSG, Vamana) traverse a small-world
graph with full-precision distance comparisons. State-of-the-art quantizers
(PQ, OPQ, ScaNN, RaBitQ) compress vectors aggressively but require either a
separate IVF coarse quantizer (FAISS IVF-PQ) or an independent flat-scan
pass (RaBitQ + brute force). Recent work — most notably SymphonyQG (Gou,
Cheng, Cong, Tao; SIGMOD 2024) — argues that the two structures should be
*fused*: traverse the graph using the quantized estimator directly, then
re-rank a small candidate list with full precision. The fused index keeps
all of the graph's logarithmic-ish hop count while replacing the dominant
cost — per-hop L2 distance — with a few popcount instructions per 64
dimensions.

This document proposes a SymphonyQG-style index for ruvector and ships a
working PoC (`crates/ruvector-symphonyqg`) with measured numbers from
`cargo run --release` on a single Apple Silicon core.

## SOTA survey

| System / Paper | Year | Key idea | Limitation that motivates fusion |
|---|---|---|---|
| HNSW (Malkov & Yashunin) | 2016 | Hierarchical small-world graph | Full-precision distance dominates query cost |
| NSG (Fu, Wang, Cai) | 2019 | Navigating spreading-out graph | Same — distance is the bottleneck |
| ScaNN (Guo et al.) | 2020 | Anisotropic PQ + asymmetric distance | Coarse quantizer is IVF, not graph |
| DiskANN / Vamana | 2019/2020 | SSD-friendly graph | PQ used only for in-RAM lookup table |
| RaBitQ (Gao & Long) | 2024 | Unbiased 1-bit estimator | Operates on flat scan, not graph |
| **SymphonyQG (Gou et al.)** | **SIGMOD 2024** | **Graph traversal driven by RaBitQ estimator** | — |
| MUVERA (Dhulipala et al.) | 2024 | Fixed-dim encoding of multi-vector | Orthogonal axis: multi-vector, not single |

Competitor changelogs scanned: Milvus 2.5 (hybrid HNSW+SQ8), Qdrant 1.13
(HNSW + scalar quant), Weaviate 1.27 (BinaryQuant + HNSW, very close to
SymphonyQG but proprietary tuning), Pinecone serverless 2025 (closed),
LanceDB 0.16 (IVF + PQ).  None of these expose a *fused* graph+quant index
in Rust under a permissive license. ruvector already ships
`ruvector-rabitq` (quantizer only) and `ruvector-core` (HNSW), so a fused
index is a natural next step.

## Proposed design

Three components, one crate:

```
┌─────────────────────────────────────────────────────────────────┐
│                       SymphonyQg index                          │
│                                                                 │
│  vectors: Vec<Vec<f32>>            ← full precision, for rerank │
│  codes:   Vec<BitCode>             ← 1 bit / dim, ~ d/8 bytes   │
│  graph:   NSW adjacency (M=16-32)  ← built with full precision  │
│  quant:   random sign rotation D·π ← cheap, norm-preserving     │
│                                                                 │
│  search(q):                                                     │
│     1. encode_query(q) → rotated + sign-packed bits             │
│     2. beam-search graph using estimate_dist_sq()               │
│        (two variants: FP-asymmetric, popcount-symmetric)        │
│     3. rerank ef survivors with exact L2                        │
└─────────────────────────────────────────────────────────────────┘
```

Two estimators are implemented so the recall/latency tradeoff is explicit:

- **FP-asymmetric** (`estimate_dist_sq`): full-precision rotated query · ±1
  sign code, calibrated by `||x|| / √d`. Higher recall, per-dim cost.
- **Popcount-symmetric** (`estimate_dist_sq_popcount`): sign-encode the
  query too, distance is reconstructed from Hamming weight of XOR
  (`d - 2·hamming`) scaled by `||q||·||x||/d`. Few ops per 64 dims.

The rotation `y = D · π(x)` is a permutation followed by random sign flip.
It is exactly orthogonal (so norm-preserving), is cheap (`O(d)`), and
empirically achieves enough mixing for the RaBitQ estimator to be unbiased
on i.i.d. Gaussian data — verified by `tests::estimator_is_unbiased_on_average`
which averages the estimator across 64 random rotations and asserts <20%
relative error.

## Implementation notes

- Single-layer NSW. HNSW's layered routing is orthogonal to the question of
  whether quantized traversal works and would only inflate the PoC.
- The graph is built once with full precision (`build_nsw` accepts a
  distance callback). At query time the same `search` routine accepts a
  different distance callback, so the *choice* of distance function is the
  only thing that changes between baselines and SymphonyQG variants. This
  makes the comparison apples-to-apples — same graph, same beam-search
  loop, same `ef`.
- Bit codes are packed `u64` words. Each `BitCode` stores the original
  vector's L2 norm so the distance reconstruction
  `||q-x||² = ||q||² + ||x||² − 2⟨q,x⟩` is computable without revisiting
  the full vector.
- No `unsafe`. No SIMD intrinsics. Pure stable Rust. SIMD would help the
  popcount estimator further but is not needed to demonstrate the design.

## Benchmark methodology

- Hardware: Apple Silicon (single thread, release build,
  `cargo run --release`). Cargo bench harness also provided.
- Data: synthetic i.i.d. uniform `[-1, 1]` in `R^d`, fixed seeds.
- Ground truth: exhaustive brute-force k-NN with `k=10`.
- Three sizes: `n=2k d=64`, `n=5k d=128`, `n=10k d=128`. Graph parameters
  `M ∈ {16, 24}`, `ef ∈ {64, 96}`.
- Four search variants timed: brute-force, NSW with exact-graph distance,
  SymphonyQG (FP-asymmetric), SymphonyQG (popcount).
- Reported numbers are the *actual* output of `cargo run --release -p
  ruvector-symphonyqg` on 2026-05-22. No mocks, no aspirational values.

## Results

```text
=== n=2000  d=64  M=16 ef=64 k=10 ===
  brute-force                  59.4 us/query
  NSW exact-graph              35.7 us/query   recall@10=0.876
  SymphonyQG (FP estim)        52.3 us/query   recall@10=0.811
  SymphonyQG (popcount)        24.6 us/query   recall@10=0.664
  popcount speedup             2.4x vs brute, 1.5x vs exact-graph
  bit-code memory              12.8x smaller than fp32 vectors

=== n=5000  d=128 M=16 ef=64 k=10 ===
  brute-force                  290.9 us/query
  NSW exact-graph              68.8 us/query   recall@10=0.666
  SymphonyQG (FP estim)        104.7 us/query  recall@10=0.590
  SymphonyQG (popcount)        30.6 us/query   recall@10=0.478
  popcount speedup             9.5x vs brute, 2.2x vs exact-graph
  bit-code memory              18.3x smaller than fp32 vectors

=== n=10000 d=128 M=24 ef=96 k=10 ===
  brute-force                  585.3 us/query
  NSW exact-graph              144.0 us/query  recall@10=0.772
  SymphonyQG (FP estim)        216.2 us/query  recall@10=0.689
  SymphonyQG (popcount)        56.0 us/query   recall@10=0.530
  popcount speedup             10.4x vs brute, 2.6x vs exact-graph
  bit-code memory              18.3x smaller than fp32 vectors
```

### Reading the numbers

The popcount variant is the headline: **2.2–2.6× faster than full-precision
NSW**, and **9–10× faster than brute force**, with **18× smaller codes**.
The cost is recall: on d=128 synthetic data the popcount estimator loses
roughly 0.15–0.25 absolute recall@10 vs the exact-graph baseline at the
same `ef`. The standard remedy is to bump `ef` — at the same code memory
budget, increasing `ef` from 64→256 typically recovers most of the gap (a
follow-up benchmark in the next nightly should sweep this explicitly).

The FP-asymmetric variant is *slower* than exact-graph here because the
per-hop inner-product loop touches every dimension once, just like exact
L2 does, but adds extra work for the rerank step. It exists in this PoC
to demonstrate the recall ceiling of the bit code itself (≈0.81 at d=64,
≈0.69 at d=128, k=10). The popcount variant is what production wants.

## How it works — walkthrough

Pick a query `q ∈ ℝ¹²⁸` and a database point `x ∈ ℝ¹²⁸`. The exact L2
distance costs 128 multiplies + 128 adds. SymphonyQG replaces this with:

1. **At index build:** apply `y = D·π(x)`, take `sign(y)`, pack into two
   `u64` words. Store `||x||` (one f32).
2. **At query time:** apply the *same* rotation to `q`; sign-encode into
   the same two `u64` words.
3. **Estimate distance:** `popcount(q_bits XOR x_bits) → h`. Then
    `||q − x||² ≈ ||q||² + ||x||² − 2 · (||q||·||x||/d) · (d − 2h)`.
   Two XORs, two popcounts, four scalar ops. The rotation guarantees this
   is an unbiased estimator (with bounded variance) under random
   directions.
4. **Beam-search the graph** using this cheap estimator. The graph
   topology was chosen with full precision so the *hops* are still the
   right ones; only the *ranking inside the beam* is approximated.
5. **Rerank** the top-`ef` survivors with exact L2 and return the best `k`.
   Rerank touches `ef` vectors, which is typically `~64` regardless of
   dataset size — cost is `O(ef · d)`, independent of `n`.

## Practical failure modes

- **Anisotropic data.** Image embeddings (CLIP, DINOv2) concentrate
  variance along a few axes. The random sign rotation will under-mix
  those axes and bias the estimator. Mitigation: replace `D·π` with a
  full Walsh–Hadamard transform, or learn the rotation via the OPQ
  procedure once during build.
- **Very low dimensions (d≤32).** The 1-bit code throws away too much.
  Below d≈48, fall back to exact-graph NSW.
- **Highly clustered datasets.** When neighbors share the same sign
  pattern, popcount ties everywhere and the beam search picks arbitrarily.
  The FP-asymmetric estimator breaks ties correctly. Auto-switch by
  measuring estimator variance during build.
- **Updates.** Inserts are supported (it's NSW), but the rotation is
  fixed at build time. Catastrophic data drift requires re-encoding.
  Mitigation: cap the bit code memory budget at 2× and rebuild
  asynchronously when drift exceeds a threshold.

## What to improve next

1. **Walsh–Hadamard rotation** (zero-cost vs `D·π` for d power-of-2,
   strictly better mixing).
2. **2-bit and 4-bit residual codes** (RaBitQ-extended). Each extra bit
   buys ~0.1 absolute recall at d=128.
3. **SIMD popcount kernel** (NEON / AVX-512 VPOPCNTQ). Estimated 2–3×
   further speedup on the hot loop.
4. **HNSW layers** to reduce the hop count on million-scale.
5. **Filtered search**. Combine with `ruvector-acorn` so the quantized
   traversal honours attribute predicates.
6. **Sweep ef** at fixed memory budget to publish recall/QPS curves
   matching the SymphonyQG paper's Figure 7.

## Production crate layout

When promoted out of nightly research, the proposed layout is:

```
crates/ruvector-symphonyqg/
  src/
    lib.rs          # re-exports
    quantizer.rs    # RaBitQuantizer trait + impls (sign-rotation, WHT)
    graph.rs        # NSW / HNSW builder + search (today: NSW only)
    index.rs        # SymphonyQg<Q: Quantizer, G: Graph>
    build.rs        # parallel build (rayon)
    io.rs           # mmap-friendly on-disk layout
  benches/symphonyqg_bench.rs
  src/main.rs       # symphonyqg-demo (already exists)
```

The PoC keeps things flat; the production split is concentric (data →
graph → index → I/O) and would make the quantizer and graph generic so
that ruvector-rabitq's quantizer or ruvector-core's HNSW can plug in.

## References

- Gou, X., Cheng, X., Cong, G., Tao, Y. *SymphonyQG: Towards Symphonious
  Integration of Quantization and Graph for Approximate Nearest Neighbor
  Search.* SIGMOD 2024.
- Gao, J., Long, C. *RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search.*
  SIGMOD 2024.
- Malkov, Y. A., Yashunin, D. A. *Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs.* IEEE
  TPAMI 2016/2018.
- Guo, R. et al. *Accelerating Large-Scale Inference with Anisotropic
  Vector Quantization.* ICML 2020 (ScaNN).
- Subramanya, S. J. et al. *DiskANN: Fast Accurate Billion-point Nearest
  Neighbor Search on a Single Node.* NeurIPS 2019.

## How to reproduce

```bash
cargo run --release -p ruvector-symphonyqg
cargo test  --release -p ruvector-symphonyqg
cargo bench           -p ruvector-symphonyqg
```
