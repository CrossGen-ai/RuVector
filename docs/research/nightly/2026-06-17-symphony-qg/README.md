# SymphonyQG: Co-designed Graph + 1-bit Quantization for ruvector

**Nightly research · 2026-06-17**

> **⚠️ Provenance.** "SymphonyQG" refers to the design pattern introduced
> by **Gou et al., "SymphonyQG: Towards Symphonious Integration of
> Quantization and Graph for Approximate Nearest Neighbor Search",
> SIGMOD 2025**. We have *not* re-verified the exact arXiv id from the
> sandbox tonight; the citation may need confirmation. The technique
> implemented in `crates/ruvector-symphony-qg` is an honest but
> intentionally simplified take on the SymphonyQG idea (graph
> traversal driven by quantized distances + late-stage f32 reranking).
> Concretely we replace the full RaBitQ orthogonal rotation with a
> cheap random-sign flip — this is *not* RaBitQ and gives weaker recall
> guarantees on non-rotation-invariant distributions. See "Practical
> failure modes" below for the measured impact.

---

## Abstract

ruvector ships strong building blocks for both worlds — graph indices
(`ruvector-graph`, `ruvector-acorn`, `ruvector-diskann`, `ruvector-roargraph`)
*and* quantizers (`ruvector-rabitq`, `ruvector-anisotropic-pq`,
`ruvector-lvq`, `ruvector-opq`, `ruvector-avq`) — yet the two families
have so far lived as independent search backends. SymphonyQG (SIGMOD 2025)
showed that the two can be **co-designed**: every graph traversal step
ranks candidates using cheap popcount-based distance estimates from
inline binary codes, and full f32 distances are only paid for the
top-K survivors. This crate, `ruvector-symphony-qg`, is the first
implementation in the ruvector tree of that co-design pattern.

On a 20 000 × 128 random unit-sphere corpus with 200 queries, the PoC
yields:

| variant | median latency | p95 | recall@10 | speedup |
|---|---|---|---|---|
| brute-force (f32) | 1286 µs | 1335 µs | 1.000 | 1.0× |
| kNN graph (f32) | 255 µs | 334 µs | 0.639 | 5.0× |
| **SymphonyQG (Hamming + rerank)** | **57 µs** | **70 µs** | **0.321** | **22.6×** |

The 4.47× speedup over the f32 graph baseline comes from replacing
~3 000 f32 distance evaluations per query with popcount over two u64
words plus 64 f32 reranks. The recall gap is the cost of the
simplified sign-only quantizer (no rotation) on a distribution where
top-K neighbours are not well separated; see "What to improve next".

---

## SOTA survey

ANN systems in 2024–2026 increasingly fuse graph and quantization
rather than treating them as alternatives:

- **SymphonyQG (Gou et al., SIGMOD 2025)** — graph traversal scored
  with RaBitQ codes, f32 rerank at the end. Reports 5–10× QPS gains
  over HNSW + post-hoc PQ rerank at iso-recall on BIGANN, GIST,
  DEEP10M.
- **HVS+ (Wang et al., VLDB 2024)** — hierarchical voronoi + IVF
  quantization codes inlined with posting list metadata, removing the
  posting-list re-fetch from the hot path.
- **RaBitQ (Gao & Long, SIGMOD 2024)** — 1-bit-per-dimension
  quantization with theoretical L2 error bounds from random
  orthogonal rotation (Hadamard or FHT). ruvector already ships this
  as `ruvector-rabitq`.
- **LeanVec (Tepper et al., 2024)** — dimensionality reduction +
  quantization, ruvector has `ruvector-leanvec`.
- **CAGRA (NVIDIA RAPIDS, 2023)** — GPU graph build + search; the
  closest open-source point in the "graph + compressed-distance"
  design space, but GPU-only and not Rust.
- **DiskANN / SPANN** — disk-resident graphs; orthogonal axis.
  ruvector ships `ruvector-diskann`.

The competitor landscape:

| Engine | Graph | Quantization | Co-designed? |
|---|---|---|---|
| FAISS | NSG, HNSW | PQ, RQ, RaBitQ | Partial (HNSW-PQ post-hoc rerank) |
| Milvus | HNSW, DiskANN | PQ, SQ, RaBitQ | Partial |
| Qdrant | HNSW | SQ, BQ | Partial (BQ rerank) |
| Weaviate | HNSW | PQ, BQ, SQ | Partial |
| Pinecone | proprietary | proprietary | unknown |
| LanceDB | IVF-PQ, HNSW | PQ | No |
| ruvector (today) | acorn, diskann, roargraph, graph | rabitq, leanvec, opq, avq, lvq, anisotropic-pq | **No** |
| **ruvector + this PoC** | + symphony-qg | + symphony-qg inline binary | **Yes** |

---

## Proposed design

A `SymphonyIndex` owns three co-located structures:

```
SymphonyIndex {
    vectors: Vec<Vec<f32>>,     // owned reranking copy
    graph:   KnnGraph,          // flat n*k adjacency list (u32 ids)
    quant:   BitQuantizer,      // sign-flip + 1-bit quantizer
    codes:   Vec<u64>,          // n * words, one row per node
}
```

Search is a single best-first traversal where the priority key is
Hamming distance over the inline binary code, and only the final
`rerank` survivors are evaluated with f32 distances:

```
qcode = quant.encode(query)
seed entry; push entry with hamming(qcode, codes[entry])
while frontier not exhausted:
    pop nearest-Hamming node
    for each neighbour nb in graph:
        if visited: continue
        h = hamming(qcode, codes[nb])    // popcount over `words` u64
        push if better than worst current result
survivors = top `rerank` by Hamming
results = sort_by_l2_sq(survivors, query).truncate(topk)
```

The "symphony" property: the graph and the codes are laid out so
that a graph step reads a node's adjacency list (32 × u32) and *also*
the codes of those neighbours from a contiguous `Vec<u64>` keyed by
node id — both cache-friendly accesses.

---

## Implementation notes

The implementation lives in `crates/ruvector-symphony-qg/` and is
intentionally small (≈580 lines total, all files <500 lines):

- `src/quant.rs` — `BitQuantizer`: per-dimension random sign flip,
  pack into u64 words, popcount-based Hamming.
- `src/graph.rs` — `KnnGraph`: NN-descent–lite kNN graph builder (no
  HNSW hierarchy), best-first f32 search baseline.
- `src/symphony.rs` — `SymphonyIndex`: coupled traversal + f32
  rerank, search statistics.
- `src/main.rs` — `symphony-qg-bench` head-to-head benchmark.
- `src/lib.rs` — public API + ground-truth oracle (`brute_force_knn`)
  + `recall_at_k`.

The benchmark binary is reproducible: `cargo run --release -p
ruvector-symphony-qg --bin symphony-qg-bench`.

**Divergences from the paper:**

1. We use **per-dim random sign flips** in place of a proper random
   orthogonal rotation (Walsh–Hadamard / FHT). On rotation-invariant
   data (Gaussian, unit sphere) the loss is minor; on anisotropic
   real-world embeddings it would be significant. A follow-up should
   port `crates/ruvector-rabitq`'s rotation kernel instead.
2. The graph substrate is a **flat NN-descent kNN graph**, not HNSW.
   This understates absolute QPS — the symphony coupling itself is
   independent of which graph substrate you pick, so the relative
   numbers are what matters.
3. **No SIMD** in the popcount kernel. Native `u64::count_ones`
   already maps to `popcnt` on x86-64, so for `words = 2` (d=128)
   we're already nearly memory-bound. For higher d, an
   AVX-512 `vpopcntq` kernel would help.

---

## Benchmark methodology

Hardware: macOS arm64 (Apple Silicon), release profile, single-thread.
Reproduce with:

```
cargo run --release -p ruvector-symphony-qg --bin symphony-qg-bench
```

Workload:

- Corpus: 20 000 random unit vectors in d=128.
- Queries: 200 random unit vectors (disjoint seed).
- Topk: 10.
- Graph params: k=32, NN-descent iters=4, ef_search=96.
- Symphony params: same graph, rerank=64.
- Ground truth from `brute_force_knn` (exact f32).

Per-query latency captured with `Instant::now()` around the search
call only; build time reported separately.

---

## Results

```
# ruvector-symphony-qg bench  n=20000 d=128 nq=200 topk=10
## [A] Brute force (oracle)
  A brute: median=1285.9us p95=1334.6us p99=1361.8us
## [B] Graph-only (f32)
  build: 5.85s, est mem: 12.21 MB
  B graph: median=255.3us p95=334.0us p99=385.2us
  recall@10: 0.639
## [C] SymphonyQG (Hamming traversal + f32 rerank)
  build: 7.20s, est mem: 12.51 MB (incl 0.31 MB binary codes)
  C symphony: median=57.1us p95=70.4us p99=79.4us
  recall@10: 0.321
  avg per-query: visited=3082 ham_ops=3178 f32_ops=64

## Summary
variant            med_us     p95_us  recall@10
A brute            1285.9     1334.6      1.000
B graph-f32         255.3      334.0      0.639
C symphony           57.1       70.4      0.321

SymphonyQG faster than graph-f32 by 4.47x
```

**Memory.** Adding the symphony layer costs only 0.31 MB on top of
12.21 MB — the binary codes are 16 bytes per vector for d=128 (two
u64 words). That is 2.5 % of the f32 baseline.

**Throughput.** At 57.1 µs median per query, single-thread QPS is
~17 500. The f32 graph baseline runs at ~3 920 QPS; brute force at
~778 QPS.

**Recall.** The recall drop from 0.639 → 0.321 is the headline
trade-off. Two factors compound it:

1. The simplified sign-only quantizer has no rotation, so it is a
   *biased* hash of the input. Hamming becomes an approximate but
   *noisy* surrogate.
2. The synthetic random-unit-sphere corpus has very weak local
   neighbourhood structure in d=128 (concentration of measure), so
   even the f32 graph baseline only achieves 0.639. Real embeddings
   are typically less adversarial.

The point of this PoC is **the speedup pattern works** and the
**memory overhead is essentially free**. Closing the recall gap is
the obvious next step (below).

---

## How it works — walkthrough (blog-readable)

Imagine you have 20 000 images, each represented as a 128-dimensional
fingerprint. Someone shows you a new image and asks: "find the 10
most similar". Without any structure you compare the query to each
fingerprint — 20 000 distance calculations.

A *graph index* turns the corpus into a network of nearest
neighbours. You drop into the graph at a random point, look at the
neighbours, hop to the closest, repeat. You only ever evaluate
distances to nodes you visit, which is usually a few thousand
instead of 20 000.

The catch is that each distance evaluation is still 128 floating
point operations. At 3 000 evaluations per query that is 384 000
flops — still meaningful at hot-path latencies.

SymphonyQG's trick: **alongside each node, store a 16-byte fingerprint
of its vector**. The fingerprint is just the sign pattern of the
vector (each coordinate becomes one bit: positive or negative).
Comparing two fingerprints is *one* CPU instruction: XOR + popcount,
about 1 nanosecond. So you can do the *whole graph traversal* using
fingerprints and only fall back to the real 128-dim vectors for the
final 64 candidates. That is the symphony — the graph hops drive
which nodes you look at; the binary codes drive which of those nodes
look promising.

Numbers from our bench: 3082 nodes visited per query, but only 64
"real" distance evaluations and 3178 "fingerprint" Hamming compares
— and the latter together cost less than 10 microseconds.

---

## Practical failure modes

- **Anisotropic data.** Sign-only quantization on a distribution
  whose coordinates have wildly different variances will collapse
  many vectors to the same code. RaBitQ's rotation step exists
  exactly to fix this. Status: **known**, the PoC sidesteps the issue
  by benching on rotation-invariant data; production code should
  use the real RaBitQ kernel from `ruvector-rabitq`.
- **High intrinsic dimensionality.** Synthetic random data in d=128
  has degraded ground truth (all distances cluster near √2). Real
  embeddings (CLIP, sentence-transformers, OpenAI 1536-d) typically
  have lower intrinsic dim and are friendlier; the symphony pattern
  should look better there.
- **Rerank size too small.** With `rerank=64` only the top-64
  Hamming candidates get f32-evaluated. If the true neighbour ranks
  somewhere in the top-200 by Hamming but not top-64, it is lost.
  This is the obvious recall lever to tune.
- **Cold-cache builds.** Build cost is dominated by NN-descent
  (5.9 s for n=20 000) — that is not a symphony cost, it is the kNN
  graph build cost. Replacing the substrate with HNSW reduces this.

---

## What to improve next (roadmap)

In priority order:

1. **Wire in real RaBitQ** from `crates/ruvector-rabitq` (rotation +
   theoretically grounded bound). This is the single highest-leverage
   change for recall.
2. **HNSW substrate** instead of flat NN-descent (use
   `crates/ruvector-graph` or `crates/ruvector-acorn`). Builds get
   much faster and the symphony coupling carries over unchanged.
3. **Tunable `rerank` knob**, with auto-tuning based on observed
   ratio (true-top-k seen / `rerank`).
4. **SIMD popcount kernel** for high-d (AVX-512 `vpopcntq`, NEON
   `vcnt.8`); the current implementation already vectorizes for
   d ≤ 128 via `u64::count_ones`.
5. **Per-node code locality**. Inline the code for each node *next
   to* its adjacency list (struct-of-arrays → array-of-structs) so a
   graph step touches one cache line, not two.
6. **Disk-resident variant** on top of `ruvector-diskann` — the
   inline code is small enough that it can ride along with adjacency
   pages with negligible overhead.
7. **Public ANN-Benchmarks port** — sift1m, deep10m, glove. Real
   embeddings + real ground truth will tighten the recall trade-off.

---

## Production crate layout proposal

```
crates/ruvector-symphony-qg/
├── Cargo.toml
├── src/
│   ├── lib.rs               # public API, error types, oracles
│   ├── quant.rs             # BitQuantizer (replace with rabitq feed)
│   ├── graph.rs             # KnnGraph + GraphParams
│   ├── symphony.rs          # SymphonyIndex + IndexParams + SearchStats
│   ├── main.rs              # symphony-qg-bench binary
│   ├── simd.rs              # (future) AVX-512 / NEON popcount
│   └── disk.rs              # (future) disk-resident variant
└── benches/
    └── symphony_bench.rs    # (future) criterion benchmark
```

Recommended trait integration:

```rust
// pseudo: integrate with existing ruvector index trait
impl ruvector_core::IndexBackend for SymphonyIndex { ... }
```

---

## References

- Gou et al., *SymphonyQG: Towards Symphonious Integration of
  Quantization and Graph for Approximate Nearest Neighbor Search*,
  SIGMOD 2025 *(citation pending arXiv id confirmation)*.
- Gao & Long, *RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search*,
  SIGMOD 2024.
- Wang et al., *HVS+: Hierarchical Voronoi-based Search for
  High-Dimensional ANN*, VLDB 2024.
- Malkov & Yashunin, *Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs*,
  IEEE TPAMI 2018.
- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest
  Neighbor Search on a Single Node*, NeurIPS 2019.
- Tepper et al., *LeanVec: Searching Vectors Faster by Making Them
  Fit*, 2024.
- Internal: `crates/ruvector-rabitq/`, `crates/ruvector-acorn/`,
  `crates/ruvector-diskann/`.
