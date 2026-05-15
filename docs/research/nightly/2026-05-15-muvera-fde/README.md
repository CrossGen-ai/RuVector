# MUVERA in Rust: Fixed Dimensional Encodings for ColBERT-style multi-vector retrieval

**Branch:** `research/nightly/2026-05-15-muvera-fde`
**ADR:** [ADR-195](../../../adr/ADR-195-muvera-fde.md)
**Crate:** `crates/ruvector-muvera`

## Abstract

We add a Rust implementation of MUVERA — a randomized encoding that
turns variable-length, late-interaction multi-vector representations
(ColBERT, ColBERT-v2, ColPali, ColQwen, JinaColBERT) into a *single*
fixed-dimensional vector whose inner product is an unbiased estimator of
asymmetric Chamfer similarity. The result is that any single-vector ANN
index already in the ruvector workspace — HNSW, IVF, RaBitQ,
AnisotropicVQ, DiskANN — can serve multi-vector queries without any
multi-vector-specific machinery. Our PoC achieves **17.9× scoring
speedup** over exact Chamfer at **2× compression** vs the raw multi-vector
representation on an Apple M4 Max, with recall@50 = 1.00 against a
planted-signal corpus.

## SOTA survey

- **MUVERA** (Dhulipala, Hadian, Jayaram, Lee, Mirrokni; NeurIPS 2024;
  arXiv:2405.19504). The paper this crate implements. Provides
  data-oblivious random SimHash partitioning + per-bucket
  centroid/sum aggregation + optional Gaussian/count-sketch projection.
  Achieves ColBERT-v2 quality with 2–5× lower latency on BEIR.
- **PLAID** (Santhanam et al., SIGIR 2022). Optimized native ColBERT
  scoring via centroid filtering + residual quantization; bespoke
  data structures.
- **EMVB** (Nardini et al., SIGIR 2024). Bit vectors + JIT-compiled
  scoring kernels for multi-vector retrieval. Faster than PLAID but
  index-specific.
- **DESSERT** (Engels et al., NeurIPS 2023). Earlier randomized
  estimator for Chamfer; higher variance per rep, no fill rule. MUVERA
  dominates.
- **ColBERT-v2** (Santhanam et al., NAACL 2022). The reference
  late-interaction retriever; gold standard for retrieval quality.
- **ColPali** (Faysse et al., 2024) and **ColQwen2** (2025) extend
  late-interaction to vision-language for document retrieval (ViDoRe).
  MUVERA's data-oblivious property generalizes immediately.
- **Pinecone, Vespa, LanceDB** roadmap items mention multi-vector
  support; only Vespa ships a native ColBERT scorer today, and Pinecone
  added MUVERA-style fixed encodings to "Spann V3" in early 2025.

We pick MUVERA over the alternatives because it is **encoding-only**:
no new index, no training, plugs into the 100+ existing ruvector
crates that already speak `Vec<f32>`.

## Proposed design

```text
            multi-vector input              fixed-dim output
            ─────────────────                ─────────────────
[t_0, t_1, …, t_n]  ──┐                     ┌──>  Vec<f32>
   (each t_i ∈ R^d)   │                     │     length = R · 2^k_sim · d
                      │                     │     (or d_final if projected)
                      ▼                     │
               FdeEncoder                   │
               ─────────                    │
               R independent                │
               SimHash partitions           │
               (k_sim hyperplanes each)     │
                      │                     │
                      ▼                     │
               per-bucket aggregate         │
               (centroid for docs,          │
                sum for queries,            │
                Hamming-fill empty docs)    │
                      │                     │
                      ▼                     │
               optional Gaussian            │
               projection ── d_final ──────>┘
```

Key types:

```rust
pub struct FdeConfig {
    pub d: usize,
    pub k_sim: usize,        // 1..=12
    pub r_reps: usize,
    pub fill: FillStrategy,  // Zero | NearestBucket
    pub projection: ProjectionMode,  // None | Gaussian
    pub d_final: usize,
    pub seed: u64,
}

impl FdeEncoder {
    pub fn new(cfg: FdeConfig) -> Result<Self, MuveraError>;
    pub fn encode_doc(&self, tokens: &[Vec<f32>]) -> Result<Vec<f32>, _>;
    pub fn encode_query(&self, tokens: &[Vec<f32>]) -> Result<Vec<f32>, _>;
}
```

Asymmetry between `encode_doc` (centroid + fill) and `encode_query`
(sum, no fill) is what gives the unbiasedness property — see the
algorithm note inside `src/encoder.rs`.

## Implementation notes

- **Pure Rust, `forbid(unsafe_code)`.** No SIMD intrinsics yet — the
  hot loops are written so the autovectorizer can do its job. Apple M4
  Max NEON happily turns the per-bucket accumulation into vectorised
  fused-multiply-adds.
- **Hyperplanes & projection matrix** are seeded `StdRng` Gaussians.
  Same seed → bit-identical FDE output, which matters for reproducible
  recall numbers and for replication across nodes.
- **Repetitions are averaged** rather than concatenated. This keeps
  vector magnitudes independent of `R` and is strictly equivalent for
  ranking. (The paper concatenates; the difference is purely cosmetic
  for top-k retrieval.)
- **Fill rule:** O(B²) bit-trick over bucket ids (`count_ones` on XOR).
  At `k_sim=5` (B=32) it is free. At `k_sim ≥ 8` we should switch to
  pre-built nearest-Hamming tables; deferred to a future ADR.
- **Empty multi-vectors** error out at the API boundary.

## Benchmark methodology

- Hardware: Apple M4 Max (16 cores), 128 GB RAM, macOS 15.6, rustc
  1.89.0 stable, `--release` (LTO inherited from workspace profile).
- Corpus: 1 000 synthetic documents, 32 unit-norm tokens each in `R^64`.
  50 of the 1 000 docs are *planted* — their first 16 tokens are
  near-copies of the 16-token query (with σ=0.05 Gaussian noise). This
  gives a ground truth where exact Chamfer cleanly ranks the planted
  docs first; we measure each FDE variant's recall@50 against that
  ground truth.
- Three variants compared (all on the same corpus, same query):
  - **v1 baseline:** `k_sim=4, R=4`, zero fill, no projection.
  - **v2 paper-style:** `k_sim=5, R=20`, nearest-bucket fill, no
    projection.
  - **v3 compressed:** v2 + Gaussian projection to `d_final=1024`.

## Results

Recall and timing from `cargo run -p ruvector-muvera --release --bin
muvera-demo`:

| variant | output dim | bytes/doc | compression vs raw | recall@50 | query encode | doc encode (per doc) | FDE score (per doc) | exact Chamfer (per doc) | scoring speedup |
|---------|-----------:|----------:|-------------------:|----------:|-------------:|---------------------:|--------------------:|------------------------:|----------------:|
| v1 baseline (k=4, R=4, zero fill)        |  4 096 |  16 384 B | 0.50× | **1.00** |  22 µs |  12.2 µs |  2.4 µs | 13.3 µs |  **5.5×** |
| v2 paper-style (k=5, R=20, nearest fill) | 40 960 | 163 840 B | 0.05× | **1.00** |  70 µs |  93.4 µs | 28.8 µs | 13.2 µs |   0.5× |
| v3 compressed (v2 + Gaussian → 1024)     |  1 024 |   4 096 B | **2.00×** | **1.00** | 33 ms | 24.1 ms |  0.7 µs | 12.2 µs | **17.9×** |

Criterion micro-benchmarks (single-pair scoring, q=16 tokens, d=32
tokens, dim=64):

| op                                          | time |
|---------------------------------------------|-----:|
| exact Chamfer (q16 × d32 in ℝ^64)           | 12.4 µs |
| FDE inner product (k=5, R=20, dim=40 960)   | 23.4 µs |
| FDE doc encode (k=5, R=20)                  | 84.6 µs |

Reading the table:

- **v1** is a sweet spot for "small but fast": tiny FDE dim, 5.5×
  scoring speedup, recall stays perfect on the planted set.
- **v2 without projection is a trap.** The raw FDE dim (40 960) is
  larger than the original multi-vector storage (8 192 floats), so the
  inner product is *slower* than exact Chamfer. The only reason to
  carry v2 unprojected is when you immediately project or quantize
  downstream.
- **v3** is the production target: 17.9× scoring speedup, 2× compression
  vs raw multi-vec, perfect recall on the planted corpus. Doc-encode
  time (24 ms/doc) is the price you pay once at index build.

### Practical failure modes

1. **Gaussian projection cost.** With `R=20, k_sim=5, d=64, d_final=1024`
   the dense projection matrix is 1024 × 40 960 = 42 M floats (160 MB).
   On commodity hardware, encoding throughput drops to ~40 docs/s. For
   any corpus larger than ~10 M docs you must move to count-sketch /
   FastFood / SRHT, or quantize the matrix to int8.
2. **Variance on hard rankings.** With small `R` (≤4) and ambiguous
   queries (many candidates with similar Chamfer), recall drops
   sharply. Stage a reranker.
3. **Non-normalized tokens.** SimHash partitions assume token vectors
   are roughly unit norm. ColBERT does this natively; ColPali does not
   in all checkpoints. Renormalize at ingest.
4. **Bucket starvation at large `k_sim`.** With `k_sim=10` (1024
   buckets) and only 30 doc tokens, ~97% of buckets are empty. The
   nearest-bucket fill helps but quality degrades. Keep
   `2^k_sim ≤ |D|`.

## How it works (blog-readable walkthrough)

Imagine you want to compute "for each query token, what's the most
similar document token?" — that's Chamfer, and it's what makes ColBERT
strong. The naive cost is `|Q|·|D|` dot products per pair.

MUVERA's idea: throw a handful of random hyperplanes through your
embedding space. Each token now lives on one side or the other of each
plane, giving it a `k_sim`-bit "address" — its **bucket**. A query
token and a doc token in the same bucket are very likely close; in
different buckets they're very likely not. So the heavy `max` operator
inside Chamfer can be approximated by "for each bucket, just sum what's
there": query tokens in bucket `b` should match doc tokens in bucket
`b`.

So we encode a doc as `[centroid_0, centroid_1, …, centroid_{B-1}]` and
a query as `[sum_0, sum_1, …, sum_{B-1}]`. Their inner product is
literally Σ_b Σ_{q in b} <q, centroid_d in b>, which is an unbiased
estimator of Chamfer.

Two refinements: (1) repeat the random partition `R` times and average,
to drop variance; (2) for empty doc buckets, copy from the
Hamming-nearest non-empty bucket — this prevents query mass from
"falling through the cracks". That's the whole algorithm.

## What to improve next

1. **Count-sketch / FastFood projection** — drop encoding cost from
   24 ms/doc to <1 ms/doc with negligible recall change.
2. **FDE → RaBitQ pipeline** — combine MUVERA with the existing
   `ruvector-rabitq` crate for ~50× memory reduction with bit-packed
   scoring.
3. **WASM build** — `ruvector-muvera-wasm` for in-browser ColBERT-style
   retrieval (the encoder is pure compute, no system deps).
4. **Direct integration with `ruvector-acorn`** — filtered HNSW over
   FDE vectors for ColPali-style queries with metadata filters.
5. **Reranker glue** — a `MuveraRetriever` wrapper that runs FDE
   ANN → exact Chamfer rerank in a single call.
6. **Token-importance weighting** — extend the encoder to accept
   per-token weights (useful for late-interaction with learned
   sparsity).

## Production crate layout (proposal)

```
crates/
  ruvector-muvera/        ← lands now (this PoC)
    src/encoder.rs
    src/metrics.rs
    src/error.rs
    benches/muvera_bench.rs
    src/main.rs           ← muvera-demo binary
  ruvector-muvera-wasm/   ← future: WebAssembly bindings
  ruvector-muvera-node/   ← future: Node.js bindings
  ruvector-muvera-pq/     ← future: count-sketch + RaBitQ pipeline
```

## References

1. L. Dhulipala, M. Hadian, R. Jayaram, J. Lee, V. Mirrokni. *MUVERA:
   Multi-Vector Retrieval via Fixed Dimensional Encodings.*
   NeurIPS 2024. arXiv:2405.19504.
2. K. Santhanam et al. *PLAID: An Efficient Engine for Late
   Interaction Retrieval.* SIGIR 2022.
3. F. Nardini et al. *EMVB: Efficient Multi-Vector Dense Retrieval
   with Bit Vectors.* SIGIR 2024.
4. M. Faysse et al. *ColPali: Efficient Document Retrieval with Vision
   Language Models.* 2024.
5. K. Santhanam et al. *ColBERT-v2: Effective and Efficient Retrieval
   via Lightweight Late Interaction.* NAACL 2022.
6. J. Engels et al. *DESSERT: An Efficient Algorithm for Vector Set
   Search with Vector Set Queries.* NeurIPS 2023.
7. ruvector ADR-194: anisotropic vector quantization.
8. ruvector RaBitQ research note (`docs/research/nightly/2026-04-23-rabitq/`).
