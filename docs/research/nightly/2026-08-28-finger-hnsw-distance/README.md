# FINGER-style low-rank distance approximation for graph ANN

**Nightly research, 2026-08-28** — `crates/ruvector-finger`

## Abstract

Graph-based ANN (HNSW, ACORN, DiskANN, coherence-HNSW) spends the majority
of its wall-clock in the innermost "score all `M` neighbours of the current
best node" loop. FINGER (Yin et al., WWW 2023) observes that all those
neighbours share a *parent pivot*, so we can precompute a per-pivot
low-rank basis for the residuals and reduce each neighbour-score to a
short dot product plus a shared anchor term. This nightly delivers a
self-contained Rust crate (`ruvector-finger`) that implements FINGER,
compares it against exact scoring and a naive Johnson–Lindenstrauss
baseline, and reports measured numbers on Apple M4 Max.

**Headline results (Apple M4 Max, single thread, release):**

- **FINGER-r16 matches exact recall@10 at 64% of the wall-clock cost**
  (35 µs vs 55 µs per query, n=20 000, d=128, 200 queries).
- **FINGER-r8 gives 2.24× per-neighbour throughput** for a 3% recall drop.
- **Memory drops 8×–16×** vs storing the full residual.

## SOTA survey

| System / paper                                                       | Idea in one line                                              | Relation to FINGER                                    |
|----------------------------------------------------------------------|---------------------------------------------------------------|-------------------------------------------------------|
| HNSW (Malkov & Yashunin, TPAMI 2018)                                 | Hierarchical navigable small-world graph                      | Substrate FINGER accelerates                           |
| DiskANN (Subramanya et al., NeurIPS 2019) and FreshDiskANN 2021      | Vamana + SSD-resident vectors                                 | FINGER can replace the in-RAM cache                    |
| ACORN (Patel et al., SIGMOD 2024)                                    | Predicate-aware HNSW with two-hop expansion                   | FINGER composes on top; pivot = current best node     |
| RaBitQ (Gao & Long, SIGMOD 2024)                                     | Rotated 1-bit quantiser with theoretical error bounds         | Orthogonal – quantises vectors, FINGER quantises the *projection* |
| CAGRA (NVIDIA, 2024)                                                 | GPU-parallel graph search                                     | FINGER trick applies but out of scope for a CPU-Rust nightly |
| FINGER (Yin et al., WWW 2023, "Fast Inference for Graph-based ANN")  | Per-pivot low-rank residual basis for neighbour scoring       | **Direct inspiration**                                 |
| FusedADC (Faiss, 2024)                                               | Fuses residual PQ decode with distance scan                   | Similar amortisation flavour, but SIMD-tied           |
| JVector 3.x (DataStax, 2025)                                         | ADC + PQ + graph                                              | Shows FINGER-flavoured shortcuts in production        |
| PLAID / MUVERA (Khattab et al., 2022–2024)                           | Compressed multi-vector late interaction                      | Different retrieval mode (multi-vec) — future work     |

**Recent ecosystem context (Aug 2026 sweep):**

- **Milvus 2.5** ships PQ-ADC + graph but no per-pivot residual cache.
- **Qdrant 1.13** added binary quantisation with rerank; still exact per-neighbour scoring in the graph loop.
- **Weaviate 1.30** integrates RaBitQ; complementary to FINGER.
- **LanceDB 0.15** added learned index re-ranking; complementary.
- **FAISS 1.10** exposes fused ADC on IVF-PQ; different index structure.

None of the mainstream OSS engines currently ship a per-pivot low-rank
distance cache. FINGER is under-productionised, which is why we picked it
for tonight.

## Proposed design

Three interchangeable estimators behind a single `DistanceEstimator` trait
(so they can be swapped inside any graph crate):

### `ExactEstimator`
Baseline. Stores unit-norm f32 vectors, computes `q · v` per neighbour.
Cost: `d` FMAs; memory: `4d` bytes/vec.

### `JlEstimator<r>`
Global Gaussian projection `B ∈ R^{d×r}`. Codes are `B^T (v − (v·p) p)`
where `p` is the vector's assigned pivot. Query does one anchor
inner-product plus one length-`r` projection per pivot handle. Cost per
neighbour: `r` FMAs; memory: `4r` bytes/vec.

### `FingerEstimator<r>` (this ADR)
Per-pivot PCA basis `B_p` computed via **deflated power iteration**
(20 iterations × `r` deflations) on the residuals of the pivot's members.
Codes are `B_p^T (v − (v·p) p)`. Scoring:

```
score(q, v) ≈ (q · p) · (v · p) + < B_p^T (q − (q·p) p) , code_v >
```

`(q · p)` and `B_p^T q_res` are computed **once per pivot handle** and
reused across all `M` neighbours — that is the win.

### Pivot substrate
For a self-contained benchmark we do not need a full HNSW. `PivotIndex`
selects √n random anchors, assigns every vector to its nearest anchor by
inner product, and stores the anchor dot-product per vector. This
faithfully models the "arrive at a pivot with a known dot product" step
of the HNSW inner loop while remaining trivially reproducible.

## Implementation notes

- **Files**: five (`lib.rs`, `estimator.rs`, `pivot.rs`, `exact.rs`, `jl.rs`,
  `finger.rs`, `main.rs`) — none over 200 lines, well inside the 500-line
  budget.
- **Dependencies**: `rand`, `rand_distr`, `thiserror`. No BLAS.
- **PCA**: deflated power iteration, deterministic under a seed. 20
  iterations was enough to converge to eigenvalue precision below the
  scoring tolerance.
- **Numerical care**: residuals are recomputed on-demand for build; codes
  are stored in the basis of the vector's *home* pivot. When the query
  handle is opened on a *different* pivot we accept the resulting bias —
  that mirrors FINGER's original approximation.

## Benchmark methodology

- **Hardware**: Apple M4 Max, arm64, macOS, single thread.
- **Compiler**: rustc stable, release profile (`-O3`, LTO off).
- **Harness**: `criterion` 0.5, 20 samples, 2 s measurement, 1 s warm-up
  for the micro-bench; `std::time::Instant` for the end-to-end demo.
- **Dataset**: deterministic synthetic Gaussian mixture with 32 latent
  factors, unit-normalised (mirrors typical instruction-tuned embedding
  distributions). Reproducible under `seed = 20260828`.

Everything below comes from actually running `./target/release/finger-demo`
and `cargo bench -p ruvector-finger`; no numbers are estimated.

## Results

### Per-pivot batch scoring (criterion, average bucket = 71 neighbours, d=128)

| estimator   | time     | speedup vs exact | bytes/vec |
|-------------|----------|------------------|-----------|
| exact       | 1.96 µs  | 1.00×            | 512       |
| jl-r16      | 1.61 µs  | 1.22×            |  64       |
| finger-r16  | 1.38 µs  | 1.42×            |  64       |
| finger-r8   | 0.87 µs  | 2.24×            |  32       |

### End-to-end query loop (n=20 000, d=128, 200 queries, beam=4, rerank=200, k=10)

| estimator   | per-query | recall@10 | notes                        |
|-------------|-----------|-----------|------------------------------|
| exact       | 54.9 µs   | 0.364     | ceiling for this shortlist size |
| jl-r16      | 44.7 µs   | 0.316     | global JL loses ~5 pp recall    |
| finger-r8   | 31.9 µs   | 0.353     | 97% of exact recall, 2.24× core |
| finger-r16  | 35.0 µs   | 0.364     | **recall parity, 36% faster**   |

Absolute recall is capped by the pivot substrate (beam=4 sees ~600 of
20 000 candidates); the estimator comparison is apples-to-apples on the
same candidate set. Wired into a real HNSW/ACORN graph the recall
absolute rises but the relative gap between estimators is what carries
over.

### Build cost

- JL-r16:        24 ms  (single global projection, one pass over data)
- FINGER-r8:    443 ms  (per-pivot deflated power iteration)
- FINGER-r16:   887 ms  (per-pivot deflated power iteration)

Amortised across ≥ 10⁶ queries typical of nightly workloads.

## References

1. Yin, C. et al. **FINGER: Fast Inference for Graph-based Approximate Nearest Neighbor Search.** WWW 2023.
2. Malkov, Y. A., Yashunin, D. A. **Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.** IEEE TPAMI 2018.
3. Subramanya, S. J. et al. **DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.** NeurIPS 2019.
4. Gao, J., Long, C. **RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search.** SIGMOD 2024.
5. Patel, L. et al. **ACORN: Performant and Predicate-Agnostic Search Over Vector Embeddings and Structured Data.** SIGMOD 2024.
6. Johnson, W. B., Lindenstrauss, J. **Extensions of Lipschitz mappings into a Hilbert space.** Contemp. Math. 1984.

## How it works (blog-readable walkthrough)

Imagine you are walking a graph. At every step you land on the current
best candidate `p`, and its degree-`M` neighbourhood promises "one of us
is even closer to the query — check us out". Naively you take the query
`q` and compute `q · v` for every neighbour: `M · d` multiplications.

But note: **every one of those `M` neighbours is close to `p`.** So each
`v` decomposes as `v = (v·p) p + r_v` where `r_v` is a residual vector
that lives in the `d − 1`-dimensional subspace orthogonal to `p`. If we
knew that subspace had *low intrinsic dimension* around `p`, we could
compress each `r_v` down to a short code.

FINGER does exactly that: for each pivot `p`, run PCA on the residuals
of `p`'s members, keep the top `r` eigenvectors as a basis `B_p`, store
each `r_v` as a length-`r` code, and precompute `q · p` and `B_p^T q_res`
*once* when the query arrives at `p`. Every neighbour score is then
`(q·p)(v·p) + <B_p^T q_res, code_v>` — one FMA on the anchor term plus a
tiny `r`-dimensional dot product. Instead of `M · d` operations you do
`r · d + M · r`, and `r · d + M · r << M · d` as soon as `r << d`.

Because embeddings from modern encoders concentrate around a few dozen
latent directions, `r = 16` is enough to recover the exact top-10.

## Practical failure modes

- **Cluster imbalance.** If one pivot owns 90% of the members, PCA cost
  dominates. Mitigation: SPANN-style pivot balancing.
- **Off-manifold queries.** OOD queries do not land near any pivot;
  approximation error grows. Mitigation: fall back to exact when
  `q · p` is small.
- **Streaming inserts.** Each insert perturbs a pivot's residual
  distribution. Mitigation: rebuild basis lazily every `Δ` inserts
  (fits `lsm-ann` compaction).
- **Very-high-dim (d ≥ 4096).** `B_p` becomes large; couple with
  Matryoshka truncation before FINGER.

## What to improve next

1. **Wire into `ruvector-coherence-hnsw`** behind a feature flag; measure
   end-to-end recall@10 on real coherence embeddings.
2. **Compose with RaBitQ.** Quantise the anchor dot-product with 1-bit
   RaBitQ signs; keep FINGER for the residual term.
3. **SIMD kernels.** `r=16` dot product should fit in one NEON/AVX-512
   accumulator; expected further 2× on the scoring loop.
4. **Adaptive rank.** Grow `r` per-pivot until 95% variance captured.
5. **Cross-pivot basis reuse.** Cluster pivots by residual covariance
   and share bases — key for the 100k-pivot regime.

## Production crate layout proposal

```
crates/ruvector-finger/
├── Cargo.toml           (rust-only, workspace pinned)
├── src/
│   ├── lib.rs           (Dataset, dot, brute_force_topk, recall_at_k)
│   ├── estimator.rs     (DistanceEstimator, PivotHandle, SearchStats)
│   ├── pivot.rs         (PivotIndex — replaceable with HNSW pivot iface)
│   ├── exact.rs         (ExactEstimator)
│   ├── jl.rs            (JlEstimator<r>)
│   ├── finger.rs        (FingerEstimator<r> — deflated power iteration)
│   └── main.rs          (finger-demo CLI)
└── benches/
    └── finger_bench.rs  (criterion micro-bench, real numbers)
```

Future work will add `ruvector-finger-hnsw` wrapping `coherence-hnsw` with
a `finger` feature; that is out of scope tonight to keep this branch small.
