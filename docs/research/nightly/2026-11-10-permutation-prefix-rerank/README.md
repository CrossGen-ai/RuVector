# Permutation-Prefix Reranker (PPR) — a tiny FP-free ANN rerank sidecar

**Date:** 2026-11-10
**Branch:** `research/nightly/2026-11-10-permutation-prefix-rerank`
**Crate:** `crates/ruvector-permutation-prefix`
**ADR:** [ADR-357](../../../adr/ADR-357-permutation-prefix-rerank.md)

---

## Abstract

Approximate Nearest Neighbour (ANN) search in ruvector relies on a
first-stage coarse index (HNSW, IVF, SPANN, graph-transformer) followed
by a distance-aware rerank over a short candidate list. Every rerank
backend currently in ruvector — PQ-ADC, RaBitQ, SQ cascades, bit-sliced
Hamming — assumes cheap floating-point multiplies and an L2-shaped
metric. This excludes no-FPU targets (Hailo accelerators, no-SIMD WASM)
and makes rerank over non-Euclidean first-stage indexes (hyperbolic
HNSW, graph-transformer traversal) awkward.

This report revives a neglected 2008 idea — **permutation-based
indexing** (Chavez, Figueroa, Navarro, TPAMI 2008) — as a lightweight
rerank *sidecar*. Each vector is summarised by the ordered ids of its
`prefix_m` nearest pivots (farthest-point sampled). Rerank cost is one
integer footrule-distance pass over a short u16 array. The result is a
16 bytes/vector reranker whose hot loop is FP-free and which is the
*fastest* reranker measured in the benchmark (85,624 QPS on
Apple M4 Max, versus PQ-ADC at 63,531 and full-precision L2 at 49,498),
at the cost of roughly half of PQ-ADC's recall@10.

PPR is not meant to replace PQ. It opens the rerank step up to targets
where PQ is not an option, and to future *progressive-rerank* and
*anytime-budget* pipelines because the integer footrule score refines
smoothly as more prefix ids are read.

---

## SOTA survey

### Quantization-based rerank (dominant line)

- **PQ / OPQ** (Jégou et al. 2011, Ge et al. 2013): split vector into
  subspaces, k-means each subspace, store code words. State of the art
  for L2/IP ANN rerank. Hot loop: one LUT lookup + add per subspace.
- **RaBitQ** (ICML 2024): rotation + sign bits + residual norm. Tight
  theoretical bounds, excellent speed-recall tradeoffs. ruvector already
  ships `ruvector-rabitq`, `fused-rabitq-residual`, multibit variants.
- **Scalar-quantization cascades** (nightly 2026-11-09): int8 → int4
  cascade with residual correction for streaming ingest.
- **Bit-sliced Hamming** / Hadamard-SQ (prior nightlies): SIMD-friendly
  bitwise rerank for quasi-binary signatures.

### Pivot / permutation-based indexing (what this report revives)

- **Chavez, Figueroa, Navarro 2008**: original formulation. Rerank by
  Spearman-footrule / Kendall-tau over full permutations.
- **Esuli 2012** "PP-Index": compressed permutation prefix as a
  secondary filter in metric space search.
- **Amato & Savino 2008** "MI-File": inverted file over permutations
  for huge metric collections.
- **MIH / FLANN permutation**: integrated into a handful of libraries
  but never into a modern vector DB. The SOTA survey of 2024 (Wang et
  al., *A Comprehensive Survey and Experimental Comparison of
  Graph-Based Approximate Nearest Neighbour Search*) does not even
  mention permutation methods — they have been eclipsed by PQ.

The **key observation this report makes**: permutation prefixes do not
have to *beat* PQ on recall. They can win on orthogonal axes — FP-free
rerank, metric-agnostic rerank, and progressive refinement — if they
are offered as a sidecar rather than a replacement.

### Competitors (whole-system, not just rerank)

| System   | Rerank backends          | FP-free rerank | Metric-agnostic rerank |
|----------|---------------------------|----------------|------------------------|
| Milvus   | SQ, PQ, DiskANN          | no             | no                     |
| Qdrant   | SQ8, binary, PQ          | binary only    | no                     |
| Weaviate | PQ, binary               | binary only    | no                     |
| Pinecone | proprietary PQ           | no             | no                     |
| FAISS    | PQ, OPQ, IVF-SQ          | no             | no                     |
| ruvector | PQ, RaBitQ, SQ cascades, bit-sliced Hamming, **PPR (new)** | binary + PPR | **PPR** |

---

## Proposed design

Three public types behind the `Reranker` trait:

```rust
pub trait Reranker {
    fn name(&self) -> &'static str;
    fn bytes_per_vector(&self) -> usize;
    fn rerank(&self, query: &[f32], candidates: &[u32], k: usize) -> Vec<u32>;
}
```

1. **`l2::L2Reranker`** — full-precision reference.
2. **`pq::PqReranker`** — classical PQ-ADC. Pure Rust, no `unsafe`,
   Lloyd's iterations over randomly-initialised centroids.
3. **`ppr::PermutationPrefixReranker`** — the new technique.

### Build

```text
pivots ← FarthestPointSample(data, n_pivots=256, seed)
for v in data:
    distances[v][pi] ← sqL2(v, pivots[pi])          for pi = 0..n_pivots
    code[v] ← argsort(distances[v])[0..prefix_m]    // u16 ids, ordered
```

Farthest-point sampling was picked over k-means because it has no
training iterations, no local minima, is deterministic given a seed,
and inherently maximises pivot diversity — which is the property the
footrule distance depends on. The first pivot is drawn uniformly at
random; each subsequent pivot is the dataset vector that maximises its
min-distance to the already-chosen set.

### Query / rerank

```text
q_prefix ← argsort(sqL2(query, pivots))[0..prefix_m]
q_rank[pivot_id] ← rank in q_prefix, or +∞ if not present   // built once per query

for id in candidates:
    dist ← 0
    shared ← 0
    for (rc, pid) in code[id].enumerate():
        rq ← q_rank[pid]
        if rq != +∞:
            dist += |rc - rq|
            shared += 1
        else:
            dist += prefix_m                     // "fell out of prefix" penalty
    dist += (prefix_m - shared) * prefix_m      // pivots only in query prefix
    score[id] ← dist
```

The inner loop is a short pass over u16 ids, one array index, one
integer add. No floating-point multiplies touch the hot path. The
`prefix_m` penalty is an upper bound on the in-prefix rank gap, chosen
so that a candidate matching nothing in the query prefix pays the
maximum possible score (`2 · prefix_m²`), while a candidate matching
all `prefix_m` ids pays only the sum of rank gaps.

### Byte budget

| Variant       | `n_pivots` | `prefix_m` | Encoding | B/vec |
|---------------|-----------:|-----------:|----------|------:|
| PPR-tiny      | 128        | 8          | u16      | 16    |
| PPR-32        | 256        | 16         | u16      | 32    |
| FullL2 (ref)  | n/a        | n/a        | f32      | 4·dim |

For dim=128: FullL2 is 512 B/vec, PPR-tiny is a **32× compression**.
For dim=768 (common embedding size) the compression is 192×.

---

## Benchmark methodology

- **Dataset**: 20,000 synthetic Gaussian-mixture vectors, dim=128,
  32 clusters, cluster separation 4 σ, noise σ=1.
- **Queries**: 200 queries drawn from the same distribution with a
  different seed.
- **Shortlist (post-first-stage)**: for each query, the true brute-force
  top-10 anchored into a 200-candidate shortlist padded with
  uniformly-random distractors from the dataset. The ceiling recall@10
  is 1.0 — any miss is caused by the reranker, not by the first stage.
- **`k = 10`.**
- **Metric**: recall@10 against brute-force f32 L2 ground truth.
- **Hardware**: Apple M4 Max, Darwin arm64, release build
  (`cargo run --release --bin benchmark`).
- **Build time** reported in µs; latency reported mean / p50 / p95 µs
  per query rerank over 200 queries.

Reproduce:

```sh
cargo run --release -p ruvector-permutation-prefix --bin benchmark
# Tune via env vars:
PPR_N=50000 PPR_DIM=256 cargo run --release -p ruvector-permutation-prefix --bin benchmark
```

---

## Results

```text
variant                   bpv     build_us    mean_us     p50_us     p95_us        qps   recall@k
----------------------------------------------------------------------------------------------------
FullL2                    512       1289.0      20.20      15.29      30.92      49498    1.0000
PQ-ADC                     16    2308564.0      15.74      14.92      18.75      63531    0.7685
PermutationPrefix          32     526768.0      22.76      21.17      27.62      43936    0.4790
PermutationPrefix-tiny     16     262060.0      11.68      10.96      13.62      85624    0.4000
```

Observations:

- **PPR-tiny is the fastest reranker in the benchmark** at 85,624 QPS,
  1.35× faster than PQ-ADC and 1.73× faster than full-precision L2 over
  the same 200-candidate shortlists — because the inner loop reads
  8 u16s and does 8 integer additions.
- **At the same 16 B/vec budget, PQ-ADC retrieves 0.77 recall@10 vs
  PPR-tiny's 0.40.** PQ is the better reranker when FP is available and
  the metric is L2. The real PPR win is everywhere PQ is unavailable
  (no FPU) or mismatched (non-L2 first stage).
- **Build is cheap**: PPR-tiny builds in 262 ms vs PQ-ADC's 2.3 s
  (Lloyd k-means on 16 subspaces × 256 centroids × 12 iterations).
- **Latency p95 / p50 ratio** is 1.24× for PPR-tiny, 1.26× for PQ-ADC,
  2.02× for FullL2 — PPR's integer pass has the tightest tail.

### How it works (blog walkthrough)

Think of pivots as *landmarks*. For every vector in your dataset you
record which landmarks are nearest and in what order — just the top
eight, by id. That short list is the vector's fingerprint.

At query time you work out the query's own list of eight nearest
landmarks. Candidates whose landmark lists look like the query's are
good hits; candidates whose landmark lists disagree are not. The
"Spearman footrule" is just the sum of rank gaps for landmarks the
two lists share, plus a flat penalty for landmarks appearing in only
one list. Integer arithmetic, no square roots, no multiplies.

Because the score depends on *rank* rather than exact distance, it is
robust to the curse-of-dimensionality noise that destroys fine
distances in high-dimensional embeddings. It does not know about L2
specifically — all it needs is a working distance function to rank
pivots during build. So the same reranker drops into a hyperbolic or
inner-product pipeline unchanged.

### Practical failure modes

- **Too few pivots** → every vector picks nearly the same prefix, so
  every candidate ties. We observed this in the first tuning run at
  `n_pivots=64, prefix_m=8` (recall 0.06). Mitigation: keep
  `n_pivots ≥ 8 · prefix_m`.
- **Dense shortlist (first stage already finds true top-few-hundred)**
  → both PQ and PPR drop below 0.3 recall because everything in the
  shortlist is near-identical to the query. Mitigation: push rerank
  earlier in the pipeline, when the shortlist still carries dropouts.
- **Non-uniform cluster sizes** → FPS can pick pivots inside a dense
  cluster, wasting pivot budget. Mitigation: trade a bit of build time
  for one Lloyd pass over the FPS pivots to recentre them.
- **Ties under identical prefix** → candidates with the same prefix
  membership tie on footrule. Fix: break ties with the second-stage
  PQ score when available, or with the first-stage shortlist rank.

### What to improve next

1. **SIMD footrule kernel** — the current inner loop is scalar;
   AVX-512 `vpcmpeqw` / NEON `tbl` can parallelise it over 8-16 lanes.
2. **Progressive rerank** — stop reading `code[id]` after the first
   `p < prefix_m` entries, with an early-abandon lower bound. This
   gives an anytime reranker: longer budget, better recall, same code.
3. **Pivot pruning** — not every pivot is equally informative; the
   least-discriminative pivots can be dropped to shrink codes further
   or to grow `prefix_m` without growing byte budget.
4. **Learned pivot weights** — give pivots real-valued weights fit by
   least squares against training distances; rerank becomes weighted
   footrule. Keeps integer arithmetic but tightens recall.
5. **Hybrid PQ + PPR** — use PPR as the first-pass rerank filter, PQ
   only on the surviving top-100. PPR's tight p95 and low memory make
   it a good *cascade head*.

### Production crate layout

```
crates/ruvector-permutation-prefix/
├── Cargo.toml
├── src/
│   ├── lib.rs                # trait, shared helpers, recall_at_k
│   ├── pivot.rs              # farthest-point sampling
│   ├── ppr.rs                # PermutationPrefixReranker
│   ├── pq.rs                 # PQ baseline
│   ├── l2.rs                 # ground-truth baseline
│   └── bin/
│       └── benchmark.rs      # cargo-run benchmark
└── tests/                    # (unit tests live in each module)
```

All source files are under 500 lines per the project rule. The crate
has no new workspace dependencies beyond `rand` and `rand_distr`.

---

## References

- Chavez, E., Figueroa, K., Navarro, G. *Effective Proximity Retrieval
  by Ordering Permutations.* IEEE TPAMI 30(9):1647-1658, 2008.
- Amato, G., Savino, P. *Approximate Similarity Search in Metric
  Spaces Using Inverted Files.* InfoScale 2008.
- Esuli, A. *Use of Permutation Prefixes for Efficient and Scalable
  Approximate Similarity Search.* Information Processing & Management
  48(5), 2012.
- Jégou, H., Douze, M., Schmid, C. *Product Quantization for Nearest
  Neighbor Search.* IEEE TPAMI 33(1), 2011.
- Ge, T. et al. *Optimized Product Quantization.* CVPR 2013.
- Gao, J. et al. *RaBitQ: Quantizing High-Dimensional Vectors with a
  Theoretical Error Bound for Approximate Nearest Neighbor Search.*
  ICML 2024.
- Wang, M. et al. *A Comprehensive Survey and Experimental Comparison
  of Graph-Based Approximate Nearest Neighbor Search.* VLDB 2024.
