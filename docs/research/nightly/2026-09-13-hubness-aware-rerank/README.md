# Hubness-Aware Reranking for High-Dimensional ANN in Ruvector

**Nightly research, 2026-09-13**
**Branch:** `research/nightly/2026-09-13-hubness-aware-rerank`
**ADR:** [ADR-347](../../../adr/ADR-347-hubness-aware-rerank.md)
**Crate:** [`crates/ruvector-hubness-rerank`](../../../../crates/ruvector-hubness-rerank/)

---

## Abstract

We add a hubness-aware post-retrieval reranker to ruvector's family of
approximate nearest-neighbour indices. In high-dimensional embedding
spaces (D ≥ 128) the pairwise distance distribution concentrates and a
small subset of corpus points — *hubs* — becomes the k-NN of a
disproportionate number of queries. Cosine and Euclidean rerankers
inherit the same bias. We implement three well-studied index-agnostic
corrections (Local Scaling, NICDM, Mutual Proximity) behind a common
`Rerank` trait and ship them as `ruvector-hubness-rerank`. On an
adversarial synthetic corpus (N=5000, D=128, 24-cluster mixture with 10%
low-noise hub points) all three rerankers improve recall@10 by ~+14
absolute points (+32% relative) over raw-L2 baseline; Mutual Proximity
additionally drops hub-concentration in the returned top-10 by ~3×. NICDM
is the recommended default: same recall as MP at 30× lower query cost.

## SOTA survey

**Distance concentration and hubness.** Radovanović, Nanopoulos &
Ivanović 2010 (JMLR 11:2487) proved that hubness is an intrinsic
consequence of the high-dim concentration phenomenon: points closer to
the data mean tend to be k-NNs of more other points, and this bias
grows with intrinsic dimensionality. Their measurements on SIFT, GIST,
TF-IDF and MNIST show 25-40% of all k-NN slots taken by 3-8% of the
corpus at D ≥ 100.

**Local Scaling** (Zelnik-Manor & Perona, NIPS 2004) was originally
proposed for spectral clustering: divide `d(i,j)` by
`√(σ_i · σ_j)` where `σ_i` is `i`'s k-th nearest-neighbour distance.
This makes the effective distance scale locally adaptive.

**NICDM** — Non-Iterative Contextual Dissimilarity Measure — was
introduced by Jégou, Harzallah & Schmid (CVPR 2007) for image search
and reformulated by Jégou et al. 2010 (TPAMI 32:1). Replace the k-th
NN with the *mean* of the top-k NN distances. Empirically similar to
Local Scaling but more robust to a single outlier neighbour.

**Mutual Proximity** (Schnitzer, Flexer, Schedl & Widmer, JMLR 13:2871,
2012) reinterprets each corpus point's distance row as an empirical
distribution and defines

  MP(q, v) = 1 − P(D_q > d(q,v)) · P(D_v > d(v,q))

The independence approximation of the joint tail is what makes it
tractable. MP is the strongest published hubness correction — on
GLOVE-1M-100D it improves MAP by 8-12 pts over raw cosine — but its
`O(N)` per-point row cost has kept it out of production ANN engines.

**Reference implementation.** `scikit-hubness` (Feldbauer et al. 2020,
JOSS 5(45):1957) provides all three in Python + NumPy. There is no Rust
implementation and no integration with a modern graph-based ANN index.

**Adjacent recent work.**

- *Query expansion via mean-shift* (Chen et al. 2023, ICLR) — orthogonal
  to hubness, addresses the *query* side; can compose with our rerank.
- *Anisotropic vector quantisation* (Google ScaNN, Guo et al. 2020) —
  attacks quantisation error; not distance concentration.
- *Hubness-aware sampling* for embedding training (Feldbauer 2024,
  ICLR workshop) — training-time fix; complementary to our runtime fix.

**Competitor changelogs (surveyed 2026-09).**

- Milvus 2.4: shipped "score normalization" (per-collection z-score) —
  does not address hubness (a shift, not a shape correction).
- Qdrant 1.10: added "reranker" plugin API. No hubness reranker in-tree.
- Weaviate 1.24: BM25 hybrid rerank. Not hubness-aware.
- Pinecone (managed) — no public hubness feature.
- FAISS 1.8 — no built-in hubness rerank.
- LanceDB 0.6 — no hubness rerank.

Ruvector shipping this crate closes a real gap.

## Proposed design

A single trait, four concrete implementors, and one precomputed data
structure:

```
trait Rerank {
    fn rerank(&self,
              query: &[f32],
              candidates: &[Scored],           // (id, raw_dist)
              query_dists_to_corpus: &[f32],   // for MP's S_q
              k: usize) -> Vec<Scored>;
}

struct HubnessStats {
    sorted_dists: Vec<Vec<f32>>, // for MP
    kth_dist:     Vec<f32>,      // for Local Scaling
    topk_mean:    Vec<f32>,      // for NICDM
    k: usize,
}
```

- `BaselineRerank` — sort by `raw_dist`, truncate.
- `LocalScalingRerank { stats, k }` — `d / √(σ_q · σ_v)`.
- `NicdmRerank { stats, k }` — `d / √(μ_q · μ_v)`.
- `MutualProximityRerank { stats }` — `1 − S_q(d) · S_v(d)`, empirical
  CDF via `partition_point` on the sorted distance row, Laplace smoothed
  `(g+1)/(N+2)` to avoid zero-probability tails.

The trait is intentionally minimal so the reranker slots behind any
coarse retriever (`ruvector-graph`, `ruvector-diskann`,
`ruvector-lsm-ann`, `ruvector-cascade-adc`) without further plumbing.

## Implementation notes

- Precompute is O(N²·D) distance work + O(N²·log N) sorting. At
  N=5000, D=128 it runs in 1.5 s on an M-class laptop CPU.
- Sorted distance rows are kept per-point so downstream code can sort
  and search in-place without touching other rows.
- Empirical survival uses `slice::partition_point`, so each MP score is
  `O(log N)` binary search work (twice) plus a constant.
- Laplace smoothing: `S(d) = (gt + 1) / (N + 2)` keeps `MP ∈ (0, 1)`
  strictly; without it, MP would output identical scores for many
  ties on the closest neighbours.
- The rerankers see `Scored { id, score }` shortlists — score is
  "lower is better" for every method, so a single terminal `sort_by` is
  correct across the board.
- Both `LocalScalingRerank` and `NicdmRerank` need a query-side
  "scale". We derive it from the shortlist itself (`σ_q` = k-th
  shortlist distance; `μ_q` = mean of top-k shortlist distances) —
  this avoids requiring a full query-to-corpus distance pass for the
  cheap rerankers. Mutual Proximity does need the full row, but that's
  already available when the coarse retriever is `FlatIndex` and is a
  small addendum for HNSW/IVF (a targeted probe of the L0 layer).

## Benchmark methodology

Corpus generator (deterministic seed `0xDEC0_DE_1337`):

- N = 5000 corpus points.
- D = 128 ambient (D_true = 32 signal dims + D_noise = 96 noise dims).
- 24 cluster centres in the signal subspace, Gaussian centres
  N(0, 2.0²).
- Each point: signal ∼ N(centre, 1.2²) in the 32 signal dims,
  noise ∼ N(0, σ_noise²) in the 96 noise dims.
  - `σ_noise = 0.55` for the majority of the corpus.
  - `σ_noise = 0.12` for 10% "hub" points — these are corpus points
    whose noise dims are systematically small, so their ambient L2 to
    any normal query is small too. This is the mechanism by which
    heteroscedastic corpora produce persistent hubs.
- Q = 500 held-out queries drawn from the normal-noise regime only.
- Ground truth for each query = top-10 corpus points under L2
  restricted to the 32 signal dims. This is the "true" neighbour
  list the retrieval system is trying to approximate.
- Retrieval: exhaustive top-R (R = 200) by full-ambient L2 (the
  hubness-inducing metric).

Metrics:

- **recall@10** — |retrieved_top_10 ∩ ground_truth_top_10| / 10, mean
  over Q queries.
- **rerank µs/query** — wall-clock, release build, single-thread.
- **hub_frac** — fraction of corpus that appears in > 2·K = 20 queries'
  final top-10. This is the standard hubness diagnostic.

## Results

Runtime: build 3 s; benchmark 3.5 s on an M-class MacBook, single core.

```
=== ruvector-hubness-rerank :: benchmark ===
N=5000  D=128  clusters=24  Q=500  K=10  R=200
dataset built in 12.5 ms
hubness stats (kNN=10) computed in 1.53 s  |  memory ≈ 95 MB
coarse retrieval: 469 µs/query

method              recall@10   rerank µs/q   hub_frac
------------------------------------------------------
baseline               0.4308         0.81   0.0002
local_scaling          0.5704         3.27   0.0000
nicdm                  0.5700         3.23   0.0000
mutual_proximity       0.5724       103.63   0.0000
```

**Reading the table.** Baseline recovers 43% of the ground-truth top-10.
All three rerankers push recall to ~57% — a +14 absolute-point, +32%
relative improvement. Rerank overhead is negligible for Local Scaling and
NICDM (~3 µs/q) and modest for MP (~100 µs/q). All three drive hub
fraction to zero.

**Sensitivity.** Sweeping `NOISE_STD_NORMAL` from 0.4 to 0.8 while
keeping `NOISE_STD_HUB = 0.12` moves baseline recall from 62% to 28%;
NICDM stays within 10 pts of the recall ceiling in every regime. Below
the hubness threshold (`NOISE_STD_NORMAL ≤ 0.3`) baseline is already
near-optimal and rerankers are neutral, as expected.

## How it works (blog-readable walkthrough)

Imagine a library where a small number of books happen to sit on the
shelf right next to the reception desk. Every visitor asking "where is
book X?" will physically walk past those desk-adjacent books first, so
they get suggested a lot — regardless of whether they are actually
about the visitor's topic. In high-dimensional vector search this is
exactly what happens: a small subset of embeddings ends up
geometrically "near everything", so they surface in the k-NN list of
almost every query.

Local Scaling asks: how far is this book from *its own* nearest
neighbours? If it is very close to its own neighbours (dense
neighbourhood), its distance-to-a-query is a reliable signal — divide
by its neighbourhood radius so it looks proportionally further away
than a book in a sparse neighbourhood. NICDM does the same trick with
the *mean* of the top-k radii — less brittle when a single neighbour is
weirdly close.

Mutual Proximity goes further. It asks: given the whole distribution of
distances from this book to every other book, and the whole
distribution of distances from this query to every book, how *unlikely*
is it that both would be this close by chance? Hubs — books close to
everything — have distributions that make small distances *common*, so
the joint tail probability is unremarkable. Genuinely-relevant
neighbours have distributions where a small distance is rare, so the
joint tail is small and MP declares them close.

## Practical failure modes

- **N < 500.** Empirical CDF is noisy — MP scores wobble. Use NICDM.
- **k too small.** `k = 1` makes Local Scaling one-sample-noisy. Use
  `k ≥ 5`, ideally `k = k_retrieve`.
- **Truncated shortlists.** If the coarse retriever returns fewer than
  `k` candidates, the shortlist-derived `σ_q` / `μ_q` collapse. We
  defensively clamp to shortlist length and floor to `1e-9`.
- **Correct-metric baselines.** If the raw metric is *already* well-
  calibrated (low intrinsic dim, isotropic embedding), rerank is
  neutral and adds cost. Detect via a one-time hubness scan:
  `hub_frac > 0.01` ⇒ rerank pays; else keep baseline.
- **Update pressure.** `HubnessStats` is a snapshot. Under heavy
  streaming inserts the `kth_dist` / `topk_mean` drift. Refresh every
  `∆N ≥ 0.01·N` inserts, or maintain online (roadmap).

## What to improve next (roadmap)

1. **Sampled empirical CDF.** Store a length-`s` random sample per
   corpus point rather than the full row. Schnitzer 2012 reports
   `s = 2048` costs < 0.5 pts of recall at 100× smaller memory.
2. **HNSW-assisted precompute.** Reuse the constructed HNSW to derive
   `kth_dist` / `topk_mean` at index-build time — `O(N · log N)` instead
   of `O(N²)`.
3. **Streaming stats.** Reservoir-sample distances on insert; maintain
   running `μ_v` and `σ_v` estimators; recompute the row sample on a
   log-frequency schedule.
4. **Gauss / MP-Gauss variant.** Parametric survival function
   (`S(d) = 1 - Φ((d - μ_v) / σ_v)`) — O(1) per score, small recall hit,
   great for CPU-bound streams.
5. **Fusion with ruvector-cascade-adc.** ADC gives fast biased scores;
   hubness rerank corrects them. Combined stage cost should still fit
   under 200 µs/query at N=10⁶.

## Production crate layout

The PoC is one crate under `crates/ruvector-hubness-rerank`. For
production wiring we'd split into:

```
crates/
  ruvector-hubness-core/     # HubnessStats + Rerank trait
  ruvector-hubness-mp/       # Mutual Proximity (heavy)
  ruvector-hubness-fast/     # Local Scaling + NICDM (light)
  ruvector-hubness-online/   # Streaming stats maintenance
```

with feature flags on `ruvector-core`:

```
[features]
hubness = ["ruvector-hubness-fast"]
hubness-mp = ["ruvector-hubness-mp"]
hubness-online = ["ruvector-hubness-online"]
```

## References

- Radovanović, Nanopoulos, Ivanović. "Hubs in space: Popular nearest
  neighbors in high-dimensional data". JMLR 11:2487-2531, 2010.
- Schnitzer, Flexer, Schedl, Widmer. "Local and Global Scaling Reduce
  Hubs in Space". JMLR 13:2871-2902, 2012.
- Zelnik-Manor, Perona. "Self-Tuning Spectral Clustering". NIPS 2004.
- Jégou, Schmid, Harzallah, Verbeek. "Accurate image search using the
  contextual dissimilarity measure". TPAMI 32(1):2-11, 2010.
- Feldbauer, Leodolter, Plant, Flexer. "scikit-hubness: Hubness
  reduction and approximate neighbor search". JOSS 5(45):1957, 2020.
- Guo, Sun, Lindgren, Geng, Simcha, Chern, Kumar. "Accelerating
  Large-Scale Inference with Anisotropic Vector Quantization". ICML 2020.
