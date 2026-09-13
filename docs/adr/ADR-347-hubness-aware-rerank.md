# ADR-347: Hubness-Aware Reranking for High-Dimensional ANN

- **Status**: Accepted (nightly research, 2026-09-13)
- **Nightly branch**: `research/nightly/2026-09-13-hubness-aware-rerank`
- **Crate**: `crates/ruvector-hubness-rerank`

## Context

Ruvector's ANN indices (HNSW, IVF-PQ, DiskANN, LSM-ANN) all rank candidates
by the raw metric they were built on — typically Euclidean or cosine on
learned embeddings whose ambient dimensionality is 384-1536. In such
high-dim spaces the pairwise distance distribution concentrates:
`Var(d) / E[d]² → 0` as `D → ∞`. A well-documented consequence is
**hubness** (Radovanović et al. 2010, JMLR 11:2487): a small number of
corpus points appear in the k-NN of a disproportionate number of queries.
Empirical measurements on GloVe-300, SIFT-1M and LAION-CLIP show 3–8% of
corpus points collecting 25–40% of all k-NN slots.

Hubs pull three failure modes into any retrieval system that ranks by
raw distance:

1. **Recall degradation** — hubs displace semantically-correct neighbours.
2. **Answer collapse** — the same handful of documents surface for
   unrelated queries (bad for RAG diversity).
3. **Reranker amplification** — because dot-product / cosine rerankers use
   the same biased metric, they cannot correct the bias.

Ruvector already has reranking hooks (`ruvector-rabitq`,
`ruvector-fused-rabitq-residual`, `ruvector-cascade-adc`,
`ruvector-gnn-rerank`) but none of them addresses hubness. The
literature offers three well-studied, cheap, index-agnostic corrections:
Local Scaling (Zelnik-Manor & Perona 2004), NICDM (Jégou et al. 2010),
and Mutual Proximity (Schnitzer et al. 2012, JMLR 13:2871). All three
operate purely on the retrieved shortlist and per-point neighbourhood
statistics — they compose with any existing coarse retriever.

## Decision

Ship `ruvector-hubness-rerank` as a first-class reranking module,
exposing three concrete rerankers plus a baseline, all behind a single
`Rerank` trait. It is index-agnostic: it consumes an `(id, raw_dist)`
shortlist and returns a re-scored, re-sorted top-k.

Precomputed corpus statistics live in `HubnessStats`:

- `sorted_dists[v]` — full sorted distance row for each corpus point,
  used by Mutual Proximity's empirical survival function.
- `kth_dist[v]` — k-th nearest-neighbour distance, used by Local Scaling.
- `topk_mean[v]` — mean of top-k NN distances, used by NICDM.

The default reranker is **NICDM** (marginally better mean recall than
Local Scaling in our benchmark, and 30× cheaper than Mutual Proximity)
with Mutual Proximity available for latency-tolerant offline pipelines
(re-scoring, evaluation harnesses, "diamond mode" search).

## Consequences

**Positive**

- +14 pts absolute recall@10 (+32% relative) on the crate's synthetic
  hubness-inducing dataset (baseline 43.08% → NICDM 57.00% → MP 57.24%).
- Hub fraction in top-10 drops from 0.0002 to 0.0000 in the benchmark
  (MP flattens the hub concentration completely).
- Rerank cost is a small tax on top of the coarse retriever:
  - NICDM / Local Scaling: ~3.3 µs / query on R=200 shortlist.
  - Mutual Proximity: ~104 µs / query on R=200, N=5000 (binary
    search into a length-N sorted row, twice).

**Negative**

- Memory: `HubnessStats::sorted_dists` is `O(N²)` = 95 MB at N=5000. For
  N=10⁶ that is 4 TB — infeasible.
- Compute: precompute is `O(N² · D)` distance work. 1.5 s at N=5000,
  D=128; scales quadratically. Approximation is required at scale.
- Correction is a *statistical* filter, not a semantic fix: if the raw
  metric is fundamentally wrong (e.g. cosine on unnormalised vectors)
  hubness rerank cannot rescue it.

## Alternatives considered

1. **Discard hubs at index-build time.** Rejected — hubs are query-
   dependent; the same point may be a hub for one query set and a
   perfectly good neighbour for another.
2. **Retrain the embedding to be more isotropic.** Correct in principle
   but out of scope: ruvector consumes third-party embeddings.
3. **Simhash/ternary rerank (already in ruvector-ternary /
   ruvector-simhash-prefilter).** Orthogonal — those attack quantisation
   error, not distance concentration.
4. **DisSim / GLOSH-style rerank.** Requires density estimation, which
   is more expensive than MP and offers no measurable extra recall in
   published comparisons (Feldbauer 2018, `scikit-hubness` docs).

## Roadmap (production hardening)

The PoC uses an `O(N²)` full-row cache. Production paths:

- **Sampled empirical CDF** — store a fixed-size random sample (e.g.
  `s = 2048`) of each row and evaluate the survival function on that.
  Loss < 0.5 recall pts at s=1024 in Schnitzer 2012's own experiments.
- **On-the-fly stats via HNSW self-search** — reuse the built HNSW to
  compute each point's k-NN distances, giving `kth_dist` and
  `topk_mean` at insertion time for `O(N · log N)`.
- **Streaming update** — treat `kth_dist` / `topk_mean` as an online
  statistic; refresh on insert/delete via reservoir sampling.
- **Feature-gate MP behind `hubness-mp` (heavy, offline) vs NICDM /
  Local Scaling behind `hubness-fast` (production, on-line).**

## References

See `docs/research/nightly/2026-09-13-hubness-aware-rerank/README.md`.
