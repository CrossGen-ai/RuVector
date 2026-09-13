# ruvector-hubness-rerank

Hubness-aware reranking for high-dimensional approximate nearest neighbour search.

In high dimensions the pairwise distance distribution concentrates and a
small number of points — **hubs** — appear in the kNN list of a
disproportionate number of queries (Radovanović et al. 2010). Standard
metric reranking cannot undo this: the raw distance *is* the biased score.
This crate implements three post-retrieval score corrections that use each
point's own neighbourhood statistics to counter the bias, plus a baseline
control.

## Methods

| Reranker | Correction | Cost / query |
|---|---|---|
| `BaselineRerank` | raw distance (control) | O(R log R) |
| `LocalScalingRerank` | `d(q,v) / √(σ_q · σ_v)` (Zelnik-Manor & Perona 2004) | O(R log R) |
| `NicdmRerank` | `d(q,v) / √(μ_q · μ_v)` (Jégou et al. 2010) | O(R log R) |
| `MutualProximityRerank` | `1 - S_q(d) · S_v(d)` (Schnitzer et al. 2012) | O(R log N) |

`σ_v` = k-th NN distance of `v`; `μ_v` = mean of top-k NN distances of `v`;
`S_v` = empirical survival function of `v`'s distance row.

## Benchmark (`cargo run --release --bin benchmark`)

Synthetic corpus: N=5000, D=128 (D_true=32 signal + D_noise=96), 24
clusters, 10% low-noise "hub" points. Ground truth = top-10 in the
signal subspace only. Retrieval uses full ambient L2.

```
method                recall@10     rerank µs/q     hub_frac
--------------------------------------------------------------
baseline                 0.4308            0.81       0.0002
local_scaling            0.5704            3.27       0.0000
nicdm                    0.5700            3.23       0.0000
mutual_proximity         0.5724          103.63       0.0000
```

+14 pts recall@10 (+32% relative) across all three rerankers; MP shows the
strongest hub suppression per query at ~100 µs on a shortlist of R=200.

## Layout

- `src/distance.rs`   — `Distance` trait, `L2`.
- `src/index.rs`      — flat exact-kNN index and `coarse_top_r`.
- `src/hubness.rs`    — corpus-wide precomputed hubness statistics.
- `src/rerank.rs`     — the four rerankers behind a shared `Rerank` trait.
- `src/bin/benchmark.rs` — the benchmark above; runs in <2 s after build.

## License

Apache-2.0.
