# ruvector-permutation-prefix

Ultra-compact permutation-prefix reranker for ANN pipelines.

Each vector stores the ordered ids of its top-`m` nearest pivots
(farthest-point sampled). Rerank is one integer Spearman-footrule pass
per candidate — FP-free, metric-agnostic, 16-32 B/vec.

See [`docs/research/nightly/2026-11-10-permutation-prefix-rerank/README.md`](../../docs/research/nightly/2026-11-10-permutation-prefix-rerank/README.md)
and [ADR-357](../../docs/adr/ADR-357-permutation-prefix-rerank.md).

Run the benchmark:

```sh
cargo run --release -p ruvector-permutation-prefix --bin benchmark
```
