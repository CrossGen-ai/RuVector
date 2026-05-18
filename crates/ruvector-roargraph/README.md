# ruvector-roargraph

Projected bipartite graph index for out-of-distribution (OOD) approximate
nearest-neighbour search, based on RoarGraph (Chen et al., VLDB 2024).

## Quick start

```bash
cargo run --release -p ruvector-roargraph --bin roargraph-demo
cargo test -p ruvector-roargraph
```

## Research document

See `docs/research/nightly/2026-05-18-roargraph/README.md` for algorithm
details, benchmark results, and SOTA survey.

## ADR

See `docs/adr/ADR-194-roargraph.md`.

## Results (Apple M4 Max, --release)

| Variant | recall@10 | QPS |
|---------|-----------|-----|
| Brute-force | 100.0% | 8,522 |
| Baseline (base-to-base k-NN) | 11.3% | 27,539 |
| **RoarGraph** | **100.0%** | **54,088** |

OOD dataset: N=5,000, dim=64, GMM-B queries shifted by 3.0σ from GMM-A base.
