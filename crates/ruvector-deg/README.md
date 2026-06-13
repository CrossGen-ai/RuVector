# ruvector-deg

DEG — Dynamic Exploration Graph. A single-layer proximity graph ANN index
with Relative-Neighborhood-Graph (RNG) edge optimization on insert.

Backends (all behind the `AnnIndex` trait):

| Backend     | Description                                              |
| ----------- | -------------------------------------------------------- |
| `BruteForce`| Exact linear scan, used for ground truth + baseline      |
| `KnnGraph`  | Static k-NN graph (O(N²) build) + greedy beam search     |
| `Deg`       | DEG: incremental inserts, RNG pruning, multi-entry beam  |

Run the demo (real numbers from `cargo run --release`):

```
cargo run -p ruvector-deg --release --bin deg-demo
cargo test  -p ruvector-deg --release
cargo bench -p ruvector-deg
```

Numbers from the nightly run on Apple Silicon (n=5000, dim=64, k=10):

```
BruteForce  recall=1.0000  qps=11363.6
KnnGraph    recall=0.1640  qps=29070.5
DEG         recall=0.5810  qps=24861.6 (ef=64)
DEG-noRNG   recall=0.2815  qps=26217.5

DEG ef_search sweep:
ef=32   recall=0.5125  qps=42250.6
ef=64   recall=0.6300  qps=27821.1
ef=128  recall=0.6760  qps=18490.6
ef=256  recall=0.8625  qps=10157.7
ef=512  recall=0.9490  qps=5793.7
```

See `docs/research/nightly/2026-06-13-deg-dynamic-exploration-graph/README.md`
for the full research write-up and `docs/adr/ADR-211-deg-dynamic-exploration-graph.md`.
