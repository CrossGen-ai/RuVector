# ruvector-nndescent

Bulk kNN-graph construction via **NN-Descent** (Dong, Charikar & Li, WWW 2011)
for HNSW base-layer bootstrap. Trait-based builders, real benchmarks,
zero mocks.

## Quick start

```bash
cargo run --release -p ruvector-nndescent --bin benchmark
cargo test --release -p ruvector-nndescent
```

## Variants measured

| Variant              | Graph recall@16 | Distance-ops vs brute | Wall-time vs brute |
|----------------------|----------------:|----------------------:|-------------------:|
| BruteForce           |           1.000 |                100.0% |              1.00× |
| NnDescent-Basic      |           0.772 |                  8.4% |              3.28× |
| NnDescent-LocalJoin  |           0.989 |                 13.9% |              1.84× |

Numbers from N=8 000, dim=64, K=16 on Apple M4 Max — see
`docs/research/nightly/2026-06-22-nndescent-bulk-hnsw/README.md`.

## API

```rust
use ruvector_nndescent::{
    BruteForceBuilder, NnDescentBuilder, NnDescentConfig,
    DistanceCounter, KnnGraphBuilder, SeededHnsw, SeededHnswConfig,
};

let counter = DistanceCounter::new();
let cfg = NnDescentConfig {
    rho: 0.5, delta: 1e-3, max_iters: 12, use_reverse: true, seed: 7,
};
let graph = NnDescentBuilder::new(cfg).build(&vectors, 16, &counter);

let hnsw = SeededHnsw::new(&vectors, &graph, SeededHnswConfig {
    ef_search: 64,
    entry_points: SeededHnswConfig::spread_entries(vectors.len(), 13),
});
let top16 = hnsw.search(&query, 16);
```

## Status

Nightly research crate. See ADR-268 for the design rationale and the
production-crate roadmap (Rayon parallelism, SIMD distance, Vamana
α-pruning, HNSW upper-layer attachment, Panorama distance bounds,
SPFresh integration).
