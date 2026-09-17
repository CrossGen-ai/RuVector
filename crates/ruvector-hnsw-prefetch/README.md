# ruvector-hnsw-prefetch

Software-prefetched HNSW neighbour traversal for ruvector. Three
swappable backends behind a common trait:

- `NoPrefetch` — baseline
- `Prefetch1` — fixed one-hop prefetch
- `Adaptive` — depth grows on accepted neighbours, shrinks on rejected

## Run

```
cargo test --release -p ruvector-hnsw-prefetch
cargo run  --release -p ruvector-hnsw-prefetch --bin benchmark
```

## Latest measured numbers (n=20k, dim=128, m=16, ef=64, k=10)

| Backend      | avg latency (µs) | dist calls / q | recall@10 |
|--------------|------------------|----------------|-----------|
| NoPrefetch   | 63.42            | 898.0          | 0.268     |
| Prefetch1    | 62.52            | 898.0          | 0.268     |
| Adaptive     | 59.19            | 898.0          | 0.268     |

Prefetch depth changes only wall-clock latency; distance calls and
recall are unaffected — a correctness invariant enforced by the unit
tests.

See `docs/research/nightly/2026-09-17-hnsw-neighbor-prefetch/` for the
full write-up and ADR-346 for the design record.
