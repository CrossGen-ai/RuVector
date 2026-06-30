# ruvector-cache-block-hnsw

Block-packed adjacency layout for HNSW — pack each node's whole neighbour list
into a single 64-byte cache line, optionally with per-edge distance sketches
for early-reject, behind a swappable trait.

```bash
cargo test    --release -p ruvector-cache-block-hnsw   # 3/3 pass
cargo run     --release -p ruvector-cache-block-hnsw --bin cache-block-bench
N=50000 NQ=300 cargo run --release -p ruvector-cache-block-hnsw --bin cache-block-bench
```

See `docs/research/nightly/2026-06-30-cache-block-hnsw/README.md` for the full
write-up and `docs/adr/ADR-272-cache-block-hnsw.md` for the decision record.
