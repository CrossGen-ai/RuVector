# ruvector-succinct-hnsw

Memory-efficient NSW/HNSW adjacency stores as swappable backends over the
same graph topology.

Backends:

- `dense_vec_vec_u32` — baseline `Vec<Vec<u32>>`.
- `delta_varbyte` — sorted deltas encoded as unsigned LEB128 into a flat
  blob + `Vec<u32>` offsets.
- `reordered_delta_varbyte` — BFS re-permutes node ids so neighbours
  cluster in id space, then delta+VarByte encoded (deltas shrink → varint
  buckets down-shift → payload drops).

Run the benchmark:

```
cargo run --release -p ruvector-succinct-hnsw --bin benchmark
```

Run tests:

```
cargo test -p ruvector-succinct-hnsw
```

See `docs/research/nightly/2026-09-08-succinct-adjacency-hnsw/README.md`
for the full research write-up with benchmark tables.
