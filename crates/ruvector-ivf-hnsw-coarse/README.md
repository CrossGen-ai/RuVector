# ruvector-ivf-hnsw-coarse

HNSW coarse quantizer for IVF: `O(log C)` nprobe centroid selection instead of
the standard `O(C)` full brute-force scan of the centroid table.

Three swappable backends behind the `IvfCoarse` trait:

| Backend            | nprobe selection cost | Recall vs Flat |
|--------------------|-----------------------|----------------|
| `FlatCoarse`       | `O(C · d)`            | 100% (oracle)  |
| `HnswCoarse`       | `O(ef · log C · d)`   | ≥99% at ef=32  |
| `RandomProjCoarse` | `O(b · d + C)` bits   | ~90% (baseline)|

`C` = number of IVF centroids. Standard IVF sets `C ≈ √N`, so at
`N=10M` you get `C ≈ 3200`, and the flat coarse scan alone is a
measurable slice of query latency at low nprobe. Replacing that scan
with an HNSW over centroids gives sub-millisecond coarse routing at
`C=4096` while keeping ≥99% recall of the top-nprobe centroids.

See `docs/research/nightly/2026-09-24-ivf-hnsw-coarse/README.md` for
full methodology, SOTA survey, and benchmark results.

Run:
```
cargo run --release -p ruvector-ivf-hnsw-coarse
```

Test:
```
cargo test -p ruvector-ivf-hnsw-coarse
```
