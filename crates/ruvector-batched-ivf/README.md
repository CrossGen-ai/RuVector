# ruvector-batched-ivf

Multi-query batched IVF scan with shared posting-list traversal.

Serving ANN queries one at a time refetches the same posting lists
from DRAM over and over whenever a batch of queries probes overlapping
centroids. This crate groups queries by probed list and scans each
list once per batch, updating every subscribing query's top-k heap
while the posting-list vectors are hot in cache.

Three semantically equivalent scanners are provided:

- `serial_scan` — one query at a time, no sharing (baseline).
- `batched_list_major_scan` — walk each list once; update all
  subscribing queries.
- `batched_tiled_scan` — list-major with a 32-vector inner tile so
  every subscribing query sweeps data that is live in L1.

See `docs/research/nightly/2026-10-04-batched-ivf-shared-posting/` and
`docs/adr/ADR-354-batched-ivf-shared-posting.md` for design,
benchmarks, and failure modes.

## Run the demo

```
cargo run --release -p ruvector-batched-ivf --example demo
```

Produces a table comparing all three strategies on a 50 000 × 128-D
IVF index at batch sizes 1–256.

## Correctness

All three strategies return the same top-k for the same probe plan.
Enforced by `cargo test -p ruvector-batched-ivf`.
