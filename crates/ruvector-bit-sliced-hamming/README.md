# ruvector-bit-sliced-hamming

Bit-sliced (bit-transposed) storage for high-throughput Hamming top-K search
over binary embedding vectors, with three swappable backends behind the
`HammingTopK` trait:

- `NaiveRowMajor` — row-major popcount scan, correctness oracle.
- `BlockedRowMajor` — cache-blocked row-major with online top-K max-heap.
- `BitSlicedEarlyTerm` — bit-transposed layout with dim-band pruning and a
  provably lossless lower-bound / upper-bound early termination.

Run the benchmark:

```sh
cargo run --release -p ruvector-bit-sliced-hamming --bin benchmark
```

See `docs/research/nightly/2026-09-18-bit-sliced-hamming-scan/README.md`
for design, SOTA survey, and measured numbers.
