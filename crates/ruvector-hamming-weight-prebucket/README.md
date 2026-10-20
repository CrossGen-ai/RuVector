# ruvector-hamming-weight-prebucket

Pre-bucket binary codes by their popcount `w(x) = pop(x) in [0..D]`, then at
query time use the triangle lower bound

```
H(x, q) >= |w(x) - w(q)|
```

to visit buckets in order of `|w - w_q|` and stop as soon as that distance
exceeds the current k-th-best Hamming τ. Exact (recall@k = 1.0 against full
scan) and composes with any inner Hamming scanner.

Backends shipped:
* `FullScan`         — baseline, visits every code.
* `WeightedScan`     — pre-bucketed scan (this work).
* `WeightedBitSlice` — pre-bucket + flat packed layout for a tight inner loop.

Benchmark (run yourself; real numbers are also in the research doc):

```bash
cargo run --release -p ruvector-hamming-weight-prebucket --bin benchmark
```
