# ruvector-polysemous-pq-hamming-prefilter

Pure-Rust prototype of **polysemous Product Quantization codes** (Douze,
Jégou, Schmid; ECCV 2016): a PQ encoder whose centroid labels are
reordered so that Hamming distance between codes correlates with L2
distance between the centroids they index. A two-stage query path uses
Hamming as a cheap prefilter and ADC as a rerank over survivors.

## Build & run

```bash
cd crates/ruvector-polysemous-pq-hamming-prefilter
cargo test  --release
cargo run   --release --bin polysemous-bench
```

Runs offline, no network, no mocks. Numbers below reproduce on Apple
M4 Max (16 core, macOS 24.6.0), Rust 1.90, seeded RNG.

## What it does

1. Train standard PQ (`m` subspaces × `ksub` centroids, Lloyd k-means).
2. For each subquantizer, fit a permutation of centroid labels by
   **simulated annealing** minimising
   `Σ (hamming(π(i), π(j)) − α · L2(c_i, c_j))²`.
3. Encode every database vector into `m`-byte remapped codes.
4. Query: XOR+popcount the query code against each stored code, keep
   those under threshold `ht`, run ADC on the survivors, return top-k.

## Headline numbers (50 000 × 64 dim MoG, 32× compression)

| threshold | recall@10 | survivors / 50 000 | latency ms/query | speedup vs plain ADC |
|-----------|-----------|--------------------|------------------|----------------------|
| 24        | 0.074     | 1 506              | 0.103            | **4.15×**            |
| 28        | 0.131     | 9 564              | 0.210            | **2.03×**            |
| 32        | 0.158     | 27 519             | 0.463            | 0.92×                |
| ADC-only  | 0.159     | 50 000             | 0.428            | 1.00× (reference)    |

`ht = 28` is the Pareto sweet spot on this workload: 82 % of ADC’s
recall for 2× the throughput. See `docs/research/nightly/2026-11-21-polysemous-pq-hamming-prefilter/README.md`
for full methodology, references, and roadmap.

## Status

Nightly research PoC. Working Rust, passing tests, real `cargo run`
benchmark numbers. See the companion ADR
`docs/adr/ADR-polysemous-pq-hamming-prefilter.md`.
