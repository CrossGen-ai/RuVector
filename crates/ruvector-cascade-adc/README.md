# ruvector-cascade-adc

Cascade Asymmetric Distance Computation (Cascade-ADC): a two-stage
progressive-precision PQ scanner. Stage 1 sweeps every code at 4-bit
precision using a tiny L1-resident LUT; Stage 2 refines the top ρ·N
survivors at 8-bit precision. Recall matches the full 8-bit baseline while
throughput is competitive or better in pure-scalar Rust.

## Layout

```
codes8 : n × m bytes  (256 centroids / subspace)
codes4 : n × ⌈m/2⌉ bytes  (16 centroids / subspace, packed 2/byte)
parent : m × 256 bytes  (fine→coarse mapping)
```

The 4-bit codebook is a *coherent coarsening* of the 8-bit codebook: a
second k-means pass reduces the 256 fine centroids to 16 coarse ones per
subspace. Because the two levels share geometry, a candidate that scores
well at 4-bit tends to also score well at 8-bit — which is what makes
Stage-1 pruning safe.

## Run the bench

```bash
cargo run --release -p ruvector-cascade-adc --bin cascade-bench
```

Results on an Apple M4 Max with n=200,000 d=128 m=32 k=10:

```
full-8bit         qps=412  recall@10=0.4625  32 B/vec
full-4bit         qps=497  recall@10=0.1685  16 B/vec
cascade ρ=0.05    qps=464  recall@10=0.4595  48 B/vec
cascade ρ=0.10    qps=428  recall@10=0.4625  48 B/vec
cascade ρ=0.20    qps=399  recall@10=0.4625  48 B/vec
```

## Tests

```bash
cargo test --release -p ruvector-cascade-adc
```

Verifies scanner return shape, distance monotonicity, and that cascade
recall never falls below the full 4-bit floor and lands within 5 pp of the
full 8-bit ceiling.
