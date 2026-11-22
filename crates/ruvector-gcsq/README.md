# ruvector-gcsq — Gray Code Scalar Quantization

Per-dimension uniform scalar quantization whose bin indices are assigned via
the reflected binary Gray code. Because adjacent bins differ by exactly one
Hamming bit, bitpacked Hamming distance over GCSQ codes correlates tightly
with L2 distance — useful as a cache-friendly reranker over PQ or HNSW
shortlists.

## Why it works

Plain binary SQ encodes bin 1 → `001`, bin 2 → `010`. These adjacent bins
differ by **two** Hamming bits — a non-monotonic lie between integer
proximity and bit-flip count. Gray code encodes 1 → `001`, 2 → `011`, so
each step costs exactly one flip. Summed over dimensions, Hamming distance
is proportional to total bin-index displacement, making it an unbiased
piecewise-linear proxy for coordinate-wise L1 (and a reasonable one for L2
at small drifts).

## Running

```bash
cargo test   -p ruvector-gcsq --release
cargo run    -p ruvector-gcsq --release --example demo
cargo bench  -p ruvector-gcsq --bench gcsq_bench
```

## Quality (gaussian n=5000, dim=128, measured)

| backend    | bits/dim | code_bytes | spearman ρ | recall@10 |
|------------|----------|------------|-----------:|----------:|
| gcsq       |        2 |         32 |      0.630 |     0.132 |
| binary_sq  |        2 |         32 |      0.324 |     0.044 |
| simhash    |        2 |         32 |      0.462 |     0.088 |

GCSQ at 2 bits/dim is the Pareto winner: best ρ/byte, and encoding is
~60× faster than bit-matched SimHash (no Gaussian matmul).
