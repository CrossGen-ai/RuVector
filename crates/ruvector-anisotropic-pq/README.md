# ruvector-anisotropic-pq

Score-aware Product Quantization for MIPS. Three swappable trainers
behind a `PqTrainer` trait:

| Trainer                | Loss                                         | Best when                                     |
|------------------------|----------------------------------------------|-----------------------------------------------|
| `L2Trainer`            | `‖s − c‖²`                                  | Baseline / pure L2 nearest-neighbor           |
| `NormWeightedTrainer`  | weighted by `‖x‖²`                          | MIPS, heavy-tail norm data (RECOMMENDED)      |
| `AnisotropicTrainer(η)`| `‖s − c‖² + (η−1)((s−c)·u)²` (per-subvector) | Research / follow-up baseline (see caveats)   |

## Quickstart

```rust
use ruvector_anisotropic_pq::{NormWeightedTrainer, PqTrainer};

let cb = NormWeightedTrainer::default()
    .train(&data, n, /*dim*/ 128, /*m*/ 16, /*k*/ 256)?;
let codes = cb.encode(&query_vector);
let recon = cb.decode(&codes);
```

## Benchmark

```
cargo run --release -p ruvector-anisotropic-pq --bin aniso-pq-bench
```

On Apple M4 Max, `n=10,000 d=128 m=16 k=256` (16 B/vec, 32× compression):

| Variant             | recall@10 | Δ vs L2        |
|---------------------|-----------|----------------|
| L2-PQ               | 0.6700    | —              |
| **NormWeighted-PQ** | **0.7770**| **+10.70 pp**  |
| Anisotropic η=2     | 0.6390    | −3.10 pp       |
| Anisotropic η=6     | 0.5820    | −8.80 pp       |

See [`docs/research/nightly/2026-08-17-anisotropic-pq/`](../../docs/research/nightly/2026-08-17-anisotropic-pq/)
and [ADR-305](../../docs/adr/ADR-305-anisotropic-pq.md) for full context,
including the honest negative result on per-subvector anisotropic loss
and the roadmap to full-vector + OPQ.
