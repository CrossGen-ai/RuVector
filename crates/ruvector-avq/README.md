# ruvector-avq

Anisotropic Vector Quantization (AVQ) — score-aware product-quantization
codebooks for inner-product ANN search (Guo et al., ScaNN, ICML 2020).

Three variants share the same [`Quantizer`] trait so they can be swapped
in production and their footprint / recall trade-offs measured on
identical corpora:

| Variant       | Loss                                     | Side channel |
| ------------- | ---------------------------------------- | ------------ |
| `PqMse`       | plain MSE (baseline)                     | –            |
| `AvqScoreAware` | `η ‖e_∥‖² + ‖e_⊥‖²` (anisotropic)      | –            |
| `AvqNorm`     | anisotropic, on unit-normalised vectors  | 4 B (f32 norm) |

## Usage

```rust
use ruvector_avq::{QuantizerConfig, AnisotropicConfig, Quantizer};
use ruvector_avq::avq::AvqScoreAware;

let cfg = QuantizerConfig { m: 16, ks: 256, iters: 12, seed: 42 };
let mut q = AvqScoreAware::new(cfg, AnisotropicConfig::default(), 128)?;
q.train(&train_vecs)?;
let codes = q.encode(&base_vecs)?;
let scores = q.adc(&query, &codes)?;
```

## Benchmark

```bash
cargo run --release -p ruvector-avq --bin avq-benchmark
```

Prints Recall@10, encode time, ADC latency, and bytes-per-vector across
`M ∈ {8, 16}` for all three variants on a deterministic 20 k × 128-D
mixture-of-Gaussians corpus.
