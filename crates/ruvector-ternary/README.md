# ruvector-ternary

Ternary `{-1, 0, +1}` vector encoding for high-recall ANN prefilters.

**Idea.** Standard 1-bit binary quantization spends the same bit on every
coordinate, whether the coordinate carries a strong sign or is close to zero
noise. Ternary encoding adds a per-vector magnitude threshold so ambiguous
coordinates *abstain*. Distance is a fused 4-op kernel per 64-D chunk:

```text
popcount( (sign_a ^ sign_b) & mask_a & mask_b )
```

Two `xor`s, two `and`s, one `popcnt` — no floats, autovectorizes on x86
(`popcnt`) and aarch64 (`cnt`+`addv`).

## Try it

```bash
cargo test    -p ruvector-ternary
cargo run --release -p ruvector-ternary --bin benchmark
```

## Layout

| File | Purpose |
| --- | --- |
| `src/lib.rs` | `Encoder` / `Distance` traits |
| `src/binary.rs` | 1-bit baseline |
| `src/ternary.rs` | ternary encoder + fused distance |
| `src/int8.rs` | int8 scalar-quant baseline |
| `src/bin/benchmark.rs` | recall + throughput harness |
| `tests/integration.rs` | end-to-end recall assertions |

See `docs/research/nightly/2026-09-11-ternary-vector-search/README.md` for
the write-up and real numbers.
