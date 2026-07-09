# ruvector-aisaq

**AISAQ — All-in-Storage ANNS with Quantization** for the ruvector workspace.
Rust proof-of-concept of Kioxia's AISAQ design (arXiv:2404.06004, 2024):
put PQ codes on SSD, keep only the graph in RAM, and let the OS page cache
serve hot working sets.

See `docs/research/nightly/2026-07-09-aisaq-all-in-storage-quantization/`
for the full write-up and `docs/adr/ADR-272-aisaq-all-in-storage-quantization.md`
for the design rationale.

## Quick start

```bash
cargo build --release -p ruvector-aisaq
cargo test  --release -p ruvector-aisaq
cargo run   --release -p ruvector-aisaq --bin aisaq-bench
```

## Layout

* `src/pq.rs` — 8-bit product quantiser (Lloyd's, k=256/subspace).
* `src/graph.rs` — brute-force k-NN graph + beam search.
* `src/backends.rs` — three `DistanceBackend` impls: RAM f32, RAM PQ, mmap PQ.
* `examples/bench.rs` — real benchmark reported in the research doc.
* `tests/smoke.rs` — invariants: AISAQ matches in-RAM PQ byte-for-byte.

## Measured (M4 Max, N=20 000, D=128, M=16, R=32, beam=64)

| variant       | heap RAM     | µs/query | recall@10 |
|---------------|--------------|----------|-----------|
| flat-f32-ram  | 10,240,000 B |   125.65 |    0.7425 |
| pq-ram        |    451,072 B |    60.97 |    0.2535 |
| pq-disk-aisaq |    147,456 B |    85.24 |    0.2535 |
