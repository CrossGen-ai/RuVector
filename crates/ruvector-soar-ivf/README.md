# ruvector-soar-ivf

SOAR (Spilling with Orthogonality-Amplified Residuals) IVF index.

Implements the SOAR partition-spilling scheme from Sun et al., *SOAR: Improved
Indexing for Approximate Nearest Neighbor Search* (NeurIPS 2024, Google Research)
in pure Rust with no external dependencies.

See `docs/research/nightly/2026-08-12-soar-ivf-orthogonality-residuals/` and
`docs/adr/ADR-298-soar-ivf-orthogonality-residuals.md` for design & benchmarks.

Run the benchmark:

```
cargo run --release -p ruvector-soar-ivf --bin benchmark
```

Run tests:

```
cargo test -p ruvector-soar-ivf
```
