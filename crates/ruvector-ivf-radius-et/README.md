# ruvector-ivf-radius-et

IVF probe-loop early termination via per-cell radius + Euclidean triangle
inequality.

For any query `q`, cell `c` with centroid `mu_c` and radius
`r_c = max_{p in c} ||p - mu_c||`, every point `p in c` satisfies

```
||q - p|| >= ||q - mu_c|| - r_c
```

Scan cells in **increasing** `LB_c = max(0, ||q - mu_c|| - r_c)`.
Terminate the moment `LB_next >= tau_k` (the current k-th-best distance):
every unvisited cell has `LB >= LB_next >= tau_k`, so no unvisited point
can beat the top-k. This is **exact w.r.t. the IVF partition** — same
answer as full-scan IVF, strictly fewer cells touched.

Backends (all behind the `IvfSearcher` trait):

* `FixedNprobe { nprobe }` — baseline: scan the top-`nprobe` centroid-closest
  cells, no cut.
* `RadiusET` — scan cells in LB order, stop at `LB >= tau_k`. Exact.
* `HybridNprobeET { max_nprobe }` — LB-sorted scan with a cap. Behaves like
  FixedNprobe at small caps (approximate), becomes exact once the cap is
  large enough for the LB cut to trigger first.

Reproduce the real numbers from the research doc:

```bash
cargo run --release -p ruvector-ivf-radius-et --bin benchmark
cargo test --release -p ruvector-ivf-radius-et --lib
```

See `docs/research/nightly/2026-10-21-ivf-radius-early-terminate/README.md`
and `docs/adr/ADR-357-ivf-radius-early-terminate.md`.
