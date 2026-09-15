# ruvector-bidir-hnsw

Reverse-neighbor densification for HNSW-style graph search.

Provides a lightweight NSW-style directed graph builder plus a **reverse
adjacency sidecar**: for each node `v`, we record up to `r_max` in-neighbors.
Query-time beam search can then expand candidates by `Adj[u] ∪ RevAdj[u]`,
raising recall at fixed `ef` without altering the forward graph.

Three modes compared in `cargo run --release -p ruvector-bidir-hnsw --bin bench`:

- `baseline`         — vanilla forward-only expansion.
- `expand-reverse`   — union of forward and reverse neighbors at query time.
- `symmetrize-build` — pre-symmetrised forward adjacency (bounded to `m_max`).

See `docs/research/nightly/2026-09-15-hnsw-bidirectional-densify/README.md`
and `docs/adr/ADR-346-hnsw-bidirectional-densify.md`.
