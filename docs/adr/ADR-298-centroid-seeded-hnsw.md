# ADR-298: Centroid-Seeded Entry Points for Graph-Based ANN

- **Status:** Accepted (research prototype)
- **Date:** 2026-08-10 (rescued and landed 2026-08-11)
- **Crate:** `crates/ruvector-centroid-seeded-hnsw/`
- **Research doc:** `docs/research/nightly/2026-08-10-centroid-seeded-hnsw/README.md`

## Context

Graph-based ANN indices (HNSW, NSG, Vamana, DiskANN) descend greedily from an entry point. HNSW pins the top-layer node; NSG/Vamana use the global medoid; random-seeding is common in reference implementations and quick prototypes. Entry choice is decoupled from the graph itself but influences hop-count, distance-call budget, and — crucially for beam=1 or tight-ef regimes — recall.

## Decision

Introduce a `SeedStrategy` trait split from the search engine, with three shippable strategies:
- `RandomSeeder` (baseline)
- `CentroidSeeder { k, m: 1 }` (k-means medoids, closest cluster)
- `MultiCentroidSeeder { k, m }` (top-m clusters, multi-start descent)

Fit is offline (Lloyd + k-means++, deterministic RNG). Serving cost per query = `k·D` extra distance ops plus `m` graph-descent restarts.

## Consequences

**Positive.** Clean separation of *where to start* from *how to walk*, so any downstream engine (HNSW/NSG/Vamana) can adopt this without index-format change. Empirical: on a 2k×32 gaussian-mixture kNN graph, beam=1 recall rises from ~0.0 → ~0.67 with negligible per-query overhead once centroids are cached.

**Negative.** k-means fit is O(iters·N·k·D); centroids drift under inserts. On already-strong indices at wide `ef`, the recall lift is small — the win becomes tail-latency, not recall.

## Alternatives considered

- **Learned entry-point selector (MLP over query).** Higher ceiling, but bespoke training + drift risk.
- **Hierarchical partition trees (KD/ball) as entry.** Similar effect, worse cache behavior in high-D.
- **Do nothing (single global medoid, NSG-style).** Simple, but ignores query locality entirely.
