# ADR-353: SOAR Orthogonality-Aware Spilled IVF Assignment

## Status

**Proposed, with pure-Rust PoC implemented and verified.** 2026-10-22.
Nightly research branch: `research/nightly/2026-10-22-soar-orthogonal-spill-ivf`.
PoC crate: `crates/ruvector-soar-ivf/`.
Research doc:
[`docs/research/nightly/2026-10-22-soar-orthogonal-spill-ivf/README.md`](../research/nightly/2026-10-22-soar-orthogonal-spill-ivf/README.md).

### Verification legend

- **[V]** Verified in this session by running `cargo test --release` and
  `cargo run --release --example bench`.
- **[D]** Design claim, consistent with the referenced 2024 Google paper,
  not independently re-derived.
- **[U]** Unverified at scale (synthetic data only, 50 k vectors).

## Context

ruvector already carries ~190 crates and ~40 prior nightly experiments
covering most of the IVF/HNSW/PQ design space (OPQ, anisotropic PQ,
RaBitQ, Matryoshka, FastScan, hubness-rerank, …). One axis none of them
touches is **how a vector is assigned to one or more posting lists at
build time** — every IVF variant in the tree uses classic
nearest-centroid assignment with optional random spill.

The 2024 NeurIPS paper by Sun, Guo, Simcha, Krichene, Wu — authors of
Google's ScaNN — formalises a build-time trick that lifts recall at low
`nprobe` dramatically with negligible memory cost: **assign a secondary
centroid chosen so its residual is orthogonal to the primary residual**.
Open-source ScaNN has shipped this as `spilling_strategy = SOAR` since
early 2025, so it is now production-grade in Google's own code.

ruvector has no equivalent, and the technique is strictly complementary
to every scoring-side improvement (quantisers, early abandon, rerank
budget) already in the repo. **[D]**

### Prior ruvector work this stacks on top of

- `ruvector-ivf-ball-pruning` (2026-10-14) — query-side list pruning.
  SOAR is build-side. Fully stackable.
- `ruvector-pq-adc-early-abandon` (2026-10-15) — scoring-side early
  abandon. Orthogonal (literally).
- `ruvector-online-ivf-centroid-refresh` (2026-10-05) — centroid
  maintenance. SOAR re-assignment on refresh is natural future work.
- `ruvector-rabitq` family — SOAR + RaBitQ is a listed roadmap item.

## Decision

**Adopt SOAR as the recommended spilled-assignment strategy for new IVF
indexes in ruvector**, exposed behind a pluggable `IvfAssigner` trait so
callers can pick `PrimaryOnly`, `RandomSpill`, or `Soar { lambda }`.

Default parameters:
- `lambda = 1.0` (recall is within 0.001 across λ ∈ {0.5, 1.0, 2.0} in
  our measurements; **[V]** — see research doc results table).
- Secondary spill only (not r-way), as the single-replica case already
  captures most of the gain in the paper and keeps posting-list memory
  tight.

Measured gain on the PoC (synthetic 50 k × dim-64 Gaussian mixture, 256
centroids, 500 queries, k=10) **[V]**:

| strategy      | recall@10 nprobe=1 | recall@10 nprobe=2 |
|---------------|-------------------:|-------------------:|
| primary-only  |             0.7170 |             0.9454 |
| random-spill  |             0.7182 |             0.9458 |
| **soar(λ=1)** |         **0.9326** |         **0.9944** |

The random-spill row proves the gain comes from the orthogonality
constraint, not from doubling list capacity.

Memory cost: +0.19 MiB (extra u32 postings) on 12.2 MiB of f32 data —
1.5 %. Build-time cost: 0.22 s → 0.68 s (3×, from the O(n·k·d)
secondary scan). **[V]**

## Consequences

### Positive

- +20+ percentage points of recall at `nprobe=1` on in-domain data,
  which translates to being able to run with a *lower* `nprobe` for the
  same recall target, lowering query cost in production.
- Build-side only — no change to query path, so existing IVF scoring,
  rerank, and quantiser code is reused unchanged.
- Trait-based pluggability means ruvector keeps `PrimaryOnly` and
  `RandomSpill` as first-class citizens for benchmarking and legacy
  tenants.
- Deterministic (seeded RNG for k-means and random-spill), so results
  are reproducible across runs.

### Negative

- **Build time 3× primary-only** at `k = 256`. Scales as O(n·k·d) in the
  naive implementation. **[V]** Mitigation: top-M candidate restriction
  (roadmap item 1 in the research doc).
- **Query latency ≈ 2× primary-only at the same `nprobe`** because each
  probe scores twice as many candidates. Mitigation: run SOAR at lower
  `nprobe` for the same recall target.
- **Posting-list memory +100 %** (every vector has two id entries).
  Still only +1.5 % of overall index size because posting lists are
  dwarfed by the vectors themselves. **[V]**
- **Not yet validated at scale.** PoC is synthetic 50 k. Public SIFT1M
  + LAION reruns are a roadmap item. **[U]**

### Neutral

- λ is forgiving — a single default value (1.0) works. **[V]**
- `safe` crate, `#![forbid(unsafe_code)]`.
- Four passing unit tests, no mocks.

## Alternatives considered

### Classic nearest-centroid assignment (`PrimaryOnly`)
Keep what ruvector does today. **Rejected as the default** because of
the 20+ pp recall gap at `nprobe=1`. Kept as a selectable
`IvfAssigner` for backwards compatibility and ablation.

### Random spill to a second centroid (`RandomSpill`)
Spill every vector to a deterministic-random second centroid. **Rejected
on evidence**: measured gain over `PrimaryOnly` is +0.1 pp — within
noise. **[V]** Kept as an ablation to prove SOAR's gain is from
orthogonality.

### Multi-centroid spill (r-way SOAR)
The paper reports further gains at r=3. **Deferred to roadmap item 2**:
adds another O(r) factor to build time and triples posting-list memory
for a smaller incremental gain. Revisit after scale validation.

### SPANN-style "probe multiple neighbours of centroid" at query time
Query-side alternative that spills probe cost into the query path.
Doesn't help the fundamental correlation problem SOAR solves and
competes with existing ruvector `ivf-ball-pruning` and
`ivf-radius-early-terminate` work. Keep SOAR complementary to those.

### Graph-based routing (DiskANN, HNSW)
Different index family. Not an alternative — ruvector has both; this
ADR is specifically about improving the IVF family.

## Follow-ups

- M1: Top-M candidate restriction (production-grade build time).
- M2: Rayon parallel build.
- M3: SIFT1M + LAION-400M-subset validation runs.
- M4: SOAR + RaBitQ fused assigner (shared research with
  `ruvector-rabitq` family).
- M5: Expose through the Python SDK (ADR-352) once the above lands.
