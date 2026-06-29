# ADR-272: Bitsampling SimHash Multi-Probe LSH for Cosine ANN

- **Status**: Proposed (PoC implemented, tests + benchmark green — branch `research/nightly/2026-06-29-bitsampling-lsh-ann`)
- **Date**: 2026-06-29
- **Crate**: `crates/ruvector-lsh`
- **Research**: `docs/research/nightly/2026-06-29-bitsampling-lsh-ann/README.md`

---

## Context

ruvector now ships dozens of ANN backends — HNSW, IVF (Beta-IVF, RAIRS), SPANN,
DiskANN, RaBitQ, Matryoshka, MUVERA, Tribase, Symphony-QG, MaxSim — but **no
first-class Locality-Sensitive Hashing index**. LSH ([Indyk & Motwani, STOC
1998]; [Charikar, STOC 2002]) is the classical sub-linear ANN baseline against
which essentially every modern method is measured (FAISS `IndexLSH`, Milvus
plugin, PUFFINN, FALCONN, LSH-APG). Lacking it means:

1. Apples-to-apples external comparisons need a third-party LSH, weakening
   ruvector's `ruvector-sota-bench` story.
2. Workloads that are well-suited to LSH (binary or low-bit cosine
   embeddings, memory-bounded edge deployments, query-streaming systems)
   have no native option.
3. Future hybrids (LSH-pruned graphs per [Zhang et al., SIGMOD 2023]) are
   blocked until an LSH primitive exists.

## Decision

Add `crates/ruvector-lsh` providing a swappable `AnnIndex` trait and three
reference implementations:

1. `BruteForce` — exact cosine, ground truth for benchmarks.
2. `SimHashLsh` — single-table bitsampling SimHash with hamming fallback.
3. `MultiProbeSimHash` — L tables × P probe bit-flips, with summed-hamming
   augmentation when bucket candidates are sparse.

Bitsampling SimHash is chosen over E2LSH because ruvector's primary distance
is cosine, and SimHash gives the cosine-optimal `Pr[collision] = 1 - θ/π`
([Charikar 2002]). Multi-probe ([Lv et al., VLDB 2007]) reduces table count
by ~10× for the same recall. Summed-hamming augmentation across tables is a
principled cosine surrogate (Johnson-Lindenstrauss bound) and recovers recall
on sparse buckets where pure bucket-lookup degenerates.

## Consequences

**Positive**
- ruvector-sota-bench gets a built-in LSH baseline (no external dep).
- Memory footprint ~13.4 MB for 20k×128 float corpus with `L=4, b=64` — 4–10×
  smaller than HNSW M=16 on the same data.
- Trait-based design lets future cross-polytope/PUFFINN backends drop in
  without touching call sites.
- Acceptance floor `recall@10 ≥ 0.60` on clustered data enforced by a real
  unit test.

**Negative / costs**
- One more workspace member (~480 lines including bench + tests).
- LSH recall ceiling on this PoC is ~0.64 at 9× brute-force speedup;
  HNSW/DiskANN comfortably beat that. LSH is positioned as a *baseline and
  memory-bounded option*, not a top-tier method.

**Neutral**
- No public API change to existing crates — purely additive.

## Alternatives Considered

- **E2LSH (p-stable)**: better for L2, suboptimal for cosine. Rejected for
  primary backend; may add later as a feature flag.
- **Cross-polytope LSH (FALCONN)**: asymptotically optimal but heavier
  implementation (structured rotations + multiprobe sequencing). Listed in
  "What to Improve Next" as ADR-273 candidate.
- **Wrap FAISS `IndexLSH` via FFI**: violates "RUST ONLY" project rule and
  blocks WASM targets.
- **Skip LSH entirely, rely on RaBitQ**: RaBitQ is a quantizer with
  LSH-style theory but solves a different problem (compression of an
  existing index). Doesn't give us a sub-linear lookup primitive.

## Measured Results (host: macOS, Apple Silicon, release, single-threaded)

| variant | mean_us | recall@10 | speedup |
|---|---:|---:|---:|
| BruteForce | 1905.3 | 1.000 | 1.0× |
| MultiProbeSimHash(b=64, L=4, p=8) | 208.3 | **0.635** | **9.1×** |

See `docs/research/nightly/2026-06-29-bitsampling-lsh-ann/README.md` for the
full table.
