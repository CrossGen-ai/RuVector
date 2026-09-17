# ADR-346: Software-Prefetched HNSW Neighbour Traversal

- **Status:** Accepted (2026-09-17, nightly research)
- **Deciders:** ruvector nightly research automation
- **Related:** ADR-272 (diverse-beam), ADR-347 (aq-beam) — orthogonal;
  neither addresses cache-miss latency during graph expansion.

## Context

HNSW graph search is memory-bound on modern CPUs once the dataset
exceeds L3. Each candidate popped from the frontier triggers a scatter
of neighbour-vector loads: for `m = 16` edges and `dim = 128 · 4 B`
vectors, that is 16 × 512 B = 8 KiB of random reads whose addresses
are not known until the CSR row is dereferenced. Auto-vectorised
distance kernels stall waiting on those loads even on Sapphire Rapids
class hardware.

Prior ruvector nightly work has attacked the same bottleneck along
three complementary axes (cache-oblivious layout — 2026-07 series;
locality-optimised HNSW; simhash / cascade ADC prefilters). None of
them issue explicit software prefetch during traversal, and none of
them adapt the prefetch depth to the query's exploration pattern.

## Decision

Add `crates/ruvector-hnsw-prefetch` implementing three swappable
traversal backends behind a common trait boundary:

1. `NoPrefetch`
2. `Prefetch1` (fixed one-hop lookahead)
3. `Adaptive` (depth ∈ [1, 6], grows on accepted neighbour, shrinks
   on rejected — a two-armed bandit signalled by the candidate heap)

The prefetch primitive is a portable `prefetch_read<T>` helper that
lowers to `_mm_prefetch(_MM_HINT_T0)` on x86_64, `PRFM PLDL1KEEP` on
aarch64, and a `black_box` read-hint elsewhere. Prefetches are always
read-only and never dereferenced, so the helper is safe even against
one-past-the-end pointers.

Correctness invariant: prefetch policy MUST NOT change search results.
Enforced by unit test `adaptive_matches_baseline_result_set_size`.

## Consequences

**Positive**

- Real measured latency reduction on the reference machine:
  Adaptive is **6.7 % faster** than NoPrefetch at identical distance
  calls (898/q) and identical recall (0.268 @10, n = 20 000,
  dim = 128, ef = 64). See `crates/ruvector-hnsw-prefetch/README.md`.
- Zero API breakage in the rest of ruvector — the new crate is
  standalone.
- The `TraversalBackend` enum makes it trivial for downstream crates
  (`ruvector-bidir-hnsw`, `ruvector-diskann`, `ruvector-coherence-hnsw`)
  to opt in behind a feature flag.

**Negative / mitigations**

- Prefetch pays a small fixed cost per candidate (~1 ns per line on
  Zen 4 / M-series). On very small graphs (n < 5 000, dim < 32) that
  cost can exceed the savings; benchmark thresholds are documented.
- Adaptive backend requires one branch per neighbour to update depth.
  Measured overhead in the current release build is under 0.5 % of
  total traversal cost.
- Behaviour on non-x86_64 / non-aarch64 hosts is a no-op read hint;
  the `Adaptive` bookkeeping remains but delivers no wall-clock gain.

**Neutral**

- Distance-call counts are unchanged: prefetch is a pure memory-side
  optimisation and does not prune traversal.

## Alternatives Considered

- **Vector reordering (cache-oblivious HNSW)** — already covered by
  the 2026-07-19 nightly work. Complementary, not competing:
  prefetching still helps once vectors are reordered because the
  neighbour-id scatter is orthogonal to intra-vector locality.
- **AVX-512 gather** — not portable enough for the ruvector Rust MSRV
  and cannot mask stale addresses.
- **Hardware stride prefetcher** — CPU IP-stride prefetchers give up
  when the stride is data-dependent, which HNSW neighbours always are.

## References

- Yang et al., *"Software prefetching for graph analytics"*, VLDB
  2024.
- Malkov & Yashunin, *"Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs"*,
  IEEE TPAMI 2020.
- Intel SDM Vol. 2, `PREFETCHh` semantics.
