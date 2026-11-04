# ADR-355: Binary Centroid Probe Shortlist for IVF

- **Status:** Proposed
- **Date:** 2026-11-04
- **Scope:** `crates/ruvector-core` IVF probe-selection path; new crate
  `crates/ruvector-bcps` holds the implementation and will be wired in
  as a `ProbeSelector` behind a cargo feature.

## Context

Every IVF query begins with a probe-selection step: compute distances
from the query to all `nlist` coarse centroids and keep the top
`nprobe`. For `nlist = 2^16, d = 128` this is 8M FMA ops reading
32 MB of float32 per query — on the measured hardware (Apple M4 Max,
release build) this is **3.75 ms per query**, incurred *before any
posting-list work*.

ruvector's IVF variants currently use `IndexFlatL2` or `HnswProbe`
for this step. Flat is cache-friendly at small `nlist` but scales
linearly; HNSW-over-centroids scales well but costs memory and adds
build complexity. We want a third option that is small, fast, and
effective in the `nlist ∈ [2^12, 2^17]` band where many real
deployments live.

Recent work on 1-bit vector quantization (SimHash 2002, RaBitQ 2024)
showed that angular distance can be estimated cheaply from
XOR+popcount over rotated sign-bits. The literature applies this to
*data* vectors; the probe table has been under-explored.

## Decision

Add a `BinarySignatureSelector` probe selector that:

1. Mean-centers the centroid table at build time.
2. Applies `K` stacked SRHT rotations (sign-flip followed by Fast
   Walsh-Hadamard Transform) to each centroid and takes the sign of
   each coordinate, producing a `K · next_pow2(d)`-bit signature.
3. At query time, applies the same rotations to the query, takes
   signs, computes Hamming distance to every centroid signature with
   `(x ^ y).count_ones()`, shortlists the top `M` by Hamming.
4. Recomputes true L2 on the `M` survivors and returns the top
   `nprobe`.

Default knobs: `K = 4`, `M = 8 · nprobe` for routine use,
`M = 16 · nprobe` for strict recall.

The selector is implemented in `ruvector-bcps` as a workspace-sibling
crate with no dependency on `ruvector-core`. Integration is a thin
adapter in a follow-up PR behind the feature flag `bcps-probe`.

## Consequences

**Positive**

- 3.8–6.4× lower latency on the probe-selection step at
  recall@nprobe ≥ 0.86 in the measured regime.
- 8–32× lower memory for the probe-selection path (64 bytes per
  centroid at `K=4, d=128` vs 512 bytes raw float32).
- No `unsafe`, no new system dependencies. Pure Rust, portable to
  wasm with no changes.
- Orthogonal to the data-scan path — stacks on top of PQ, SQ,
  RaBitQ, or any other data-vector quantization.

**Negative**

- Fixed per-query overhead (rotation + signature build) is
  ~0.5–2 µs. For `nlist < 2^11` the exact L2 baseline is faster.
- Recall is not free: tuning `(K, M)` is now a config surface for
  operators. We mitigate with sensible defaults and a tuning recipe
  in the research doc.
- Signatures must be rebuilt on centroid updates or mean drift. For
  streaming IVF variants (`ruvector-online-ivf-centroid-refresh`)
  this adds a maintenance hook.

**Neutral**

- At `nlist ≥ 2^18` the better complement is HNSW-over-centroids;
  BCPS can be used *inside* the HNSW layers but that is a separate
  investigation.

## Alternatives

1. **HNSW-over-centroids.** Already available
   (`ruvector-ivf-hnsw-coarse`). Lower memory than a full float
   table, but still larger than BCPS and requires graph construction.
   Better for `nlist ≥ 2^18`; we recommend keeping both and letting
   the operator choose by `nlist`.
2. **ScaNN-style anisotropic PQ of centroids.** Higher recall per
   bit than 1-bit signatures, but ~10× the arithmetic. BCPS wins on
   speed; PQ-of-centroids wins on recall. We may add it as a third
   selector in the future.
3. **Full random-Gaussian projection + sign.** Strongest LSH
   guarantee but `O(d²)` rotation cost, which negates the speedup.
4. **Residual re-ranking only (no shortlist).** Skip the signature
   pass; rely on PQ residual rerank downstream. Does not help —
   probe selection remains O(nlist · d).

## Implementation plan

1. (This ADR) Land `ruvector-bcps` standalone crate with the
   selector, benchmarks, demo. — Done on this branch.
2. Wire a `BcpsSelector` adapter into `ruvector-core::ivf::probe`
   behind cargo feature `bcps-probe`. — Follow-up PR.
3. Add a `probe_selector` field to the ruvector IVF config with
   variants `{ Flat, HnswCoarse, Bcps { k, m_mult } }`.
4. Extend the `ruvector-sota-bench` harness to sweep `(k, m_mult)`
   on a real embedding dataset (SIFT1M, GIST1M) and publish the
   recall/latency curve against FAISS `IndexIVF::quantizer`.
