# ADR-352: Elias-Fano Compressed IVF Posting Lists

## Status

Proposed. New, opt-in crate `ruvector-ef-ivf-posting` (2026-09-30 nightly).
Not wired into any existing IVF crate yet — this ADR proposes the codec
trait and the eventual switch-over; the follow-up nightly should integrate
into `ruvector-ivf-hnsw-coarse` and/or `ruvector-spann` behind a config
flag.

## Context

Every IVF-family index in ruvector — `ruvector-spann`,
`ruvector-ivf-hnsw-coarse`, `ruvector-soar-ivf`,
`ruvector-crackivf-adaptive`, `ruvector-norm-bucket-ivf` — stores per-
centroid posting lists as raw `Vec<u32>` or equivalent. At production
scales this dominates the non-vector memory of the index:

* nlist=65,536, avg 15,360 docs/list, 1B vectors → ~4 GB of raw ID storage
* nlist=2,048, avg 500K docs/list, 1B vectors → ~4 GB
* Compared to PQ-compressed vector storage (~16–32 bytes/vector), raw IDs
  are up to 25% of the total footprint.

The IR community has known for 50+ years how to compress monotone integer
sequences near-optimally: Elias-Fano encoding (Elias 1974, Fano 1971),
revived for modern search engines by Vigna's *Quasi-Succinct Indices*
(WSDM 2013). EF stores each element in `~2 + ⌈log₂(U/n)⌉` bits, provably
within a constant of the information-theoretic minimum for the class of
strictly increasing sequences of `n` values in `[0, U)`.

Vector-search systems have not adopted this: FAISS, Milvus, Qdrant, and
Weaviate all keep raw int IDs in their posting analogues. Adopting EF is
a pure engineering win with no algorithmic risk.

## Decision

Introduce `ruvector-ef-ivf-posting`, a self-contained crate defining:

```rust
pub trait PostingCodec: Send + Sync {
    fn name(&self) -> &'static str;
    fn encode(&self, sorted: &[u32]) -> Vec<u8>;
    fn decode(&self, bytes: &[u8], out: &mut Vec<u32>);
}
```

with three implementations shipped in v0.1:

1. `RawU32` — baseline; the current behavior in bytes-oriented form.
2. `VByteDelta` — 7-bit continuation gap encoding; matches Lucene / Milvus.
3. `EliasFano` — near-optimal for monotone `u32` sequences.

The trait is deliberately minimal (no `next_geq` yet) so the initial
integration is a mechanical replacement of `Vec<u32>` in IVF cluster
storage. `next_geq`, partitioned EF, and rank/select LUTs are follow-ups.

## Consequences

**Positive**
* Measured **3.25×** compression on dense/uniform IVF (256×4K, U=1M),
  **3.80×** on very-dense (64×16K, U=1.5M), **1.95×** on sparse/wide
  (1024×500, U=10M) — Apple M4 Max, release build.
* Compression ratios are hardware-independent and reproducible from seed.
* Trait-based design lets each IVF crate pick codec at runtime; existing
  callers pass `Arc<RawU32>` and keep current behavior byte-for-byte.
* 7 randomized roundtrip tests, including EF-beats-raw density check.

**Negative / Cost**
* Decode is 2–15× slower per ID than raw. At 47 ns/id (dense) this is
  still cheap next to a PQ ADC scan (40–200 ns) or a full-float distance
  (>500 ns), so total query latency will not measurably regress once IVF
  actually does distance work per decoded ID. On a decode-only benchmark
  the slowdown is real.
* Not suitable for streaming-mutation IVF paths. Encoding is immutable;
  inserts must batch and periodically re-encode. This matches the
  freeze-and-merge pattern used by Milvus segments and Lucene tiers.
* Tiny lists (n < 64) pay header overhead. Callers should keep small
  lists as raw and encode only past a threshold.

**Neutral**
* No changes to public IVF APIs until a follow-up wires the trait in.
* Codec identity is stored in the encoded blob header (via crate-level
  choice), so on-disk format is codec-tagged from day one.

## Alternatives Considered

1. **Keep raw `Vec<u32>`.** Simplest, fastest single-list decode, but
   forfeits 2–4× memory on the largest index component. Rejected.
2. **VByte-delta only.** Ships in Lucene and Milvus; ~2× compression, easy
   to `next_geq`. Rejected as *sole* option because EF beats it by
   15–25% on dense lists at similar decode speed. Kept as one of the
   three shipped codecs for A/B and for regimes (sparse/wide) where EF's
   advantage narrows.
3. **Partitioned Elias-Fano (Ottaviano & Venturini 2014).** Better
   compression on clumpy real-world IDs (10–25% further). Deferred: needs
   a partition-selection heuristic and doubles the codec-surface for a
   marginal gain. Correct next step after this ADR lands.
4. **FastPFor / SIMD-BP128.** Byte-aligned SIMD codecs from Lemire &
   Boytsov; faster decode than EF, similar compression. Deferred: adds a
   non-trivial SIMD dependency (`fastpfor-rs`) and only pays off if
   decode is on the critical path — which it usually isn't in IVF, since
   distance computation dominates.
5. **Roaring bitmaps.** Excellent for high-density lists (n / U > 1/16)
   but wasteful for typical IVF (n / U ≪ 1/16). Rejected as a general
   default; may be worth adding as a codec for very-dense clusters.

## Follow-ups

* **Next nightly:** wire `PostingCodec` into `ruvector-ivf-hnsw-coarse`
  with a `posting_codec` config field, default `RawU32`, and measure
  end-to-end query latency + memory at 1M and 10M vectors.
* **After that:** partitioned EF with 128-element blocks; NEON unary
  decode; `next_geq` rank/select LUTs.

## References

* Elias, P. "Efficient Storage and Retrieval by Content and Address of
  Static Files." JACM 21, 1974.
* Vigna, S. "Quasi-Succinct Indices." WSDM 2013.
* Ottaviano, G., & Venturini, R. "Partitioned Elias-Fano Indexes." SIGIR
  2014.
* Pibiri, G. E., & Venturini, R. "Techniques for Inverted Index
  Compression." ACM Computing Surveys 2020.
* Research doc: `docs/research/nightly/2026-09-30-ef-ivf-posting/README.md`
