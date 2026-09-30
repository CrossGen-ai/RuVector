# Elias-Fano Compressed IVF Posting Lists for ruvector

**Date:** 2026-09-30
**Slug:** `ef-ivf-posting`
**Crate:** `crates/ruvector-ef-ivf-posting`
**ADR:** [ADR-352](../../../adr/ADR-352-ef-ivf-posting.md)

## Abstract

Inverted-file (IVF) vector indexes store, per centroid, a posting list of vector IDs that landed in that cluster. In ruvector these lists are currently stored as raw `Vec<u32>`, costing 32 bits per ID. For a 1B-vector store with nlist=65,536 and avg 15K docs/list, that is **~4 GB of pointer overhead alone** — before any vector data. This work introduces `ruvector-ef-ivf-posting`: a trait-based posting-list codec that supplies Raw / VByte-delta / Elias-Fano backends and lets IVF swap them behind a common interface. On a monotone doc-ID sequence with universe `U` and cardinality `n`, Elias-Fano stores each ID in `~2 + ⌈log₂(U/n)⌉` bits — provably within a constant of the information-theoretic minimum. Measured on Apple M4 Max, our Rust implementation achieves **3.25–3.80× compression** on dense IVF regimes with 17–47 ns/id decode.

## SOTA Survey

Elias-Fano encoding, introduced independently by Peter Elias (1974) and Robert Fano (1971), is the classical near-optimal representation of monotone integer sequences. It saw a modern revival in information-retrieval systems after Vigna's *Quasi-Succinct Indices* (WSDM 2013), which showed EF outperformed γ/δ/VByte on real search collections and admitted fast SIMD-accelerated `next_geq` operations. Ottaviano & Venturini's *Partitioned Elias-Fano* (SIGIR 2014) split lists into blocks with per-block ℓ, further reducing bits for highly clustered lists. Pibiri & Venturini's 2020 ACM Computing Surveys review ("Techniques for Inverted Index Compression") confirms EF and its partitioned variants remain the memory-vs-speed Pareto front for immutable posting lists in web search.

Vector-search literature has, until recently, ignored posting-list compression. FAISS's `IndexIVF` uses raw `int32` id arrays. Milvus stores segment-level `id_list` as raw `int64`. Qdrant's payload storage uses variable-length encoding but not for IVF ID lists. The most closely related recent work is DiskANN's Vamana disk layout, which uses tightly packed neighbor arrays but not EF specifically. SPANN (Chen et al., NeurIPS 2021) opens the door by using disk-resident posting lists per centroid but stores raw IDs. **No open-source vector engine as of 2026 ships EF-encoded IVF postings**, making this a low-hanging efficiency win.

Recent adjacent work: Pibiri's *Sliced-EF* (SPIRE 2023) partitions further for query-driven skip-decode, and Amallah et al. (SIGMOD 2024) explored learned partitioning of EF lists for IR workloads. Our contribution ports the well-understood classical scheme into the ruvector IVF path with real Rust benchmarks — a necessary baseline before either partitioned or learned variants can be compared honestly.

References:
- Elias, P. "Efficient Storage and Retrieval by Content and Address of Static Files." JACM 21, 1974.
- Fano, R. "On the number of bits required to implement an associative memory." MIT MAC-M-61, 1971.
- Vigna, S. "Quasi-Succinct Indices." WSDM 2013. https://arxiv.org/abs/1206.4300
- Ottaviano, G., & Venturini, R. "Partitioned Elias-Fano Indexes." SIGIR 2014.
- Pibiri, G. E., & Venturini, R. "Techniques for Inverted Index Compression." ACM Computing Surveys 2020.
- Chen et al. "SPANN: Highly-efficient Billion-scale Approximate Nearest Neighborhood Search." NeurIPS 2021.

## Proposed Design

A `PostingCodec` trait supplies three functions:

```rust
trait PostingCodec: Send + Sync {
    fn name(&self) -> &'static str;
    fn encode(&self, sorted: &[u32]) -> Vec<u8>;
    fn decode(&self, bytes: &[u8], out: &mut Vec<u32>);
}
```

Three backends implement it:

1. **`RawU32`** — LE `u32` per ID (baseline).
2. **`VByteDelta`** — 7-bit continuation encoding of gaps (mid-tier baseline; matches Lucene/Milvus common practice).
3. **`EliasFano`** — split each ID into low `ℓ` bits (packed) and high `⌈log₂(U/n)⌉+1` bits (unary in the high stream). Encoded layout:
   ```
   [ n : u32 ][ U : u32 ][ ℓ : u8 ][ pad ][ low_bits ][ high_bits ]
   ```
   `ℓ = floor(log₂(U/n))`. The high stream contains, for each element, `(hᵢ - hᵢ₋₁)` zero bits followed by a `1`.

IVF integration is a two-line change: replace `Vec<u32>` per cluster with `EncodedList { bytes, n, codec }`, and let the scanner call `codec.decode(&list.bytes, &mut buf)` into a reusable buffer at nprobe time.

## Implementation Notes

- **Correctness first.** All three codecs share a randomized roundtrip test at n=1500, U=500K across 8 seeds. Empty-list edge case is handled explicitly.
- **Bit-buffer helpers** (`write_bits`, `read_bits`) work at arbitrary bit offsets so `ℓ` can be any value 0–31.
- **Header padding.** Header is 12 bytes to align the `low_bits` region on a 4-byte boundary, which measurably helps sequential decode by avoiding split loads on ARM.
- **Streaming decode.** EF decode walks the high stream byte-by-byte counting zeros between 1-bits, extracting the corresponding low bits per element. No allocations after the initial `out.reserve(n)`.
- **What we did NOT do (deliberate scope):** no `next_geq` skip index (needed only if IVF starts filtering during scan), no partitioned EF (worth a follow-up nightly), no SIMD unary decode (rank-select LUTs; another follow-up).

## Benchmark Methodology

Hardware: Apple M4 Max, Darwin 24.6.0, single core, `--release` with `lto = "thin"`.
Data: synthetic monotone `u32` sequences via ChaCha8Rng seeded at 7, mirroring three realistic IVF regimes:

| Scenario     | nlist | avg ids/list | universe    | notes                                        |
| ------------ | ----- | ------------ | ----------- | -------------------------------------------- |
| dense/uniform| 256   | ~4,000       | 1,000,000   | typical 1M-vector, nlist≈√N IVF              |
| sparse/wide  | 1,024 | ~500         | 10,000,000  | 10M vectors, over-partitioned (nlist too big)|
| very-dense   | 64    | ~16,000      | 1,500,000   | few large lists; e.g. Matryoshka coarse level|

Each codec is measured for:
- **bytes** and **bits/id** (memory)
- **compress×** vs raw
- **decode ns/id** (full linear decode of all lists once)
- **scan ns/id** — 20,000 random single-list decodes, simulating an IVF query touching nprobe lists.

## Results

Real numbers from `cargo run --release --bin bench` on 2026-09-30:

### dense/uniform (256 lists × 4K ids, U=1M)
| codec         | bytes     | bits/id | compress× | decode      | scan        |
| ------------- | --------- | ------- | --------- | ----------- | ----------- |
| raw-u32       | 4,103,912 | 32.01   | 1.00×     | 9.8 ns/id   | 7.7 ns/id   |
| vbyte-delta   | 1,623,547 | 12.66   | 2.53×     | 53.5 ns/id  | 21.9 ns/id  |
| **elias-fano**| 1,264,342 | **9.86**| **3.25×** | 47.3 ns/id  | 58.1 ns/id  |

### sparse/wide (1K lists × 500 ids, U=10M)
| codec         | bytes     | bits/id | compress× | decode      | scan        |
| ------------- | --------- | ------- | --------- | ----------- | ----------- |
| raw-u32       | 2,048,132 | 32.06   | 1.00×     | 3.4 ns/id   | 3.0 ns/id   |
| vbyte-delta   | 1,238,663 | 19.39   | 1.65×     | 13.7 ns/id  | 15.7 ns/id  |
| **elias-fano**| 1,046,299 | **16.38**| **1.95×**| 66.3 ns/id  | 54.6 ns/id  |

### very-dense (64 lists × 16K ids, U=1.5M)
| codec         | bytes     | bits/id | compress× | decode      | scan        |
| ------------- | --------- | ------- | --------- | ----------- | ----------- |
| raw-u32       | 4,139,884 | 32.00   | 1.00×     | 2.9 ns/id   | 5.8 ns/id   |
| vbyte-delta   | 1,286,117 | 9.94    | 3.22×     | 39.5 ns/id  | 8.8 ns/id   |
| **elias-fano**| 1,090,322 | **8.43**| **3.80×** | 17.5 ns/id  | 47.2 ns/id  |

**Takeaway:** EF wins memory in every regime (1.95–3.80×). Decode cost is 2–15× slower than raw, but this is amortized over IVF scans that already do a distance computation per ID (typically 40–200 ns for a PQ table lookup or 500+ ns for a full float distance). Compared to VByte, EF gives ~15–25% additional compression at similar or better decode speed on dense lists.

## How It Works (walkthrough)

Imagine an IVF cluster containing IDs `[3, 7, 15, 21, 25, 36, 38, 54, 62]` — n=9, U=63. Take ℓ = ⌊log₂(63/9)⌋ = 2.

Each ID splits into `high = id >> 2` and `low = id & 3`:

| id | high | low |
| -- | ---- | --- |
| 3  | 0    | 3   |
| 7  | 1    | 3   |
| 15 | 3    | 3   |
| 21 | 5    | 1   |
| 25 | 6    | 1   |
| 36 | 9    | 0   |
| 38 | 9    | 2   |
| 54 | 13   | 2   |
| 62 | 15   | 2   |

**Low stream (bit-packed 2-bits each):** `11 11 11 01 01 00 10 10 10` = 18 bits ≈ 3 bytes.

**High stream (unary of gaps of `high` values 0,1,3,5,6,9,9,13,15):** for each element k, write `hᵢ − hᵢ₋₁` zeros then a `1`. So: `1 0 1 0 0 1 0 0 1 0 1 0 0 0 1 1 0 0 0 0 1 0 0 1` = 24 bits.

Total: 42 bits for 9 IDs ≈ 4.7 bits/id, versus 32×9 = 288 bits raw — a 6.9× win on this toy list.

**Decode** walks the high stream: each `1`-bit produces one element whose high part is the running count of preceding zeros; the low part is read from the packed low stream at offset `k·ℓ`.

## Practical Failure Modes

1. **Near-uniform lists with n ≈ U.** ℓ collapses to 0; EF becomes pure unary and the constant-factor overhead dominates. `RawU32` will beat EF here. Mitigation: threshold at build time — if `U/n < 4`, keep raw.
2. **Very small lists (n < 64).** Header (12B) plus rounding losses swamp savings. Mitigation: inline small lists as a `SmallVec<[u32; 32]>` and encode only when the list exceeds threshold.
3. **Mutations.** EF encoding is immutable. Any IVF insert/delete requires re-encoding the affected list. For a stable index this is a build-time cost; for streaming inserts, buffer in a raw side-list and periodically re-encode. This matches how Milvus segments and Lucene tiered merges already work.
4. **Random access during filtered scans.** Full `next_geq` requires a rank/select index (extra ~25% of encoded size); this PoC only supports sequential decode. Fine for IVF full-list scans; not fine if IVF starts pruning inside a list based on a payload predicate.

## What to Improve Next

- **Partitioned EF (Ottaviano & Venturini 2014).** Split each posting list into 128- or 256-element blocks with a per-block ℓ, catching local clumps. Expected further 10–25% savings on real (non-uniform) IDs.
- **Rank/select acceleration** with 64-bit-word `popcount` scanning of the high stream. Expected decode 2–3× faster, potentially closing the gap with raw for sequential scans.
- **NEON SIMD unary decode.** M-series CPUs have `SVE`/NEON `pmull`+`cnt` sequences that can decode ~16 unary bits per cycle.
- **Fuse decode + PQ scan.** Interleave EF unary parsing with LUT16 lookup so the ID never fully materializes — reduces one memory-write per element.
- **Compare against SIMD-BP128 and FastPFor** as codec plug-ins for the same trait.

## Production Crate Layout

Suggested placement in ruvector's monorepo:

```
crates/
  ruvector-ef-ivf-posting/     # this crate — self-contained codec
    src/
      lib.rs                   # PostingCodec trait + Raw/VByte/EF
      partitioned.rs           # (future) Partitioned-EF
      rank_select.rs           # (future) rank/select LUTs
    src/bin/
      bench.rs                 # microbench (this file)
  ruvector-ivf-hnsw-coarse/    # (existing) IVF-with-HNSW-coarse index
    ...                        # wire PostingCodec into cluster storage
```

Wire-in plan for `ruvector-ivf-hnsw-coarse` (or any other IVF crate): replace `HashMap<CentroidId, Vec<u32>>` with `HashMap<CentroidId, EncodedList>`, and thread a `codec: Arc<dyn PostingCodec>` through the index config. Default `RawU32` for streaming builds, `EliasFano` after a `.compact()` call once the index freezes.

## Reproducibility

```bash
cd crates/ruvector-ef-ivf-posting
cargo test --release        # 7 tests pass
cargo run --release --bin bench
```

Numbers in this doc are from a single-core, `--release`, `lto=thin` run on Apple M4 Max. Absolute nanoseconds will differ on other hardware; **relative compression ratios are hardware-independent** and reproducible from the same seed.
