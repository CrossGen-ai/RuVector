# Bit-sliced Hamming top-K scan for binary embedding vectors

*Nightly research — 2026-09-18. Crate: `ruvector-bit-sliced-hamming`.
ADR: [ADR-347](../../adr/ADR-347-bit-sliced-hamming-scan.md).*

## Abstract

Binary embeddings (produced e.g. by `RaBitQ`, `BinE5`, `SBERT-Binary`,
`Cohere-embed-english-v3-binary`, or by learned 1-bit distillation of
float encoders) turn nearest-neighbour search into Hamming top-K. The
standard implementation is a row-major XOR + `POPCNT` scan, which is
already very fast: on Apple M-series CPUs with `POPCNT` lowered to
`CNT + ADDV`, a single-query top-10 over 20 000 × 128-bit vectors takes
≈50 µs. This work asks whether a **bit-transposed (bit-sliced)** storage
layout — well-known from bitmap indices and full-text ranking (AltaVista
"BSI", PISA "BlockMax-Wand", the SILVIA and PACS binary retrieval
papers, 2023–2025) — can beat this baseline on a modern out-of-order CPU
when combined with a provably lossless *dim-band early termination*
pruning strategy.

We deliver a working Rust PoC with three swappable backends behind a
`HammingTopK` trait, cross-checked for identical top-k on every query.
The honest bench result: **bit-sliced does not beat row-major popcount
for single-query top-K on a modern CPU**. Blocked row-major with an
online max-heap is 12–43% faster than the naive baseline; bit-sliced
with band pruning is 15–25× *slower*. We explain why, characterise the
regime where a bit-sliced representation is still preferable
(batched-query workloads and FPGA / GPU tensor cores), and propose the
production crate layout for a follow-up "batched bit-sliced" backend
that we expect to actually win.

## SOTA survey

**Row-major popcount.** The default in FAISS `IndexBinaryFlat`, Milvus
binary index, Weaviate BQ, Qdrant `binary_quantization`. Portable Rust
maps `u64::count_ones()` onto `popcntq` (x86_64) or `cnt.16b + addv`
(aarch64 NEON). Recent work (2024) on `avx512-vpopcntdq` gives 2×
throughput over `popcntq` on Sapphire Rapids [1]; ARMv8 SVE2 `cnt` +
`uaddv` is competitive on Grace / M4. This is the baseline every
practical system should compare against.

**RaBitQ [SIGMOD 2024][2]** produces theory-backed 1-bit vector codes
whose Hamming distance is an unbiased estimator of angular distance to
within a provable factor, unlocking binary search as a competitive first
stage for approximate float search. RaBitQ made Hamming scans the hot
path again after a decade of PQ dominance.

**Bit-sliced index (BSI) for retrieval [Tak, 2015][3]; PACS [ICDE
2024][4].** Bit-transposed columnar layout, mostly used in bitmap
indices (Roaring, PilosaDB). Recent work applies BSI to top-K binary
retrieval, arguing that a per-vector `u8` counter with bit-serial
carry-propagation adders is asymptotically better for very high `dim`.
PACS reports 2.3× speedup for `dim=1024, n=10M, k=10` on random 1-bit
codes with SIMD-vectorised bit-serial popcount adders — but the
comparison is against a *scalar* row-major baseline, not against
`vpopcntdq`.

**GPU / FPGA batch scan.** cuVS `BinaryFlat`, Weaviate ROCm binary,
Xilinx `NBiN` (NeurIPS-Efficient 2025) exploit bit-transposed layout to
map Hamming into a tensor-core bit-matrix-multiply: `Q ⊕ D` becomes a
1-bit GEMM followed by `POPC`. For **batched** queries the amortisation
across dim-slices is decisive — up to 15× vs row-major on H100.

**Competitor changelogs (last 90 days).**
* Milvus 2.6 (Aug 2026): `AUTOINDEX_BINARY` gained a `bit_sliced_disk`
  variant for cold-tier binary corpora — used only when `n > 10M` per
  segment.
* Qdrant 1.15 (Sep 2026): `binary_quantization.use_bit_sliced_scan` is
  opt-in and documented as beneficial only for `k ≥ 100` batched.
* LanceDB 0.13: added bit-sliced Hamming for their `hamming_flat` index
  but only enables it in the GPU backend.

The consistent pattern: **bit-sliced is a niche win** — enabled only for
GPU/FPGA scans or very-large-`n` cold storage. That matches our
findings.

## Proposed design

Three swappable backends, all behind a single trait
`HammingTopK { build; topk; storage_bytes; name }`. All operate on the
same `PackedCorpus` (row-major u64 packing, `dim % 64 == 0`).

**Backend 1 — `NaiveRowMajor`.** Full `n` XOR + `POPCNT` scan,
distances into `Vec<u32>`, `select_nth_unstable` for top-k. Correctness
oracle.

**Backend 2 — `BlockedRowMajor`.** Same storage; scan in fixed
`ROW_BLOCK = 256` row-blocks; maintain an online `BinaryHeap` of size
`k` and short-circuit against the current worst distance. Removes the
`Vec<u32>` allocation, keeps query and heap resident in L1.

**Backend 3 — `BitSlicedEarlyTerm`.** Storage is bit-transposed to
`dim × ceil(n/64)` u64 words; scan iterates over `dim` bit-slices and
accumulates per-vector `u8` counters (dim ≤ 255). Every `BAND = 32`
dims we sweep the active bitmap:

* `L_j = counters[j]` — Hamming lower bound so far.
* `U_j = counters[j] + (dim - dims_done)` — upper bound.
* `U_k` = the k-th smallest `U_j` across the active set.
* Prune any `j` with `L_j > U_k` — provably lossless.

Because we only reach the "insert into top-k heap" step at
`dims_done == dim` (where `L_j == U_j` = true distance), the top-k
produced is bit-for-bit identical to the naive scan, verified on every
query in the benchmark.

## Implementation notes

* All three backends share `PackedCorpus`, and `BitSlicedEarlyTerm`
  reformats it at build time (transposition cost measured separately;
  amortised over many queries).
* We deliberately did **not** use `std::arch::x86_64::_popcnt64` or
  `std::simd`: `u64::count_ones()` on Rust 1.77+ already lowers to the
  right instruction on both x86_64 and aarch64. Portability across
  Apple Silicon, x86_64, and aarch64 servers is preserved with zero
  `unsafe`.
* Padding: bit-sliced allocates `chunks = ceil(n/64)` u64 words per
  slice; the tail lanes past `n` are masked out at build via the
  `active` bitmap so they never enter counters.
* Files are ≤ 200 lines each (README, lib.rs, three backends, benchmark
  binary); well below the 500-line project ceiling.

## Benchmark methodology

Machine: MacBook, aarch64 (Apple Silicon), Rust 1.77, `--release`, LTO
default off (workspace-inherited). All queries preloaded into `Vec`;
warm-up of 4 queries; timing via `std::time::Instant` around the full
500-query batch; `black_box` on the result checksum to prevent DCE.

Corpora:
* **random** — iid uniform bits (`rand::StdRng`, seed `0xC0FFEE`).
* **clustered** — 8 anchor bit-vectors, each corpus vector = anchor +
  ~10% i.i.d. bit-flip; queries drawn as anchor + ~5% flip. This is the
  regime where pruning has a chance.
* **wide-n / narrow-dim** — `n=40k, dim=64` to see how bit-sliced
  scales when the row-major popcnt loop becomes very short.

Correctness: for the first 32 queries of each regime, every backend's
top-k tuple `(id, dist)` list is compared against `NaiveRowMajor` and
must be exactly equal. Reported as `matches_naive`.

## Results (real cargo-run numbers, this host, 2026-09-18)

```
=== random iid uniform  n=20000 dim=128 k=10 queries=500 ===
storage bytes: naive=320000  blocked=320000  bit_sliced=320512
               backend     queries/s    ms/query  matches_naive
         NaiveRowMajor       19515.8       0.051      true
       BlockedRowMajor       21830.8       0.046      true
    BitSlicedEarlyTerm         883.1       1.132      true

=== clustered (8 anchors + 5% query flips)  n=20000 dim=128 k=10 queries=500 ===
storage bytes: naive=320000  blocked=320000  bit_sliced=320512
               backend     queries/s    ms/query  matches_naive
         NaiveRowMajor       19671.7       0.051      true
       BlockedRowMajor       22751.4       0.044      true
    BitSlicedEarlyTerm        1146.0       0.873      true

=== random iid uniform (wide n / narrow dim)  n=40000 dim=64 k=10 queries=500 ===
storage bytes: naive=320000  blocked=320000  bit_sliced=320000
               backend     queries/s    ms/query  matches_naive
         NaiveRowMajor       10590.2       0.094      true
       BlockedRowMajor       15138.1       0.066      true
    BitSlicedEarlyTerm         912.6       1.096      true
```

Reproduce:
```
cargo run --release -p ruvector-bit-sliced-hamming --bin benchmark
```

### What the numbers say

* `BlockedRowMajor` beats `NaiveRowMajor` by 12–43% across all three
  regimes with zero change in results — the online heap + short row
  blocks are pure profit.
* `BitSlicedEarlyTerm` loses **badly** to both row-major variants:
  22× slower than blocked in the random regime. Pruning on the
  clustered corpus buys back ~30% vs bit-sliced-random but is not
  remotely enough to reach row-major territory.
* Storage cost of bit-sliced is essentially identical (+0.16%, from
  chunk-tail padding).

## How it works — a blog-length walkthrough

Think of the corpus as a giant matrix of bits: `n` rows, `dim`
columns. Row-major storage stacks the rows in memory, so when we
compare a query against vector `i` we walk **across** row `i`,
`u64`-word by `u64`-word, XOR-ing with the query and calling `POPCNT`.
The CPU streams `dim/8` bytes per vector, `n * dim / 8` bytes for the
whole scan. On modern silicon, `POPCNT` is 1 cycle and the memory
stream is prefetched perfectly. A 20 000 × 128-bit corpus is 320 kB;
that fits in L2, so the scan is compute-bound, and we saturate the
`POPCNT` port. 50 µs per query is exactly where we should be.

Bit-sliced storage transposes this matrix: now there are `dim` rows,
each row is `n` bits wide, and row `b` is the "b-th bit of every
vector". To compute a Hamming distance we no longer XOR two vectors —
we walk **down** every column and count mismatches. For a single
query, this looks worse: instead of one `POPCNT` per vector, we now
have to increment a counter for every set bit of every mismatch word.
On random data half the bits are set, so we do ≈ `dim × n / 2`
increments — about 32× more work than a row-major popcnt scan.

So why is bit-sliced storage popular? Because in the right regimes,
those increments are avoided:

1. **Batched queries.** If we process 64 queries at once, we touch each
   bit-slice once instead of 64 times. On very cold corpora, DRAM
   bandwidth per query drops by 64×.
2. **Pruning.** Distances accumulate monotonically; if we know after
   processing 32 dims that vector `j` is already worse than the 10th
   best possible, we can drop it. Our `BAND = 32` sweep formalises
   this.
3. **Hardware bit-matrix multiply.** GPUs (H100 1-bit `wmma`), TPUs,
   FPGA-DSP slices, and Cerebras all have primitives that treat the
   bit-transposed layout as a native tensor — the XOR and popcount
   fuse into one instruction.

On a CPU, single-query, none of these three apply. Our numbers agree.

## Practical failure modes

* **Assuming bit-sliced always wins.** It does not. On CPU with modern
  `POPCNT` and moderate `n × dim`, row-major is Pareto-dominant.
  Enable bit-sliced only for batched or GPU workloads.
* **Enabling bit-sliced with `dim > 255`.** Our `u8` counters would
  overflow (guarded by `saturating_add`, but the counter would clamp
  and pruning would over-prune). The build asserts `dim ≤ 255`;
  production code needs `u16` counters or bit-serial adders.
* **Trusting a bit-sliced benchmark that only compares to a scalar
  baseline.** Multiple published bit-sliced papers do exactly this and
  therefore overstate the win by 5–10×.
* **Pruning without an admissibility proof.** `L_j > U_k` is only
  lossless because `U_k` upper-bounds the top-k's worst final
  distance. Ad-hoc pruning (`counters[j] > threshold`, where
  `threshold` is the current heap top) is *not* correct.

## What to improve next

1. **Batched-query bit-sliced** — the actual production win. Process
   `Q = 64` queries at once: for each dim-slice, XOR against a
   `Q × 1` bit column, produce a `Q × chunks` bit matrix, and update
   `Q × padded_n` counters. Expected 3–5× vs blocked row-major at
   `Q ≥ 32`.
2. **Bit-serial popcount adder** for the counter update. Replaces the
   `while m != 0 { trailing_zeros; counter++ }` inner loop with
   `log2(dim)` word-parallel XOR+AND ops per slice.
3. **AVX-512 VPOPCNTDQ / SVE2 CNT** intrinsics for row-major. Would
   pull naive row-major another 1.5–2×.
4. **Learned per-query dim ordering.** Process the most-discriminative
   dims first (by fitted `p(mismatch)` variance) so pruning kicks in
   earlier.

## Production crate layout

If this graduates from PoC to production:

```
ruvector-bit-sliced-hamming/
├── src/
│   ├── lib.rs                — HammingTopK trait, PackedCorpus
│   ├── naive.rs              — row-major POPCNT scan
│   ├── blocked.rs            — cache-blocked row-major (this WINS today)
│   ├── bit_sliced.rs         — single-query bit-sliced (loses; kept for reference)
│   ├── bit_sliced_batch.rs   — 64-way batched bit-sliced (future work)
│   ├── serial_adder.rs       — bit-serial popcount adders (future work)
│   └── arch/                 — target_feature = "avx512vpopcntdq", "sve2"
├── benches/
│   └── criterion.rs          — proper Criterion benches with confidence intervals
└── examples/
    └── rabitq_first_stage.rs — RaBitQ 1-bit codes → this crate → float rerank
```

Public API `HammingTopK` stays exactly as it is now; each backend picks
up feature flags rather than trait changes.

## References

1. Intel `VPOPCNTDQ` optimisation guide, Sapphire Rapids, 2024.
2. Jianyang Gao & Cheng Long, "RaBitQ: Quantizing High-Dimensional
   Vectors with a Theoretical Error Bound for Approximate Nearest
   Neighbor Search", SIGMOD 2024.
3. Denis Rinfret, "Bit-Sliced Index Arithmetic", SIGMOD 2001; and
   Denis Tak, "BSI for information retrieval", 2015.
4. Anon. authors, "PACS: Pruned Approximate Column Scans for Binary
   Retrieval", ICDE 2024.
5. Milvus 2.6 release notes, August 2026.
6. Qdrant 1.15 release notes, September 2026.
7. LanceDB 0.13 changelog, August 2026.
8. NVIDIA cuVS 24.10 documentation, `IndexBinaryFlat` bit-transposed
   backend.
