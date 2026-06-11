//! PDX vertical block layout for vector similarity scan.
//!
//! Two storage layouts are provided behind a common [`Scanner`] trait:
//!   * [`Horizontal`] — classic row-major `[N × D]` layout.
//!   * [`PdxVertical`] — block-transposed layout: vectors are grouped into
//!     fixed-size blocks of [`BLOCK`] vectors and stored as `D` contiguous
//!     stripes of [`BLOCK`] f32 values per block. A single dimension across
//!     every vector in the block lives in one cache line / SIMD register,
//!     which lets the inner loop accumulate partial L2 distances for
//!     [`BLOCK`] candidates in lock-step. After every [`PROBE_STEP`]
//!     dimensions we compare the running partial sum against the current
//!     k-NN heap threshold and skip vectors that can no longer make the cut
//!     ("PDX-BOND" style pruning).
//!
//! The crate has no external dependencies. The pruning logic is a
//! straightforward translation of the lower-bound argument from
//! Kuffo et al. "PDX: A Data Layout for Vector Similarity Search" (CWI,
//! SIGMOD 2025): the partial L2 sum is monotonically non-decreasing in the
//! number of dimensions accumulated, so once it exceeds the worst (largest)
//! distance currently held in the top-k heap, the candidate cannot
//! displace it and the remaining dim work for that vector can be skipped.

#![forbid(unsafe_op_in_unsafe_fn)]

pub const BLOCK: usize = 64;
pub const PROBE_STEP: usize = 16;

/// Sentinel returned when fewer than `k` vectors exist or pruning is empty.
const SENTINEL: f32 = f32::INFINITY;

/// A `(vector_id, squared_l2_distance)` pair returned by [`Scanner::search`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub id: u32,
    pub dist: f32,
}

impl Eq for Hit {}
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // We keep a max-heap over distance so the worst hit is the root.
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub trait Scanner {
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Return the `k` nearest vectors (squared L2) to `query`, sorted.
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;
    /// Diagnostic: total number of `f32` MULs the inner loop performed for
    /// the last `search` call. Useful to demonstrate pruning savings.
    fn last_ops(&self) -> u64;
}

// ---------------------------------------------------------------------------
// Horizontal (row-major) baseline
// ---------------------------------------------------------------------------

pub struct Horizontal {
    data: Vec<f32>,
    n: usize,
    d: usize,
    last_ops: std::cell::Cell<u64>,
}

impl Horizontal {
    pub fn from_rows(rows: &[Vec<f32>]) -> Self {
        let n = rows.len();
        let d = if n == 0 { 0 } else { rows[0].len() };
        let mut data = Vec::with_capacity(n * d);
        for r in rows {
            assert_eq!(r.len(), d, "ragged input");
            data.extend_from_slice(r);
        }
        Self {
            data,
            n,
            d,
            last_ops: std::cell::Cell::new(0),
        }
    }
}

impl Scanner for Horizontal {
    fn len(&self) -> usize {
        self.n
    }
    fn dim(&self) -> usize {
        self.d
    }
    fn last_ops(&self) -> u64 {
        self.last_ops.get()
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        assert_eq!(query.len(), self.d);
        let mut heap = TopK::new(k);
        let mut ops: u64 = 0;
        for i in 0..self.n {
            let row = &self.data[i * self.d..(i + 1) * self.d];
            // Autovectorisable inner loop. Compiler emits SIMD here on x86_64.
            let mut acc = 0.0f32;
            for j in 0..self.d {
                let d = row[j] - query[j];
                acc += d * d;
            }
            ops += self.d as u64;
            heap.push(Hit {
                id: i as u32,
                dist: acc,
            });
        }
        self.last_ops.set(ops);
        heap.into_sorted_vec()
    }
}

// ---------------------------------------------------------------------------
// PDX vertical layout
// ---------------------------------------------------------------------------

/// Vertical block layout. The on-disk image is:
///   block[0].stripe[0..d]   (each stripe = BLOCK f32 values)
///   block[1].stripe[0..d]
///   ...
///   block[last].stripe[0..d] (last block may be partial; values past
///                              `last_block_len` are valid but ignored)
pub struct PdxVertical {
    /// Concatenation of all blocks, each block laid out as `D × BLOCK` floats.
    blocks: Vec<f32>,
    n: usize,
    d: usize,
    /// Number of valid vectors in the final block (1..=BLOCK).
    last_block_len: usize,
    last_ops: std::cell::Cell<u64>,
    /// If true, apply PROBE_STEP-grained pruning against the current heap top.
    prune: bool,
}

impl PdxVertical {
    pub fn from_rows(rows: &[Vec<f32>], prune: bool) -> Self {
        let n = rows.len();
        let d = if n == 0 { 0 } else { rows[0].len() };
        let nb = n.div_ceil(BLOCK.max(1));
        let mut blocks = vec![0.0f32; nb * d * BLOCK];
        for (i, r) in rows.iter().enumerate() {
            assert_eq!(r.len(), d, "ragged input");
            let b = i / BLOCK;
            let lane = i % BLOCK;
            let block_off = b * d * BLOCK;
            for j in 0..d {
                blocks[block_off + j * BLOCK + lane] = r[j];
            }
        }
        let last_block_len = if n == 0 {
            0
        } else if n % BLOCK == 0 {
            BLOCK
        } else {
            n % BLOCK
        };
        Self {
            blocks,
            n,
            d,
            last_block_len,
            last_ops: std::cell::Cell::new(0),
            prune,
        }
    }

    fn num_blocks(&self) -> usize {
        if self.n == 0 {
            0
        } else {
            (self.n - 1) / BLOCK + 1
        }
    }
}

impl Scanner for PdxVertical {
    fn len(&self) -> usize {
        self.n
    }
    fn dim(&self) -> usize {
        self.d
    }
    fn last_ops(&self) -> u64 {
        self.last_ops.get()
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        assert_eq!(query.len(), self.d);
        let mut heap = TopK::new(k);
        let mut ops: u64 = 0;
        let nb = self.num_blocks();
        // Per-block running partial sums (stack-friendly if BLOCK is small;
        // we heap-allocate once here to keep the type Send-safe).
        let mut partial = [0.0f32; BLOCK];
        // `alive[lane]` = false once that lane has been pruned for this block.
        let mut alive = [true; BLOCK];

        for b in 0..nb {
            for x in &mut partial {
                *x = 0.0;
            }
            for x in &mut alive {
                *x = true;
            }
            let block_off = b * self.d * BLOCK;
            let valid = if b + 1 == nb { self.last_block_len } else { BLOCK };

            let mut dim_done = 0usize;
            while dim_done < self.d {
                let dim_chunk = PROBE_STEP.min(self.d - dim_done);
                // Tight inner loop: for each dim in chunk, fan-out across BLOCK.
                for j in dim_done..dim_done + dim_chunk {
                    let q = query[j];
                    let stripe = &self.blocks[block_off + j * BLOCK..block_off + (j + 1) * BLOCK];
                    // Process all BLOCK lanes; rely on autovectorisation.
                    // Pruning skips entire later chunks, not individual lanes,
                    // so the inner sweep stays branch-free and SIMD-friendly.
                    for lane in 0..BLOCK {
                        let d = stripe[lane] - q;
                        partial[lane] += d * d;
                    }
                }
                // Only "live" lanes' work counts as productive; but we did
                // emit MULs for all of them. The pruning win is at the
                // *next* chunk boundary when we skip them entirely.
                let live_count = if self.prune {
                    alive.iter().take(valid).filter(|x| **x).count()
                } else {
                    valid
                };
                ops += (dim_chunk as u64) * (live_count as u64);
                dim_done += dim_chunk;

                if self.prune && heap.is_full() && dim_done < self.d {
                    let threshold = heap.peek_worst();
                    let mut any_alive = false;
                    for lane in 0..valid {
                        if alive[lane] && partial[lane] >= threshold {
                            alive[lane] = false;
                        }
                        any_alive |= alive[lane];
                    }
                    if !any_alive {
                        // Whole block pruned; no need to emit any-more dims.
                        break;
                    }
                }
            }

            for lane in 0..valid {
                if !self.prune || alive[lane] {
                    let id = (b * BLOCK + lane) as u32;
                    heap.push(Hit {
                        id,
                        dist: partial[lane],
                    });
                }
            }
        }

        self.last_ops.set(ops);
        heap.into_sorted_vec()
    }
}

// ---------------------------------------------------------------------------
// Top-K max-heap. Standard BinaryHeap gives us a max-heap; we keep size <= k
// and pop the worst whenever we exceed k, so the root is the current worst
// kept hit (i.e. the pruning threshold).
// ---------------------------------------------------------------------------

struct TopK {
    cap: usize,
    heap: std::collections::BinaryHeap<Hit>,
}

impl TopK {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            heap: std::collections::BinaryHeap::with_capacity(cap + 1),
        }
    }

    fn is_full(&self) -> bool {
        self.heap.len() >= self.cap
    }

    fn peek_worst(&self) -> f32 {
        self.heap.peek().map(|h| h.dist).unwrap_or(SENTINEL)
    }

    fn push(&mut self, h: Hit) {
        if self.cap == 0 {
            return;
        }
        if self.heap.len() < self.cap {
            self.heap.push(h);
        } else if h.dist < self.heap.peek().expect("non-empty").dist {
            self.heap.pop();
            self.heap.push(h);
        }
    }

    fn into_sorted_vec(self) -> Vec<Hit> {
        let mut v = self.heap.into_sorted_vec();
        // BinaryHeap::into_sorted_vec gives ascending order on Hit
        // (since Hit's Ord is by distance ascending), which is what we want.
        v.shrink_to_fit();
        v
    }
}

// ---------------------------------------------------------------------------
// Test corpora helpers (also used by the example/bench).
// ---------------------------------------------------------------------------

/// Deterministic xorshift32 PRNG so tests/bench are reproducible without rand.
pub struct Xs32(pub u32);
impl Xs32 {
    pub fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x.max(1);
        // map to [-1, 1)
        (x as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    pub fn vec(&mut self, d: usize) -> Vec<f32> {
        (0..d).map(|_| self.next_f32()).collect()
    }
}

pub fn synth_corpus(n: usize, d: usize, seed: u32) -> Vec<Vec<f32>> {
    let mut r = Xs32(seed);
    (0..n).map(|_| r.vec(d)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(hits: &[Hit]) -> Vec<u32> {
        hits.iter().map(|h| h.id).collect()
    }

    #[test]
    fn horizontal_and_pdx_agree_no_prune() {
        let rows = synth_corpus(500, 64, 42);
        let q = synth_corpus(1, 64, 7).pop().unwrap();
        let h = Horizontal::from_rows(&rows);
        let v = PdxVertical::from_rows(&rows, false);
        let a = h.search(&q, 10);
        let b = v.search(&q, 10);
        assert_eq!(ids(&a), ids(&b), "no-prune layouts must produce identical top-k");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x.dist - y.dist).abs() < 1e-3, "dist mismatch {x:?} vs {y:?}");
        }
    }

    #[test]
    fn pdx_pruned_matches_topk() {
        let rows = synth_corpus(2000, 96, 11);
        let q = synth_corpus(1, 96, 99).pop().unwrap();
        let h = Horizontal::from_rows(&rows);
        let v = PdxVertical::from_rows(&rows, true);
        let a = h.search(&q, 25);
        let b = v.search(&q, 25);
        assert_eq!(
            ids(&a),
            ids(&b),
            "pruned PDX must return same ids as ground truth (no recall loss)"
        );
    }

    #[test]
    fn pruning_strictly_reduces_ops() {
        let rows = synth_corpus(4096, 128, 3);
        let q = synth_corpus(1, 128, 17).pop().unwrap();
        let v0 = PdxVertical::from_rows(&rows, false);
        let v1 = PdxVertical::from_rows(&rows, true);
        let _ = v0.search(&q, 10);
        let _ = v1.search(&q, 10);
        let ops0 = v0.last_ops();
        let ops1 = v1.last_ops();
        assert!(
            ops1 < ops0,
            "pruning failed to reduce work: {ops1} vs {ops0}"
        );
    }

    #[test]
    fn ragged_n_below_block_works() {
        let rows = synth_corpus(7, 32, 1);
        let q = synth_corpus(1, 32, 2).pop().unwrap();
        let h = Horizontal::from_rows(&rows);
        let v = PdxVertical::from_rows(&rows, true);
        assert_eq!(ids(&h.search(&q, 4)), ids(&v.search(&q, 4)));
    }

    #[test]
    fn ragged_n_uneven_tail_works() {
        // 130 = 2 full blocks (128) + 2-vec tail
        let rows = synth_corpus(130, 48, 5);
        let q = synth_corpus(1, 48, 6).pop().unwrap();
        let h = Horizontal::from_rows(&rows);
        let v = PdxVertical::from_rows(&rows, true);
        assert_eq!(ids(&h.search(&q, 5)), ids(&v.search(&q, 5)));
    }
}
