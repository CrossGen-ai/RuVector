//! Public index trait and two implementations:
//! - [`FlatF32Index`] — exact L2 baseline (ground truth for recall).
//! - [`ExtendedRabitqIndex`] — B-bit rotated + packed candidates.
//!
//! Both use a bounded max-heap for top-k with `f32::total_cmp` so NaN never
//! panics. The design is trait-based so future backends (SIMD kernels, GPU
//! packed scan, disk-resident codes) can plug in behind the same API.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use serde::{Deserialize, Serialize};

use crate::error::ExtRabitqError;
use crate::quantize::{ExtendedCode, ExtendedQuantizer};
use crate::rotation::RandomRotation;
use crate::scan::QueryLut;

/// A single search hit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: u32,
    pub score: f32,
}

/// ANN index contract shared by all backends.
pub trait AnnIndex {
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Return the top-`k` ids by ascending L2 distance (`score = est. L2²`).
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, ExtRabitqError>;
}

// ----- FlatF32Index -----------------------------------------------------

#[derive(Clone, Debug)]
pub struct FlatF32Index {
    dim: usize,
    /// Row-major length `n * dim`.
    data: Vec<f32>,
}

impl FlatF32Index {
    pub fn from_vectors(dim: usize, vectors: &[Vec<f32>]) -> Result<Self, ExtRabitqError> {
        if dim == 0 {
            return Err(ExtRabitqError::InvalidDim { dim });
        }
        if vectors.is_empty() {
            return Err(ExtRabitqError::EmptyCorpus);
        }
        let mut data = Vec::with_capacity(vectors.len() * dim);
        for v in vectors {
            if v.len() != dim {
                return Err(ExtRabitqError::DimMismatch {
                    expected: dim,
                    actual: v.len(),
                });
            }
            data.extend_from_slice(v);
        }
        Ok(Self { dim, data })
    }
}

impl AnnIndex for FlatF32Index {
    fn len(&self) -> usize {
        self.data.len() / self.dim
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, ExtRabitqError> {
        if query.len() != self.dim {
            return Err(ExtRabitqError::DimMismatch {
                expected: self.dim,
                actual: query.len(),
            });
        }
        top_k(self.len(), k, |i| {
            let base = i * self.dim;
            let mut acc = 0f32;
            for d in 0..self.dim {
                let diff = self.data[base + d] - query[d];
                acc += diff * diff;
            }
            acc
        })
    }
}

// ----- ExtendedRabitqIndex ----------------------------------------------

/// B-bit index. Storage = one `ExtendedCode` per vector + shared rotation +
/// shared quantizer. Query is `f32`, no re-quantisation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtendedRabitqIndex {
    rotation: RandomRotation,
    quantizer: ExtendedQuantizer,
    codes: Vec<ExtendedCode>,
}

impl ExtendedRabitqIndex {
    pub fn build(
        dim: usize,
        bits: u32,
        seed: u64,
        vectors: &[Vec<f32>],
    ) -> Result<Self, ExtRabitqError> {
        if vectors.is_empty() {
            return Err(ExtRabitqError::EmptyCorpus);
        }
        let rotation = RandomRotation::new(dim, seed)?;
        let quantizer = ExtendedQuantizer::new(dim, bits)?;
        let mut codes = Vec::with_capacity(vectors.len());
        for v in vectors {
            let r = rotation.apply(v)?;
            codes.push(quantizer.encode(&r)?);
        }
        Ok(Self {
            rotation,
            quantizer,
            codes,
        })
    }

    pub fn bits(&self) -> u32 {
        self.quantizer.bits()
    }

    /// Bytes per code, for memory accounting.
    pub fn code_bytes(&self) -> usize {
        self.quantizer.code_bytes()
    }

    /// Total on-heap footprint of the code table (excludes rotation + norms).
    pub fn code_table_bytes(&self) -> usize {
        self.codes.len() * (self.code_bytes() + std::mem::size_of::<f32>())
    }
}

impl AnnIndex for ExtendedRabitqIndex {
    fn len(&self) -> usize {
        self.codes.len()
    }
    fn dim(&self) -> usize {
        self.quantizer.dim()
    }
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, ExtRabitqError> {
        if query.len() != self.dim() {
            return Err(ExtRabitqError::DimMismatch {
                expected: self.dim(),
                actual: query.len(),
            });
        }
        let q_rot = self.rotation.apply(query)?;
        let lut = QueryLut::new(&self.quantizer, &q_rot);
        let bits = self.quantizer.bits();
        top_k(self.codes.len(), k, |i| lut.l2_sq(&self.codes[i], bits))
    }
}

// ----- Shared top-k helper ----------------------------------------------

fn top_k(
    n: usize,
    k: usize,
    mut score: impl FnMut(usize) -> f32,
) -> Result<Vec<SearchResult>, ExtRabitqError> {
    let k = k.min(n);
    if k == 0 {
        return Ok(Vec::new());
    }
    // Max-heap keyed by (score, id) so the top is the worst-of-the-best.
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);
    for i in 0..n {
        let s = score(i);
        let item = HeapItem {
            id: i as u32,
            score: s,
        };
        if heap.len() < k {
            heap.push(item);
        } else if let Some(top) = heap.peek() {
            if s.total_cmp(&top.score) == Ordering::Less {
                heap.pop();
                heap.push(item);
            }
        }
    }
    let mut out: Vec<SearchResult> = heap
        .into_iter()
        .map(|h| SearchResult {
            id: h.id,
            score: h.score,
        })
        .collect();
    out.sort_by(|a, b| a.score.total_cmp(&b.score));
    Ok(out)
}

#[derive(Clone, Copy, Debug)]
struct HeapItem {
    id: u32,
    score: f32,
}
impl PartialEq for HeapItem {
    fn eq(&self, o: &Self) -> bool {
        self.score.total_cmp(&o.score) == Ordering::Equal && self.id == o.id
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> Ordering {
        // Max-heap on score, tie-break by id ascending.
        match self.score.total_cmp(&o.score) {
            Ordering::Equal => o.id.cmp(&self.id),
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        // Deterministic simple corpus.
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = vec![0f32; dim];
            let mut s = (seed.wrapping_add(i as u64)).wrapping_mul(2654435761);
            for d in 0..dim {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                v[d] = ((s >> 32) as i32 as f32) / 2_147_483_648.0;
            }
            out.push(v);
        }
        out
    }

    #[test]
    fn flat_finds_self() {
        let data = synth(20, 16, 1);
        let idx = FlatF32Index::from_vectors(16, &data).unwrap();
        let res = idx.search(&data[3], 1).unwrap();
        assert_eq!(res[0].id, 3);
        assert!(res[0].score < 1e-5);
    }

    #[test]
    fn extended_recall_improves_with_bits() {
        let dim = 32;
        let n = 512;
        let data = synth(n, dim, 7);
        let queries = synth(32, dim, 99);
        let flat = FlatF32Index::from_vectors(dim, &data).unwrap();
        let mut prev_recall = 0.0f32;
        for &bits in &[1u32, 2, 4] {
            let idx = ExtendedRabitqIndex::build(dim, bits, 42, &data).unwrap();
            let k = 10;
            let mut hits = 0usize;
            let mut total = 0usize;
            for q in &queries {
                let gt: std::collections::HashSet<u32> =
                    flat.search(q, k).unwrap().into_iter().map(|r| r.id).collect();
                let got = idx.search(q, k).unwrap();
                for r in &got {
                    if gt.contains(&r.id) {
                        hits += 1;
                    }
                }
                total += k;
            }
            let recall = hits as f32 / total as f32;
            assert!(
                recall + 1e-6 >= prev_recall,
                "recall regressed at bits={bits}: {recall} < {prev_recall}"
            );
            prev_recall = recall;
        }
        // 4-bit should be strongly above chance for k=10, n=512.
        assert!(prev_recall > 0.4, "4-bit recall too low: {prev_recall}");
    }
}
