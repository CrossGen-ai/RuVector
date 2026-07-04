//! Concrete draft/verifier backends: F32BruteForce, Int8BruteForce, Sign1BitDraft.

use crate::{DraftIndex, Neighbor, Result, SpecAnnError, Verifier};

// -----------------------------------------------------------------------------
// F32BruteForce — exact float32 baseline. Serves as verifier or (trivially)
// as a draft when you want a pure exact baseline for benchmarking.
// -----------------------------------------------------------------------------
pub struct F32BruteForce {
    dim: usize,
    data: Vec<f32>, // n * dim, row-major
    n: usize,
}

impl F32BruteForce {
    pub fn from_vectors(vectors: &[Vec<f32>]) -> Result<Self> {
        if vectors.is_empty() {
            return Ok(Self {
                dim: 0,
                data: Vec::new(),
                n: 0,
            });
        }
        let dim = vectors[0].len();
        let mut data = Vec::with_capacity(vectors.len() * dim);
        for v in vectors {
            if v.len() != dim {
                return Err(SpecAnnError::DimMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
            data.extend_from_slice(v);
        }
        Ok(Self {
            dim,
            data,
            n: vectors.len(),
        })
    }

    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    #[inline]
    fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            s += d * d;
        }
        s
    }
}

impl DraftIndex for F32BruteForce {
    fn draft(&self, query: &[f32], k_draft: usize) -> Result<Vec<Neighbor>> {
        if query.len() != self.dim {
            return Err(SpecAnnError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut all: Vec<Neighbor> = (0..self.n)
            .map(|i| Neighbor {
                id: i as u32,
                score: Self::sq_l2(query, self.row(i)),
            })
            .collect();
        all.sort();
        all.truncate(k_draft);
        Ok(all)
    }
    fn len(&self) -> usize {
        self.n
    }
}

impl Verifier for F32BruteForce {
    fn verify(&self, query: &[f32], candidates: &[u32]) -> Result<Vec<Neighbor>> {
        if query.len() != self.dim {
            return Err(SpecAnnError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut out: Vec<Neighbor> = candidates
            .iter()
            .map(|&id| Neighbor {
                id,
                score: Self::sq_l2(query, self.row(id as usize)),
            })
            .collect();
        out.sort();
        Ok(out)
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.n
    }
}

// -----------------------------------------------------------------------------
// Int8BruteForce — symmetric int8 quantized draft.
// Per-vector scale s = max(|v_i|) / 127. Cheap: i32 accumulator over i8*i8.
// -----------------------------------------------------------------------------
pub struct Int8BruteForce {
    dim: usize,
    data: Vec<i8>,
    scales: Vec<f32>,
    n: usize,
}

impl Int8BruteForce {
    pub fn from_vectors(vectors: &[Vec<f32>]) -> Result<Self> {
        if vectors.is_empty() {
            return Ok(Self {
                dim: 0,
                data: Vec::new(),
                scales: Vec::new(),
                n: 0,
            });
        }
        let dim = vectors[0].len();
        let mut data = Vec::with_capacity(vectors.len() * dim);
        let mut scales = Vec::with_capacity(vectors.len());
        for v in vectors {
            if v.len() != dim {
                return Err(SpecAnnError::DimMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
            let max_abs = v.iter().copied().fold(0.0f32, |a, x| a.max(x.abs())).max(1e-12);
            let s = max_abs / 127.0;
            scales.push(s);
            for &x in v {
                let q = (x / s).round().clamp(-127.0, 127.0) as i8;
                data.push(q);
            }
        }
        Ok(Self {
            dim,
            data,
            scales,
            n: vectors.len(),
        })
    }

    fn quantize_query(&self, q: &[f32]) -> (Vec<i8>, f32) {
        let max_abs = q.iter().copied().fold(0.0f32, |a, x| a.max(x.abs())).max(1e-12);
        let s = max_abs / 127.0;
        let qq: Vec<i8> = q
            .iter()
            .map(|&x| (x / s).round().clamp(-127.0, 127.0) as i8)
            .collect();
        (qq, s)
    }

    #[inline]
    fn row(&self, i: usize) -> &[i8] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

impl DraftIndex for Int8BruteForce {
    fn draft(&self, query: &[f32], k_draft: usize) -> Result<Vec<Neighbor>> {
        if query.len() != self.dim {
            return Err(SpecAnnError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let (qq, sq) = self.quantize_query(query);
        let mut all: Vec<Neighbor> = (0..self.n)
            .map(|i| {
                let row = self.row(i);
                let sd = self.scales[i];
                let mut acc = 0.0f32;
                for j in 0..self.dim {
                    let d = sd * row[j] as f32 - sq * qq[j] as f32;
                    acc += d * d;
                }
                Neighbor {
                    id: i as u32,
                    score: acc,
                }
            })
            .collect();
        all.sort();
        all.truncate(k_draft);
        Ok(all)
    }
    fn len(&self) -> usize {
        self.n
    }
}

// -----------------------------------------------------------------------------
// Sign1BitDraft — 1-bit sign approximation (RaBitQ-lite, no rotation).
// score = popcount(sign(q) XOR sign(v)) — Hamming distance proxy for L2.
// Ultra-fast: dim/64 u64 XORs per compare.
// -----------------------------------------------------------------------------
pub struct Sign1BitDraft {
    dim: usize,
    words_per_vec: usize,
    data: Vec<u64>,
    n: usize,
}

impl Sign1BitDraft {
    pub fn from_vectors(vectors: &[Vec<f32>]) -> Result<Self> {
        if vectors.is_empty() {
            return Ok(Self {
                dim: 0,
                words_per_vec: 0,
                data: Vec::new(),
                n: 0,
            });
        }
        let dim = vectors[0].len();
        let words_per_vec = (dim + 63) / 64;
        let mut data = vec![0u64; vectors.len() * words_per_vec];
        for (vi, v) in vectors.iter().enumerate() {
            if v.len() != dim {
                return Err(SpecAnnError::DimMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
            for (i, &x) in v.iter().enumerate() {
                if x >= 0.0 {
                    let w = vi * words_per_vec + i / 64;
                    data[w] |= 1u64 << (i % 64);
                }
            }
        }
        Ok(Self {
            dim,
            words_per_vec,
            data,
            n: vectors.len(),
        })
    }

    fn encode_query(&self, q: &[f32]) -> Vec<u64> {
        let mut out = vec![0u64; self.words_per_vec];
        for (i, &x) in q.iter().enumerate() {
            if x >= 0.0 {
                out[i / 64] |= 1u64 << (i % 64);
            }
        }
        out
    }
}

impl DraftIndex for Sign1BitDraft {
    fn draft(&self, query: &[f32], k_draft: usize) -> Result<Vec<Neighbor>> {
        if query.len() != self.dim {
            return Err(SpecAnnError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let qw = self.encode_query(query);
        let mut all: Vec<Neighbor> = (0..self.n)
            .map(|i| {
                let base = i * self.words_per_vec;
                let mut acc: u32 = 0;
                for w in 0..self.words_per_vec {
                    acc += (self.data[base + w] ^ qw[w]).count_ones();
                }
                Neighbor {
                    id: i as u32,
                    score: acc as f32,
                }
            })
            .collect();
        all.sort();
        all.truncate(k_draft);
        Ok(all)
    }
    fn len(&self) -> usize {
        self.n
    }
}
