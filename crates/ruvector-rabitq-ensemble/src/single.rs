//! Single-rotation 1-bit RaBitQ (asymmetric estimator).
//!
//! For each row `x`, we store `‖x‖` and `sign(R x)` as `D_pad` bits. At
//! query time we rotate `q` once and compute
//!
//! ```text
//! est_ip(q, x) = ‖x‖ · (1/√D_pad) · Σ_i sign(Rx)_i · (Rq)_i
//! ```
//!
//! which is an unbiased estimator of `<q, x>` under Haar-uniform `R` (Gao &
//! Long 2024, Prop. 3.1). Here `sign(z) ∈ {+1,-1}` with `sign(0) := +1`.

use crate::index::{finalize_topk, push_topk, Index, SearchResult};
use crate::rotation::{Rotation, RotationKind};

#[derive(Clone)]
pub struct RabitqSingle {
    pub dim: usize,
    pub rotation: Rotation,
    pub inv_sqrt_pad: f32,
    pub norms: Vec<f32>,
    /// Row-major: `n_rows * words_per_row` u64 words.
    pub codes: Vec<u64>,
    pub words_per_row: usize,
}

impl RabitqSingle {
    pub fn build(rows: &[Vec<f32>], kind: RotationKind, seed: u64) -> Self {
        let dim = rows[0].len();
        let rot = Rotation::new(dim, kind, seed);
        let pad = rot.out_dim();
        let words_per_row = pad.div_ceil(64);
        let mut codes = vec![0_u64; rows.len() * words_per_row];
        let mut norms = Vec::with_capacity(rows.len());
        let mut buf = vec![0.0_f32; pad];
        for (i, row) in rows.iter().enumerate() {
            let nrm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            norms.push(nrm);
            rot.apply_into(row, &mut buf);
            pack_signs(&buf, &mut codes[i * words_per_row..(i + 1) * words_per_row]);
        }
        Self {
            dim,
            rotation: rot,
            inv_sqrt_pad: (pad as f32).sqrt().recip(),
            norms,
            codes,
            words_per_row,
        }
    }

    /// Score of `rq` against row `id`, returning the unbiased IP estimator.
    #[inline]
    fn score_row(&self, rq: &[f32], id: usize) -> f32 {
        let row = &self.codes[id * self.words_per_row..(id + 1) * self.words_per_row];
        let pad = rq.len();
        let mut acc = 0.0_f32;
        for b in 0..pad {
            let w = row[b >> 6];
            let bit = (w >> (b & 63)) & 1;
            let s = if bit == 1 { 1.0 } else { -1.0 };
            acc += s * rq[b];
        }
        self.norms[id] * self.inv_sqrt_pad * acc
    }
}

impl Index for RabitqSingle {
    fn dim(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.norms.len()
    }
    fn aux_bytes(&self) -> usize {
        self.rotation.aux_bytes() + self.codes.len() * 8 + self.norms.len() * 4
    }
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dim);
        let rq = self.rotation.apply(query);
        let mut heap: Vec<SearchResult> = Vec::with_capacity(k);
        for id in 0..self.len() {
            let score = self.score_row(&rq, id);
            push_topk(
                &mut heap,
                k,
                SearchResult {
                    id: id as u32,
                    score,
                },
            );
        }
        finalize_topk(heap)
    }
}

/// Pack `sign(v[i]) ∈ {+1,-1}` into a bitstream where bit = 1 iff v[i] ≥ 0.
pub fn pack_signs(v: &[f32], out: &mut [u64]) {
    let pad = v.len();
    let words = pad.div_ceil(64);
    assert_eq!(out.len(), words);
    for w in out.iter_mut() {
        *w = 0;
    }
    for i in 0..pad {
        if v[i] >= 0.0 {
            out[i >> 6] |= 1u64 << (i & 63);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    fn mk_rows(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| rng.gen_range(-1.0_f32..=1.0)).collect())
            .collect()
    }

    #[test]
    fn single_builds_and_searches() {
        let rows = mk_rows(500, 48, 0xA1);
        let idx = RabitqSingle::build(&rows, RotationKind::HadamardSigned, 7);
        let q = rows[0].clone();
        let top = idx.search(&q, 5);
        assert_eq!(top.len(), 5);
        // The query equals row 0, so it should land in the top-5 with high
        // probability — hard to guarantee for a stochastic 1-bit index,
        // but a 500×48 random dataset makes this extremely robust.
        assert!(
            top.iter().any(|r| r.id == 0),
            "self-match missing: {:?}",
            top.iter().map(|r| r.id).collect::<Vec<_>>()
        );
    }
}
