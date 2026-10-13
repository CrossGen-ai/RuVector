//! K-rotation ensemble RaBitQ.
//!
//! Build K independent `RabitqSingle` indexes with seeds `seed, seed+1, …,
//! seed+K-1` (same `kind`). At query time score each row under all K
//! rotations and average. This is an unbiased estimator with K× lower
//! variance than one rotation, at the cost of K× more bits per row and
//! K rotation applies per query.
//!
//! The implementation is intentionally boring — we delegate to the
//! single-rotation primitives — so that any correctness guarantee that
//! holds for `RabitqSingle` carries straight through.

use crate::index::{finalize_topk, push_topk, Index, SearchResult};
use crate::rotation::RotationKind;
use crate::single::RabitqSingle;

#[derive(Clone)]
pub struct RabitqEnsemble {
    pub dim: usize,
    pub k_rot: usize,
    pub members: Vec<RabitqSingle>,
}

impl RabitqEnsemble {
    pub fn build(rows: &[Vec<f32>], k_rot: usize, kind: RotationKind, seed: u64) -> Self {
        assert!((1..=64).contains(&k_rot), "K must be in 1..=64");
        let dim = rows[0].len();
        let members: Vec<RabitqSingle> = (0..k_rot)
            .map(|i| RabitqSingle::build(rows, kind, seed.wrapping_add(i as u64)))
            .collect();
        Self {
            dim,
            k_rot,
            members,
        }
    }
}

impl Index for RabitqEnsemble {
    fn dim(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.members[0].len()
    }
    fn aux_bytes(&self) -> usize {
        self.members.iter().map(|m| m.aux_bytes()).sum()
    }
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        assert_eq!(query.len(), self.dim);
        // Pre-rotate the query once per member — these are reused inside
        // the per-row loop so this is the main hot cost.
        let rqs: Vec<Vec<f32>> = self
            .members
            .iter()
            .map(|m| m.rotation.apply(query))
            .collect();

        let n = self.len();
        let inv_k = (self.k_rot as f32).recip();
        let mut heap: Vec<SearchResult> = Vec::with_capacity(k);

        // We re-implement the inner score loop inline rather than calling
        // `self.members[j].score_row` K times, so the per-row average can
        // short-circuit on degenerate inputs and stays allocation-free.
        for id in 0..n {
            let mut acc = 0.0_f32;
            for j in 0..self.k_rot {
                let m = &self.members[j];
                let row = &m.codes
                    [id * m.words_per_row..(id + 1) * m.words_per_row];
                let rq = &rqs[j];
                let pad = rq.len();
                let mut s = 0.0_f32;
                for b in 0..pad {
                    let w = row[b >> 6];
                    let bit = (w >> (b & 63)) & 1;
                    let sign = if bit == 1 { 1.0 } else { -1.0 };
                    s += sign * rq[b];
                }
                // Norm cancels across rotations because all R_j are
                // orthogonal and ‖x‖ doesn't depend on j — but each
                // member stores its own `norms[id]` (identical). Use
                // member 0's norm as the canonical value.
                acc += m.norms[id] * m.inv_sqrt_pad * s;
            }
            push_topk(
                &mut heap,
                k,
                SearchResult {
                    id: id as u32,
                    score: acc * inv_k,
                },
            );
        }
        finalize_topk(heap)
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
    fn ensemble_builds_and_searches() {
        let rows = mk_rows(1_000, 64, 0xBEEF);
        let idx = RabitqEnsemble::build(&rows, 4, RotationKind::HadamardSigned, 11);
        let q = rows[7].clone();
        let top = idx.search(&q, 10);
        assert_eq!(top.len(), 10);
        assert!(top.iter().any(|r| r.id == 7));
    }

    #[test]
    fn ensemble_k1_matches_single() {
        // K=1 ensemble should be score-equivalent to RabitqSingle built
        // with the same seed on the same data (we store the same signs,
        // apply the same rotation, divide by K=1).
        let rows = mk_rows(300, 48, 42);
        let s = RabitqSingle::build(&rows, RotationKind::HadamardSingle, 2);
        let e = RabitqEnsemble::build(&rows, 1, RotationKind::HadamardSingle, 2);
        let q = rows[11].clone();
        let a = s.search(&q, 10);
        let b = e.search(&q, 10);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.id, y.id);
            assert!((x.score - y.score).abs() < 1e-5, "{} vs {}", x.score, y.score);
        }
    }
}
