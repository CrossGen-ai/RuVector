//! Rotated brute-force top-k index. Not a graph — the point is to
//! isolate the *distance oracle* and measure how it changes work vs
//! recall. Any real ANN index (HNSW, IVF, DiskANN) plugs into the same
//! trait; see ADR-273 for the wiring.

use crate::oracle::{DistanceOracle, Outcome};
use crate::rotation::RandomRotation;
use crate::AdsError;

/// A neighbor returned by the index.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    /// Row id inside the index.
    pub id: u32,
    /// Squared L2 distance (rotated space; equals original space).
    pub dist_sq: f32,
}

/// Bounded top-k min-heap (by distance ascending → we pop the *max* to
/// evict). Kept tiny and heap-free for K ≤ 128 which is the typical
/// ANN case.
#[derive(Debug)]
pub struct TopK {
    k: usize,
    items: Vec<Neighbor>,
}

impl TopK {
    /// Fresh empty heap of capacity `k`.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            items: Vec::with_capacity(k + 1),
        }
    }
    /// Current worst distance (tau). +∞ when not yet full.
    pub fn tau(&self) -> f32 {
        if self.items.len() < self.k {
            f32::INFINITY
        } else {
            self.items.iter().map(|n| n.dist_sq).fold(0.0f32, f32::max)
        }
    }
    /// Try to insert; keeps only the smallest `k`.
    pub fn push(&mut self, n: Neighbor) {
        if self.items.len() < self.k {
            self.items.push(n);
            return;
        }
        // Evict current max if `n` beats it.
        let (worst_idx, worst_d) = self
            .items
            .iter()
            .enumerate()
            .fold((0usize, f32::MIN), |(bi, bd), (i, x)| {
                if x.dist_sq > bd {
                    (i, x.dist_sq)
                } else {
                    (bi, bd)
                }
            });
        if n.dist_sq < worst_d {
            self.items[worst_idx] = n;
        }
    }
    /// Sorted-ascending final result.
    pub fn into_sorted(mut self) -> Vec<Neighbor> {
        self.items
            .sort_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap());
        self.items
    }
}

/// Rotated brute-force top-k index.
pub struct AdsIndex {
    dim: usize,
    rotation: RandomRotation,
    rotated: Vec<Vec<f32>>,
}

impl AdsIndex {
    /// Build from `vectors` using rotation seed `seed`.
    pub fn build(vectors: &[Vec<f32>], seed: u64) -> Result<Self, AdsError> {
        assert!(!vectors.is_empty(), "index must have >= 1 vector");
        let dim = vectors[0].len();
        for v in vectors {
            if v.len() != dim {
                return Err(AdsError::DimensionMismatch {
                    index_d: dim,
                    query_d: v.len(),
                });
            }
        }
        let rot = RandomRotation::new(dim, seed);
        let rotated: Vec<Vec<f32>> = vectors.iter().map(|v| rot.apply(v)).collect();
        Ok(Self {
            dim,
            rotation: rot,
            rotated,
        })
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Number of indexed vectors.
    pub fn len(&self) -> usize {
        self.rotated.len()
    }
    /// True when the index has no vectors.
    pub fn is_empty(&self) -> bool {
        self.rotated.is_empty()
    }
    /// Read-only rotation handle (used to rotate queries).
    pub fn rotation(&self) -> &RandomRotation {
        &self.rotation
    }

    /// Search using the supplied oracle. Returns top-`k` by squared L2.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        oracle: &dyn DistanceOracle,
    ) -> Result<Vec<Neighbor>, AdsError> {
        if k == 0 {
            return Err(AdsError::ZeroK);
        }
        if query.len() != self.dim {
            return Err(AdsError::DimensionMismatch {
                index_d: self.dim,
                query_d: query.len(),
            });
        }
        let rq = self.rotation.apply(query);
        let mut topk = TopK::new(k);
        for (i, rx) in self.rotated.iter().enumerate() {
            let tau = topk.tau();
            if let Outcome::Keep(d) = oracle.evaluate(&rq, rx, tau) {
                topk.push(Neighbor {
                    id: i as u32,
                    dist_sq: d,
                });
            }
        }
        Ok(topk.into_sorted())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::{AdsAdaptive, ExactL2};

    fn corpus(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        use rand::{Rng, SeedableRng};
        let mut r = rand::rngs::StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| r.gen_range(-1.0f32..1.0)).collect())
            .collect()
    }

    #[test]
    fn exact_search_finds_planted_neighbor() {
        let d = 64;
        let mut c = corpus(256, d, 11);
        let mut q = vec![0.0f32; d];
        for i in 0..d {
            q[i] = 0.1;
        }
        // Plant a very-near vector at id 7.
        c[7] = q.iter().map(|x| x + 0.001).collect();
        let idx = AdsIndex::build(&c, 42).unwrap();
        let r = idx.search(&q, 3, &ExactL2::new()).unwrap();
        assert_eq!(r[0].id, 7);
    }

    #[test]
    fn adaptive_matches_exact_on_planted_neighbor() {
        let d = 64;
        let mut c = corpus(256, d, 12);
        let mut q = vec![0.0f32; d];
        for i in 0..d {
            q[i] = -0.2;
        }
        c[42] = q.iter().map(|x| x + 0.002).collect();
        let idx = AdsIndex::build(&c, 7).unwrap();
        let ex = idx.search(&q, 1, &ExactL2::new()).unwrap();
        let ad = idx
            .search(&q, 1, &AdsAdaptive::with_epsilon_from_dim(d))
            .unwrap();
        assert_eq!(ex[0].id, 42);
        assert_eq!(ad[0].id, 42);
    }
}
