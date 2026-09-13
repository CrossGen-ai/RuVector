//! Minimal exact-kNN flat index. This is the "coarse retriever" the rerankers
//! sit behind. For the purposes of this PoC we use exact retrieval so any
//! recall difference between rerankers is attributable to the score
//! correction, not to approximation noise from HNSW/IVF.

use crate::distance::Distance;
use crate::rerank::Scored;

pub type Vector = Vec<f32>;

/// Flat, exhaustive kNN over an in-memory corpus.
pub struct FlatIndex<'a, D: Distance> {
    pub corpus: &'a [Vector],
    pub distance: D,
}

impl<'a, D: Distance> FlatIndex<'a, D> {
    pub fn new(corpus: &'a [Vector], distance: D) -> Self {
        Self { corpus, distance }
    }

    /// All pairwise distances from `q` to every corpus vector.
    pub fn distances_to(&self, q: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.corpus.len());
        for v in self.corpus {
            out.push(self.distance.dist(q, v));
        }
        out
    }

    /// Coarse top-`r` neighbours by raw distance. This is what any coarse
    /// ANN (HNSW / IVF / DiskANN) would deliver in a real deployment.
    pub fn coarse_top_r(&self, q: &[f32], r: usize) -> Vec<Scored> {
        let dists = self.distances_to(q);
        let mut scored: Vec<Scored> = dists
            .into_iter()
            .enumerate()
            .map(|(id, d)| Scored { id, score: d })
            .collect();
        // Partial sort — we only need the top-r.
        let r = r.min(scored.len());
        scored.select_nth_unstable_by(r.saturating_sub(1).max(0), |a, b| {
            a.score.partial_cmp(&b.score).unwrap()
        });
        scored.truncate(r);
        scored.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        scored
    }

    /// True kNN under the raw distance (ground truth for the raw metric).
    pub fn exact_knn(&self, q: &[f32], k: usize) -> Vec<Scored> {
        self.coarse_top_r(q, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::L2;

    #[test]
    fn coarse_top_r_is_sorted_and_bounded() {
        let corpus: Vec<Vector> = (0..20)
            .map(|i| vec![i as f32, (2 * i) as f32])
            .collect();
        let idx = FlatIndex::new(&corpus, L2);
        let q = vec![0.0, 0.0];
        let top = idx.coarse_top_r(&q, 5);
        assert_eq!(top.len(), 5);
        for w in top.windows(2) {
            assert!(w[0].score <= w[1].score);
        }
        // The closest to origin is index 0.
        assert_eq!(top[0].id, 0);
    }
}
