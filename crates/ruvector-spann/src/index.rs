//! SPANN-style IVF index: centroid coarse quantizer + posting lists.
//!
//! Build:
//!   1. Train k-means over a sample to get K centroids.
//!   2. For each base vector, compute distances to all centroids,
//!      sort ascending, then ask the `ClosurePolicy` which postings
//!      it joins. With `SpannClosure`, boundary points join multiple.
//!
//! Search:
//!   1. Find the nearest `nprobe` centroids to the query.
//!   2. Linearly scan those posting lists; keep top-k.
//!   3. Deduplicate by base id (a replicated point appears in multiple
//!      postings; we don't want to count it twice in top-k).

use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::kmeans::KMeans;
use crate::metrics::sq_l2;
use crate::policy::ClosurePolicy;

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: usize,
    pub distance: f32,
}

/// Max-heap entry by distance so the heap kicks out the worst candidate.
#[derive(Debug, Clone, Copy)]
struct HeapItem {
    id: usize,
    d: f32,
}
impl Eq for HeapItem {}
impl PartialEq for HeapItem {
    fn eq(&self, o: &Self) -> bool {
        self.d == o.d
    }
}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> Ordering {
        // NaN should not occur; treat as equal if it does.
        self.d.partial_cmp(&o.d).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

pub struct SpannIndex {
    pub data: Vec<Vec<f32>>,
    pub centroids: Vec<Vec<f32>>,
    pub postings: Vec<Vec<u32>>, // posting[c] = list of base ids
    pub dim: usize,
    /// total entries across postings (= n with single, larger with closure).
    pub total_entries: usize,
}

pub struct BuildStats {
    pub n_vectors: usize,
    pub n_postings_entries: usize,
    pub replication_factor: f32,
}

impl SpannIndex {
    pub fn build<P: ClosurePolicy + ?Sized>(
        data: Vec<Vec<f32>>,
        k_centroids: usize,
        kmeans_iters: usize,
        policy: &P,
        seed: u64,
    ) -> (Self, BuildStats) {
        assert!(!data.is_empty(), "spann: empty data");
        let dim = data[0].len();
        let km = KMeans::fit(&data, k_centroids, kmeans_iters, seed);
        let centroids = km.centroids;

        let mut postings: Vec<Vec<u32>> = vec![Vec::new(); centroids.len()];
        let mut total: usize = 0;
        let mut buf: Vec<(usize, f32)> = Vec::with_capacity(centroids.len());

        for (i, x) in data.iter().enumerate() {
            buf.clear();
            for (c, cv) in centroids.iter().enumerate() {
                buf.push((c, sq_l2(x, cv)));
            }
            buf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let ids = policy.assign(&buf);
            total += ids.len();
            for c in ids {
                postings[c].push(i as u32);
            }
        }

        let stats = BuildStats {
            n_vectors: data.len(),
            n_postings_entries: total,
            replication_factor: total as f32 / data.len() as f32,
        };

        (
            SpannIndex {
                data,
                centroids,
                postings,
                dim,
                total_entries: total,
            },
            stats,
        )
    }

    pub fn search(&self, query: &[f32], top_k: usize, nprobe: usize) -> Vec<SearchResult> {
        // 1. nprobe nearest centroids.
        let mut cd: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, sq_l2(query, c)))
            .collect();
        cd.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let probe: Vec<usize> = cd.iter().take(nprobe).map(|(i, _)| *i).collect();

        // 2. Scan postings, dedup base ids via bitset.
        let mut seen = vec![false; self.data.len()];
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(top_k + 1);

        for c in probe {
            for &bid in &self.postings[c] {
                let id = bid as usize;
                if seen[id] {
                    continue;
                }
                seen[id] = true;
                let d = sq_l2(query, &self.data[id]);
                if heap.len() < top_k {
                    heap.push(HeapItem { id, d });
                } else if let Some(top) = heap.peek() {
                    if d < top.d {
                        heap.pop();
                        heap.push(HeapItem { id, d });
                    }
                }
            }
        }

        let mut out: Vec<SearchResult> =
            heap.into_iter().map(|h| SearchResult { id: h.id, distance: h.d }).collect();
        out.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        out
    }

    /// Exact brute-force ground truth, used by recall@k evaluation.
    pub fn brute_force(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut all: Vec<SearchResult> = self
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| SearchResult { id: i, distance: sq_l2(query, v) })
            .collect();
        all.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        all.truncate(top_k);
        all
    }

    /// Memory footprint in bytes (data + centroids + postings entries).
    pub fn mem_bytes(&self) -> usize {
        let v = self.data.len() * self.dim * std::mem::size_of::<f32>();
        let c = self.centroids.len() * self.dim * std::mem::size_of::<f32>();
        let p = self.total_entries * std::mem::size_of::<u32>();
        v + c + p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{SingleAssign, SpannClosure};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn gen(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn build_search_smoke() {
        let data = gen(500, 16, 1);
        let (idx, _) = SpannIndex::build(data, 16, 10, &SingleAssign, 42);
        let q = vec![0.5; 16];
        let r = idx.search(&q, 5, 4);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn closure_increases_recall_vs_single() {
        let data = gen(2000, 16, 7);
        let queries = gen(50, 16, 8);

        let (single, _) =
            SpannIndex::build(data.clone(), 32, 12, &SingleAssign, 42);
        let (spann, stats) = SpannIndex::build(
            data.clone(),
            32,
            12,
            &SpannClosure { epsilon: 0.15, cap: 4 },
            42,
        );
        assert!(stats.replication_factor > 1.0, "closure should replicate");

        let nprobe = 2;
        let top_k = 10;
        let mut hit_single = 0usize;
        let mut hit_spann = 0usize;
        let mut total = 0usize;

        for q in &queries {
            let gt = single.brute_force(q, top_k);
            let gt_ids: Vec<usize> = gt.iter().map(|r| r.id).collect();
            let rs = single.search(q, top_k, nprobe);
            let rp = spann.search(q, top_k, nprobe);
            for r in rs {
                if gt_ids.contains(&r.id) {
                    hit_single += 1;
                }
            }
            for r in rp {
                if gt_ids.contains(&r.id) {
                    hit_spann += 1;
                }
            }
            total += top_k;
        }

        let rec_single = hit_single as f32 / total as f32;
        let rec_spann = hit_spann as f32 / total as f32;
        // SPANN closure should not REDUCE recall and almost always raises it
        // at low nprobe. Allow tiny slack for tie-breaking randomness.
        assert!(
            rec_spann + 1e-6 >= rec_single,
            "spann recall {} should be >= single recall {}",
            rec_spann,
            rec_single
        );
    }
}
