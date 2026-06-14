//! Top-level HCNNG index: build + search API.

use crate::distance::{self, Distance, Metric};
use crate::error::HcnngError;
use crate::graph::Graph;
use crate::mst::mst_plus_knn_edges;
use crate::partition::build_tree;
use crate::search::{beam_search, SearchResult};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HcnngParams {
    /// Number of partition trees to ensemble. Paper recommends 10–20.
    pub n_trees: usize,
    /// Maximum leaf bucket size before stopping recursion. 16–64 is typical.
    pub leaf_size: usize,
    /// Hard cap on per-node neighbor list size after dedup/sort. ~30 is HNSW-like.
    pub max_degree: usize,
    /// Extra intra-leaf kNN edges added per node (on top of MST). 0 disables.
    /// 3–5 is a good range; densifies the graph cheaply.
    pub knn_per_node: usize,
    /// Search beam width (ef in HNSW parlance).
    pub ef_search: usize,
    /// Deterministic seed for partition trees.
    pub seed: u64,
    /// Distance metric.
    pub metric: Metric,
}

impl Default for HcnngParams {
    fn default() -> Self {
        Self {
            n_trees: 10,
            leaf_size: 32,
            max_degree: 32,
            knn_per_node: 3,
            ef_search: 64,
            seed: 0xC0FFEE_DEADBEEF,
            metric: Metric::L2Sq,
        }
    }
}

pub struct HcnngIndex {
    vectors: Vec<Vec<f32>>,
    dim: usize,
    params: HcnngParams,
    graph: Graph,
    dist: Box<dyn Distance>,
    entries: Vec<u32>,
}

impl HcnngIndex {
    pub fn build(vectors: Vec<Vec<f32>>, params: HcnngParams) -> Result<Self, HcnngError> {
        if vectors.is_empty() {
            return Err(HcnngError::Empty);
        }
        if params.n_trees == 0 {
            return Err(HcnngError::InvalidParam("n_trees must be > 0"));
        }
        if params.leaf_size < 2 {
            return Err(HcnngError::InvalidParam("leaf_size must be >= 2"));
        }
        if params.max_degree == 0 {
            return Err(HcnngError::InvalidParam("max_degree must be > 0"));
        }
        let dim = vectors[0].len();
        for v in &vectors {
            if v.len() != dim {
                return Err(HcnngError::DimMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
        }
        let dist = distance::make(params.metric);
        let mut graph = Graph::new(vectors.len());

        // Build n_trees partition trees; for each leaf, compute MST; union edges.
        for t in 0..params.n_trees {
            let seed = params.seed.wrapping_add(t as u64 * 0x9E37_79B9_7F4A_7C15);
            let leaves = build_tree(&vectors, params.leaf_size, dist.as_ref(), seed);
            for leaf in &leaves {
                let edges = mst_plus_knn_edges(leaf, &vectors, dist.as_ref(), params.knn_per_node);
                for (u, v) in edges {
                    graph.add_edge_unique(u, v);
                }
            }
        }

        graph.finalize(&vectors, dist.as_ref(), params.max_degree);

        // Multi-entry: top-K highest-degree nodes (hubs) plus a few random
        // anchors so the beam search starts from diverse regions of the graph.
        // This is critical for recall on multi-modal data — a single hub
        // tends to lie in one cluster and the greedy walk can stall there.
        let mut by_degree: Vec<(usize, u32)> = graph
            .neighbors
            .iter()
            .enumerate()
            .map(|(i, l)| (l.len(), i as u32))
            .collect();
        by_degree.sort_by(|a, b| b.0.cmp(&a.0));
        let mut entries: Vec<u32> = by_degree
            .iter()
            .take(4)
            .map(|(_, i)| *i)
            .collect();
        // Add a few deterministic random anchors.
        let n = vectors.len();
        let mut h = params.seed;
        for _ in 0..4 {
            h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            let pick = (h as usize) % n;
            if !entries.contains(&(pick as u32)) {
                entries.push(pick as u32);
            }
        }

        Ok(Self {
            vectors,
            dim,
            params,
            graph,
            dist,
            entries,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
    pub fn params(&self) -> &HcnngParams {
        &self.params
    }
    pub fn entries(&self) -> &[u32] {
        &self.entries
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>, HcnngError> {
        if query.len() != self.dim {
            return Err(HcnngError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut visited: HashSet<u32> = HashSet::with_capacity(self.params.ef_search * 4);
        Ok(beam_search(
            query,
            &self.graph,
            &self.vectors,
            self.dist.as_ref(),
            &self.entries,
            k,
            self.params.ef_search,
            &mut visited,
        ))
    }

    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<SearchResult>, HcnngError> {
        if query.len() != self.dim {
            return Err(HcnngError::DimMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 4);
        Ok(beam_search(
            query,
            &self.graph,
            &self.vectors,
            self.dist.as_ref(),
            &self.entries,
            k,
            ef,
            &mut visited,
        ))
    }

    /// Approximate bytes used by the graph adjacency + vectors.
    pub fn memory_bytes(&self) -> usize {
        let v_bytes = self.vectors.len() * self.dim * std::mem::size_of::<f32>();
        let g_bytes: usize = self
            .graph
            .neighbors
            .iter()
            .map(|n| n.capacity() * std::mem::size_of::<u32>() + std::mem::size_of::<Vec<u32>>())
            .sum();
        v_bytes + g_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn synth(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| rng.gen_range(-1.0..1.0_f32)).collect())
            .collect()
    }

    fn brute_topk(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<u32> {
        let mut s: Vec<(f32, u32)> = data
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let mut s = 0.0;
                for j in 0..v.len() {
                    let dd = v[j] - query[j];
                    s += dd * dd;
                }
                (s, i as u32)
            })
            .collect();
        s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        s.into_iter().take(k).map(|(_, i)| i).collect()
    }

    #[test]
    fn build_and_search_small() {
        let data = synth(500, 32, 1);
        let idx = HcnngIndex::build(data.clone(), HcnngParams::default()).unwrap();
        let q = &data[10];
        let r = idx.search(q, 5).unwrap();
        assert_eq!(r.len(), 5);
        // Query equals data[10] so id=10 must be the nearest neighbor.
        assert_eq!(r[0].id, 10);
    }

    #[test]
    fn recall_at_10_reasonable() {
        let n = 1000;
        let d = 32;
        let data = synth(n, d, 7);
        let queries = synth(50, d, 11);
        let idx = HcnngIndex::build(
            data.clone(),
            HcnngParams {
                n_trees: 12,
                leaf_size: 32,
                max_degree: 32,
                knn_per_node: 3,
                ef_search: 64,
                seed: 99,
                metric: Metric::L2Sq,
            },
        )
        .unwrap();

        let k = 10;
        let mut tot = 0.0;
        for q in &queries {
            let gt: std::collections::HashSet<u32> = brute_topk(q, &data, k).into_iter().collect();
            let got = idx.search(q, k).unwrap();
            let hit = got.iter().filter(|r| gt.contains(&r.id)).count();
            tot += hit as f64 / k as f64;
        }
        let recall = tot / queries.len() as f64;
        // 12 trees + ef=64 on n=1000 should comfortably clear 0.85.
        assert!(recall > 0.85, "recall={}", recall);
    }

    #[test]
    fn rejects_empty() {
        let r = HcnngIndex::build(Vec::<Vec<f32>>::new(), HcnngParams::default());
        assert!(matches!(r, Err(HcnngError::Empty)));
    }

    #[test]
    fn rejects_dim_mismatch_query() {
        let data = synth(50, 8, 0);
        let idx = HcnngIndex::build(data, HcnngParams::default()).unwrap();
        let q = vec![0.0; 4];
        assert!(matches!(
            idx.search(&q, 3),
            Err(HcnngError::DimMismatch { .. })
        ));
    }
}
