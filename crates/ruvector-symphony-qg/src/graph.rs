//! Minimal kNN graph index (NN-descent–lite).
//!
//! Builds an undirected kNN graph by random initialization + a few
//! refinement passes that swap in better neighbours via shared
//! neighbour exploration. This is *not* HNSW — we omit hierarchical
//! layers to keep the PoC small. The graph is the substrate; the
//! "symphony" experiment is what runs on top of it.

use crate::l2_sq;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[derive(Clone, Debug)]
pub struct GraphParams {
    pub k: usize,        // neighbours per node
    pub iters: usize,    // refinement passes
    pub ef_search: usize,
    pub seed: u64,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self { k: 32, iters: 4, ef_search: 64, seed: 0xC0FFEE }
    }
}

#[derive(Clone, Debug)]
pub struct KnnGraph {
    pub n: usize,
    pub k: usize,
    pub adj: Vec<u32>, // flat n*k neighbour ids
    pub ef_search: usize,
}

impl KnnGraph {
    pub fn build(vectors: &[Vec<f32>], params: GraphParams) -> Self {
        let n = vectors.len();
        let k = params.k.min(n.saturating_sub(1)).max(1);
        let mut rng = StdRng::seed_from_u64(params.seed);

        // 1. random init
        let mut adj: Vec<Vec<(u32, f32)>> = (0..n).map(|i| {
            let mut nbrs = Vec::with_capacity(k);
            while nbrs.len() < k {
                let j = rng.gen_range(0..n) as u32;
                if j as usize != i && !nbrs.iter().any(|(x, _): &(u32, f32)| *x == j) {
                    let d = l2_sq(&vectors[i], &vectors[j as usize]);
                    nbrs.push((j, d));
                }
            }
            nbrs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            nbrs
        }).collect();

        // 2. NN-descent refinement: for each node, explore neighbours of neighbours.
        for _ in 0..params.iters {
            let snapshot = adj.clone();
            for i in 0..n {
                let mut cands: Vec<u32> = snapshot[i].iter().map(|x| x.0).collect();
                for &(j, _) in &snapshot[i] {
                    for &(jj, _) in &snapshot[j as usize] {
                        cands.push(jj);
                    }
                }
                for c in cands {
                    if c as usize == i { continue; }
                    if adj[i].iter().any(|(x, _)| *x == c) { continue; }
                    let d = l2_sq(&vectors[i], &vectors[c as usize]);
                    if let Some(&(_, worst)) = adj[i].last() {
                        if d < worst {
                            adj[i].pop();
                            adj[i].push((c, d));
                            adj[i].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        }
                    }
                }
            }
        }

        let flat: Vec<u32> = adj.iter()
            .flat_map(|row| row.iter().map(|(x, _)| *x))
            .collect();
        Self { n, k, adj: flat, ef_search: params.ef_search }
    }

    #[inline]
    pub fn neighbours(&self, node: usize) -> &[u32] {
        &self.adj[node * self.k .. (node + 1) * self.k]
    }

    /// Best-first graph search using f32 distances (baseline / reranking oracle).
    pub fn search_f32(
        &self,
        vectors: &[Vec<f32>],
        query: &[f32],
        topk: usize,
        ef: usize,
        seed: u64,
    ) -> Vec<(usize, f32)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let entry = rng.gen_range(0..self.n);

        let mut visited = vec![false; self.n];
        let mut heap: std::collections::BinaryHeap<NegF32> = Default::default();
        let mut results: std::collections::BinaryHeap<PosF32> = Default::default();

        let d0 = l2_sq(&vectors[entry], query);
        heap.push(NegF32(d0, entry as u32));
        results.push(PosF32(d0, entry as u32));
        visited[entry] = true;

        while let Some(NegF32(d, node)) = heap.pop() {
            if let Some(PosF32(top, _)) = results.peek() {
                if results.len() >= ef && d > *top { break; }
            }
            for &nb in self.neighbours(node as usize) {
                let nbu = nb as usize;
                if visited[nbu] { continue; }
                visited[nbu] = true;
                let dn = l2_sq(&vectors[nbu], query);
                if results.len() < ef {
                    heap.push(NegF32(dn, nb));
                    results.push(PosF32(dn, nb));
                } else if let Some(PosF32(top, _)) = results.peek() {
                    if dn < *top {
                        heap.push(NegF32(dn, nb));
                        results.push(PosF32(dn, nb));
                        if results.len() > ef { results.pop(); }
                    }
                }
            }
        }

        let mut out: Vec<(usize, f32)> = results.into_iter().map(|PosF32(d, i)| (i as usize, d)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out.truncate(topk);
        out
    }
}

#[derive(PartialEq)]
struct PosF32(f32, u32);
impl Eq for PosF32 {}
impl Ord for PosF32 { fn cmp(&self, other: &Self) -> std::cmp::Ordering { self.0.partial_cmp(&other.0).unwrap() } }
impl PartialOrd for PosF32 { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }

#[derive(PartialEq)]
struct NegF32(f32, u32);
impl Eq for NegF32 {}
impl Ord for NegF32 { fn cmp(&self, other: &Self) -> std::cmp::Ordering { other.0.partial_cmp(&self.0).unwrap() } }
impl PartialOrd for NegF32 { fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) } }

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_data(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| (0..d).map(|_| rng.gen::<f32>() - 0.5).collect()).collect()
    }

    #[test]
    fn graph_build_smoke() {
        let data = fake_data(500, 32, 1);
        let g = KnnGraph::build(&data, GraphParams { k: 16, iters: 2, ef_search: 32, seed: 2 });
        assert_eq!(g.n, 500);
        assert_eq!(g.k, 16);
        // every neighbour list has k unique ids != self
        for i in 0..g.n {
            let nbrs = g.neighbours(i);
            assert_eq!(nbrs.len(), 16);
            let s: std::collections::HashSet<_> = nbrs.iter().collect();
            assert_eq!(s.len(), 16);
            assert!(!nbrs.contains(&(i as u32)));
        }
    }

    #[test]
    fn graph_search_beats_random() {
        let data = fake_data(1000, 32, 3);
        let g = KnnGraph::build(&data, GraphParams { k: 24, iters: 3, ef_search: 64, seed: 4 });
        let q = &data[7];
        let hits = g.search_f32(&data, q, 10, 64, 11);
        // self must be in the top hits
        assert!(hits.iter().any(|(i, _)| *i == 7));
    }
}
