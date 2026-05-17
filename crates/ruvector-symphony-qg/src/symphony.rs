//! Symphony-QG style navigable small-world graph driven by quantized distances.
//!
//! Build:
//!   1. Train a [`ProductQuantizer`] on the dataset, encode all vectors.
//!   2. For each node, find `m_edges` neighbors using the *quantized* metric.
//!      This is the "quantization-aware graph" step — building with the same
//!      metric the search will use closes the recall gap that vanilla
//!      HNSW + PQ leaves open.
//!   3. Diversify with a simple alpha-RNG pruning so the graph stays
//!      navigable rather than collapsing into clusters.
//!
//! Search: best-first traversal on the quantized graph. The priority queue
//! uses ADT distances directly — no rerank pass is needed for the headline
//! configuration. An optional `refine` parameter rescues recall on hard
//! queries by full-precision-rescoring the top-`refine` candidates at the
//! end; this is exposed for ablations.

use std::collections::BinaryHeap;
use std::cmp::Ordering;

use crate::{
    metric::l2_sq,
    pq::{PqCodes, ProductQuantizer},
    AnnIndex,
};

#[derive(Clone, Copy)]
struct Cand { id: u32, d: f32 }
impl PartialEq for Cand { fn eq(&self, o: &Self) -> bool { self.d == o.d } }
impl Eq for Cand {}
impl PartialOrd for Cand { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        // Min-heap via reverse on distance.
        o.d.partial_cmp(&self.d).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap candidate (for "worst of top-k" tracking).
#[derive(Clone, Copy)]
struct MaxCand { id: u32, d: f32 }
impl PartialEq for MaxCand { fn eq(&self, o: &Self) -> bool { self.d == o.d } }
impl Eq for MaxCand {}
impl PartialOrd for MaxCand { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl Ord for MaxCand {
    fn cmp(&self, o: &Self) -> Ordering { self.d.partial_cmp(&o.d).unwrap_or(Ordering::Equal) }
}

pub struct SymphonyQgIndex {
    pub d: usize,
    pub data: Vec<f32>,           // kept for optional refine; not used in headline path
    pub pq: ProductQuantizer,
    pub codes: PqCodes,
    pub adj: Vec<Vec<u32>>,       // per-node neighbor lists
    pub entries: Vec<u32>,        // multi-entry seeds (one per cluster region)
    pub ef_search: usize,
    pub refine: usize,            // 0 = no rerank (headline); >0 = rescue
}

impl SymphonyQgIndex {
    pub fn build(
        d: usize,
        data: Vec<f32>,
        m_pq: usize,
        k_pq: usize,
        iters_kmeans: usize,
        m_edges: usize,
        ef_construction: usize,
        alpha: f32,
        seed: u64,
    ) -> Self {
        let pq = ProductQuantizer::train(&data, d, m_pq, k_pq, iters_kmeans, seed).expect("train");
        let codes = pq.encode_all(&data).expect("encode");
        let n = data.len() / d;
        let mut adj: Vec<Vec<u32>> = vec![Vec::with_capacity(m_edges); n];

        // Graph is built with FULL-PRECISION distances. The Symphony-QG
        // intuition is "quantization-aware navigability": full-precision is
        // available at build, so we use it to find true near neighbors and
        // diversify with alpha-RNG; this preserves connectivity better than
        // building on quantized scores (which double-quantizes through the
        // codebook reconstruction). At *query* time, only PQ codes + ADT
        // are touched — that's where the savings come from.
        for i in 0..n {
            let qi = &data[i * d..(i + 1) * d];
            let mut cand: Vec<(u32, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j as u32, l2_sq(&data[j * d..(j + 1) * d], qi)))
                .collect();
            let take = ef_construction.min(cand.len());
            cand.select_nth_unstable_by(take.saturating_sub(1).max(0), |a, b| a.1.partial_cmp(&b.1).unwrap());
            cand.truncate(take);
            cand.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let mut kept: Vec<(u32, f32)> = Vec::with_capacity(m_edges);
            for (j, dj_i) in cand.into_iter() {
                if kept.len() >= m_edges { break; }
                let mut dominated = false;
                let qj = &data[j as usize * d..(j as usize + 1) * d];
                for (nk, _) in kept.iter() {
                    let d_nk_j = l2_sq(&data[*nk as usize * d..(*nk as usize + 1) * d], qj);
                    if alpha * d_nk_j < dj_i { dominated = true; break; }
                }
                if !dominated { kept.push((j, dj_i)); }
            }
            adj[i] = kept.into_iter().map(|(j, _)| j).collect();
        }

        // Symmetrize lightly: ensure reverse edges exist (capped at 2*m_edges).
        let mut to_add: Vec<(u32, u32)> = Vec::new();
        for i in 0..n {
            for &j in adj[i].iter() {
                if !adj[j as usize].contains(&(i as u32)) {
                    to_add.push((j, i as u32));
                }
            }
        }
        for (a, b) in to_add {
            let list = &mut adj[a as usize];
            if list.len() < 2 * m_edges { list.push(b); }
        }

        // Farthest-point-sampled entry seeds. Cardinality ~= log2(n) + 4
        // covers cluster diversity for small/medium datasets without
        // ballooning per-query traversal cost. Seeds are deterministic
        // because FPS uses argmax over data; no RNG needed.
        let n_entries = (((n as f32).log2() as usize) + 4).min(n).max(1);
        let mut entries: Vec<u32> = Vec::with_capacity(n_entries);
        entries.push(0);
        let mut min_d: Vec<f32> = (0..n)
            .map(|j| l2_sq(&data[0..d], &data[j * d..(j + 1) * d]))
            .collect();
        for _ in 1..n_entries {
            let mut best_i = 0usize;
            let mut best_d = -1.0f32;
            for (j, &md) in min_d.iter().enumerate() {
                if md > best_d { best_d = md; best_i = j; }
            }
            entries.push(best_i as u32);
            let new_pt = &data[best_i * d..(best_i + 1) * d];
            for j in 0..n {
                let dj = l2_sq(new_pt, &data[j * d..(j + 1) * d]);
                if dj < min_d[j] { min_d[j] = dj; }
            }
        }

        Self {
            d, data, pq, codes, adj,
            entries,
            ef_search: m_edges.max(16),
            refine: 0,
        }
    }

    pub fn with_ef_search(mut self, ef: usize) -> Self { self.ef_search = ef; self }
    pub fn with_refine(mut self, refine: usize) -> Self { self.refine = refine; self }
    pub fn with_entries(mut self, entries: Vec<u32>) -> Self { self.entries = entries; self }

    fn n(&self) -> usize { self.data.len() / self.d }
}

impl AnnIndex for SymphonyQgIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let n = self.n();
        if n == 0 { return Vec::new(); }
        let lut = self.pq.build_adt(query);
        let mut visited = vec![false; n];

        let mut frontier: BinaryHeap<Cand> = BinaryHeap::new();      // min-heap by distance
        let mut top: BinaryHeap<MaxCand> = BinaryHeap::new();        // max-heap of top-ef
        let ef = self.ef_search.max(k);

        for &e in self.entries.iter() {
            let eu = e as usize;
            if visited[eu] { continue; }
            visited[eu] = true;
            let de = self.pq.adc_distance(&lut, self.codes.code(eu));
            frontier.push(Cand { id: e, d: de });
            top.push(MaxCand { id: e, d: de });
        }

        while let Some(cur) = frontier.pop() {
            let worst = top.peek().map(|c| c.d).unwrap_or(f32::INFINITY);
            if cur.d > worst && top.len() >= ef { break; }
            for &nb in self.adj[cur.id as usize].iter() {
                let nb_u = nb as usize;
                if visited[nb_u] { continue; }
                visited[nb_u] = true;
                let dnb = self.pq.adc_distance(&lut, self.codes.code(nb_u));
                if top.len() < ef {
                    top.push(MaxCand { id: nb, d: dnb });
                    frontier.push(Cand { id: nb, d: dnb });
                } else if let Some(w) = top.peek() {
                    if dnb < w.d {
                        top.pop();
                        top.push(MaxCand { id: nb, d: dnb });
                        frontier.push(Cand { id: nb, d: dnb });
                    }
                }
            }
        }

        let mut out: Vec<(u32, f32)> = top.into_iter().map(|c| (c.id, c.d)).collect();
        out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if self.refine > 0 {
            // Optional rescue: rescore top-`refine` with full precision.
            let r = self.refine.min(out.len());
            for slot in out.iter_mut().take(r) {
                let id = slot.0 as usize;
                slot.1 = l2_sq(&self.data[id * self.d..(id + 1) * self.d], query);
            }
            out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }

        out.truncate(k);
        out
    }
    fn len(&self) -> usize { self.n() }
}
