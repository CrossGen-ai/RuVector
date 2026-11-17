//! Inverted Multi-Index (IMI).
//!
//! Vectors are split into two halves of dimension `d/2`. k-means is run
//! on each half giving codebooks `U` and `V` of size K. Cell `(u, v)`
//! holds the ids whose halves quantize to `u_i` and `v_j`. Effective
//! cell count is K² with only 2K centroids.
//!
//! Search uses the **multi-sequence algorithm**: for a query split into
//! `q = [q_u, q_v]`, compute sorted distance lists
//!   `du[0..K]` = indices of `U` sorted by L2 to `q_u` (ascending)
//!   `dv[0..K]` = indices of `V` sorted by L2 to `q_v` (ascending)
//! then enumerate cell pairs `(du[i], dv[j])` in non-decreasing
//! `du[i].dist + dv[j].dist` with a min-heap. For each popped cell,
//! append its posting to the shortlist; stop when `shortlist_cap`
//! candidates are gathered. Finally rescore the shortlist on full
//! vectors and return top-k.

use std::collections::BinaryHeap;

use crate::data::Rng;
use crate::flat_ivf::push_top_k;
use crate::kmeans::{kmeans, l2_sq};
use crate::AnnIndex;

pub type ImiHit = (u32, f32);

pub struct ImiIndex {
    pub d: usize,
    pub dh: usize, // d/2
    pub k: usize,  // per-half codebook size
    pub u_centroids: Vec<f32>,
    pub v_centroids: Vec<f32>,
    /// `cells[u * k + v] = Vec<id>` — flat index over K² virtual cells.
    pub cells: Vec<Vec<u32>>,
    pub vectors: Vec<f32>,
    /// Target shortlist size (candidates considered after enumeration).
    pub shortlist_cap: usize,
}

impl ImiIndex {
    pub fn build(data: Vec<f32>, d: usize, k: usize, kmeans_iters: usize, seed: u64) -> Self {
        assert!(d % 2 == 0, "IMI needs even dim");
        let dh = d / 2;
        let n = data.len() / d;

        // Materialize two halves as separate contiguous buffers so
        // kmeans can treat each half as its own dataset.
        let mut u_half = Vec::<f32>::with_capacity(n * dh);
        let mut v_half = Vec::<f32>::with_capacity(n * dh);
        for i in 0..n {
            let row = &data[i * d..(i + 1) * d];
            u_half.extend_from_slice(&row[..dh]);
            v_half.extend_from_slice(&row[dh..]);
        }

        let mut rng_u = Rng::new(seed);
        let mut rng_v = Rng::new(seed.wrapping_add(0xDEAD_BEEF));
        let (u_centroids, u_assign) = kmeans(&u_half, dh, k, kmeans_iters, &mut rng_u);
        let (v_centroids, v_assign) = kmeans(&v_half, dh, k, kmeans_iters, &mut rng_v);

        let mut cells: Vec<Vec<u32>> = (0..k * k).map(|_| Vec::new()).collect();
        for i in 0..n {
            let u = u_assign[i] as usize;
            let v = v_assign[i] as usize;
            cells[u * k + v].push(i as u32);
        }

        ImiIndex {
            d,
            dh,
            k,
            u_centroids,
            v_centroids,
            cells,
            vectors: data,
            shortlist_cap: 2048,
        }
    }
}

/// Min-heap entry for the multi-sequence enumerator.
#[derive(Copy, Clone)]
struct Cell {
    score: f32,
    iu: u32,
    iv: u32,
}
impl Eq for Cell {}
impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Flip ordering to make BinaryHeap a min-heap on score.
        other.score.partial_cmp(&self.score)
    }
}
impl Ord for Cell {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl AnnIndex for ImiIndex {
    fn name(&self) -> &str {
        "imi"
    }

    fn search(&self, query: &[f32], probe: usize, k: usize) -> Vec<(u32, f32)> {
        let qu = &query[..self.dh];
        let qv = &query[self.dh..];

        // Sorted distance lists (ascending) over per-half centroids.
        let mut du: Vec<(u32, f32)> = (0..self.k as u32)
            .map(|c| (c, l2_sq(qu, &self.u_centroids[c as usize * self.dh..(c as usize + 1) * self.dh])))
            .collect();
        let mut dv: Vec<(u32, f32)> = (0..self.k as u32)
            .map(|c| (c, l2_sq(qv, &self.v_centroids[c as usize * self.dh..(c as usize + 1) * self.dh])))
            .collect();
        du.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dv.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Multi-sequence enumeration with a visited-set to avoid dupes.
        let kk = self.k;
        let mut visited = vec![false; kk * kk];
        let mut heap = BinaryHeap::<Cell>::new();
        heap.push(Cell {
            score: du[0].1 + dv[0].1,
            iu: 0,
            iv: 0,
        });
        visited[0] = true;

        // `probe` caps the number of cells enumerated; `shortlist_cap`
        // caps the number of candidates actually scored. Both are
        // tunable from outside — this gives the tuner two knobs.
        let mut cand: Vec<u32> = Vec::with_capacity(self.shortlist_cap);
        let mut enumerated = 0usize;

        while let Some(top) = heap.pop() {
            enumerated += 1;
            let iu = top.iu as usize;
            let iv = top.iv as usize;
            let u_id = du[iu].0 as usize;
            let v_id = dv[iv].0 as usize;
            let posting = &self.cells[u_id * kk + v_id];
            cand.extend_from_slice(posting);

            if cand.len() >= self.shortlist_cap || enumerated >= probe {
                break;
            }
            // Push (iu+1, iv) and (iu, iv+1) if unseen and in range.
            if iu + 1 < kk {
                let idx = (iu + 1) * kk + iv;
                if !visited[idx] {
                    visited[idx] = true;
                    heap.push(Cell {
                        score: du[iu + 1].1 + dv[iv].1,
                        iu: (iu + 1) as u32,
                        iv: iv as u32,
                    });
                }
            }
            if iv + 1 < kk {
                let idx = iu * kk + (iv + 1);
                if !visited[idx] {
                    visited[idx] = true;
                    heap.push(Cell {
                        score: du[iu].1 + dv[iv + 1].1,
                        iu: iu as u32,
                        iv: (iv + 1) as u32,
                    });
                }
            }
        }

        // Rescore candidates on full vectors.
        let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for &id in &cand {
            let sq = l2_sq(query, &self.vectors[id as usize * self.d..(id as usize + 1) * self.d]);
            push_top_k(&mut top, (id, sq), k);
        }
        top.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        top
    }

    fn memory_bytes(&self) -> usize {
        self.u_centroids.len() * 4
            + self.v_centroids.len() * 4
            + self.cells.iter().map(|p| p.len() * 4 + 24).sum::<usize>()
            + self.vectors.len() * 4
    }
}
