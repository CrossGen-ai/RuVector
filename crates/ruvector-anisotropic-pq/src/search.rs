//! Asymmetric distance computation (ADC) for MIPS.
//!
//! At query time we build one `M × K` lookup table of `<q_m, c_{m,k}>` inner
//! products. The estimated inner product between the query and a database
//! vector with code `(c_0, …, c_{M-1})` is
//!
//! ```text
//! ⟨q, x̃⟩ = Σ_m LUT[m, c_m]
//! ```
//!
//! Top-k is found with a simple bounded min-heap of size `k`.

use crate::pq::{Code, Codebook, Vector};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub id: u32,
    pub score: f32, // inner product (higher is better)
}

impl PartialEq for Hit {
    fn eq(&self, o: &Self) -> bool {
        self.score == o.score && self.id == o.id
    }
}
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Hit {
    // Reverse so BinaryHeap is a min-heap on score.
    fn cmp(&self, o: &Self) -> Ordering {
        o.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal)
    }
}

pub struct AdcSearcher<'a> {
    pub codebook: &'a Codebook,
    pub codes: &'a [Code],
}

impl<'a> AdcSearcher<'a> {
    pub fn new(codebook: &'a Codebook, codes: &'a [Code]) -> Self {
        Self { codebook, codes }
    }

    pub fn build_lut(&self, q: &[f32]) -> Vec<f32> {
        let m = self.codebook.params.m;
        let k = self.codebook.params.k;
        let s = self.codebook.sub_dim;
        let mut lut = vec![0.0_f32; m * k];
        for mi in 0..m {
            let qm = &q[mi * s..(mi + 1) * s];
            for kj in 0..k {
                let c = self.codebook.centroid(mi, kj);
                let mut d = 0.0_f32;
                for i in 0..s {
                    d += qm[i] * c[i];
                }
                lut[mi * k + kj] = d;
            }
        }
        lut
    }

    pub fn search(&self, q: &Vector, top_k: usize) -> Vec<Hit> {
        let lut = self.build_lut(q);
        let m = self.codebook.params.m;
        let k = self.codebook.params.k;
        let mut heap: BinaryHeap<Hit> = BinaryHeap::with_capacity(top_k + 1);
        for (id, code) in self.codes.iter().enumerate() {
            let mut score = 0.0_f32;
            for mi in 0..m {
                score += lut[mi * k + code[mi] as usize];
            }
            if heap.len() < top_k {
                heap.push(Hit {
                    id: id as u32,
                    score,
                });
            } else if score > heap.peek().unwrap().score {
                heap.pop();
                heap.push(Hit {
                    id: id as u32,
                    score,
                });
            }
        }
        let mut out: Vec<Hit> = heap.into_vec();
        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        out
    }
}
