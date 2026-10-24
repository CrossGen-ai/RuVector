//! FINGER-style edge sketches.
//!
//! For every directed edge (c -> n_i) we store a rank-k Johnson-Lindenstrauss
//! projection of (n_i - c) plus the squared edge length. A single
//! shared k x d random Gaussian matrix P is used across the whole graph.
//!
//! The identity we exploit:
//!
//! ```text
//!     d(q, n_i)^2 = ||q - c||^2 + ||n_i - c||^2 - 2 * <q - c, n_i - c>
//! ```
//!
//! Compute `delta = q - c` once per visited node. Then for each neighbor n_i
//! the only term that depends on the neighbor is <delta, n_i - c>, which we
//! approximate by <P * delta, P * (n_i - c)> in O(k).
//!
//! `||q - c||^2` is already available at the moment we visit c (we got there
//! by evaluating distance to c), so each neighbor costs just O(k) extra.

use crate::data::l2_sq;
use crate::graph::FlatGraph;

#[derive(Clone, Debug)]
pub struct SketchParams {
    /// Projection rank (sketch dimension).
    pub k: usize,
    /// Seed for the shared projection matrix.
    pub seed: u64,
}

impl Default for SketchParams {
    fn default() -> Self {
        Self { k: 16, seed: 0xA17EC0DE }
    }
}

#[derive(Clone, Debug)]
pub struct EdgeSketch {
    pub dim: usize,
    pub k: usize,
    pub n: usize,
    pub m: usize,

    /// Shared k x d projection matrix in row-major order.
    pub proj: Vec<f32>,

    /// Flat edge-projection table (n * m * k), row-major by (node, neighbor_slot, k).
    pub edge_proj: Vec<f32>,

    /// Squared edge length per edge, flat (n * m).
    pub edge_sq_len: Vec<f32>,

    /// Estimated bytes for the sketch structure (useful in reports).
    pub bytes: usize,
}

impl EdgeSketch {
    pub fn build(data: &[Vec<f32>], graph: &FlatGraph, params: &SketchParams) -> Self {
        let dim = graph.dim;
        let n = graph.n;
        let m = graph.m;
        let k = params.k;

        // Shared Gaussian projection matrix. Entries ~ N(0, 1/k) so
        // E[<Px, Py>] = <x, y>.
        let proj = gauss_matrix(k, dim, params.seed);

        let scale = 1.0f32 / (k as f32).sqrt();

        let mut edge_proj = vec![0.0f32; n * m * k];
        let mut edge_sq_len = vec![0.0f32; n * m];

        let mut edge_vec = vec![0.0f32; dim];
        let mut tmp = vec![0.0f32; k];

        for c in 0..n {
            let base = c * m;
            let c_vec = &data[c];
            for slot in 0..m {
                let nbr = graph.neighbors[base + slot] as usize;
                // Edge vector e = n_i - c.
                for d in 0..dim {
                    edge_vec[d] = data[nbr][d] - c_vec[d];
                }
                // ||e||^2.
                let mut s = 0.0f32;
                for d in 0..dim {
                    s += edge_vec[d] * edge_vec[d];
                }
                edge_sq_len[base + slot] = s;

                // tmp = P * e.
                project(&proj, k, dim, &edge_vec, &mut tmp);
                // Scale once at build time: store (P * e) * (1/sqrt(k))
                // so inner product requires one more * (1/sqrt(k))
                // at query time, done jointly on delta.
                let dst = &mut edge_proj[(base + slot) * k..(base + slot) * k + k];
                for i in 0..k {
                    dst[i] = tmp[i] * scale;
                }
            }
        }

        let bytes = proj.len() * 4
            + edge_proj.len() * 4
            + edge_sq_len.len() * 4;

        Self { dim, k, n, m, proj, edge_proj, edge_sq_len, bytes }
    }

    /// Approximate inner product <delta, e_i> where `delta_proj_scaled`
    /// is `(P * delta) * (1/sqrt(k))` and `edge_proj_row` is the stored
    /// (P * e_i) * (1/sqrt(k)) row of length k.
    #[inline]
    pub fn approx_ip(&self, delta_proj_scaled: &[f32], edge_proj_row: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..self.k {
            s += delta_proj_scaled[i] * edge_proj_row[i];
        }
        s
    }

    /// Full projection of `delta` into the stored basis, pre-scaled by 1/sqrt(k).
    #[inline]
    pub fn project_delta(&self, delta: &[f32], out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.k);
        let scale = 1.0f32 / (self.k as f32).sqrt();
        let mut tmp = vec![0.0f32; self.k];
        project(&self.proj, self.k, self.dim, delta, &mut tmp);
        for i in 0..self.k {
            out[i] = tmp[i] * scale;
        }
    }

    #[inline]
    pub fn edge_slot_row(&self, node: u32, slot: usize) -> &[f32] {
        let base = (node as usize * self.m + slot) * self.k;
        &self.edge_proj[base..base + self.k]
    }

    #[inline]
    pub fn edge_slot_sqlen(&self, node: u32, slot: usize) -> f32 {
        self.edge_sq_len[node as usize * self.m + slot]
    }
}

/// Fill `out` with `A * x` where `A` is `rows x cols` row-major.
fn project(a: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for r in 0..rows {
        let row = &a[r * cols..(r + 1) * cols];
        let mut s = 0.0f32;
        for c in 0..cols {
            s += row[c] * x[c];
        }
        out[r] = s;
    }
}

fn gauss_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    // JL-friendly scale: entries ~ N(0, 1) and we fold the 1/sqrt(k) scale
    // into stored sketches. This yields unbiased inner-product estimates
    // with variance ~ (||x||^2 ||y||^2 + <x,y>^2) / k.
    let mut rng = crate::data::Rng::new(seed);
    let mut m = vec![0.0f32; rows * cols];
    for v in m.iter_mut() {
        *v = rng.gauss();
    }
    m
}

/// Convenience: for a candidate node whose squared distance to q is known,
/// compute an approximate squared distance to each of its graph neighbors
/// using only the sketch. Useful for callers that want to see the raw
/// approximation output.
pub fn approx_neighbor_distances(
    sketch: &EdgeSketch,
    graph: &FlatGraph,
    data_c: &[f32],
    q: &[f32],
    c: u32,
) -> Vec<f32> {
    let m = graph.m;
    let mut delta = vec![0.0f32; sketch.dim];
    for i in 0..sketch.dim {
        delta[i] = q[i] - data_c[i];
    }
    let dq_c_sq = l2_sq(q, data_c);
    let mut delta_proj = vec![0.0f32; sketch.k];
    sketch.project_delta(&delta, &mut delta_proj);

    let mut out = Vec::with_capacity(m);
    for slot in 0..m {
        let ip = sketch.approx_ip(&delta_proj, sketch.edge_slot_row(c, slot));
        let sq_len = sketch.edge_slot_sqlen(c, slot);
        let approx = dq_c_sq + sq_len - 2.0 * ip;
        out.push(approx);
    }
    out
}
