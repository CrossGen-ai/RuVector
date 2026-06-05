//! Pluggable backend that stores the raw (or quantised) representation of
//! each vector and answers distance queries.

use crate::metric::l2_sq;

pub trait Quantizer {
    /// Add a vector under id `id`.  Ids must be unique and monotonically
    /// increasing in this PoC (id == vector index).
    fn add(&mut self, id: u32, v: &[f32]);
    /// Squared-L2 distance from `q` to the vector previously stored as `id`.
    fn distance(&self, id: u32, q: &[f32]) -> f32;
    /// Borrow the stored representation as f32 (used during splits).
    fn get(&self, id: u32) -> &[f32];
}

/// A no-op quantiser: stores raw f32 vectors.  Memory = 4 * n * d bytes.
pub struct FlatQuantizer {
    dim: usize,
    data: Vec<f32>, // row-major n × dim
}

impl FlatQuantizer {
    pub fn new(dim: usize) -> Self { Self { dim, data: Vec::new() } }
    pub fn dim(&self) -> usize { self.dim }
    pub fn bytes(&self) -> usize { self.data.len() * 4 }
}

impl Quantizer for FlatQuantizer {
    fn add(&mut self, id: u32, v: &[f32]) {
        let off = id as usize * self.dim;
        if self.data.len() < off + self.dim {
            self.data.resize(off + self.dim, 0.0);
        }
        self.data[off..off + self.dim].copy_from_slice(v);
    }
    fn distance(&self, id: u32, q: &[f32]) -> f32 {
        let off = id as usize * self.dim;
        l2_sq(&self.data[off..off + self.dim], q)
    }
    fn get(&self, id: u32) -> &[f32] {
        let off = id as usize * self.dim;
        &self.data[off..off + self.dim]
    }
}
