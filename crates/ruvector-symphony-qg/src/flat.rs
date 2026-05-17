use crate::{metric::l2_sq, AnnIndex};

pub struct FlatIndex {
    pub d: usize,
    pub data: Vec<f32>,
}

impl FlatIndex {
    pub fn new(d: usize, data: Vec<f32>) -> Self { Self { d, data } }
    pub fn n(&self) -> usize { self.data.len() / self.d }
}

impl AnnIndex for FlatIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        let n = self.n();
        let mut all: Vec<(u32, f32)> = (0..n)
            .map(|i| (i as u32, l2_sq(&self.data[i * self.d..(i + 1) * self.d], query)))
            .collect();
        all.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all.truncate(k);
        all
    }
    fn len(&self) -> usize { self.n() }
}
