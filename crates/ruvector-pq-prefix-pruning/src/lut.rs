//! Per-query LUT: for each segment m, 16 distances between the query sub-vector
//! and that segment's 16 centroids.
//!
//! Values are quantized to u8 using a per-segment scale so that partial sums
//! fit comfortably in u16 across M <= 64 segments. Quantization bias is tiny
//! for M in the usual 8..32 range; we expose it as `scale` for callers that
//! want a reproducible back-conversion.

use crate::codec::Codebook;

#[derive(Clone, Debug)]
pub struct QueryLut {
    pub m: usize,
    /// Row-major: m rows × 16 columns of u8 segment distances.
    pub table: Vec<u8>,
    /// Per-segment variance (sample variance of the 16 u8 entries).
    pub seg_variance: Vec<u32>,
    /// Minimum per-segment distance — used as the floor for pruning math.
    pub seg_min: Vec<u8>,
}

impl QueryLut {
    pub fn build(book: &Codebook, query: &[f32]) -> Self {
        assert_eq!(query.len(), book.params.d());
        let m = book.params.m;
        let d = book.params.d_sub;
        let mut raw = vec![0f32; m * 16];
        let mut max_per_seg = vec![0f32; m];
        for seg in 0..m {
            let q = &query[seg * d..(seg + 1) * d];
            let mut mx = 0f32;
            for c in 0..16 {
                let cent = book.centroid(seg, c);
                let mut s = 0f32;
                for i in 0..d {
                    let v = q[i] - cent[i];
                    s += v * v;
                }
                raw[seg * 16 + c] = s;
                if s > mx {
                    mx = s;
                }
            }
            max_per_seg[seg] = mx;
        }
        // Pick a single global scale so partial sums are comparable across
        // scanners. 255/(max over all segments) keeps precision tight.
        let global_max = max_per_seg.iter().cloned().fold(0f32, f32::max);
        let scale = if global_max > 0.0 { 255.0 / global_max } else { 1.0 };

        let mut table = vec![0u8; m * 16];
        for i in 0..m * 16 {
            let v = (raw[i] * scale).round();
            table[i] = if v >= 255.0 { 255 } else if v <= 0.0 { 0 } else { v as u8 };
        }

        let mut seg_variance = vec![0u32; m];
        let mut seg_min = vec![0u8; m];
        for seg in 0..m {
            let row = &table[seg * 16..(seg + 1) * 16];
            let mut mn = u8::MAX;
            let mut sum: u32 = 0;
            for &v in row {
                sum += v as u32;
                if v < mn {
                    mn = v;
                }
            }
            seg_min[seg] = mn;
            let mean = sum as f32 / 16.0;
            let mut var = 0f32;
            for &v in row {
                let d = v as f32 - mean;
                var += d * d;
            }
            seg_variance[seg] = (var / 16.0).round() as u32;
        }
        Self { m, table, seg_variance, seg_min }
    }

    #[inline]
    pub fn get(&self, seg: usize, code: u8) -> u8 {
        self.table[seg * 16 + (code as usize & 0x0F)]
    }

    /// Return segment indices sorted by descending variance.
    pub fn order_by_variance(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.m).collect();
        order.sort_by(|&a, &b| self.seg_variance[b].cmp(&self.seg_variance[a]));
        order
    }
}
