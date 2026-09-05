//! PQ index: owns the codebooks, the 8-bit codes, and (optionally) the packed
//! 4-bit companion codes. Provides LUT construction for asymmetric distance.

use crate::codebook::{Codebook4, Codebook8, TrainingConfig, K4, K8};

/// Static parameters describing a PQ index.
#[derive(Copy, Clone, Debug)]
pub struct PqParams {
    pub d: usize,
    pub m: usize,
}

impl PqParams {
    pub fn dsub(&self) -> usize {
        self.d / self.m
    }
}

/// PQ index with both 8-bit and 4-bit code layouts.
pub struct PqIndex {
    pub params: PqParams,
    pub fine: Codebook8,
    pub coarse: Codebook4,
    pub codes8: Vec<u8>,   // n * m
    pub codes4: Vec<u8>,   // n * ceil(m/2), two codes per byte
    pub n: usize,
}

impl PqIndex {
    /// Train on `train` and encode `data` (both row-major, `n * d`).
    pub fn build(
        train: &[f32],
        n_train: usize,
        data: &[f32],
        n: usize,
        params: PqParams,
        cfg: &TrainingConfig,
    ) -> Self {
        assert_eq!(train.len(), n_train * params.d);
        assert_eq!(data.len(), n * params.d);
        let fine = Codebook8::train(train, n_train, params.d, params.m, cfg);
        let coarse = fine.coarse(cfg);
        let codes8 = fine.encode(data, n);
        let codes4 = coarse.pack_from_fine(&codes8, n);
        Self {
            params,
            fine,
            coarse,
            codes8,
            codes4,
            n,
        }
    }

    /// Build the 8-bit LUT for a query: `m * K8` f32 entries, indexed by
    /// `[subspace * K8 + code]`, holding partial squared L2 distances.
    pub fn lut8(&self, query: &[f32]) -> Vec<f32> {
        let dsub = self.params.dsub();
        let mut lut = vec![0.0f32; self.params.m * K8];
        for j in 0..self.params.m {
            let q_off = j * dsub;
            for k in 0..K8 {
                let c_off = (j * K8 + k) * dsub;
                let mut s = 0.0f32;
                for t in 0..dsub {
                    let d = query[q_off + t] - self.fine.centroids[c_off + t];
                    s += d * d;
                }
                lut[j * K8 + k] = s;
            }
        }
        lut
    }

    /// Build the 4-bit LUT for a query: `m * K4` f32 entries. Fits comfortably
    /// in L1 for the m values typical of vector search (m ≤ 64).
    pub fn lut4(&self, query: &[f32]) -> Vec<f32> {
        let dsub = self.params.dsub();
        let mut lut = vec![0.0f32; self.params.m * K4];
        for j in 0..self.params.m {
            let q_off = j * dsub;
            for k in 0..K4 {
                let c_off = (j * K4 + k) * dsub;
                let mut s = 0.0f32;
                for t in 0..dsub {
                    let d = query[q_off + t] - self.coarse.centroids[c_off + t];
                    s += d * d;
                }
                lut[j * K4 + k] = s;
            }
        }
        lut
    }
}
