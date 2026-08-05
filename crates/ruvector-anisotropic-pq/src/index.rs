//! Flat MIPS index using PQ-ADC scoring. The trainer is pluggable through
//! [`PqCodebookTrainer`].

use crate::{argmax, dot, PqCodebook, PqCodebookTrainer, PqError, PqTrainConfig};

/// A single search result for the MIPS index.
#[derive(Debug, Clone, PartialEq)]
pub struct MipsResult {
    /// Insertion-order id.
    pub id: usize,
    /// Approximate inner-product score.
    pub score: f32,
}

/// Compressed MIPS index: flat scan over `N × M` byte codes with a per-query
/// look-up table for inner products.
pub struct AnisoPqIndex {
    codebook: PqCodebook,
    codes: Vec<u8>, // N × M row-major
    n: usize,
    trainer_name: &'static str,
}

impl AnisoPqIndex {
    /// Train a new index end-to-end from `train` and `data`. `train` may be
    /// the same as `data`; in production you'd sample.
    pub fn build<T: PqCodebookTrainer>(
        trainer: &T,
        train: &[Vec<f32>],
        data: &[Vec<f32>],
        cfg: &PqTrainConfig,
    ) -> Result<Self, PqError> {
        let codebook = trainer.train(train, cfg)?;
        let n = data.len();
        let mut codes = vec![0u8; n * codebook.m];
        let mut buf = vec![0u8; codebook.m];
        for (i, x) in data.iter().enumerate() {
            if x.len() != codebook.m * codebook.d {
                return Err(PqError::ShapeMismatch {
                    expected: codebook.m * codebook.d,
                    got: x.len(),
                });
            }
            codebook.encode(x, &mut buf);
            codes[i * codebook.m..(i + 1) * codebook.m].copy_from_slice(&buf);
        }
        Ok(Self { codebook, codes, n, trainer_name: trainer.name() })
    }

    /// Top-k MIPS search using per-query ADC LUT. Returns ids sorted by
    /// descending approximate inner product.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<MipsResult> {
        let cb = &self.codebook;
        assert_eq!(query.len(), cb.m * cb.d);
        // Look-up table: LUT[sub][code] = <query_sub, centroid[sub, code]>
        let mut lut = vec![0.0f32; cb.m * cb.k];
        for sub in 0..cb.m {
            let qs = &query[sub * cb.d..(sub + 1) * cb.d];
            for c in 0..cb.k {
                lut[sub * cb.k + c] = dot(qs, cb.centroid(sub, c));
            }
        }
        // Sum contributions across sub-spaces per vector.
        let mut scores = vec![0.0f32; self.n];
        for i in 0..self.n {
            let row = &self.codes[i * cb.m..(i + 1) * cb.m];
            let mut acc = 0.0f32;
            for sub in 0..cb.m {
                acc += lut[sub * cb.k + row[sub] as usize];
            }
            scores[i] = acc;
        }
        // Top-k (partial sort — cheap since bench k is small).
        let mut order: Vec<usize> = (0..self.n).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        order
            .into_iter()
            .take(k)
            .map(|id| MipsResult { id, score: scores[id] })
            .collect()
    }

    /// Argmax MIPS lookup (top-1 shortcut).
    pub fn argmax(&self, query: &[f32]) -> MipsResult {
        let top = self.search(query, 1);
        top.into_iter().next().unwrap_or(MipsResult { id: argmax(&[0.0]), score: 0.0 })
    }

    /// Approximate memory footprint (codes + centroids).
    pub fn memory_bytes(&self) -> usize {
        self.codes.len() + self.codebook.memory_bytes()
    }

    /// Number of database vectors.
    pub fn len(&self) -> usize { self.n }
    /// Is the index empty?
    pub fn is_empty(&self) -> bool { self.n == 0 }
    /// Trainer name used.
    pub fn trainer(&self) -> &'static str { self.trainer_name }
    /// Underlying codebook.
    pub fn codebook(&self) -> &PqCodebook { &self.codebook }
}
