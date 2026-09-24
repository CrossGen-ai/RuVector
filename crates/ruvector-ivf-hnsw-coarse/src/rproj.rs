//! Signed random-projection coarse quantizer.
//!
//! Rademacher-projection sketch (b bits per centroid). At query time, sketch
//! the query then rank centroids by Hamming distance. `top_lists` cost is
//! `O(b · d + C)` (projection + linear scan of small integer sketch), which
//! beats `O(C · d)` when `d >> b/8`. Recall degrades relative to Flat because
//! Hamming on signed projections only approximates cosine, not L2.
//!
//! This backend exists as a baseline: it is the classical LSH-style shortcut
//! that HnswCoarse must beat.

use crate::{CoarseError, IvfCoarse};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Config for [`RandomProjCoarse`].
#[derive(Debug, Clone, Copy)]
pub struct RandomProjConfig {
    /// Sketch length in bits. Rounded up to a multiple of 64.
    pub bits: usize,
    /// Seed for the Rademacher projection matrix.
    pub seed: u64,
}

impl Default for RandomProjConfig {
    fn default() -> Self {
        Self {
            bits: 128,
            seed: 0xB17E_5EED,
        }
    }
}

/// Signed random-projection coarse quantizer.
#[derive(Debug)]
pub struct RandomProjCoarse {
    cfg: RandomProjConfig,
    d: usize,
    // Row-major (bits/64 * 64 x d) Rademacher matrix ({-1, +1}).
    proj: Vec<i8>,
    // Sketch words per centroid.
    sketches: Vec<u64>,
    words_per: usize,
    c: usize,
}

impl RandomProjCoarse {
    /// Build with default config.
    pub fn new() -> Self {
        Self::with_config(RandomProjConfig::default())
    }
    /// Build with explicit config.
    pub fn with_config(cfg: RandomProjConfig) -> Self {
        Self {
            cfg,
            d: 0,
            proj: Vec::new(),
            sketches: Vec::new(),
            words_per: 0,
            c: 0,
        }
    }

    #[inline]
    fn sketch(&self, x: &[f32], out: &mut [u64]) {
        let bits = self.words_per * 64;
        out.iter_mut().for_each(|w| *w = 0);
        for b in 0..bits {
            let row = &self.proj[b * self.d..(b + 1) * self.d];
            let mut acc = 0.0f32;
            for i in 0..self.d {
                acc += x[i] * row[i] as f32;
            }
            if acc >= 0.0 {
                out[b / 64] |= 1u64 << (b % 64);
            }
        }
    }
}

impl Default for RandomProjCoarse {
    fn default() -> Self {
        Self::new()
    }
}

impl IvfCoarse for RandomProjCoarse {
    fn fit(&mut self, centroids: &[f32], d: usize) -> Result<(), CoarseError> {
        if d == 0 || centroids.len() % d != 0 {
            return Err(CoarseError::DimensionMismatch {
                expected: d,
                got: centroids.len(),
            });
        }
        self.d = d;
        self.c = centroids.len() / d;
        let bits = ((self.cfg.bits + 63) / 64) * 64;
        self.words_per = bits / 64;
        // Rademacher: {-1, +1} projection matrix.
        let mut rng = StdRng::seed_from_u64(self.cfg.seed);
        self.proj = (0..bits * d)
            .map(|_| if rng.gen::<bool>() { 1i8 } else { -1i8 })
            .collect();
        self.sketches = vec![0u64; self.c * self.words_per];
        for i in 0..self.c {
            let x = &centroids[i * d..(i + 1) * d];
            let mut tmp = vec![0u64; self.words_per];
            self.sketch(x, &mut tmp);
            self.sketches[i * self.words_per..(i + 1) * self.words_per].copy_from_slice(&tmp);
        }
        Ok(())
    }

    fn top_lists(&self, query: &[f32], nprobe: usize) -> Vec<u32> {
        assert_eq!(query.len(), self.d);
        let nprobe = nprobe.min(self.c);
        let mut qs = vec![0u64; self.words_per];
        self.sketch(query, &mut qs);
        let mut scored: Vec<(u32, u32)> = (0..self.c)
            .map(|i| {
                let s = &self.sketches[i * self.words_per..(i + 1) * self.words_per];
                let mut h = 0u32;
                for w in 0..self.words_per {
                    h += (s[w] ^ qs[w]).count_ones();
                }
                (h, i as u32)
            })
            .collect();
        let pivot = nprobe.saturating_sub(1).min(scored.len() - 1);
        scored.select_nth_unstable_by(pivot, |a, b| a.0.cmp(&b.0));
        let mut head: Vec<(u32, u32)> = scored.into_iter().take(nprobe).collect();
        head.sort_by_key(|x| x.0);
        head.into_iter().map(|(_, i)| i).collect()
    }

    fn name(&self) -> &'static str {
        "RandomProjCoarse"
    }

    fn extra_bytes(&self) -> usize {
        self.proj.len() + self.sketches.len() * 8
    }
}
