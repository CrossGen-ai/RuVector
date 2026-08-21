//! Cascade retrieval:
//!
//!   query --[coarse oracle: scan all N]--> top `probe_k`
//!         --[fine oracle: score probe_k]--> top `k`
//!
//! Both oracles operate over the same index space (`i in 0..N`), so the
//! cascade is embarrassingly simple: sort a `Vec<(f32, u32)>` twice. The
//! payoff is bandwidth — the coarse pass is what dominates a flat scan,
//! and Hamming/INT8 both shrink that pass by 8x / 4x respectively.

use crate::oracle::DistanceOracle;

#[derive(Debug, Clone, Copy)]
pub struct CascadeConfig {
    /// Final `k` results returned to the caller.
    pub k: usize,
    /// Shortlist size passed from coarse to fine oracle.
    /// Setting `probe_k = k` disables the rerank step.
    pub probe_k: usize,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self { k: 10, probe_k: 100 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Hit {
    pub id: u32,
    pub score: f32,
}

/// A cascade owns two oracles: `coarse` (typically Hamming / INT8) and
/// `fine` (typically FP32). Either oracle may be omitted by passing the
/// same oracle twice — in which case `search` degenerates to a flat scan.
pub struct Cascade<C: DistanceOracle, F: DistanceOracle> {
    pub coarse: C,
    pub fine: F,
    pub cfg: CascadeConfig,
}

impl<C: DistanceOracle, F: DistanceOracle> Cascade<C, F> {
    pub fn new(coarse: C, fine: F, cfg: CascadeConfig) -> Self {
        assert_eq!(coarse.len(), fine.len(), "oracle length mismatch");
        assert!(cfg.k > 0);
        assert!(cfg.probe_k >= cfg.k);
        Self { coarse, fine, cfg }
    }

    pub fn len(&self) -> usize { self.fine.len() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn footprint_bytes(&self) -> usize {
        self.coarse.footprint_bytes() + self.fine.footprint_bytes()
    }

    /// Return the top-`k` hits, cascading coarse->fine.
    pub fn search(&mut self, query: &[f32]) -> Vec<Hit> {
        let n = self.len();
        if n == 0 { return Vec::new(); }
        self.coarse.prime(query);
        let probe_k = self.cfg.probe_k.min(n);

        // Coarse pass: partial top-`probe_k`.
        let mut scored: Vec<(f32, u32)> = Vec::with_capacity(n);
        for i in 0..n {
            scored.push((self.coarse.score(i), i as u32));
        }
        // select_nth_unstable_by is O(N) on average.
        scored.select_nth_unstable_by(probe_k - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap()
        });
        let shortlist = &mut scored[..probe_k];

        // Fine pass: exact reranking on the shortlist only.
        self.fine.prime(query);
        for entry in shortlist.iter_mut() {
            entry.0 = self.fine.score(entry.1 as usize);
        }
        shortlist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        shortlist.iter()
            .take(self.cfg.k)
            .map(|&(s, id)| Hit { id, score: s })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::{Fp32Oracle, HammingOracle};

    fn synth_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        // xorshift for determinism (no external rng dep).
        let mut s = seed | 1;
        let mut next = || {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ((s >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (0..n).map(|_| (0..dim).map(|_| next()).collect()).collect()
    }

    #[test]
    fn cascade_beats_hamming_only_recall() {
        let dim = 64;
        let n = 512;
        let db = synth_dataset(n, dim, 0xC0FFEE);
        let queries = synth_dataset(50, dim, 0xBEEF);

        // Ground truth via FP32 flat scan.
        let mut truth = Fp32Oracle::from_vectors(&db);
        let ks = 10;
        let mut gt_hits: Vec<Vec<u32>> = Vec::new();
        for q in &queries {
            truth.prime(q);
            let mut all: Vec<(f32, u32)> =
                (0..n).map(|i| (truth.score(i), i as u32)).collect();
            all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            gt_hits.push(all.into_iter().take(ks).map(|(_, i)| i).collect());
        }

        // Hamming-only (probe_k = k, no rerank).
        let mut ham_only = Cascade::new(
            HammingOracle::from_vectors(&db),
            HammingOracle::from_vectors(&db),
            CascadeConfig { k: ks, probe_k: ks },
        );
        let mut ham_recall = 0.0;
        for (q, gt) in queries.iter().zip(gt_hits.iter()) {
            let hits = ham_only.search(q);
            let got: std::collections::HashSet<u32> = hits.iter().map(|h| h.id).collect();
            let hit_ct = gt.iter().filter(|g| got.contains(g)).count();
            ham_recall += hit_ct as f32 / ks as f32;
        }
        ham_recall /= queries.len() as f32;

        // Cascade: Hamming coarse over probe_k=64, FP32 rerank.
        let mut casc = Cascade::new(
            HammingOracle::from_vectors(&db),
            Fp32Oracle::from_vectors(&db),
            CascadeConfig { k: ks, probe_k: 200 },
        );
        let mut casc_recall = 0.0;
        for (q, gt) in queries.iter().zip(gt_hits.iter()) {
            let hits = casc.search(q);
            let got: std::collections::HashSet<u32> = hits.iter().map(|h| h.id).collect();
            let hit_ct = gt.iter().filter(|g| got.contains(g)).count();
            casc_recall += hit_ct as f32 / ks as f32;
        }
        casc_recall /= queries.len() as f32;

        assert!(
            casc_recall > ham_recall + 0.10,
            "cascade should lift recall meaningfully: casc={casc_recall:.3} ham={ham_recall:.3}"
        );
        assert!(casc_recall > 0.80, "cascade recall too low: {casc_recall:.3}");
    }
}
