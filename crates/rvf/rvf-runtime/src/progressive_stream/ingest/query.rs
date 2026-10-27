//! Query policy for a stream that has not completed.
//!
//! Measured (B2, 100k x 128): `ProgressiveIndex` layers A and A+B answer worse
//! than an exact scan over the vectors that have already arrived (A 0.197 and
//! A+B 0.032 against 0.197 and 0.197 for the scan), and the scan is cheap
//! enough to run under a latency bound. So before Layer C is published the
//! default answer is the exact scan over the verified vectors, as long as the
//! scan fits `scan_latency_bound` (calibrated from the measured per-vector
//! cost on this machine, not assumed); beyond it, and for stores too big to
//! scan within the bound, the layers answer. Once Layer C is published it
//! answers (recall 0.95 at ~0.3 ms).
//!
//! Every answer states what it covers: `coverage_vectors` is the fraction of
//! the stream's vectors that had arrived, `QueryPath` says how it was made,
//! and `sealed` stays false until `Complete`.

use super::*;
use rvf_types::quality::BudgetReport;
use std::time::{Duration, Instant};

/// Which engine answers a query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QueryMode {
    /// Scan the arrived vectors while that fits the latency bound and Layer C
    /// is absent; otherwise layers.
    #[default]
    Auto,
    /// Always the published layers (the pre-policy behaviour).
    LayersOnly,
    /// Always the exact scan over arrived vectors.
    ScanOnly,
}

/// Rows timed when calibrating the scan cost.
const CALIBRATION_ROWS: usize = 2048;

impl<'p> StreamIngest<'p> {
    /// Latency bound for the exact-scan path (default 25 ms).
    pub fn set_scan_latency_bound(&mut self, bound: Duration) {
        self.scan_bound = bound;
    }

    /// Largest number of visible vectors the exact scan is allowed to cover,
    /// from the measured cost per vector (calibrated on first use).
    pub fn scan_threshold(&self, q: &[f32]) -> usize {
        let mut ns = self.scan_ns_per_vec.get();
        if ns <= 0.0 {
            let rows = self.vectors.len().min(CALIBRATION_ROWS);
            if rows == 0 {
                return usize::MAX;
            }
            let t = Instant::now();
            let _ = self.vectors.exact_topk_rows(q, 10, rows, &|_| false);
            ns = (t.elapsed().as_nanos() as f64 / rows as f64).max(0.5);
            self.scan_ns_per_vec.set(ns);
        }
        (self.scan_bound.as_nanos() as f64 / ns) as usize
    }

    /// k-NN over what is verified so far, using [`QueryMode::Auto`].
    pub fn query(&self, q: &[f32], k: usize, ef: usize) -> StreamAnswer {
        self.query_with(q, k, ef, QueryMode::Auto)
    }

    /// k-NN with an explicit engine choice (benchmarks, tests).
    pub fn query_with(&self, q: &[f32], k: usize, ef: usize, mode: QueryMode) -> StreamAnswer {
        let t0 = Instant::now();
        let layers = self.layers();
        let state = self.verifier.state();
        // `sealed` is a derived release: re-check the epoch pin now (ADR-353 section 7),
        // so a session superseded since completion is never reported as final.
        let sealed = state == StreamState::Complete
            && self
                .pins
                .is_none_or(|st| self.verifier.recheck_pin(st).is_ok());
        let visible = self.vectors.len();
        let ok_query = !self.unpublished && q.len() == self.vectors.dimension_hint() && visible > 0;
        let can_layers = layers.layer_a && ok_query;
        let scan = ok_query
            && match mode {
                QueryMode::ScanOnly => true,
                QueryMode::LayersOnly => false,
                QueryMode::Auto => {
                    !layers.layer_c && (!can_layers || visible <= self.scan_threshold(q))
                }
            };
        let (mut hits, path) = if scan {
            let t = Instant::now();
            let h = self
                .vectors
                .exact_topk(q, k, &|id| self.deleted.contains(&id));
            let per = t.elapsed().as_nanos() as f64 / visible as f64;
            let old = self.scan_ns_per_vec.get();
            self.scan_ns_per_vec.set(if old <= 0.0 {
                per
            } else {
                0.5 * old + 0.5 * per
            });
            (h, QueryPath::ArrivedScan)
        } else if can_layers {
            let h = self
                .index
                .search(q, k + self.deleted.len().min(k), ef, &self.vectors);
            (h, QueryPath::Layers)
        } else {
            (Vec::new(), QueryPath::Empty)
        };
        hits.retain(|(id, _)| !self.deleted.contains(id) && self.vectors.contains(*id));
        hits.truncate(k);
        let rq = match (path, layers.layer_b, layers.layer_c) {
            (QueryPath::ArrivedScan, _, _) => RetrievalQuality::BruteForceBudgeted,
            (_, _, true) => RetrievalQuality::Full,
            (_, true, false) => RetrievalQuality::Partial,
            _ => RetrievalQuality::LayerAOnly,
        };
        let full = sealed && layers.layer_c && path == QueryPath::Layers;
        let quality = if hits.is_empty() {
            ResponseQuality::Unreliable
        } else if full {
            ResponseQuality::Verified
        } else {
            ResponseQuality::Usable
        };
        let degradation = (!full).then_some(DegradationReport {
            fallback_path: if path == QueryPath::ArrivedScan {
                FallbackPath::SafetyNetSelective
            } else {
                FallbackPath::None
            },
            reason: DegradationReason::IndexNotLoaded { available: layers },
            guarantee_lost: if path == QueryPath::ArrivedScan {
                "exact over the verified vectors received so far only (see coverage_vectors); not sealed"
            } else {
                "index layers and vectors still arriving; not sealed"
            },
        });
        let dim = self.vectors.dimension_hint();
        let expected = if dim > 0 && self.vec_stored_bytes > 0 {
            self.vec_stored_bytes as f64 / (8 + 4 * dim) as f64
        } else {
            0.0
        };
        let coverage_vectors = if expected > 0.0 {
            (visible as f64 / expected).min(1.0)
        } else {
            f64::from(u8::from(sealed))
        };
        let body_len = self.verifier.descriptor().map_or(0, |d| d.header.body_len);
        let coverage_bytes = if body_len > 0 {
            (self.verifier.verified_bytes() as f64 / body_len as f64).min(1.0)
        } else {
            0.0
        };
        StreamAnswer {
            envelope: QualityEnvelope {
                results: hits
                    .into_iter()
                    .map(|(id, distance)| SearchResult {
                        id,
                        distance,
                        retrieval_quality: rq,
                        metadata: None,
                    })
                    .collect(),
                quality,
                evidence: SearchEvidenceSummary {
                    layers_used: layers,
                    safety_net_candidate_count: if path == QueryPath::ArrivedScan {
                        visible as u32
                    } else {
                        0
                    },
                    ..Default::default()
                },
                budgets: BudgetReport {
                    total_us: t0.elapsed().as_micros() as u64,
                    linear_scan_count: if path == QueryPath::ArrivedScan {
                        visible as u64
                    } else {
                        0
                    },
                    ..Default::default()
                },
                degradation,
            },
            sealed,
            state,
            verified_bytes: self.verifier.verified_bytes(),
            visible_vectors: visible,
            path,
            coverage_vectors,
            coverage_bytes,
        }
    }
}
