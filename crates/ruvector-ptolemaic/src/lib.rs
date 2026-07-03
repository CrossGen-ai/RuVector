//! # ruvector-ptolemaic
//!
//! **Ptolemaic Pivot Pruning** for exact k-nearest-neighbor search in Euclidean
//! metric spaces. Uses [Ptolemy's inequality] with a small set of pre-selected
//! pivots to produce a *tighter* lower bound on `d(q, x)` than the classical
//! triangle inequality, cutting the number of distance-comparison operations
//! (DCOs) required per query.
//!
//! ## Background
//!
//! For a query `q`, candidate `x`, and pivots `p1, p2` in Euclidean space,
//! the **triangle inequality** gives:
//!
//! ```text
//! d(q, x) >= | d(q, p1) - d(x, p1) |          (single pivot)
//! ```
//!
//! Ptolemy's inequality, valid in all Euclidean (and more generally, Ptolemaic)
//! spaces, states that for any four points `q, x, p1, p2`:
//!
//! ```text
//! d(q, x) * d(p1, p2) + d(q, p2) * d(x, p1) >= d(q, p1) * d(x, p2)
//! ```
//!
//! Solving for `d(q, x)` yields a lower bound:
//!
//! ```text
//! d(q, x) >= [ d(q, p1) * d(x, p2) - d(q, p2) * d(x, p1) ] / d(p1, p2)
//! ```
//!
//! and the symmetric bound by swapping `p1 <-> p2`. Taking the max of both
//! (and of the triangle bound) gives a strictly tighter lower bound whenever
//! the pivots are geometrically informative.
//!
//! ## Why it matters
//!
//! During k-NN search we maintain a "current radius" `tau` (the distance to
//! the k-th best candidate so far). Any candidate `x` with
//! `lower_bound(q, x) >= tau` can be pruned without evaluating the true
//! (expensive, high-dimensional) distance. Tighter lower bounds ⇒ more pruning
//! ⇒ fewer DCOs ⇒ faster search. The trade-off is a small O(#pivots) table of
//! precomputed candidate-to-pivot distances.
//!
//! ## Backends provided
//!
//! Three backends implement the [`KnnIndex`] trait so callers can swap in and
//! out and benchmark:
//!
//! * [`LinearScan`] — baseline; computes true distance for every point.
//! * [`TrianglePivot`] — pivot pruning with the triangle inequality.
//! * [`PtolemaicPivot`] — pivot pruning with Ptolemy's inequality (this crate's
//!   contribution).
//!
//! All three return **identical, exact** k-NN results by construction — pruning
//! is only a speed optimisation.
//!
//! [Ptolemy's inequality]: https://en.wikipedia.org/wiki/Ptolemy%27s_inequality

use serde::{Deserialize, Serialize};
use std::cell::Cell;

pub mod dataset;
pub mod pivot;

pub use dataset::{gen_dataset, gen_queries};

/// Errors returned by the crate.
#[derive(Debug, thiserror::Error)]
pub enum PtolemaicError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("empty dataset")]
    EmptyDataset,
    #[error("k={k} is larger than dataset size {n}")]
    KTooLarge { k: usize, n: usize },
    #[error("num_pivots={pivots} must be >=2 for Ptolemaic pruning")]
    TooFewPivots { pivots: usize },
}

/// A neighbour returned by a k-NN query: `(id, true_distance)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u32,
    pub distance: f32,
}

impl Eq for Neighbor {}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Per-query search statistics — the currency the paper is written in.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SearchStats {
    /// Number of full distance-comparison operations (DCOs) performed on
    /// dataset points during the query — the dominant cost in high-D.
    pub dcos: u64,
    /// Number of candidates pruned by lower-bound before any DCO.
    pub pruned: u64,
    /// Number of candidates for which the lower bound was computed
    /// (includes both pruned and later-evaluated points).
    pub bounds_checked: u64,
}

/// Uniform interface across the three backends.
pub trait KnnIndex {
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Return the k nearest neighbours to `query` and the stats for the call.
    fn search(&self, query: &[f32], k: usize) -> Result<(Vec<Neighbor>, SearchStats), PtolemaicError>;
}

// Metric

/// Squared-Euclidean distance. We keep sqrt out of the hot path when possible,
/// but Ptolemy's inequality is stated in *true* Euclidean distance so lower
/// bounds must be compared in that space. We therefore use `f32::sqrt` for the
/// bound calculation and cache pivot distances (`d`, not `d^2`).
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    // Auto-vectorises well on x86_64 & aarch64 without unsafe.
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc.sqrt()
}

// Backend 0: Linear scan baseline

pub struct LinearScan {
    data: Vec<f32>,
    n: usize,
    dim: usize,
    dcos: Cell<u64>,
}

impl LinearScan {
    pub fn new(data: Vec<f32>, dim: usize) -> Result<Self, PtolemaicError> {
        if dim == 0 || data.is_empty() {
            return Err(PtolemaicError::EmptyDataset);
        }
        if data.len() % dim != 0 {
            return Err(PtolemaicError::DimensionMismatch {
                expected: dim,
                got: data.len(),
            });
        }
        let n = data.len() / dim;
        Ok(Self {
            data,
            n,
            dim,
            dcos: Cell::new(0),
        })
    }

    #[inline]
    fn point(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

impl KnnIndex for LinearScan {
    fn len(&self) -> usize {
        self.n
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn search(&self, query: &[f32], k: usize) -> Result<(Vec<Neighbor>, SearchStats), PtolemaicError> {
        if query.len() != self.dim {
            return Err(PtolemaicError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k > self.n {
            return Err(PtolemaicError::KTooLarge { k, n: self.n });
        }
        let mut heap: TopK = TopK::new(k);
        let mut stats = SearchStats::default();
        for i in 0..self.n {
            let d = euclidean(query, self.point(i));
            stats.dcos += 1;
            heap.push(Neighbor {
                id: i as u32,
                distance: d,
            });
        }
        self.dcos.set(self.dcos.get() + stats.dcos);
        Ok((heap.into_sorted(), stats))
    }
}

// Shared: pivot table

/// A precomputed table of distances `d(x_i, p_j)` for every dataset point
/// and every pivot. Stored row-major: `pivot_dist[i * P + j]`.
pub struct PivotTable {
    pub pivots: Vec<Vec<f32>>, // P pivots of length dim
    pub pivot_pairwise: Vec<f32>, // P*P pairwise pivot distances
    pub pivot_dist: Vec<f32>,  // N*P
    pub n: usize,
    pub p: usize,
    pub dim: usize,
}

impl PivotTable {
    pub fn build(data: &[f32], dim: usize, pivots: Vec<Vec<f32>>) -> Result<Self, PtolemaicError> {
        let n = data.len() / dim;
        let p = pivots.len();
        if p < 2 {
            return Err(PtolemaicError::TooFewPivots { pivots: p });
        }
        for pv in &pivots {
            if pv.len() != dim {
                return Err(PtolemaicError::DimensionMismatch {
                    expected: dim,
                    got: pv.len(),
                });
            }
        }
        let mut pivot_dist = vec![0.0f32; n * p];
        for i in 0..n {
            let xi = &data[i * dim..(i + 1) * dim];
            for j in 0..p {
                pivot_dist[i * p + j] = euclidean(xi, &pivots[j]);
            }
        }
        let mut pivot_pairwise = vec![0.0f32; p * p];
        for i in 0..p {
            for j in 0..p {
                pivot_pairwise[i * p + j] = euclidean(&pivots[i], &pivots[j]);
            }
        }
        Ok(Self {
            pivots,
            pivot_pairwise,
            pivot_dist,
            n,
            p,
            dim,
        })
    }

    #[inline]
    pub fn dist(&self, point_id: usize, pivot_id: usize) -> f32 {
        self.pivot_dist[point_id * self.p + pivot_id]
    }

    #[inline]
    pub fn pair(&self, i: usize, j: usize) -> f32 {
        self.pivot_pairwise[i * self.p + j]
    }
}

// Backend 1: Triangle pivot pruning

pub struct TrianglePivot {
    data: Vec<f32>,
    dim: usize,
    n: usize,
    table: PivotTable,
}

impl TrianglePivot {
    pub fn build(data: Vec<f32>, dim: usize, pivots: Vec<Vec<f32>>) -> Result<Self, PtolemaicError> {
        let table = PivotTable::build(&data, dim, pivots)?;
        let n = data.len() / dim;
        Ok(Self { data, dim, n, table })
    }

    #[inline]
    fn point(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

impl KnnIndex for TrianglePivot {
    fn len(&self) -> usize {
        self.n
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn search(&self, query: &[f32], k: usize) -> Result<(Vec<Neighbor>, SearchStats), PtolemaicError> {
        if query.len() != self.dim {
            return Err(PtolemaicError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k > self.n {
            return Err(PtolemaicError::KTooLarge { k, n: self.n });
        }
        // Precompute query-to-pivot distances (P DCOs, amortised over N).
        let mut qp = Vec::with_capacity(self.table.p);
        let mut stats = SearchStats::default();
        for pv in &self.table.pivots {
            qp.push(euclidean(query, pv));
            stats.dcos += 1;
        }
        let mut heap = TopK::new(k);
        for i in 0..self.n {
            // Triangle lower bound across all pivots.
            let mut lb = 0.0f32;
            for j in 0..self.table.p {
                let cand = (qp[j] - self.table.dist(i, j)).abs();
                if cand > lb {
                    lb = cand;
                }
            }
            stats.bounds_checked += 1;
            if heap.len() >= k && lb >= heap.tau() {
                stats.pruned += 1;
                continue;
            }
            let d = euclidean(query, self.point(i));
            stats.dcos += 1;
            heap.push(Neighbor {
                id: i as u32,
                distance: d,
            });
        }
        Ok((heap.into_sorted(), stats))
    }
}

// Backend 2: Ptolemaic pivot pruning  (this crate's contribution)

pub struct PtolemaicPivot {
    data: Vec<f32>,
    dim: usize,
    n: usize,
    table: PivotTable,
}

impl PtolemaicPivot {
    pub fn build(data: Vec<f32>, dim: usize, pivots: Vec<Vec<f32>>) -> Result<Self, PtolemaicError> {
        let table = PivotTable::build(&data, dim, pivots)?;
        let n = data.len() / dim;
        Ok(Self { data, dim, n, table })
    }

    #[inline]
    fn point(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

/// Estimated memory (bytes) required for the pivot table.
pub fn pivot_table_bytes(n: usize, p: usize, dim: usize) -> usize {
    // pivot vectors + N*P + P*P, all f32
    p * dim * 4 + n * p * 4 + p * p * 4
}

impl KnnIndex for PtolemaicPivot {
    fn len(&self) -> usize {
        self.n
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn search(&self, query: &[f32], k: usize) -> Result<(Vec<Neighbor>, SearchStats), PtolemaicError> {
        if query.len() != self.dim {
            return Err(PtolemaicError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            });
        }
        if k > self.n {
            return Err(PtolemaicError::KTooLarge { k, n: self.n });
        }
        let p = self.table.p;
        // Precompute d(q, pj)
        let mut qp = Vec::with_capacity(p);
        let mut stats = SearchStats::default();
        for pv in &self.table.pivots {
            qp.push(euclidean(query, pv));
            stats.dcos += 1;
        }
        // Cache pivot pairwise distances for fast access.
        let mut heap = TopK::new(k);
        for i in 0..self.n {
            // Triangle lower bound (single pivot) — cheap floor.
            let mut lb = 0.0f32;
            for j in 0..p {
                let t = (qp[j] - self.table.dist(i, j)).abs();
                if t > lb {
                    lb = t;
                }
            }
            // Ptolemaic lower bound over all ordered pivot pairs.
            //   d(q,x) >= | d(q,pa)*d(x,pb) - d(q,pb)*d(x,pa) | / d(pa,pb)
            // (absolute value used because either ordering can dominate).
            // We check all unordered pairs and keep the max.
            for a in 0..p {
                for b in (a + 1)..p {
                    let dab = self.table.pair(a, b);
                    if dab <= f32::EPSILON {
                        continue; // degenerate pivots
                    }
                    let num =
                        (qp[a] * self.table.dist(i, b) - qp[b] * self.table.dist(i, a)).abs();
                    let cand = num / dab;
                    if cand > lb {
                        lb = cand;
                    }
                }
            }
            stats.bounds_checked += 1;
            if heap.len() >= k && lb >= heap.tau() {
                stats.pruned += 1;
                continue;
            }
            let d = euclidean(query, self.point(i));
            stats.dcos += 1;
            heap.push(Neighbor {
                id: i as u32,
                distance: d,
            });
        }
        Ok((heap.into_sorted(), stats))
    }
}

// Bounded top-k with tau (the current k-th distance)

struct TopK {
    k: usize,
    items: Vec<Neighbor>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            items: Vec::with_capacity(k + 1),
        }
    }
    fn len(&self) -> usize {
        self.items.len()
    }
    /// The current radius (k-th smallest distance so far). If the heap isn't
    /// full yet, returns +inf so no pruning triggers.
    fn tau(&self) -> f32 {
        if self.items.len() < self.k {
            f32::INFINITY
        } else {
            // Max element in a max-heap style structure. We use a sorted
            // Vec keeping it tiny (k is typically <=100).
            self.items[self.items.len() - 1].distance
        }
    }
    fn push(&mut self, n: Neighbor) {
        if self.items.len() < self.k {
            // insertion sort into ascending order
            let pos = self
                .items
                .binary_search_by(|x| x.distance.partial_cmp(&n.distance).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_else(|e| e);
            self.items.insert(pos, n);
        } else if n.distance < self.items[self.items.len() - 1].distance {
            let pos = self
                .items
                .binary_search_by(|x| x.distance.partial_cmp(&n.distance).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_else(|e| e);
            self.items.pop();
            self.items.insert(pos, n);
        }
    }
    fn into_sorted(self) -> Vec<Neighbor> {
        // items are already ascending by construction
        self.items
    }
}

