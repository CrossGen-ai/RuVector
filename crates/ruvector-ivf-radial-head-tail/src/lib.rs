//! Radial Head-Tail Quantization for IVF cells (IVF-RHT).
//!
//! Within each IVF posting list, vectors are split by radial distance from the
//! cell centroid. The "head" (close to centroid) is quantized aggressively
//! (4-bit per dim) because small residuals have low information. The "tail"
//! (far from centroid) uses a higher precision (8-bit per dim) because far
//! residuals dominate the query distance error budget.
//!
//! Trait-based so the backend quantizer can be swapped.

pub mod quant;

use quant::{Quantizer, Sq4, Sq8};
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;

#[derive(Clone, Copy, Debug)]
pub struct BuildParams {
    pub n_cells: usize,
    pub head_fraction: f32, // fraction of vectors assigned to head (close to centroid)
    pub kmeans_iters: usize,
    pub seed: u64,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self {
            n_cells: 64,
            head_fraction: 0.5,
            kmeans_iters: 10,
            seed: 0xC0FFEE,
        }
    }
}

/// How to scan a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScanMode {
    /// Full scan: decode head and tail.
    Full,
    /// Head-only scan: skip the tail (faster, lower recall).
    HeadOnly,
    /// Radial-adaptive: if query is close to centroid, scan head only; else full.
    Adaptive,
}

/// One IVF cell with head/tail split residuals.
pub struct Cell {
    pub centroid: Vec<f32>,
    pub head_ids: Vec<u32>,
    pub tail_ids: Vec<u32>,
    pub head_codes: Vec<u8>, // Sq4 packed
    pub tail_codes: Vec<u8>, // Sq8
    pub head_params: quant::SqParams,
    pub tail_params: quant::SqParams,
    /// Max ||x-c|| of the head (used for Adaptive scan threshold).
    pub head_radius: f32,
}

pub struct IvfRht {
    pub dim: usize,
    pub cells: Vec<Cell>,
}

impl IvfRht {
    pub fn build(data: &[Vec<f32>], params: BuildParams) -> Self {
        assert!(!data.is_empty());
        let dim = data[0].len();
        let mut rng = Pcg64Mcg::new(params.seed as u128);

        // Init centroids: random distinct points.
        let mut idxs: Vec<usize> = (0..data.len()).collect();
        idxs.shuffle(&mut rng);
        let mut centroids: Vec<Vec<f32>> = idxs
            .into_iter()
            .take(params.n_cells)
            .map(|i| data[i].clone())
            .collect();

        // Lloyd's k-means.
        let mut assign = vec![0usize; data.len()];
        for _ in 0..params.kmeans_iters {
            // Assign step.
            for (i, x) in data.iter().enumerate() {
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for (c_i, c) in centroids.iter().enumerate() {
                    let d = l2_sq(x, c);
                    if d < best_d {
                        best_d = d;
                        best = c_i;
                    }
                }
                assign[i] = best;
            }
            // Update step.
            let mut sums = vec![vec![0.0f32; dim]; params.n_cells];
            let mut counts = vec![0usize; params.n_cells];
            for (i, x) in data.iter().enumerate() {
                let c = assign[i];
                for d in 0..dim {
                    sums[c][d] += x[d];
                }
                counts[c] += 1;
            }
            for c in 0..params.n_cells {
                if counts[c] > 0 {
                    for d in 0..dim {
                        centroids[c][d] = sums[c][d] / counts[c] as f32;
                    }
                }
            }
        }

        // Compute residuals, split head/tail by radial distance.
        let mut cell_items: Vec<Vec<(u32, Vec<f32>, f32)>> = (0..params.n_cells).map(|_| Vec::new()).collect();
        for (i, x) in data.iter().enumerate() {
            let c = assign[i];
            let mut r = vec![0.0f32; dim];
            for d in 0..dim {
                r[d] = x[d] - centroids[c][d];
            }
            let rad2: f32 = r.iter().map(|v| v * v).sum();
            cell_items[c].push((i as u32, r, rad2));
        }

        let mut cells = Vec::with_capacity(params.n_cells);
        for (c_i, items) in cell_items.into_iter().enumerate() {
            let centroid = centroids[c_i].clone();
            if items.is_empty() {
                cells.push(Cell {
                    centroid,
                    head_ids: vec![],
                    tail_ids: vec![],
                    head_codes: vec![],
                    tail_codes: vec![],
                    head_params: quant::SqParams::zero(dim),
                    tail_params: quant::SqParams::zero(dim),
                    head_radius: 0.0,
                });
                continue;
            }
            let mut sorted = items;
            sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
            let split = ((sorted.len() as f32) * params.head_fraction).round() as usize;
            let split = split.clamp(0, sorted.len());
            let (head, tail) = sorted.split_at(split);

            let head_residuals: Vec<&[f32]> = head.iter().map(|(_, r, _)| r.as_slice()).collect();
            let tail_residuals: Vec<&[f32]> = tail.iter().map(|(_, r, _)| r.as_slice()).collect();
            let head_params = quant::SqParams::fit(&head_residuals, dim);
            let tail_params = quant::SqParams::fit(&tail_residuals, dim);
            let head_codes = Sq4::encode_many(&head_residuals, &head_params);
            let tail_codes = Sq8::encode_many(&tail_residuals, &tail_params);
            let head_ids: Vec<u32> = head.iter().map(|(i, _, _)| *i).collect();
            let tail_ids: Vec<u32> = tail.iter().map(|(i, _, _)| *i).collect();
            let head_radius = head.last().map(|(_, _, r2)| r2.sqrt()).unwrap_or(0.0);

            cells.push(Cell {
                centroid,
                head_ids,
                tail_ids,
                head_codes,
                tail_codes,
                head_params,
                tail_params,
                head_radius,
            });
        }

        IvfRht { dim, cells }
    }

    /// Return top-k results (id, distance_sq) for `query` with the given scan mode.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize, mode: ScanMode) -> Vec<(u32, f32)> {
        // Probe the nprobe closest cells.
        let mut cell_dists: Vec<(usize, f32)> = self
            .cells
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_sq(query, &c.centroid)))
            .collect();
        cell_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut heap: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
        for (c_i, qc_dist2) in cell_dists.into_iter().take(nprobe) {
            let cell = &self.cells[c_i];
            let scan_head = true;
            let scan_tail = match mode {
                ScanMode::Full => true,
                ScanMode::HeadOnly => false,
                ScanMode::Adaptive => qc_dist2.sqrt() > cell.head_radius * 0.5,
            };

            // Precompute q-c once.
            let qc: Vec<f32> = (0..self.dim).map(|d| query[d] - cell.centroid[d]).collect();

            if scan_head && !cell.head_ids.is_empty() {
                let n = cell.head_ids.len();
                let mut decoded = vec![0.0f32; self.dim];
                for i in 0..n {
                    Sq4::decode_into(&cell.head_codes, i, &cell.head_params, &mut decoded);
                    let d2 = l2_sq_sub(&qc, &decoded);
                    push_topk(&mut heap, cell.head_ids[i], d2, k);
                }
            }
            if scan_tail && !cell.tail_ids.is_empty() {
                let n = cell.tail_ids.len();
                let mut decoded = vec![0.0f32; self.dim];
                for i in 0..n {
                    Sq8::decode_into(&cell.tail_codes, i, &cell.tail_params, &mut decoded);
                    let d2 = l2_sq_sub(&qc, &decoded);
                    push_topk(&mut heap, cell.tail_ids[i], d2, k);
                }
            }
        }
        heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        heap.truncate(k);
        heap
    }

    /// Memory footprint in bytes (approx).
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0usize;
        for c in &self.cells {
            total += c.centroid.len() * 4;
            total += c.head_codes.len();
            total += c.tail_codes.len();
            total += c.head_ids.len() * 4 + c.tail_ids.len() * 4;
            total += c.head_params.scale.len() * 4 + c.head_params.offset.len() * 4;
            total += c.tail_params.scale.len() * 4 + c.tail_params.offset.len() * 4;
        }
        total
    }
}

fn push_topk(heap: &mut Vec<(u32, f32)>, id: u32, d: f32, k: usize) {
    if heap.len() < k {
        heap.push((id, d));
        if heap.len() == k {
            // Sort once to establish worst at end.
            heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
    } else {
        if d < heap[k - 1].1 {
            heap[k - 1] = (id, d);
            // Local bubble up.
            let mut i = k - 1;
            while i > 0 && heap[i - 1].1 > heap[i].1 {
                heap.swap(i, i - 1);
                i -= 1;
            }
        }
    }
}

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[inline]
fn l2_sq_sub(qc: &[f32], decoded_r: &[f32]) -> f32 {
    // ||q - x||^2 = ||(q-c) - (x-c)||^2
    let mut s = 0.0f32;
    for i in 0..qc.len() {
        let d = qc[i] - decoded_r[i];
        s += d * d;
    }
    s
}

/// Baseline uniform SQ8 IVF (for comparison).
pub mod baseline {
    use super::*;

    pub struct IvfSq8 {
        pub dim: usize,
        pub cells: Vec<BaselineCell>,
    }

    pub struct BaselineCell {
        pub centroid: Vec<f32>,
        pub ids: Vec<u32>,
        pub codes: Vec<u8>,
        pub params: quant::SqParams,
    }

    impl IvfSq8 {
        pub fn build(data: &[Vec<f32>], params: BuildParams) -> Self {
            let dim = data[0].len();
            let mut rng = Pcg64Mcg::new(params.seed as u128);
            let mut idxs: Vec<usize> = (0..data.len()).collect();
            idxs.shuffle(&mut rng);
            let mut centroids: Vec<Vec<f32>> = idxs
                .into_iter()
                .take(params.n_cells)
                .map(|i| data[i].clone())
                .collect();
            let mut assign = vec![0usize; data.len()];
            for _ in 0..params.kmeans_iters {
                for (i, x) in data.iter().enumerate() {
                    let mut best = 0usize;
                    let mut best_d = f32::INFINITY;
                    for (c_i, c) in centroids.iter().enumerate() {
                        let d = l2_sq(x, c);
                        if d < best_d {
                            best_d = d;
                            best = c_i;
                        }
                    }
                    assign[i] = best;
                }
                let mut sums = vec![vec![0.0f32; dim]; params.n_cells];
                let mut counts = vec![0usize; params.n_cells];
                for (i, x) in data.iter().enumerate() {
                    for d in 0..dim {
                        sums[assign[i]][d] += x[d];
                    }
                    counts[assign[i]] += 1;
                }
                for c in 0..params.n_cells {
                    if counts[c] > 0 {
                        for d in 0..dim {
                            centroids[c][d] = sums[c][d] / counts[c] as f32;
                        }
                    }
                }
            }
            let mut cells: Vec<BaselineCell> = (0..params.n_cells)
                .map(|c_i| BaselineCell {
                    centroid: centroids[c_i].clone(),
                    ids: vec![],
                    codes: vec![],
                    params: quant::SqParams::zero(dim),
                })
                .collect();
            let mut tmp: Vec<Vec<(u32, Vec<f32>)>> = (0..params.n_cells).map(|_| Vec::new()).collect();
            for (i, x) in data.iter().enumerate() {
                let c = assign[i];
                let mut r = vec![0.0f32; dim];
                for d in 0..dim {
                    r[d] = x[d] - centroids[c][d];
                }
                tmp[c].push((i as u32, r));
            }
            for (c_i, items) in tmp.into_iter().enumerate() {
                let residuals: Vec<&[f32]> = items.iter().map(|(_, r)| r.as_slice()).collect();
                let p = quant::SqParams::fit(&residuals, dim);
                let codes = Sq8::encode_many(&residuals, &p);
                cells[c_i].ids = items.iter().map(|(i, _)| *i).collect();
                cells[c_i].codes = codes;
                cells[c_i].params = p;
            }
            IvfSq8 { dim, cells }
        }

        pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(u32, f32)> {
            let mut cd: Vec<(usize, f32)> = self
                .cells
                .iter()
                .enumerate()
                .map(|(i, c)| (i, l2_sq(query, &c.centroid)))
                .collect();
            cd.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            let mut heap: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
            for (c_i, _) in cd.into_iter().take(nprobe) {
                let cell = &self.cells[c_i];
                let qc: Vec<f32> = (0..self.dim).map(|d| query[d] - cell.centroid[d]).collect();
                let mut decoded = vec![0.0f32; self.dim];
                for i in 0..cell.ids.len() {
                    Sq8::decode_into(&cell.codes, i, &cell.params, &mut decoded);
                    let d2 = l2_sq_sub(&qc, &decoded);
                    push_topk(&mut heap, cell.ids[i], d2, k);
                }
            }
            heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            heap.truncate(k);
            heap
        }

        pub fn memory_bytes(&self) -> usize {
            let mut t = 0usize;
            for c in &self.cells {
                t += c.centroid.len() * 4 + c.codes.len() + c.ids.len() * 4
                    + c.params.scale.len() * 4 + c.params.offset.len() * 4;
            }
            t
        }
    }
}

/// Exact brute-force for ground truth.
pub fn exact_topk(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut dists: Vec<(u32, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, x)| (i as u32, l2_sq(query, x)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

/// Recall@k between a predicted top-k and the ground-truth top-k.
pub fn recall_at_k(pred: &[(u32, f32)], truth: &[(u32, f32)]) -> f32 {
    let truth_set: std::collections::HashSet<u32> = truth.iter().map(|(i, _)| *i).collect();
    let hit = pred.iter().filter(|(i, _)| truth_set.contains(i)).count();
    hit as f32 / truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_data(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Pcg64Mcg::new(seed as u128);
        (0..n)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
            .collect()
    }

    #[test]
    fn builds_and_searches() {
        let data = gen_data(2000, 32, 1);
        let p = BuildParams {
            n_cells: 16,
            head_fraction: 0.5,
            kmeans_iters: 5,
            seed: 7,
        };
        let idx = IvfRht::build(&data, p);
        let q = &data[0];
        let r = idx.search(q, 10, 4, ScanMode::Full);
        assert_eq!(r.len(), 10);
        // The query itself should be in top-10.
        assert!(r.iter().any(|(i, _)| *i == 0));
    }

    #[test]
    fn full_scan_better_recall_than_head_only() {
        let data = gen_data(2000, 32, 2);
        let queries = gen_data(50, 32, 3);
        let p = BuildParams {
            n_cells: 16,
            head_fraction: 0.5,
            kmeans_iters: 5,
            seed: 7,
        };
        let idx = IvfRht::build(&data, p);
        let mut full_r = 0.0;
        let mut head_r = 0.0;
        for q in &queries {
            let truth = exact_topk(&data, q, 10);
            full_r += recall_at_k(&idx.search(q, 10, 8, ScanMode::Full), &truth);
            head_r += recall_at_k(&idx.search(q, 10, 8, ScanMode::HeadOnly), &truth);
        }
        full_r /= queries.len() as f32;
        head_r /= queries.len() as f32;
        assert!(full_r >= head_r - 0.01, "full={full_r} head={head_r}");
    }
}
