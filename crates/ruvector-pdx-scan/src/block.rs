//! Storage layouts.
//!
//! Two concrete backends implement the [`VectorStore`] trait:
//!
//! * [`HorizontalStore`] — flat `n * d` row-major `f32` buffer. Baseline.
//! * [`PdxStore`] — vectors grouped into blocks of [`BLOCK_LEN`] where each
//!   dimension is contiguous across the whole block (`d * BLOCK_LEN` per
//!   block, plus a possibly-shorter trailing block).
//!
//! The two are indexed by the same *external* id (0..n). PDX also exposes
//! block-local coordinates for the pruned scan.

/// Vectors per PDX block. 64 is the value used in the PDX paper and matches
/// two full 256-bit SIMD registers on AVX2 (or one 512-bit register on
/// AVX-512). Kept as a `const` (not generic) so callers see a stable ABI.
pub const BLOCK_LEN: usize = 64;

/// Common interface between horizontal and PDX storage.
pub trait VectorStore {
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    /// Copy vector `id` into `out` (assumed length `dim()`).
    fn get(&self, id: usize, out: &mut [f32]);
}

/// Row-major `n * d` `f32` store. This is the "horizontal" baseline.
#[derive(Clone)]
pub struct HorizontalStore {
    dim: usize,
    data: Vec<f32>,
}

impl HorizontalStore {
    pub fn from_iter<I, V>(dim: usize, iter: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: AsRef<[f32]>,
    {
        let mut data = Vec::new();
        for v in iter {
            let s = v.as_ref();
            assert_eq!(s.len(), dim, "vector length must match dim");
            data.extend_from_slice(s);
        }
        Self { dim, data }
    }

    #[inline]
    pub fn row(&self, id: usize) -> &[f32] {
        let d = self.dim;
        &self.data[id * d..(id + 1) * d]
    }
}

impl VectorStore for HorizontalStore {
    fn len(&self) -> usize { self.data.len() / self.dim }
    fn dim(&self) -> usize { self.dim }
    fn get(&self, id: usize, out: &mut [f32]) {
        out.copy_from_slice(self.row(id));
    }
}

/// PDX (dimension-partitioned) block store.
///
/// Layout: a `Vec` of blocks, each block a `Vec<f32>` of length
/// `dim * block_len_actual` where `block_len_actual == BLOCK_LEN` for all
/// full blocks and `== n % BLOCK_LEN` for the tail block. Within a block,
/// dimension `j`'s values for the block's vectors are at
/// `block[j*bl .. (j+1)*bl]`.
///
/// The `perm` field records the *dimension permutation* used at build time
/// (variance-descending is the default; see `DimensionOrder::VarianceDesc`).
/// This lets the pruned scan walk the highest-variance dimensions first
/// without touching the query at scan time — the query is permuted once at
/// scan start.
#[derive(Clone)]
pub struct PdxStore {
    dim: usize,
    n: usize,
    /// Blocks: outer len = ceil(n / BLOCK_LEN). Each inner Vec is
    /// column-major within the block (dim-major).
    blocks: Vec<Vec<f32>>,
    /// perm[k] = original-dimension-id that we store at PDX slot k.
    /// Inverse map for reconstruction when needed.
    perm: Vec<u32>,
}

impl PdxStore {
    /// Build a PDX store from a horizontal store, using the given dimension
    /// permutation. `perm.len() == dim`.
    pub fn from_horizontal(h: &HorizontalStore, perm: Vec<u32>) -> Self {
        let dim = h.dim();
        let n = h.len();
        assert_eq!(perm.len(), dim);
        let num_blocks = (n + BLOCK_LEN - 1) / BLOCK_LEN;
        let mut blocks = Vec::with_capacity(num_blocks);
        for b in 0..num_blocks {
            let start = b * BLOCK_LEN;
            let end = (start + BLOCK_LEN).min(n);
            let bl = end - start;
            let mut block = vec![0.0f32; dim * bl];
            for (k, &orig_dim) in perm.iter().enumerate() {
                let od = orig_dim as usize;
                for (local, id) in (start..end).enumerate() {
                    block[k * bl + local] = h.row(id)[od];
                }
            }
            blocks.push(block);
        }
        Self { dim, n, blocks, perm }
    }

    #[inline]
    pub fn num_blocks(&self) -> usize { self.blocks.len() }

    #[inline]
    pub fn block(&self, b: usize) -> &[f32] { &self.blocks[b] }

    /// Number of vectors in block `b` (== BLOCK_LEN except possibly the last).
    #[inline]
    pub fn block_size(&self, b: usize) -> usize {
        let start = b * BLOCK_LEN;
        let end = (start + BLOCK_LEN).min(self.n);
        end - start
    }

    /// Permutation used for dimension ordering: `perm[k]` is the original
    /// dimension index stored at PDX slot `k`.
    pub fn perm(&self) -> &[u32] { &self.perm }

    /// Estimate resident bytes (f32 data only, ignoring Vec headers/perm).
    pub fn bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.len() * 4).sum()
    }
}

impl VectorStore for PdxStore {
    fn len(&self) -> usize { self.n }
    fn dim(&self) -> usize { self.dim }
    fn get(&self, id: usize, out: &mut [f32]) {
        let b = id / BLOCK_LEN;
        let local = id % BLOCK_LEN;
        let bl = self.block_size(b);
        let block = &self.blocks[b];
        for k in 0..self.dim {
            let od = self.perm[k] as usize;
            out[od] = block[k * bl + local];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> HorizontalStore {
        HorizontalStore::from_iter(4, vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10., 11., 12.],
        ])
    }

    #[test]
    fn horizontal_roundtrip() {
        let h = tiny();
        assert_eq!(h.len(), 3);
        assert_eq!(h.dim(), 4);
        let mut buf = vec![0f32; 4];
        h.get(1, &mut buf);
        assert_eq!(buf, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn pdx_roundtrip_identity_perm() {
        let h = tiny();
        let perm: Vec<u32> = (0..4).collect();
        let p = PdxStore::from_horizontal(&h, perm);
        let mut buf = vec![0f32; 4];
        for id in 0..h.len() {
            p.get(id, &mut buf);
            assert_eq!(buf, h.row(id).to_vec(), "id {id} roundtrip");
        }
    }

    #[test]
    fn pdx_roundtrip_reversed_perm() {
        let h = tiny();
        let perm: Vec<u32> = (0..4).rev().collect();
        let p = PdxStore::from_horizontal(&h, perm);
        let mut buf = vec![0f32; 4];
        for id in 0..h.len() {
            p.get(id, &mut buf);
            assert_eq!(buf, h.row(id).to_vec(), "reversed perm roundtrip {id}");
        }
    }
}
