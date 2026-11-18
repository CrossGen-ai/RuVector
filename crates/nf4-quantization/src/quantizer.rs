//! Core `Quantizer` trait and the packed 4-bit code container.

/// Packed 4-bit codes (two per byte). Length-prefixed to recover the
/// original vector dimension without external bookkeeping.
#[derive(Debug, Clone, PartialEq)]
pub struct Code4 {
    pub dim: u32,
    pub scale: f32,
    pub bias: f32,
    pub bytes: Vec<u8>,
}

impl Code4 {
    pub fn new(dim: usize, scale: f32, bias: f32) -> Self {
        let n_bytes = (dim + 1) / 2;
        Self {
            dim: dim as u32,
            scale,
            bias,
            bytes: vec![0u8; n_bytes],
        }
    }

    #[inline]
    pub fn set_nibble(&mut self, i: usize, v: u8) {
        let byte = i >> 1;
        let shift = ((i & 1) as u8) * 4;
        self.bytes[byte] =
            (self.bytes[byte] & !(0x0F << shift)) | ((v & 0x0F) << shift);
    }

    #[inline]
    pub fn get_nibble(&self, i: usize) -> u8 {
        let byte = i >> 1;
        let shift = ((i & 1) as u8) * 4;
        (self.bytes[byte] >> shift) & 0x0F
    }

    /// Bytes per code (including the 8-byte header for scale+bias+dim).
    pub fn storage_bytes(&self) -> usize {
        self.bytes.len() + 12
    }
}

/// A trait abstracting over NF4 variants so callers can benchmark
/// side-by-side with no `dyn` cost in the hot loop.
pub trait Quantizer: Send + Sync {
    /// Encode one vector.
    fn encode(&self, x: &[f32]) -> Code4;

    /// Reconstruct one vector into a provided buffer.
    fn decode_into(&self, code: &Code4, out: &mut [f32]);

    /// Asymmetric squared L2: query stays FP32, database is 4-bit.
    /// Default implementation decodes lazily; variants may specialize.
    fn asym_l2_sq(&self, q: &[f32], code: &Code4) -> f32 {
        debug_assert_eq!(q.len(), code.dim as usize);
        let scale = code.scale;
        let bias = code.bias;
        let mut acc = 0f32;
        for (i, &qi) in q.iter().enumerate() {
            let xi = scale * self.level_at(code.get_nibble(i)) + bias;
            let d = qi - xi;
            acc += d * d;
        }
        acc
    }

    /// Level lookup for level `k` ∈ 0..16. Variants with the same
    /// codebook share this implementation.
    fn level_at(&self, k: u8) -> f32;
}
