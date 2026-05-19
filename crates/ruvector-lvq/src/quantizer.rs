use crate::distance::Metric;
use crate::error::LvqError;

/// Codes are stored as packed bytes (4-bit packs two-per-byte; 8-bit is one).
pub type Code = Vec<u8>;

/// One encoded vector + its per-vector metadata.
#[derive(Debug, Clone)]
pub struct Encoded {
    pub code: Code,
    /// Per-vector scale Δ such that x ≈ scale*q + bias + mean.
    pub scale: f32,
    /// Per-vector bias (the floor of the quantized range, after centering).
    pub bias: f32,
    /// ‖x̂‖² where x̂ is the decoded vector. Pre-computed so L2 can be
    /// expressed as ‖q‖² - 2⟨q,x̂⟩ + ‖x̂‖².
    pub decoded_sq_norm: f32,
    /// Optional residual encoding (second LVQ level).
    pub residual: Option<Box<Encoded>>,
}

/// Trait for quantizers that ingest f32 vectors and decode/score them.
pub trait Quantizer: Send + Sync {
    fn name(&self) -> &'static str;
    fn dim(&self) -> usize;
    /// Bits per stored component (not counting per-vector overhead).
    fn bits_per_component(&self) -> u8;
    /// Bytes used by one stored vector's code (excluding the constant
    /// per-vector overhead: scale + bias + sq_norm = 12 bytes).
    fn code_bytes(&self) -> usize;

    fn fit(&mut self, training: &[Vec<f32>]) -> Result<(), LvqError>;
    fn encode(&self, v: &[f32]) -> Result<Encoded, LvqError>;
    fn decode(&self, e: &Encoded, out: &mut [f32]);

    /// Asymmetric distance: query is f32, db is the encoded vector.
    /// `q_sq_norm` is ‖q‖² (caller can hoist it across many calls).
    fn distance(&self, q: &[f32], q_sq_norm: f32, e: &Encoded, metric: Metric) -> f32;
}

/// Total bytes for one stored vector including per-vector overhead.
#[inline]
pub fn total_bytes(code_bytes: usize) -> usize {
    // scale (4) + bias (4) + sq_norm (4) = 12 bytes overhead
    code_bytes + 12
}
