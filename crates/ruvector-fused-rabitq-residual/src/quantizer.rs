//! Common `Quantizer` trait so scan/index code is quantizer-agnostic.

/// Precomputed per-query state (rotated query + its squared norm).
pub struct QueryCtx {
    pub q_rot: Vec<f32>,
    pub q_norm_sq: f32,
}

/// Vector quantizer: encode a raw f32 vector to a byte code, and score a
/// prepared query against a stored code in approximate squared-L2.
pub trait Quantizer: Sync + Send {
    /// Size in bytes of one encoded vector.
    fn code_bytes(&self) -> usize;

    /// Human-readable identifier used in benchmark output.
    fn name(&self) -> &'static str;

    /// Encode a raw vector to its byte code.
    fn encode(&self, v: &[f32]) -> Vec<u8>;

    /// Prepare per-query state (typically: rotate query, cache ||q||²).
    fn prepare_query(&self, q: &[f32]) -> QueryCtx;

    /// Approximate squared-L2 distance between the stored `code` and the
    /// query prepared in `ctx`.
    fn distance(&self, code: &[u8], ctx: &QueryCtx) -> f32;
}
