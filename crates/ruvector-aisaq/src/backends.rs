//! Swappable distance backends.
//!
//! Same graph, same beam search, three storage decisions:
//!
//! | Variant       | Where PQ codes live | Where vectors live | RAM footprint per point |
//! |---------------|--------------------|--------------------|------------------------|
//! | `FlatF32Ram`  | (n/a)              | RAM                | `D * 4` B              |
//! | `PqRam`       | RAM                | (discarded)        | `M` B                  |
//! | `PqDisk`      | SSD (mmap)         | (discarded)        | `~0` B (mmap-backed)   |
//!
//! `PqDisk` is AISAQ: only the graph stays in memory. Reads
//! into the mmap trigger page-ins that get cached by the OS,
//! but the resident set of *our* process is bounded by the
//! page cache, not by `n * M`.

use std::path::PathBuf;

use memmap2::Mmap;

use crate::l2_sq;
use crate::pq::ProductQuantizer;

/// Something that can score a candidate node against the last
/// query passed to `prepare_query`.
pub trait DistanceBackend {
    fn prepare_query(&mut self, q: &[f32]);
    fn dist(&self, id: u32) -> f32;
    fn ram_bytes(&self) -> usize;
    fn label(&self) -> &'static str;
}

/// Raw f32 vectors in RAM — recall-perfect baseline.
pub struct FlatF32Ram {
    pub vectors: Vec<f32>,
    pub d: usize,
    query: Vec<f32>,
}

impl FlatF32Ram {
    pub fn new(vectors: Vec<f32>, d: usize) -> Self {
        let query = vec![0.0f32; d];
        Self { vectors, d, query }
    }
}

impl DistanceBackend for FlatF32Ram {
    fn prepare_query(&mut self, q: &[f32]) {
        self.query.copy_from_slice(q);
    }
    fn dist(&self, id: u32) -> f32 {
        let off = id as usize * self.d;
        l2_sq(&self.query, &self.vectors[off..off + self.d])
    }
    fn ram_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
    }
    fn label(&self) -> &'static str { "flat-f32-ram" }
}

/// PQ codes in RAM, ADC scoring.
pub struct PqRam {
    pub pq: ProductQuantizer,
    pub codes: Vec<u8>,
    lut: Vec<f32>,
}

impl PqRam {
    pub fn new(pq: ProductQuantizer, codes: Vec<u8>) -> Self {
        let lut = vec![0.0f32; pq.m * ProductQuantizer::K];
        Self { pq, codes, lut }
    }
}

impl DistanceBackend for PqRam {
    fn prepare_query(&mut self, q: &[f32]) {
        self.lut = self.pq.build_lut(q);
    }
    fn dist(&self, id: u32) -> f32 {
        let off = id as usize * self.pq.m;
        self.pq.adc(&self.codes[off..off + self.pq.m], &self.lut)
    }
    fn ram_bytes(&self) -> usize {
        self.codes.len() * std::mem::size_of::<u8>()
            + self.pq.codebooks.len() * std::mem::size_of::<f32>()
    }
    fn label(&self) -> &'static str { "pq-ram" }
}

/// **AISAQ**: PQ codes memory-mapped from disk. The graph is in
/// RAM but the code array is not. Reads hit the SSD (or the OS
/// page cache) on first touch and stay warm for hot pages.
pub struct PqDisk {
    pub pq: ProductQuantizer,
    pub mmap: Mmap,
    pub path: PathBuf,
    pub file_bytes: usize,
    lut: Vec<f32>,
}

impl PqDisk {
    /// Write `codes` to `path` and mmap it back read-only. Callers
    /// should invoke [`Self::advise_random`] before search-heavy loops.
    pub fn create(pq: ProductQuantizer, codes: &[u8], path: PathBuf) -> std::io::Result<Self> {
        use std::io::Write;
        let mut f = std::fs::File::create(&path)?;
        f.write_all(codes)?;
        f.sync_all()?;
        drop(f);
        let f = std::fs::File::open(&path)?;
        let mmap = unsafe { Mmap::map(&f)? };
        let file_bytes = codes.len();
        let lut = vec![0.0f32; pq.m * ProductQuantizer::K];
        Ok(Self { pq, mmap, path, file_bytes, lut })
    }

    /// Hint to the kernel that access is random — reduces read-ahead
    /// wasted work on small PQ code fetches.
    #[cfg(unix)]
    pub fn advise_random(&self) -> std::io::Result<()> {
        self.mmap.advise(memmap2::Advice::Random)
    }
    #[cfg(not(unix))]
    pub fn advise_random(&self) -> std::io::Result<()> { Ok(()) }
}

impl DistanceBackend for PqDisk {
    fn prepare_query(&mut self, q: &[f32]) {
        self.lut = self.pq.build_lut(q);
    }
    fn dist(&self, id: u32) -> f32 {
        let off = id as usize * self.pq.m;
        // SAFETY: mmap byte range is valid for `file_bytes`; id is bounded by caller.
        let code = &self.mmap[off..off + self.pq.m];
        self.pq.adc(code, &self.lut)
    }
    fn ram_bytes(&self) -> usize {
        // Only the LUT + codebooks are truly resident in *our* heap.
        self.lut.len() * std::mem::size_of::<f32>()
            + self.pq.codebooks.len() * std::mem::size_of::<f32>()
    }
    fn label(&self) -> &'static str { "pq-disk-aisaq" }
}
