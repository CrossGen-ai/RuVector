//! Differential check shared by `tests/tail_scan_differential.rs` and the
//! `fuzz/` target (included with `#[path]`, so there is one definition).
//!
//! The runtime must not carry its own tail scanner: its manifest location has
//! to be derivable from `rvf-wire`'s scanner on the same bytes.
//!
//! Invariants, for any input `data`:
//!  I1  A runtime hit found in aligned mode is the FIRST aligned wire
//!      candidate (extent inside the file) whose payload parses as a runtime
//!      manifest. Hence it is never an offset the wire scanner would not visit.
//!  I2  If the newest aligned wire candidate parses, the runtime returns
//!      exactly `rvf_wire::find_latest_manifest(data)`.
//!  I3  A byte-granular (legacy) hit is the newest byte-granular candidate
//!      that parses, and is never older than the aligned answer.
//!  I4  Byte-granular mode is used only when the first segment header says
//!      legacy (`checksum_algo = 0`) or when the aligned scan saw no manifest
//!      header at all.

use rvf_runtime::read_path::{locate_manifest_in_bytes, manifest_payload_parses};
use rvf_types::{SEGMENT_HEADER_SIZE, SEGMENT_MAGIC};
use rvf_wire::tail_scan::{
    extent_fits, find_latest_manifest, manifest_candidates, split_root_page, ScanMode,
};

const MAX_READ_PAYLOAD: u64 = 256 * 1024 * 1024;

fn parseable(data: &[u8], off: usize, plen: u64) -> bool {
    let start = off + SEGMENT_HEADER_SIZE;
    plen <= MAX_READ_PAYLOAD
        && start
            .checked_add(plen as usize)
            .is_some_and(|end| end <= data.len() && manifest_payload_parses(&data[start..end]))
}

/// Candidates in `mode`, newest first, extent inside the file.
fn candidates(data: &[u8], mode: ScanMode) -> Vec<(usize, u64)> {
    let (body, _) = split_root_page(data);
    manifest_candidates(body, mode)
        .filter(|(o, h)| extent_fits(*o, h, data.len()))
        .map(|(o, h)| (o, h.payload_length))
        .collect()
}

fn first_header_is_legacy(data: &[u8]) -> bool {
    data.len() >= SEGMENT_HEADER_SIZE
        && data[..4] == SEGMENT_MAGIC.to_le_bytes()
        && data[4] == 1
        && data[0x20] == 0
}

pub fn check(data: &[u8]) -> Result<Option<(u64, bool)>, String> {
    let got = locate_manifest_in_bytes(data);
    let aligned = candidates(data, ScanMode::Aligned);
    let aligned_ok: Vec<_> = aligned
        .iter()
        .copied()
        .filter(|(o, p)| parseable(data, *o, *p))
        .collect();
    let byte_ok: Vec<_> = candidates(data, ScanMode::ByteGranular)
        .into_iter()
        .filter(|(o, p)| parseable(data, *o, *p))
        .collect();

    match got {
        None => {
            // Nothing found: no parseable candidate may exist in the mode used.
            if first_header_is_legacy(data) {
                if let Some(b) = byte_ok.first() {
                    return Err(format!("legacy mode missed parseable candidate at {}", b.0));
                }
            } else if let Some(a) = aligned_ok.first() {
                return Err(format!("aligned mode missed parseable candidate at {}", a.0));
            }
            Ok(None)
        }
        Some((off, legacy)) => {
            let off = off as usize;
            if !legacy {
                // I1
                match aligned_ok.first() {
                    Some((o, _)) if *o == off => {}
                    other => return Err(format!("I1: runtime {off} vs first parseable aligned {other:?}")),
                }
                // I2
                if let Some((o, p)) = aligned.first() {
                    if parseable(data, *o, *p) {
                        let w = find_latest_manifest(data).map_err(|e| format!("{e:?}"))?;
                        if w.0 != off {
                            return Err(format!("I2: wire {} vs runtime {off}", w.0));
                        }
                    }
                }
            } else {
                // I3
                match byte_ok.first() {
                    Some((o, _)) if *o == off => {}
                    other => return Err(format!("I3: runtime {off} vs first parseable byte {other:?}")),
                }
                if let Some((a, _)) = aligned_ok.first() {
                    if *a > off {
                        return Err(format!("I3: aligned {a} newer than legacy {off}"));
                    }
                }
                // I4
                if !first_header_is_legacy(data) && !aligned.is_empty() {
                    return Err("I4: byte-granular used though aligned candidates exist".into());
                }
            }
            Ok(Some((off as u64, legacy)))
        }
    }
}
