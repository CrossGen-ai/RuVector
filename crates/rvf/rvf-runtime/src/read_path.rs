//! Progressive read logic for the RVF runtime.
//!
//! Boot sequence:
//! 1. Seek to EOF - 4096, parse Level 0 root manifest
//! 2. Extract hotset pointers, mmap hot segments
//! 3. Background: parse Level 1 -> full segment directory
//! 4. On-demand: load cold segments as queries need them

use crate::options::DistanceMetric;
use crate::write_path::{SEG_HASH_ENTRY_LEN, SEG_HASH_TRAILER_MARKER};
use rvf_types::{
    FileIdentity, SegmentHeader, SEGMENT_ALIGNMENT, SEGMENT_HEADER_SIZE, SEGMENT_MAGIC,
};
use rvf_wire::reader::read_segment_header;
use rvf_wire::tail_scan::{extent_fits, manifest_candidates, split_root_page, ScanMode};
use std::io::{self, Read, Seek, SeekFrom};

/// In-memory vector storage. The contiguous-slab implementation lives in
/// [`crate::vector_slab`]; re-exported here so existing consumers
/// (`store`, `index_path`, `rabitq_path`) keep their import path.
pub(crate) use crate::vector_slab::VectorData;

/// A parsed segment directory entry.
#[derive(Clone, Debug)]
pub(crate) struct SegDirEntry {
    pub seg_id: u64,
    pub offset: u64,
    pub payload_length: u64,
    pub seg_type: u8,
}

/// Parsed manifest data from the file.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(crate) struct ParsedManifest {
    pub epoch: u32,
    pub dimension: u16,
    pub total_vectors: u64,
    pub profile_id: u8,
    /// Distance metric decoded from byte [19] of the manifest header.
    ///
    /// Stores written before this field existed have 0x00 there (reserved),
    /// which decodes as `DistanceMetric::L2` — the backward-compatible default.
    pub metric: DistanceMetric,
    pub segment_dir: Vec<SegDirEntry>,
    pub deleted_ids: Vec<u64>,
    pub file_identity: Option<FileIdentity>,
    /// Per-segment committed hashes (signed stores only): `(segment_id, hash)`.
    pub seg_hashes: Option<Vec<(u64, [u8; 16])>>,
}

/// A manifest located by the tail scan, with the evidence a policy check needs.
#[derive(Clone, Debug)]
pub(crate) struct LocatedManifest {
    pub manifest: ParsedManifest,
    /// File offset of the MANIFEST_SEG header.
    pub offset: u64,
    pub header: SegmentHeader,
    /// Raw manifest payload (hashed by the Strict policy check).
    pub payload: Vec<u8>,
    /// Scan mode that found it. `ByteGranular` means legacy dialect.
    pub mode: ScanMode,
}

/// Choose the scan mode for a file from its first segment header.
///
/// Every legacy-dialect file starts with a segment stamped `checksum_algo = 0`
/// (the runtime wrote its CRC-rotation hash under that label), and legacy
/// segments are unpadded, so only the byte-granular mode can find their later
/// manifests. Anything else is wire dialect: 64-byte aligned.
fn mode_for_file<R: Read + Seek>(reader: &mut R, file_size: u64) -> io::Result<ScanMode> {
    if file_size < SEGMENT_HEADER_SIZE as u64 {
        return Ok(ScanMode::Aligned);
    }
    reader.seek(SeekFrom::Start(0))?;
    let mut first = [0u8; SEGMENT_HEADER_SIZE];
    reader.read_exact(&mut first)?;
    Ok(match read_segment_header(&first) {
        Ok(h) if h.checksum_algo == 0 => ScanMode::ByteGranular,
        _ => ScanMode::Aligned,
    })
}

/// Scan backwards from EOF to find and parse the latest valid manifest.
///
/// The scan itself is `rvf_wire::tail_scan` (the single tail scanner); this
/// function only supplies windows of the file and the runtime's payload
/// parser. Wire-dialect files are scanned at 64-byte alignment. Legacy-dialect
/// files (first segment `checksum_algo = 0`) are scanned byte-granular, the
/// documented fallback for unpadded segments. If an aligned scan finds no
/// manifest header anywhere (and only then) the byte-granular scan is tried
/// and the result reported as such; a parse failure of an aligned candidate
/// never triggers the fallback.
pub(crate) fn find_latest_manifest<R: Read + Seek>(
    reader: &mut R,
) -> io::Result<Option<LocatedManifest>> {
    let file_size = reader.seek(SeekFrom::End(0))?;
    if file_size < SEGMENT_HEADER_SIZE as u64 {
        return Ok(None);
    }
    let mode = mode_for_file(reader, file_size)?;
    let (found, saw_any) = scan_all_windows(reader, file_size, mode)?;
    if found.is_some() {
        return Ok(found);
    }
    if mode == ScanMode::Aligned && !saw_any {
        return Ok(scan_all_windows(reader, file_size, ScanMode::ByteGranular)?.0);
    }
    Ok(None)
}

/// Widen the backward scan window until a manifest parses or the file start
/// is reached. Returns the manifest (if any) and whether any header candidate
/// was seen at all.
fn scan_all_windows<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    mode: ScanMode,
) -> io::Result<(Option<LocatedManifest>, bool)> {
    // The manifest grows with segment count, so it can extend arbitrarily far
    // back from EOF: 64 KB -> 1 MB -> 16 MB -> whole file.
    const SCAN_WINDOWS: [u64; 4] = [64 << 10, 1 << 20, 16 << 20, u64::MAX];
    let mut prev_scan_size = 0u64;
    let mut saw_any = false;
    for window in SCAN_WINDOWS {
        let scan_size = window.min(file_size);
        if scan_size == prev_scan_size {
            break; // Already scanned the whole file.
        }
        let (found, saw) = scan_tail_for_manifest(reader, file_size, scan_size, mode)?;
        saw_any |= saw;
        if found.is_some() {
            return Ok((found, saw_any));
        }
        prev_scan_size = scan_size;
    }
    Ok((None, saw_any))
}

/// Scan the final `scan_size` bytes of the file for the latest valid manifest.
fn scan_tail_for_manifest<R: Read + Seek>(
    reader: &mut R,
    file_size: u64,
    scan_size: u64,
    mode: ScanMode,
) -> io::Result<(Option<LocatedManifest>, bool)> {
    // Round the window start DOWN to a segment boundary so alignment within
    // the buffer is alignment within the file.
    let scan_start = (file_size - scan_size) & !(SEGMENT_ALIGNMENT as u64 - 1);
    reader.seek(SeekFrom::Start(scan_start))?;
    let mut buf = vec![0u8; (file_size - scan_start) as usize];
    reader.read_exact(&mut buf)?;

    // A trailing Level-0 root page is not a segment: never scan inside it.
    let (body, _page) = split_root_page(&buf);
    let mut saw = false;
    for (i, header) in manifest_candidates(body, mode) {
        saw = true;
        // Reject implausible payload lengths to prevent OOM.
        if header.payload_length > MAX_READ_PAYLOAD {
            continue;
        }
        let payload_length = header.payload_length as usize;
        let payload_start = i + SEGMENT_HEADER_SIZE;
        let Some(payload_end) = payload_start.checked_add(payload_length) else {
            continue;
        };
        let file_offset = scan_start + i as u64;
        let payload: Vec<u8> = if payload_end <= buf.len() {
            buf[payload_start..payload_end].to_vec()
        } else {
            // Payload extends beyond our buffer -- but must exist in the file.
            if !extent_fits(
                usize::try_from(file_offset).unwrap_or(usize::MAX),
                &header,
                usize::try_from(file_size).unwrap_or(usize::MAX),
            ) {
                continue;
            }
            reader.seek(SeekFrom::Start(file_offset + SEGMENT_HEADER_SIZE as u64))?;
            let mut p = vec![0u8; payload_length];
            if reader.read_exact(&mut p).is_err() {
                continue;
            }
            p
        };
        if let Some(manifest) = parse_manifest_payload(&payload) {
            return Ok((
                Some(LocatedManifest {
                    manifest,
                    offset: file_offset,
                    header,
                    payload,
                    mode,
                }),
                saw,
            ));
        }
    }
    Ok((None, saw))
}

/// Locate the latest manifest in a complete in-memory file with the same
/// code the store uses at open. Returns `(offset, legacy)` where `legacy`
/// means the byte-granular mode found it.
///
/// Public only so the differential test and the fuzz target can compare the
/// runtime against `rvf-wire`'s scanner on identical bytes; not a stable API.
#[doc(hidden)]
pub fn locate_manifest_in_bytes(data: &[u8]) -> Option<(u64, bool)> {
    let mut cursor = io::Cursor::new(data);
    find_latest_manifest(&mut cursor)
        .ok()
        .flatten()
        .map(|m| (m.offset, m.mode == ScanMode::ByteGranular))
}

/// Whether `payload` parses as a runtime manifest. Test/fuzz support only.
#[doc(hidden)]
pub fn manifest_payload_parses(payload: &[u8]) -> bool {
    parse_manifest_payload(payload).is_some()
}

/// Parse a manifest payload into structured data.
fn parse_manifest_payload(payload: &[u8]) -> Option<ParsedManifest> {
    // Minimum header: epoch(4) + dim(2) + total_vectors(8) + seg_count(4) + profile(1) + pad(3) = 22
    if payload.len() < 22 {
        return None;
    }

    let epoch = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let dimension = u16::from_le_bytes([payload[4], payload[5]]);
    let total_vectors = u64::from_le_bytes([
        payload[6],
        payload[7],
        payload[8],
        payload[9],
        payload[10],
        payload[11],
        payload[12],
        payload[13],
    ]);
    let seg_count = u32::from_le_bytes([payload[14], payload[15], payload[16], payload[17]]);
    let profile_id = payload[18];
    // Byte [19] encodes the distance metric (was reserved zero in older stores).
    // DistanceMetric::from_id(0) == L2, so old stores boot correctly.
    let metric = DistanceMetric::from_id(payload[19]);

    let mut offset = 22; // past header (4+2+8+4+1+3)

    // Validate that seg_count does not exceed what the payload can actually hold.
    // Each directory entry is 25 bytes, so seg_count * 25 + 22 must fit in the payload.
    let max_possible_entries = payload.len().saturating_sub(22) / 25;
    if (seg_count as usize) > max_possible_entries {
        return None;
    }

    // Parse segment directory.
    let mut segment_dir = Vec::with_capacity(seg_count as usize);
    for _ in 0..seg_count {
        if offset + 25 > payload.len() {
            return None;
        }
        let seg_id = u64::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
            payload[offset + 4],
            payload[offset + 5],
            payload[offset + 6],
            payload[offset + 7],
        ]);
        let seg_offset = u64::from_le_bytes([
            payload[offset + 8],
            payload[offset + 9],
            payload[offset + 10],
            payload[offset + 11],
            payload[offset + 12],
            payload[offset + 13],
            payload[offset + 14],
            payload[offset + 15],
        ]);
        let plen = u64::from_le_bytes([
            payload[offset + 16],
            payload[offset + 17],
            payload[offset + 18],
            payload[offset + 19],
            payload[offset + 20],
            payload[offset + 21],
            payload[offset + 22],
            payload[offset + 23],
        ]);
        let stype = payload[offset + 24];
        segment_dir.push(SegDirEntry {
            seg_id,
            offset: seg_offset,
            payload_length: plen,
            seg_type: stype,
        });
        offset += 25;
    }

    // Parse deletion bitmap.
    let mut deleted_ids = Vec::new();
    if offset + 4 <= payload.len() {
        let del_count = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
        offset += 4;
        for _ in 0..del_count {
            if offset + 8 > payload.len() {
                break;
            }
            let did = u64::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
                payload[offset + 4],
                payload[offset + 5],
                payload[offset + 6],
                payload[offset + 7],
            ]);
            deleted_ids.push(did);
            offset += 8;
        }
    }

    // Try to parse FileIdentity trailer (backward-compatible).
    // Look for magic marker 0x46494449 ("FIDI") followed by 68 bytes.
    let file_identity = if offset + 4 + 68 <= payload.len() {
        let marker = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
        if marker == 0x4649_4449 {
            offset += 4;
            let fi_data: &[u8; 68] = payload[offset..offset + 68].try_into().ok()?;
            offset += 68;
            Some(FileIdentity::from_bytes(fi_data))
        } else {
            None
        }
    } else {
        None
    };

    // Optional per-segment hash trailer (signed stores). A present-but-
    // malformed trailer rejects the candidate rather than being ignored: an
    // ignored trailer would silently downgrade a signed store to unchecked.
    let mut seg_hashes = None;
    if offset + 8 <= payload.len() {
        let marker = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
        if marker == SEG_HASH_TRAILER_MARKER {
            let count = u32::from_le_bytes([
                payload[offset + 4],
                payload[offset + 5],
                payload[offset + 6],
                payload[offset + 7],
            ]) as usize;
            let body = &payload[offset + 8..];
            if count.checked_mul(SEG_HASH_ENTRY_LEN)? != body.len() {
                return None;
            }
            let mut entries = Vec::with_capacity(count);
            for chunk in body.chunks_exact(SEG_HASH_ENTRY_LEN) {
                let sid = u64::from_le_bytes(chunk[..8].try_into().ok()?);
                let mut h = [0u8; 16];
                h.copy_from_slice(&chunk[8..]);
                entries.push((sid, h));
            }
            seg_hashes = Some(entries);
        }
    }

    Some(ParsedManifest {
        epoch,
        dimension,
        total_vectors,
        profile_id,
        metric,
        segment_dir,
        deleted_ids,
        file_identity,
        seg_hashes,
    })
}

/// Read a VEC_SEG payload and return (id, vector) pairs.
pub(crate) fn read_vec_seg_payload(payload: &[u8]) -> Option<Vec<(u64, Vec<f32>)>> {
    if payload.len() < 6 {
        return None;
    }

    let dimension = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    let vector_count =
        u32::from_le_bytes([payload[2], payload[3], payload[4], payload[5]]) as usize;

    // Checked u64 arithmetic: on 32-bit targets (wasm32) a crafted
    // `vector_count` can wrap `vector_count * (8 + bytes_per_vec)` in
    // usize, pass this length check, and panic on out-of-bounds reads in
    // the loop below.
    let bytes_per_vec = (dimension as u64) * 4;
    let expected_size = (vector_count as u64)
        .checked_mul(8 + bytes_per_vec)
        .and_then(|v| v.checked_add(6))?;
    if (payload.len() as u64) < expected_size {
        return None;
    }

    let mut result = Vec::with_capacity(vector_count);
    let mut offset = 6;

    for _ in 0..vector_count {
        let vec_id = u64::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
            payload[offset + 4],
            payload[offset + 5],
            payload[offset + 6],
            payload[offset + 7],
        ]);
        offset += 8;

        let mut vec_data = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            let val = f32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]);
            vec_data.push(val);
            offset += 4;
        }

        result.push((vec_id, vec_data));
    }

    Some(result)
}

/// Maximum allowed payload size when reading segments (256 MiB).
/// This prevents a malicious payload_length field from causing OOM.
const MAX_READ_PAYLOAD: u64 = 256 * 1024 * 1024;

/// Read a segment's payload from the file given its offset.
///
/// Validates magic, enforces a maximum payload size, and verifies the
/// content hash before returning the data.
pub(crate) fn read_segment_payload<R: Read + Seek>(
    reader: &mut R,
    seg_offset: u64,
) -> io::Result<(SegmentHeader, Vec<u8>)> {
    reader.seek(SeekFrom::Start(seg_offset))?;

    let mut hdr_buf = [0u8; SEGMENT_HEADER_SIZE];
    reader.read_exact(&mut hdr_buf)?;

    let magic = u32::from_le_bytes([hdr_buf[0], hdr_buf[1], hdr_buf[2], hdr_buf[3]]);
    if magic != SEGMENT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid segment magic",
        ));
    }

    let payload_length = u64::from_le_bytes([
        hdr_buf[0x10],
        hdr_buf[0x11],
        hdr_buf[0x12],
        hdr_buf[0x13],
        hdr_buf[0x14],
        hdr_buf[0x15],
        hdr_buf[0x16],
        hdr_buf[0x17],
    ]);

    // Enforce maximum payload size to prevent OOM from crafted files.
    if payload_length > MAX_READ_PAYLOAD {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "segment payload too large: {} bytes (max {})",
                payload_length, MAX_READ_PAYLOAD
            ),
        ));
    }

    let header = SegmentHeader {
        magic,
        version: hdr_buf[0x04],
        seg_type: hdr_buf[0x05],
        flags: u16::from_le_bytes([hdr_buf[0x06], hdr_buf[0x07]]),
        segment_id: u64::from_le_bytes([
            hdr_buf[0x08],
            hdr_buf[0x09],
            hdr_buf[0x0A],
            hdr_buf[0x0B],
            hdr_buf[0x0C],
            hdr_buf[0x0D],
            hdr_buf[0x0E],
            hdr_buf[0x0F],
        ]),
        payload_length,
        timestamp_ns: u64::from_le_bytes([
            hdr_buf[0x18],
            hdr_buf[0x19],
            hdr_buf[0x1A],
            hdr_buf[0x1B],
            hdr_buf[0x1C],
            hdr_buf[0x1D],
            hdr_buf[0x1E],
            hdr_buf[0x1F],
        ]),
        checksum_algo: hdr_buf[0x20],
        compression: hdr_buf[0x21],
        reserved_0: u16::from_le_bytes([hdr_buf[0x22], hdr_buf[0x23]]),
        reserved_1: u32::from_le_bytes([
            hdr_buf[0x24],
            hdr_buf[0x25],
            hdr_buf[0x26],
            hdr_buf[0x27],
        ]),
        content_hash: {
            let mut h = [0u8; 16];
            h.copy_from_slice(&hdr_buf[0x28..0x38]);
            h
        },
        uncompressed_len: u32::from_le_bytes([
            hdr_buf[0x38],
            hdr_buf[0x39],
            hdr_buf[0x3A],
            hdr_buf[0x3B],
        ]),
        alignment_pad: u32::from_le_bytes([
            hdr_buf[0x3C],
            hdr_buf[0x3D],
            hdr_buf[0x3E],
            hdr_buf[0x3F],
        ]),
    };

    // payload_length is guaranteed <= MAX_READ_PAYLOAD (256 MiB) which fits in usize.
    let mut payload = vec![0u8; payload_length as usize];
    reader.read_exact(&mut payload)?;

    // Verify the content hash if it is non-zero (zero hash means "not set").
    // Dual-accept: the algorithm named by the header selects the dialect
    // (see `crate::hashing`); an unknown algorithm is refused.
    if header.content_hash != [0u8; 16]
        && !crate::hashing::verify_payload_hash(
            header.checksum_algo,
            &payload,
            &header.content_hash,
        )
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "segment content hash mismatch",
        ));
    }

    Ok((header, payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_manifest() {
        assert!(parse_manifest_payload(&[]).is_none());
        assert!(parse_manifest_payload(&[0u8; 10]).is_none());
    }

    #[test]
    fn vec_seg_round_trip() {
        // Build a VEC_SEG payload: dim=2, count=2, vectors.
        let dim: u16 = 2;
        let count: u32 = 2;
        let mut payload = Vec::new();
        payload.extend_from_slice(&dim.to_le_bytes());
        payload.extend_from_slice(&count.to_le_bytes());
        // Vector 0: id=10, [1.0, 2.0]
        payload.extend_from_slice(&10u64.to_le_bytes());
        payload.extend_from_slice(&1.0f32.to_le_bytes());
        payload.extend_from_slice(&2.0f32.to_le_bytes());
        // Vector 1: id=20, [3.0, 4.0]
        payload.extend_from_slice(&20u64.to_le_bytes());
        payload.extend_from_slice(&3.0f32.to_le_bytes());
        payload.extend_from_slice(&4.0f32.to_le_bytes());

        let result = read_vec_seg_payload(&payload).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 10);
        assert_eq!(result[0].1, vec![1.0, 2.0]);
        assert_eq!(result[1].0, 20);
        assert_eq!(result[1].1, vec![3.0, 4.0]);
    }

    #[test]
    fn vec_seg_rejects_oversized_counts_without_panicking() {
        // Maximal dimension and vector_count: the size product
        // (~1.1e15 bytes) wraps a 32-bit usize. The checked u64 size
        // computation must reject the payload (None), never pass the
        // length check and read out of bounds.
        let mut payload = Vec::new();
        payload.extend_from_slice(&u16::MAX.to_le_bytes()); // dimension
        payload.extend_from_slice(&u32::MAX.to_le_bytes()); // vector_count
        payload.extend_from_slice(&[0u8; 32]); // a little body, far too short
        assert!(read_vec_seg_payload(&payload).is_none());

        // A count that wraps usize-32 to a tiny value (dim 0 -> 8 bytes
        // per record; count * 8 wraps at 2^29 records on 32-bit).
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&(1u32 << 29).to_le_bytes());
        payload.extend_from_slice(&[0u8; 64]);
        assert!(read_vec_seg_payload(&payload).is_none());
    }
}
