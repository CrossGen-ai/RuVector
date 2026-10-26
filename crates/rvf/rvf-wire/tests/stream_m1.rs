//! Segment-granular (M1) and chunked (M2) behaviour of the RVSD verifier.
mod common;
use common::*;
use rvf_types::stream::*;
use rvf_types::ErrorCode as E;
use rvf_wire::stream::*;

fn bundle(body: &[u8], log2: u8) -> StreamBundle {
    StreamDescriptorBuilder::new(FILE_ID, key(1))
        .chunk_log2(log2)
        .build(body)
        .unwrap()
}

fn both_profiles() -> [u8; 2] {
    [STREAM_CHUNK_LOG2_SEGMENT, 12]
}

#[test]
fn accepts_in_file_order_and_releases_every_byte() {
    for log2 in both_profiles() {
        let body = sample_body(1);
        let b = bundle(&body, log2);
        let keys = [pk(1)];
        let r = run_frames(b.frames_file_order(&body), policy(&keys));
        assert!(r.errors.is_empty(), "{:?}", r.errors);
        assert_eq!(r.state, StreamState::Complete);
        assert_eq!(r.complete, Some(IdentityStatus::Verified));
        assert_subset(&r, &body);
        let total: usize = r.released.iter().map(|(_, c)| c.len()).sum();
        assert_eq!(total, body.len());
        assert_eq!(
            b.entries.last().unwrap().seg_type,
            STREAM_SEG_TYPE_ROOT_PAGE
        );
    }
}

#[test]
fn stage_order_with_requires_mask_completes_unverifiable_identity() {
    let body = sample_body(1);
    // file order of segments: meta0 manifest0 wasm2 index1 vec3 quant1 root2
    let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .chunk_log2(12)
        .stages(&[0, 0, 2, 1, 3, 1, 2])
        .requires_lower_stages(true)
        .build(&body)
        .unwrap();
    let keys = [pk(1)];
    let r = run_frames(b.frames_stage_order(&body), policy(&keys));
    assert!(r.errors.is_empty(), "{:?}", r.errors);
    assert_eq!(r.state, StreamState::Complete);
    assert_eq!(r.complete, Some(IdentityStatus::Unverifiable));
    assert_subset(&r, &body);
    // File order breaks the requires gate: first wasm chunk needs stages 0,1.
    let r = run_frames(b.frames_file_order(&body), policy(&keys));
    assert!(r
        .errors
        .iter()
        .any(|(_, e)| *e == E::StreamRequirementsUnmet));
    assert_ne!(r.state, StreamState::Complete);
    assert_subset(&r, &body);
}

#[test]
fn finish_identity_over_persisted_body() {
    let body = sample_body(1);
    let b = StreamDescriptorBuilder::new(FILE_ID, key(1))
        .chunk_log2(12)
        .stages(&[1, 1, 0, 0, 0, 0, 0])
        .build(&body)
        .unwrap();
    let keys = [pk(1)];
    let mut v = StreamVerifier::new(policy(&keys));
    for f in b.frames_stage_order(&body) {
        v.push(&f).unwrap();
    }
    assert_eq!(v.identity_status(), Some(IdentityStatus::Unverifiable));
    let mut bad = body.clone();
    bad[10] ^= 1;
    let mut v2 = StreamVerifier::new(policy(&keys));
    for f in b.frames_stage_order(&body) {
        v2.push(&f).unwrap();
    }
    assert_eq!(
        v2.finish_identity(|o, buf| {
            buf.copy_from_slice(&bad[o as usize..o as usize + buf.len()]);
            true
        }),
        Err(E::StreamIdentityMismatch)
    );
    assert_eq!(
        v.finish_identity(|o, buf| {
            buf.copy_from_slice(&body[o as usize..o as usize + buf.len()]);
            true
        }),
        Ok(())
    );
    assert_eq!(v.identity_status(), Some(IdentityStatus::Verified));
}

fn first_chunk_frame(frames: &[Vec<u8>], seg_id: u64, idx: u32) -> usize {
    frames
        .iter()
        .position(|f| f[0] == 1 && f[1..9] == seg_id.to_le_bytes() && f[9..13] == idx.to_le_bytes())
        .unwrap()
}

#[test]
fn flipped_chunk_is_never_released_and_blocks_its_segment() {
    for log2 in both_profiles() {
        let body = sample_body(1);
        let b = bundle(&body, log2);
        let keys = [pk(1)];
        let frames: Vec<Vec<u8>> = b.frames_file_order(&body).collect();
        let at = first_chunk_frame(&frames, 5, 0);
        let mut bad = frames.clone();
        bad[at][30] ^= 0x10;
        let r = run_frames(bad, policy(&keys));
        assert!(r
            .errors
            .iter()
            .any(|(i, e)| *i == at && *e == E::StreamChunkHashMismatch));
        assert_subset(&r, &body);
        assert!(
            !r.released
                .iter()
                .any(|(o, _)| *o == b.entries[4].body_offset),
            "flipped chunk released"
        );
        assert_ne!(r.state, StreamState::Complete);
        // with a zero fault budget the stream is poisoned at the flip
        let mut pol = policy(&keys);
        pol.fault_budget = 0;
        let mut frames2 = frames.clone();
        frames2[at][30] ^= 0x10;
        let r = run_frames(frames2, pol);
        assert_eq!(r.state, StreamState::Poisoned(E::StreamChunkHashMismatch));
        assert_eq!(
            r.released
                .last()
                .map(|(o, _)| *o < b.entries[4].body_offset),
            Some(true)
        );
    }
}

#[test]
fn retry_within_budget_recovers() {
    let body = sample_body(1);
    let b = bundle(&body, 12);
    let keys = [pk(1)];
    let mut frames: Vec<Vec<u8>> = b.frames_file_order(&body).collect();
    let at = first_chunk_frame(&frames, 5, 1);
    let good = frames[at].clone();
    frames[at][40] ^= 1;
    frames.insert(at + 1, good);
    let r = run_frames(frames, policy(&keys));
    assert_eq!(r.state, StreamState::Complete);
    assert_eq!(r.errors.len(), 1);
}

#[test]
fn size_lies_poison_immediately() {
    let body = sample_body(1);
    let b = bundle(&body, 12);
    let keys = [pk(1)];
    let frames: Vec<Vec<u8>> = b.frames_file_order(&body).collect();
    let at = first_chunk_frame(&frames, 5, 0);
    let mut lie = frames.clone();
    lie[at][13] ^= 1; // declared len differs from payload
    let r = run_frames(lie, policy(&keys));
    assert_eq!(r.state, StreamState::Poisoned(E::StreamFrameSizeLie));
    let mut short = frames.clone();
    let nl = short[at].len() - 1;
    short[at].truncate(nl);
    short[at][13..17].copy_from_slice(&((nl - 17) as u32).to_le_bytes());
    let r = run_frames(short, policy(&keys));
    assert_eq!(r.state, StreamState::Poisoned(E::StreamFrameSizeLie));
}

#[test]
fn duplicate_out_of_order_and_unknown_are_counted_faults() {
    let body = sample_body(1);
    let b = bundle(&body, 12);
    let keys = [pk(1)];
    let frames: Vec<Vec<u8>> = b.frames_file_order(&body).collect();
    let at = first_chunk_frame(&frames, 5, 0);
    let mut dup = frames.clone();
    dup.insert(at + 1, frames[at].clone());
    let r = run_frames(dup, policy(&keys));
    assert!(r.errors.iter().any(|(_, e)| *e == E::StreamUnexpectedFrame));
    assert_eq!(r.state, StreamState::Complete); // one counted fault, then the stream proceeds
    assert_eq!(
        r.released.len(),
        run_frames(frames.clone(), policy(&keys)).released.len()
    );
    let mut ooo = frames.clone();
    ooo.swap(at, at + 1); // chunk 1 before chunk 0
    let r = run_frames(ooo, policy(&keys));
    assert_ne!(r.state, StreamState::Complete);
    assert_subset(&r, &body);
    let mut unk = frames.clone();
    unk.insert(1, vec![0x7F, 1, 2, 3]);
    let r = run_frames(unk, policy(&keys));
    assert_eq!(r.errors[0].1, E::StreamUnexpectedFrame);
    assert_eq!(r.state, StreamState::Complete);
}

#[test]
fn splice_from_another_file_with_same_key_is_rejected() {
    let body = sample_body(1);
    let keys = [pk(1)];
    let a = bundle(&body, 12);
    let mut body_b = body.clone();
    let off = a.entries[4].body_offset as usize + 200;
    body_b[off] ^= 0x55; // different content at the same position
    let b = StreamDescriptorBuilder::new(*b"another-file-id!", key(1))
        .chunk_log2(12)
        .build(&body_b)
        .unwrap();
    let fa: Vec<Vec<u8>> = a.frames_file_order(&body).collect();
    let fb: Vec<Vec<u8>> = b.frames_file_order(&body_b).collect();
    let at = first_chunk_frame(&fa, 5, 0);
    // B's chunk is validly hashed under B's descriptor, signed by the same key.
    let rb = run_frames(fb.clone(), policy(&keys));
    assert_eq!(rb.state, StreamState::Complete);
    let mut spliced = fa.clone();
    spliced[at] = fb[at].clone();
    let r = run_frames(spliced, policy(&keys));
    assert!(r
        .errors
        .iter()
        .any(|(i, e)| *i == at && *e == E::StreamChunkHashMismatch));
    assert_ne!(r.state, StreamState::Complete);
    // also: file B's table for a segment of file A
    let t = fa.iter().position(|f| f[0] == 2).unwrap();
    let mut s2 = fa.clone();
    s2[t] = fb[t].clone();
    let r = run_frames(s2, policy(&keys));
    assert!(r
        .errors
        .iter()
        .any(|(_, e)| *e == E::StreamTableHashMismatch));
}

#[test]
fn wrong_identity_signed_by_trusted_key_is_caught_at_completion() {
    // A signer that lies about rvf_identity: all leaves match, identity does not.
    let body = sample_body(1);
    let keys = [pk(1)];
    let b = bundle(&body, 12);
    let mut d = b.descriptor[..b.descriptor.len() - 64].to_vec();
    d[0x30] ^= 1;
    let sig = rvf_crypto::sign_descriptor(&d, &key(1));
    d.extend_from_slice(&sig);
    let mut frames: Vec<Vec<u8>> = b.frames_file_order(&body).collect();
    frames[0] = descriptor_frame(&d);
    let r = run_frames(frames, policy(&keys));
    assert_eq!(r.state, StreamState::Poisoned(E::StreamIdentityMismatch));
    assert!(r
        .errors
        .iter()
        .any(|(_, e)| *e == E::StreamIdentityMismatch));
    assert_eq!(
        r.released.iter().map(|(_, c)| c.len()).sum::<usize>(),
        body.len() - 4096 + 4096 - last_chunk_len(&b)
    );
}

fn last_chunk_len(b: &StreamBundle) -> usize {
    let e = b.entries.last().unwrap();
    b.header
        .chunk_len(
            e.stored_len,
            b.header.chunk_count(e.stored_len).unwrap() - 1,
        )
        .unwrap() as usize
}

#[test]
fn weak_checksum_on_executable_is_poisoned() {
    use rvf_types::SegmentFlags;
    let f = SegmentFlags::empty();
    let mut body = rvf_wire::writer::write_segment_with_algo(0x10, &payload(9, 5000), f, 1, 1); // XXH3 on Wasm
    body.extend(root_page());
    let keys = [pk(1)];
    let b = bundle(&body, 12);
    let r = run_frames(b.frames_file_order(&body), policy(&keys));
    assert_eq!(r.state, StreamState::Poisoned(E::StreamWeakChecksum));
    assert!(r.released.is_empty());
}

#[test]
fn suspend_and_resume_never_complete_early() {
    let body = sample_body(1);
    let b = bundle(&body, 12);
    let keys = [pk(1)];
    let mut v = StreamVerifier::new(policy(&keys));
    let frames: Vec<Vec<u8>> = b.frames_file_order(&body).collect();
    for f in &frames[..frames.len() / 2] {
        v.push(f).unwrap();
    }
    assert_eq!(v.end_of_input(), StreamState::Suspended);
    assert_eq!(v.push(&frames[frames.len() / 2]), Err(E::StreamBadState));
    assert!(v.resume());
    for f in &frames[frames.len() / 2..] {
        v.push(f).unwrap();
    }
    assert_eq!(v.state(), StreamState::Complete);
    assert_eq!(v.push(&frames[1]), Err(E::StreamBadState));
}

#[test]
fn sidecar_descriptor_constructor() {
    let body = sample_body(1);
    let b = bundle(&body, 12);
    let keys = [pk(1)];
    let mut v = StreamVerifier::from_descriptor(&b.descriptor, policy(&keys)).unwrap();
    for f in b.frames_file_order(&body).skip(1) {
        v.push(&f).unwrap();
    }
    assert_eq!(v.state(), StreamState::Complete);
    assert!(StreamVerifier::from_descriptor(&b.descriptor[1..], policy(&keys)).is_err());
}
