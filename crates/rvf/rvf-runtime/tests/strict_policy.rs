//! ADR-354 item 5 / ADR-033 sections 3.5 and 4: `Strict` is real.
//!
//! Runs against the real `RvfStore::create_signed` / `open_with_policy`, not a
//! model of them.

use rvf_runtime::options::WitnessConfig;
use rvf_runtime::root_page::{self, PageContent};
use rvf_runtime::{QueryOptions, RvfOptions, RvfStore};
use rvf_types::ed25519::Ed25519Keypair;
use rvf_types::security::{SecurityError, SecurityPolicy};
use rvf_types::{FileIdentity, RvfError, SegmentType, SEGMENT_HEADER_SIZE};
use rvf_wire::hash::compute_shake256_128;
use rvf_wire::tail_scan::split_root_page;
use rvf_wire::writer::calculate_padded_size;
use std::path::{Path, PathBuf};

const SECRET: [u8; 32] = [0x11; 32];
const OTHER_SECRET: [u8; 32] = [0x22; 32];

fn public(secret: &[u8; 32]) -> [u8; 32] {
    Ed25519Keypair::from_secret(secret).public_key()
}

fn opts(trusted: &[[u8; 32]]) -> RvfOptions {
    RvfOptions {
        dimension: 4,
        trusted_signers: trusted.to_vec(),
        ..Default::default()
    }
}

fn rows(n: usize, seed: f32) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| vec![seed + i as f32, 1.0, 2.0, 3.0])
        .collect()
}

fn signed_store(dir: &Path) -> PathBuf {
    let path = dir.join("signed.rvf");
    let mut s = RvfStore::create_signed(&path, opts(&[]), &SECRET).unwrap();
    for b in 0..2u64 {
        let v = rows(6, b as f32 * 100.0);
        let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
        let ids: Vec<u64> = (0..6).map(|i| b * 10 + i).collect();
        s.ingest_batch(&refs, &ids, None).unwrap();
    }
    s.close().unwrap();
    path
}

/// (offset, type, payload_len) of every segment, wire-dialect walk.
fn segments(bytes: &[u8]) -> Vec<(usize, u8, usize)> {
    let (body, _) = split_root_page(bytes);
    let mut off = 0;
    let mut out = Vec::new();
    while off < body.len() {
        let h = rvf_wire::read_segment_header(&body[off..]).unwrap();
        out.push((off, h.seg_type, h.payload_length as usize));
        off += calculate_padded_size(SEGMENT_HEADER_SIZE, h.payload_length as usize);
    }
    out
}

fn page_of(bytes: &[u8]) -> [u8; 4096] {
    bytes[bytes.len() - 4096..].try_into().unwrap()
}

fn set_page(bytes: &mut [u8], page: &[u8; 4096]) {
    let n = bytes.len();
    bytes[n - 4096..].copy_from_slice(page);
}

fn security(e: RvfError) -> SecurityError {
    match e {
        RvfError::Security(s) => s,
        other => panic!("expected a Security error, got {other:?}"),
    }
}

fn open_err(path: &Path, trusted: &[[u8; 32]], policy: SecurityPolicy) -> RvfError {
    match RvfStore::open_with_policy(path, opts(trusted), policy) {
        Err(e) => e,
        Ok(_) => panic!("{policy:?} opened a file it must refuse"),
    }
}

#[test]
fn signed_store_opens_under_strict_and_paranoid_and_queries() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let bytes = std::fs::read(&p).unwrap();
    assert_eq!(bytes.len() % 64, 0);
    assert!(split_root_page(&bytes).1.is_some(), "ends in an RVM0 page");
    for policy in [
        SecurityPolicy::Strict,
        SecurityPolicy::Paranoid,
        SecurityPolicy::WarnOnly,
    ] {
        let s = RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), policy).unwrap();
        assert_eq!(s.security_policy(), policy);
        assert!(!s.legacy_dialect());
        assert_eq!(s.status().total_vectors, 12);
        let r = s
            .query(&[0.0, 1.0, 2.0, 3.0], 3, &QueryOptions::default())
            .unwrap();
        assert_eq!(r[0].id, 0);
    }
    // open_with_options reads RvfOptions.security_policy (default Strict).
    assert!(RvfStore::open_with_options(&p, opts(&[public(&SECRET)])).is_ok());
    assert!(RvfStore::open_with_options(&p, opts(&[])).is_err());
}

#[test]
fn unsigned_manifest_is_refused_under_strict_and_accepted_under_lower_policies() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("plain.rvf");
    let mut s = RvfStore::create(&p, opts(&[])).unwrap();
    s.ingest_batch(&[&[1.0, 2.0, 3.0, 4.0]], &[1], None)
        .unwrap();
    s.close().unwrap();

    for policy in [SecurityPolicy::Strict, SecurityPolicy::Paranoid] {
        let e = security(open_err(&p, &[public(&SECRET)], policy));
        assert!(matches!(e, SecurityError::UnsignedManifest { .. }), "{e:?}");
        assert_eq!(
            rvf_runtime::store::security_error_code(&e) as u16,
            0x0800,
            "stable code"
        );
    }
    for policy in [SecurityPolicy::WarnOnly, SecurityPolicy::Permissive] {
        let s = RvfStore::open_with_policy(&p, opts(&[]), policy).unwrap();
        assert_eq!(s.status().total_vectors, 1);
    }
    // The explicit lower policy that existing callers get implicitly.
    assert!(RvfStore::open(&p).is_ok());
    assert!(RvfStore::open_readonly(&p).is_ok());
}

// ---- ADR-033 section 3.5: Malicious Tail Manifest ---------------------------------

/// Rewrite pointer fields in the page, FIX the CRC32C, do NOT re-sign.
fn tampered_pointer_file(dir: &Path) -> (PathBuf, u64) {
    let p = signed_store(dir);
    let mut bytes = std::fs::read(&p).unwrap();
    let segs = segments(&bytes);
    let manifest_off = segs
        .iter()
        .rev()
        .find(|s| s.1 == SegmentType::Manifest as u8)
        .unwrap()
        .0;
    let other = segs
        .iter()
        .find(|s| s.1 == SegmentType::Vec as u8)
        .unwrap()
        .0;
    let mut page = page_of(&bytes);
    page[0x008..0x010].copy_from_slice(&(other as u64).to_le_bytes()); // redirect L1
    root_page::reseal_crc(&mut page);
    assert!(root_page::view(&page).crc_ok, "attacker kept the CRC valid");
    set_page(&mut bytes, &page);
    std::fs::write(&p, &bytes).unwrap();
    (p, manifest_off as u64)
}

#[test]
fn malicious_tail_manifest_fails_strict_and_paranoid_deterministically() {
    let d = tempfile::tempdir().unwrap();
    let (p, manifest_off) = tampered_pointer_file(d.path());
    for policy in [SecurityPolicy::Strict, SecurityPolicy::Paranoid] {
        let e = security(open_err(&p, &[public(&SECRET)], policy));
        match e {
            SecurityError::InvalidSignature {
                manifest_offset,
                rejection_phase,
            } => {
                assert_eq!(manifest_offset, manifest_off);
                assert_eq!(
                    rejection_phase, "signature_verification",
                    "not content_hash"
                );
            }
            other => panic!("{other:?}"),
        }
        // Stable code, same on every run.
        assert_eq!(
            rvf_runtime::store::security_error_code(&security(open_err(
                &p,
                &[public(&SECRET)],
                policy
            ))) as u16,
            0x0103
        );
    }
}

#[test]
fn malicious_tail_manifest_warnonly_opens_and_serves_committed_data() {
    // The page's L1 pointer is advisory here (the manifest is located by the
    // single tail scanner), so redirecting it cannot redirect reads. WarnOnly
    // warns about the signature and serves data that still matches the
    // manifest's committed hashes.
    let d = tempfile::tempdir().unwrap();
    let (p, _) = tampered_pointer_file(d.path());
    let s =
        RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::WarnOnly).unwrap();
    assert_eq!(s.status().total_vectors, 12);
}

#[test]
fn permissive_ignores_everything_by_design() {
    let d = tempfile::tempdir().unwrap();
    let (p, _) = tampered_pointer_file(d.path());
    assert!(RvfStore::open_with_policy(&p, opts(&[]), SecurityPolicy::Permissive).is_ok());
}

#[test]
fn resigned_with_a_different_key_is_unknown_signer_not_invalid_signature() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let mut bytes = std::fs::read(&p).unwrap();
    let old = root_page::view(&page_of(&bytes));
    // Attacker rebuilds the page byte-for-byte with their own key.
    let id = FileIdentity::zeroed();
    let forged = root_page::build_signed_page(
        &PageContent {
            l1_offset: old.l1_offset,
            l1_length: old.l1_length,
            l1_content_hash: old.l1_content_hash,
            total_vectors: 12,
            dimension: 4,
            profile: 0,
            epoch: 1,
            identity: &id,
        },
        &OTHER_SECRET,
    );
    set_page(&mut bytes, &forged);
    std::fs::write(&p, &bytes).unwrap();

    let e = security(open_err(&p, &[public(&SECRET)], SecurityPolicy::Strict));
    match e {
        SecurityError::UnknownSigner {
            actual_signer,
            expected_signer,
            ..
        } => {
            assert_eq!(
                actual_signer,
                root_page::signer_fingerprint(&public(&OTHER_SECRET))
            );
            assert_eq!(
                expected_signer,
                Some(root_page::signer_fingerprint(&public(&SECRET)))
            );
        }
        other => panic!("{other:?}"),
    }
    // Distinct from "no signature" and from "bad signature".
    assert_eq!(rvf_runtime::store::security_error_code(&e) as u16, 0x0802);
    // An empty trust store cannot vouch for anyone.
    let e = security(open_err(&p, &[], SecurityPolicy::Strict));
    assert!(matches!(
        e,
        SecurityError::UnknownSigner {
            expected_signer: None,
            ..
        }
    ));
}

#[test]
fn forged_fingerprint_claiming_the_trusted_signer_is_invalid_signature() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let mut bytes = std::fs::read(&p).unwrap();
    let mut page = page_of(&bytes);
    // Sign with the attacker's key, then overwrite the fingerprint with the
    // trusted one (before re-signing so the page is internally consistent
    // except for the key).
    page[root_page::OFF_SIGNER_FINGERPRINT..root_page::OFF_SIGNER_FINGERPRINT + 16]
        .copy_from_slice(&root_page::signer_fingerprint(&public(&SECRET)));
    let sig = rvf_types::ed25519::ed25519_sign(&OTHER_SECRET, &root_page::signing_bytes(&page));
    page[0x09C..0x09C + 64].copy_from_slice(&sig);
    root_page::reseal_crc(&mut page);
    set_page(&mut bytes, &page);
    std::fs::write(&p, &bytes).unwrap();
    let e = security(open_err(&p, &[public(&SECRET)], SecurityPolicy::Strict));
    assert!(
        matches!(
            e,
            SecurityError::InvalidSignature {
                rejection_phase: "signature_verification",
                ..
            }
        ),
        "{e:?}"
    );
}

#[test]
fn stale_page_replay_after_a_planted_manifest_is_refused() {
    // Attacker keeps the genuine page (valid signature!) but plants a newer
    // manifest segment in front of it. The tail scan finds the plant; the
    // signed page commits to a different offset and hash.
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let bytes = std::fs::read(&p).unwrap();
    let (body, page) = split_root_page(&bytes);
    let segs = segments(&bytes);
    let (m_off, _, m_len) = *segs
        .iter()
        .rev()
        .find(|s| s.1 == SegmentType::Manifest as u8)
        .unwrap();
    // Copy the real manifest payload with one directory byte flipped.
    let mut payload = body[m_off + 64..m_off + 64 + m_len].to_vec();
    payload[22 + 8] ^= 0x01; // a directory offset
                             // Same algorithm as a genuine signed manifest (SHAKE-256-128), so the
                             // only thing wrong with the plant is that the page does not commit to it.
    let planted = rvf_wire::writer::write_segment_with_algo(
        SegmentType::Manifest as u8,
        &payload,
        rvf_types::SegmentFlags::empty(),
        500,
        2,
    );
    let mut out = body.to_vec();
    let planted_off = out.len();
    out.extend_from_slice(&planted);
    out.extend_from_slice(page.unwrap());
    std::fs::write(&p, &out).unwrap();

    let e = security(open_err(&p, &[public(&SECRET)], SecurityPolicy::Strict));
    match e {
        SecurityError::ContentHashMismatch {
            pointer_name,
            seg_offset,
            ..
        } => {
            assert_eq!(pointer_name, "l1_manifest");
            assert_eq!(seg_offset, planted_off as u64);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn rewritten_manifest_under_an_intact_page_is_refused() {
    // Same-offset, same-length manifest rewrite with a recomputed header
    // hash: the signature still verifies, the committed L1 hash does not.
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let mut bytes = std::fs::read(&p).unwrap();
    let segs = segments(&bytes);
    let (m_off, _, m_len) = *segs
        .iter()
        .rev()
        .find(|s| s.1 == SegmentType::Manifest as u8)
        .unwrap();
    bytes[m_off + 64 + 22 + 8] ^= 0x01; // flip a directory offset byte
    let new_hash = compute_shake256_128(&bytes[m_off + 64..m_off + 64 + m_len]);
    bytes[m_off + 0x28..m_off + 0x38].copy_from_slice(&new_hash);
    std::fs::write(&p, &bytes).unwrap();
    let e = security(open_err(&p, &[public(&SECRET)], SecurityPolicy::Strict));
    assert!(
        matches!(
            e,
            SecurityError::ContentHashMismatch {
                pointer_name: "l1_manifest",
                ..
            }
        ),
        "{e:?}"
    );
}

// ---- segment authentication --------------------------------------------------------

fn swap_vec_payload(path: &Path, recompute_header_hash: bool) {
    let mut bytes = std::fs::read(path).unwrap();
    let segs = segments(&bytes);
    let (off, _, len) = *segs.iter().find(|s| s.1 == SegmentType::Vec as u8).unwrap();
    bytes[off + 64 + len - 1] ^= 0x40; // alter a vector value
    if recompute_header_hash {
        // Keyless algo-2 hash: the attacker can recompute it.
        let h = compute_shake256_128(&bytes[off + 64..off + 64 + len]);
        bytes[off + 0x28..off + 0x38].copy_from_slice(&h);
    }
    std::fs::write(path, &bytes).unwrap();
}

#[test]
fn vector_segment_with_recomputed_header_hash_is_refused_by_the_committed_hash() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    swap_vec_payload(&p, true);
    for policy in [
        SecurityPolicy::Strict,
        SecurityPolicy::Paranoid,
        SecurityPolicy::WarnOnly,
    ] {
        let e = security(open_err(&p, &[public(&SECRET)], policy));
        assert!(
            matches!(
                e,
                SecurityError::ContentHashMismatch {
                    pointer_name: "segment_content_hash",
                    ..
                }
            ),
            "{policy:?}: {e:?}"
        );
    }
    // Permissive has no committed-hash enforcement: wrong data is served.
    assert!(RvfStore::open_with_policy(&p, opts(&[]), SecurityPolicy::Permissive).is_ok());
}

#[test]
fn corrupt_vector_segment_without_hash_fix_is_refused_everywhere() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    swap_vec_payload(&p, false);
    for policy in [
        SecurityPolicy::Strict,
        SecurityPolicy::Paranoid,
        SecurityPolicy::WarnOnly,
    ] {
        assert!(RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), policy).is_err());
    }
}

#[test]
fn index_segment_must_be_authentic_not_silently_rebuilt() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("idx.rvf");
    let mut o = opts(&[]);
    o.witness = WitnessConfig {
        witness_ingest: false,
        witness_delete: false,
        witness_compact: false,
        audit_queries: false,
    };
    let mut s = RvfStore::create_signed(&path, o, &SECRET).unwrap();
    let v = rows(1100, 0.0);
    let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
    s.ingest_batch(&refs, &(0..1100).collect::<Vec<u64>>(), None)
        .unwrap();
    // Force an index build so close() persists an INDEX_SEG.
    let _ = s
        .query(&[1.0, 1.0, 2.0, 3.0], 5, &QueryOptions::default())
        .unwrap();
    s.close().unwrap();

    let mut bytes = std::fs::read(&path).unwrap();
    let segs = segments(&bytes);
    let &(off, _, len) = segs
        .iter()
        .rev()
        .find(|s| s.1 == SegmentType::Index as u8)
        .expect("an INDEX_SEG must have been persisted (>= 1024 vectors + a query)");
    bytes[off + 64 + len / 2] ^= 0x01;
    let h = compute_shake256_128(&bytes[off + 64..off + 64 + len]);
    bytes[off + 0x28..off + 0x38].copy_from_slice(&h);
    std::fs::write(&path, &bytes).unwrap();
    let e = security(open_err(&path, &[public(&SECRET)], SecurityPolicy::Strict));
    assert!(
        matches!(e, SecurityError::ContentHashMismatch { .. }),
        "{e:?}"
    );
}

#[test]
fn paranoid_reads_every_segment_strict_defers_unread_ones() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("k.rvf");
    let mut s = RvfStore::create_signed(&path, opts(&[]), &SECRET).unwrap();
    s.ingest_batch(&[&[1.0, 2.0, 3.0, 4.0]], &[1], None)
        .unwrap();
    s.embed_kernel(0, 0, 0, &[0x7f; 256], 0, None).unwrap();
    s.close().unwrap();

    let mut bytes = std::fs::read(&path).unwrap();
    let segs = segments(&bytes);
    let (off, _, len) = *segs
        .iter()
        .find(|s| s.1 == SegmentType::Kernel as u8)
        .unwrap();
    bytes[off + 64 + len - 1] ^= 0x01;
    let h = compute_shake256_128(&bytes[off + 64..off + 64 + len]);
    bytes[off + 0x28..off + 0x38].copy_from_slice(&h);
    std::fs::write(&path, &bytes).unwrap();

    let trusted = [public(&SECRET)];
    let strict = RvfStore::open_with_policy(&path, opts(&trusted), SecurityPolicy::Strict).unwrap();
    let e = strict.extract_kernel().unwrap_err();
    // Executable bytes are refused at the point of use.
    assert!(
        matches!(e, RvfError::Code(_) | RvfError::Security(_)),
        "{e:?}"
    );
    assert!(strict.extract_kernel().is_err());
    let e = security(open_err(&path, &trusted, SecurityPolicy::Paranoid));
    assert!(
        matches!(e, SecurityError::ContentHashMismatch { .. }),
        "{e:?}"
    );
}

// ---- writing signed stores ------------------------------------------------------------

#[test]
fn signed_store_stays_strict_valid_across_append_delete_compact_and_audit() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("w.rvf");
    let o = RvfOptions {
        witness: WitnessConfig {
            audit_queries: true,
            ..Default::default()
        },
        ..opts(&[])
    };
    let mut s = RvfStore::create_signed(&path, o, &SECRET).unwrap();
    let trusted = [public(&SECRET)];
    let check = |expect: u64| {
        let r =
            RvfStore::open_with_policy(&path, opts(&trusted), SecurityPolicy::Paranoid).unwrap();
        assert_eq!(r.status().total_vectors, expect);
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() % 64, 0);
        assert_eq!(split_root_page(&bytes).1.map(<[u8]>::len), Some(4096));
    };

    let v = rows(8, 0.0);
    let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
    s.ingest_batch(&refs, &(0..8).collect::<Vec<u64>>(), None)
        .unwrap();
    // Readers can open a live writer's file (no lock for read-only).
    check(8);
    s.delete(&[1, 2]).unwrap();
    check(6);
    s.query_audited(&[0.0, 1.0, 2.0, 3.0], 2, &QueryOptions::default())
        .unwrap();
    check(6);
    s.compact().unwrap();
    check(6);
    s.close().unwrap();

    // Reopen for writing under the same key, append, close.
    let mut w =
        RvfStore::open_signed(&path, opts(&trusted), SecurityPolicy::Strict, &SECRET).unwrap();
    w.ingest_batch(&[&[9.0, 9.0, 9.0, 9.0]], &[77], None)
        .unwrap();
    w.close().unwrap();
    check(7);
}

#[test]
fn read_only_policy_open_cannot_write_and_signed_open_needs_a_signing_policy() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let mut s =
        RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Strict).unwrap();
    assert!(s
        .ingest_batch(&[&[1.0, 2.0, 3.0, 4.0]], &[500], None)
        .is_err());
    assert!(RvfStore::open_signed(
        &p,
        opts(&[public(&SECRET)]),
        SecurityPolicy::Permissive,
        &SECRET
    )
    .is_err());
    // open_signed itself verifies first: an untrusted key list is refused.
    assert!(RvfStore::open_signed(&p, opts(&[]), SecurityPolicy::Strict, &SECRET).is_err());
}

#[test]
fn plain_open_of_a_signed_file_cuts_the_page_on_write_and_the_file_is_then_unsigned() {
    let d = tempfile::tempdir().unwrap();
    let p = signed_store(d.path());
    let mut s = RvfStore::open(&p).unwrap(); // Permissive: no key, nothing verified
    s.ingest_batch(&[&[5.0, 5.0, 5.0, 5.0]], &[900], None)
        .unwrap();
    s.close().unwrap();
    let bytes = std::fs::read(&p).unwrap();
    // No stale RVM0 page is left behind in the middle of the file.
    assert!(split_root_page(&bytes).1.is_none());
    assert!(!bytes
        .chunks(64)
        .any(|c| c[..4] == rvf_types::ROOT_MANIFEST_MAGIC.to_le_bytes()));
    // Honest outcome: no longer acceptable under Strict, fine under Permissive.
    let e = security(open_err(&p, &[public(&SECRET)], SecurityPolicy::Strict));
    assert!(matches!(e, SecurityError::UnsignedManifest { .. }));
    assert_eq!(
        RvfStore::open_readonly(&p).unwrap().status().total_vectors,
        13
    );
}
