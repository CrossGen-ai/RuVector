//! Security policy state and the policy-checked open path.
//!
//! ADR-033 section 4 / ADR-354 item 5. Plain `create`, `open` and
//! `open_readonly` keep their historical behaviour, which is exactly
//! [`SecurityPolicy::Permissive`]: nothing is verified. The policy-aware
//! entry points are [`RvfStore::create_signed`], [`RvfStore::open_with_policy`],
//! [`RvfStore::open_with_options`] and [`RvfStore::open_signed`].
//!
//! Chain of trust under `Strict`/`Paranoid`:
//!
//! 1. the file ends in a Level-0 root page (`RVM0`); absent or `sig_algo = 0`
//!    is `UnsignedManifest`;
//! 2. the page CRC32C holds, and its Ed25519 signature verifies under one of
//!    the caller's trusted signers (`UnknownSigner` when the page names a
//!    signer that is not trusted, `InvalidSignature` when it verifies under
//!    none of the matching keys);
//! 3. the page commits to the manifest the tail scan found: same offset,
//!    same length, and SHAKE-256-128 of the manifest payload equals the
//!    signed `l1_content_hash` (`ContentHashMismatch` otherwise);
//! 4. the signed manifest carries a per-segment hash trailer; every segment
//!    read afterwards (vectors, index, kernels, wasm, ...) must have the
//!    committed SHAKE-256-128 hash, so a swapped segment is refused even if
//!    its own header hash was recomputed by the attacker.
//!
//! Which directory did we use? The runtime's own manifest directory, extended
//! with an optional trailer, not `rvf-manifest`'s `SegmentDirEntry`: that
//! codec has 64-byte entries in a different TLV container that nothing in the
//! runtime reads or writes, and adopting it would be a format change, not a
//! policy change.

use super::*;
use rvf_types::security::{SecurityError, SecurityPolicy};
use rvf_wire::hash::compute_shake256_128;
use std::io;

use crate::root_page;

/// Manifest facts the signed page must be re-bound to at the next commit.
#[derive(Clone, Copy)]
pub(super) struct LastManifest {
    pub offset: u64,
    /// Header + payload length of the MANIFEST segment.
    pub len: u64,
    /// SHAKE-256-128 of its payload.
    pub hash: [u8; 16],
}

/// Per-store security state.
pub(super) struct SecurityState {
    pub policy: SecurityPolicy,
    pub trusted_signers: Vec<[u8; 32]>,
    /// Secret key; present only for stores that sign (never printed).
    pub signing_key: Option<[u8; 32]>,
    /// True while the file ends in a root page this store wrote or verified.
    pub root_page_present: AtomicBool,
    /// Committed content hash by segment offset, when enforcement is on.
    pub committed: Option<HashMap<u64, [u8; 16]>>,
    /// Set when any legacy-dialect byte layout or hash was encountered.
    pub legacy_dialect: AtomicBool,
    pub last_manifest: Option<LastManifest>,
}

impl SecurityState {
    /// Historical behaviour of `create`/`open`/`open_readonly`.
    pub(super) fn permissive() -> Self {
        Self {
            policy: SecurityPolicy::Permissive,
            trusted_signers: Vec::new(),
            signing_key: None,
            root_page_present: AtomicBool::new(false),
            committed: None,
            legacy_dialect: AtomicBool::new(false),
            last_manifest: None,
        }
    }

    /// A store that signs every commit with `key` (created by `create_signed`).
    pub(super) fn signer(key: [u8; 32]) -> Self {
        Self::with(SecurityPolicy::Strict, Vec::new(), Some(key))
    }

    pub(super) fn with(
        policy: SecurityPolicy,
        trusted: Vec<[u8; 32]>,
        key: Option<[u8; 32]>,
    ) -> Self {
        Self {
            policy,
            trusted_signers: trusted,
            signing_key: key,
            ..Self::permissive()
        }
    }
}

/// A committed-hash check failed on read.
#[derive(Debug)]
pub(super) struct CommitMismatch {
    pub offset: u64,
    pub expected: Option<[u8; 16]>,
    pub actual: [u8; 16],
}

impl std::fmt::Display for CommitMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "segment at 0x{:X} does not match its committed hash",
            self.offset
        )
    }
}

impl std::error::Error for CommitMismatch {}

/// Map a Security error to its stable wire code (ADR-033 section 3.5: the
/// code, not the prose, is the contract).
pub fn security_error_code(e: &SecurityError) -> ErrorCode {
    match e {
        SecurityError::UnsignedManifest { .. } => ErrorCode::UnsignedManifest,
        SecurityError::InvalidSignature { .. } => ErrorCode::InvalidSignature,
        SecurityError::UnknownSigner { .. } => ErrorCode::UnknownSigner,
        SecurityError::ContentHashMismatch { .. } => ErrorCode::ContentHashMismatch,
        SecurityError::EpochDriftExceeded { .. } => ErrorCode::EpochDriftExceeded,
        SecurityError::Level1InvalidSignature { .. } => ErrorCode::Level1InvalidSignature,
    }
}

fn read_trailing_page(file: &File) -> Option<[u8; root_page::PAGE_SIZE]> {
    let len = file.metadata().ok()?.len();
    if len < (root_page::PAGE_SIZE + SEGMENT_HEADER_SIZE) as u64 {
        return None;
    }
    let mut reader = BufReader::new(file);
    reader
        .seek(SeekFrom::Start(len - root_page::PAGE_SIZE as u64))
        .ok()?;
    let mut page = [0u8; root_page::PAGE_SIZE];
    reader.read_exact(&mut page).ok()?;
    root_page::view(&page).magic_ok.then_some(page)
}

/// The signature-level checks (steps 1-3 plus the dialect/trailer rules).
fn check_signed_root(
    file: &File,
    trusted: &[[u8; 32]],
    located: &read_path::LocatedManifest,
) -> Result<(), SecurityError> {
    let off = located.offset;
    let unsigned = SecurityError::UnsignedManifest {
        manifest_offset: off,
    };
    let invalid = |phase: &'static str| SecurityError::InvalidSignature {
        manifest_offset: off,
        rejection_phase: phase,
    };
    let page = read_trailing_page(file).ok_or_else(|| unsigned.clone())?;
    let v = root_page::view(&page);
    if v.sig_algo == 0 {
        return Err(unsigned);
    }
    if !v.crc_ok {
        return Err(invalid("level0_crc32c"));
    }
    if v.sig_algo != root_page::SIG_ALGO_ED25519
        || v.sig_len as usize != root_page::ED25519_SIGNATURE_LEN
    {
        return Err(invalid("signature_algorithm"));
    }
    let matching: Vec<&[u8; 32]> = trusted
        .iter()
        .filter(|k| root_page::signer_fingerprint(k) == v.signer_fingerprint)
        .collect();
    if matching.is_empty() {
        return Err(SecurityError::UnknownSigner {
            manifest_offset: off,
            actual_signer: v.signer_fingerprint,
            expected_signer: trusted.first().map(root_page::signer_fingerprint),
        });
    }
    if !matching
        .iter()
        .any(|k| root_page::verify_signature(&page, k))
    {
        return Err(invalid("signature_verification"));
    }
    // The page is authentic. Now bind it to the manifest we are about to use.
    if located.mode != rvf_wire::tail_scan::ScanMode::Aligned
        || located.manifest.dialect != write_path::MANIFEST_DIALECT_WIRE
        || located.header.checksum_algo != crate::hashing::ALGO_SHAKE256_128
    {
        return Err(invalid("legacy_dialect"));
    }
    let actual = compute_shake256_128(&located.payload);
    let l1_len = SEGMENT_HEADER_SIZE as u64 + located.payload.len() as u64;
    if v.l1_offset != off || v.l1_length != l1_len || v.l1_content_hash != actual {
        return Err(SecurityError::ContentHashMismatch {
            pointer_name: "l1_manifest",
            expected_hash: v.l1_content_hash,
            actual_hash: actual,
            seg_offset: off,
        });
    }
    if located.manifest.seg_hashes.is_none() {
        return Err(invalid("segment_hashes_missing"));
    }
    Ok(())
}

impl RvfStore {
    /// Create a store whose every commit ends in a signed Level-0 root page.
    ///
    /// All segments are hashed with SHAKE-256-128 (never the fast XXH3 hash,
    /// which is integrity only), and each manifest carries a per-segment hash
    /// trailer that the signed page commits to. `signing_key` is an Ed25519
    /// secret key (ADR-033 names ML-DSA-65; only Ed25519 exists in this
    /// codebase today). Fails closed without the `ed25519` feature.
    ///
    /// Crash behaviour: a signed commit is "segments, manifest, then page".
    /// A crash before the page is written leaves a file that opens under
    /// `Permissive` and is refused (`UnsignedManifest`) under `Strict`.
    pub fn create_signed(
        path: &Path,
        options: RvfOptions,
        signing_key: &[u8; 32],
    ) -> Result<Self, RvfError> {
        if !cfg!(feature = "ed25519") {
            return Err(err(ErrorCode::InvalidSignature));
        }
        Self::create_inner(path, options, Some(*signing_key))
    }

    /// Open under an explicit security policy (ADR-033 section 4.5).
    ///
    /// Trust anchors come from `options.trusted_signers`. `Permissive` is
    /// exactly [`RvfStore::open`]. Every other policy opens **read-only**: a
    /// writer without the signing key would break the signed chain; use
    /// [`RvfStore::open_signed`] to write.
    ///
    /// Failures are `RvfError::Security(..)` with the structured fields
    /// (offset, phase, hashes) and are logged at WARN.
    pub fn open_with_policy(
        path: &Path,
        options: RvfOptions,
        policy: SecurityPolicy,
    ) -> Result<Self, RvfError> {
        if policy == SecurityPolicy::Permissive {
            return Self::open(path);
        }
        let state = SecurityState::with(policy, options.trusted_signers.clone(), None);
        Self::open_readonly_inner(path, &mut HashSet::new(), state)
    }

    /// [`RvfStore::open_with_policy`] using `options.security_policy` (which
    /// defaults to `Strict`).
    pub fn open_with_options(path: &Path, options: RvfOptions) -> Result<Self, RvfError> {
        let policy = options.security_policy;
        Self::open_with_policy(path, options, policy)
    }

    /// Open a signed store for writing. Requires a policy that checks
    /// signatures; every later commit re-signs with `signing_key`.
    pub fn open_signed(
        path: &Path,
        options: RvfOptions,
        policy: SecurityPolicy,
        signing_key: &[u8; 32],
    ) -> Result<Self, RvfError> {
        if !cfg!(feature = "ed25519") || !policy.requires_signature() {
            return Err(err(ErrorCode::InvalidSignature));
        }
        let state =
            SecurityState::with(policy, options.trusted_signers.clone(), Some(*signing_key));
        Self::open_inner(path, state)
    }

    /// True if any legacy-dialect (pre ADR-354) layout was encountered:
    /// unpadded segments, a CRC-rotation hash labelled `checksum_algo = 0`, or
    /// a manifest without the wire-dialect marker. Under `Strict` a legacy
    /// file never opens, so this is mainly an audit/migration signal.
    pub fn legacy_dialect(&self) -> bool {
        self.security.legacy_dialect.load(Ordering::Relaxed)
    }

    /// The policy this store was opened under.
    pub fn security_policy(&self) -> SecurityPolicy {
        self.security.policy
    }

    /// Writer cursor positioned at EOF. A trailing root page (ours) is cut
    /// first so new segments never land behind it; the next commit re-seals.
    pub(super) fn append_cursor<'a>(
        file: &'a File,
        security: &SecurityState,
    ) -> Result<BufWriter<&'a File>, RvfError> {
        if security.root_page_present.swap(false, Ordering::AcqRel) {
            let len = file
                .metadata()
                .map_err(|_| err(ErrorCode::FsyncFailed))?
                .len();
            file.set_len(len.saturating_sub(root_page::PAGE_SIZE as u64))
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
        }
        let mut w = BufWriter::new(file);
        w.seek(SeekFrom::End(0))
            .map_err(|_| err(ErrorCode::FsyncFailed))?;
        Ok(w)
    }

    /// Read a segment payload, applying the policy's committed-hash rule.
    pub(super) fn read_seg<R: Read + Seek>(
        &self,
        reader: &mut R,
        offset: u64,
    ) -> io::Result<(rvf_types::SegmentHeader, Vec<u8>)> {
        let (header, payload) = read_path::read_segment_payload(reader, offset)?;
        if header.checksum_algo == 0 {
            self.security.legacy_dialect.store(true, Ordering::Relaxed);
        }
        if let Some(committed) = &self.security.committed {
            let expected = committed.get(&offset).copied();
            let ok = header.checksum_algo == crate::hashing::ALGO_SHAKE256_128
                && expected == Some(header.content_hash);
            if !ok {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    CommitMismatch {
                        offset,
                        expected,
                        actual: header.content_hash,
                    },
                ));
            }
        }
        Ok((header, payload))
    }

    /// Convert a failed [`Self::read_seg`] into the right `RvfError`: a
    /// committed-hash mismatch becomes the structured security error.
    pub(super) fn map_read_error(e: &io::Error, fallback: ErrorCode) -> RvfError {
        if let Some(m) = e.get_ref().and_then(|i| i.downcast_ref::<CommitMismatch>()) {
            return RvfError::Security(SecurityError::ContentHashMismatch {
                pointer_name: "segment_content_hash",
                expected_hash: m.expected.unwrap_or([0; 16]),
                actual_hash: m.actual,
                seg_offset: m.offset,
            });
        }
        err(fallback)
    }

    /// Policy check run by `boot` once the manifest is located. Sets up
    /// committed-hash enforcement.
    pub(super) fn apply_open_policy(
        &mut self,
        located: &read_path::LocatedManifest,
    ) -> Result<(), RvfError> {
        let manifest = &located.manifest;
        if located.mode == rvf_wire::tail_scan::ScanMode::ByteGranular
            || manifest.dialect != write_path::MANIFEST_DIALECT_WIRE
            || manifest
                .segment_dir
                .iter()
                .any(|e| e.offset % rvf_types::SEGMENT_ALIGNMENT as u64 != 0)
        {
            self.security.legacy_dialect.store(true, Ordering::Relaxed);
        }

        let policy = self.security.policy;
        if policy == SecurityPolicy::Permissive {
            return Ok(());
        }
        if let Err(e) = check_signed_root(&self.file, &self.security.trusted_signers, located) {
            log::warn!(
                "rvf: {} policy rejected {}: {e}",
                policy_name(policy),
                self.path.display()
            );
            if policy.requires_signature() {
                return Err(RvfError::Security(e));
            }
            // WarnOnly: continue; committed hashes (if the manifest has them)
            // are still enforced below.
        } else {
            self.security
                .root_page_present
                .store(true, Ordering::Release);
        }

        if let Some(entries) = &manifest.seg_hashes {
            let by_id: HashMap<u64, [u8; 16]> = entries.iter().copied().collect();
            let mut by_offset = HashMap::new();
            for e in &manifest.segment_dir {
                if let Some(h) = by_id.get(&e.seg_id) {
                    by_offset.insert(e.offset, *h);
                }
            }
            self.security.committed = Some(by_offset);
        }
        if self.security.signing_key.is_some() {
            self.security.last_manifest = Some(LastManifest {
                offset: located.offset,
                len: SEGMENT_HEADER_SIZE as u64 + located.payload.len() as u64,
                hash: compute_shake256_128(&located.payload),
            });
        }
        Ok(())
    }

    /// `Paranoid`: read every directory segment once so a bad hash fails the
    /// open instead of a later query.
    pub(super) fn verify_all_segments(&self) -> Result<(), RvfError> {
        if self.security.policy != SecurityPolicy::Paranoid {
            return Ok(());
        }
        for &(_, offset, _, _) in &self.segment_dir {
            let mut reader = BufReader::new(&self.file);
            self.read_seg(&mut reader, offset)
                .map_err(|e| Self::map_read_error(&e, ErrorCode::InvalidChecksum))?;
        }
        Ok(())
    }

    /// Append the signed root page for the last committed manifest.
    pub(super) fn seal_root_page(&mut self) -> Result<(), RvfError> {
        let Some(key) = self.security.signing_key else {
            return Ok(());
        };
        #[cfg(feature = "ed25519")]
        {
            let last = self
                .security
                .last_manifest
                .ok_or_else(|| err(ErrorCode::InvalidManifest))?;
            let live =
                (self.vectors.len() as u64).saturating_sub(self.deletion_bitmap.count() as u64);
            let page = root_page::build_signed_page(
                &root_page::PageContent {
                    l1_offset: last.offset,
                    l1_length: last.len,
                    l1_content_hash: last.hash,
                    total_vectors: live,
                    dimension: self.options.dimension,
                    profile: self.options.profile,
                    epoch: self.epoch,
                    identity: &self.file_identity,
                },
                &key,
            );
            let mut w = Self::append_cursor(&self.file, &self.security)?;
            w.write_all(&page)
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            w.flush().map_err(|_| err(ErrorCode::FsyncFailed))?;
            drop(w);
            self.file
                .sync_all()
                .map_err(|_| err(ErrorCode::FsyncFailed))?;
            self.security
                .root_page_present
                .store(true, Ordering::Release);
            Ok(())
        }
        #[cfg(not(feature = "ed25519"))]
        {
            let _ = key;
            Err(err(ErrorCode::InvalidSignature))
        }
    }

    /// Seed a freshly created `SegmentWriter` for this store's security mode.
    pub(super) fn configure_writer(
        &self,
        writer: &mut SegmentWriter,
        hashes: Option<&[(u64, [u8; 16])]>,
    ) {
        if self.security.signing_key.is_some() {
            writer.use_cryptographic_hash();
            for (id, h) in hashes.unwrap_or(&[]) {
                writer.record_hash(*id, *h);
            }
        }
    }
}

fn policy_name(p: SecurityPolicy) -> &'static str {
    match p {
        SecurityPolicy::Permissive => "Permissive",
        SecurityPolicy::WarnOnly => "WarnOnly",
        SecurityPolicy::Strict => "Strict",
        SecurityPolicy::Paranoid => "Paranoid",
    }
}
