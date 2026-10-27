# ADR-354: RVF dialect convergence and an effective Strict policy (project Ofer prerequisites)

- **Status**: Proposed
- **Date**: 2026-10-26
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-033, ADR-284, ADR-353; RVM ADR-149, ADR-155, ADR-160
- **Tags**: rvf, interoperability, security-policy, tail-scan, ofer

## Context

Project Ofer's research found that the RVF stack is not interoperable end to end, reproduced on this machine (dossier section 1.7, reproducer `/data/scratch/ofer-research/interop`):

- A file written by `rvf_runtime::RvfStore` (the producer behind `rvf-server`) fails `rvm_rvf::walk` and `verify` with `BadMagic`: the runtime emits unpadded segments and labels a CRC32-rotation hash `checksum_algo = 0`, while the wire registry reads algo 0 as XXH3-128 and `rvm-rvf` walks on 64-byte boundaries. `rvf-runtime/src/hashing.rs` itself documents the incompatibility and says migration needs a version bump plus a dual-accept reader.
- A signed container from `rvf-forge-core` fails the same way because it ends in a 4096-byte Level-0 root page (`RVM0`) that `rvm-rvf::walk` does not skip. With that page stripped it verifies.
- Two tail scanners disagree on the same bytes (`rvf-wire` scans at 64-byte alignment; `rvf-runtime` scans byte-by-byte): the same bug class as ZIP end-of-central-directory parser differentials.
- "Verify the root manifest" (ADR-284 1.1, ADR-155 2.1) is a presence check. `RvfOptions.security_policy` defaults to `Strict` but is never read. `RvfStore::open_with_policy` and `create_signed` (ADR-033 section 4) do not exist. The Level-0 root signature helper has no caller outside authoring.

ADR-353 needs a trustworthy body to stream. These defects must be fixed first; they are worth fixing regardless of streaming.

## Decision

1. **One wire dialect.** The `rvf-wire` registry is authoritative: `checksum_algo` 1 = XXH3-128, 2 = SHAKE-256-128; segments are padded to 64 bytes. New `rvf-runtime` writes use the wire dialect. Anything that will be signed or released to an executor or index consumer uses algo 2 or a descriptor commitment (ADR-353), never algo 0 or 1 as an authenticity claim.
2. **Dual-accept legacy reader.** `rvf-runtime` keeps reading legacy-dialect files (algo 0 with the CRC32-rotation hash, unpadded segments) and marks them `legacy_dialect`. Under `Strict` a legacy-dialect segment cannot feed an executable or an index consumer.
3. **Root page.** `rvm-rvf::walk` recognizes a trailing 4096-byte Level-0 root page, as `rvf-forge-core` already does, and reports it as a root-page region instead of failing.
4. **One tail scanner.** The `rvf-wire` scanner is the single implementation; `rvf-runtime` calls it. A differential test and a fuzz target assert both code paths agree on the same bytes. The stream path never uses tail scanning for authority (ADR-353).
5. **Strict is real.** `security_policy` is read. `RvfStore::open_with_policy` exists. Under `Strict`: an unsigned manifest is refused (`UnsignedManifest = 0x0800`); the Level-0 signature is verified, not just parsed; model and index segments that feed queries must be authenticated (signature or descriptor-bound hash), not only header-hashed.
6. **Interop test in CI in both repos.** The reproducer's cases become tests: a `RvfStore`-written file and a Forge-authored file with root page must both be accepted by `rvm-rvf` verification (RVM side) and by the wire reader (ruvector side). Shared byte fixtures are checked into both repos.

## Compatibility

Existing files stay readable. Writers change; a writer-side dialect switch ships behind a version bump as `hashing.rs` already requires. A legacy-dialect file served to RVM needs the descriptor path or a re-write to the wire dialect.

## Acceptance

- The two reproducer cases pass in CI on both sides.
- A differential fuzz target finds no input on which the two scanners disagree.
- A tampered or unsigned manifest is refused under `Strict` and accepted only under an explicit lower policy; the ADR-033 section 3.5 "Malicious Tail Manifest" test runs against the real `open_with_policy`.
- Dual-accept: every file written before this change still opens and queries identically.

## Consequences and risks

Writer dialect change touches the hot path of `rvf-runtime`; mitigated by dual-accept and by round-trip tests over the existing test corpus. Making `Strict` effective can break callers that relied on it being ignored; mitigated by an explicit lower policy and a release note.

## Implementation notes and residual risks (runtime side, project Ofer)

Implemented in `rvf-runtime` 0.4.0. Outcomes of the adversarial review that shape what "Strict is real" means:

- **Strict is opt-in.** `RvfStore::open` / `open_readonly` remain `Permissive`. No in-tree consumer (`rvf-launch`, `rvf-cli` ingest/delete/compact, `rvf-server`) opts into `Strict`; none was changed silently. `rvf-launch::extract_kernel` now reports `signature_unverified` (and warns on stderr) when it extracts a kernel from a file that carries a signed root page it did not verify. Callers that execute extracted bytes must open with `open_with_policy` first.
- **Non-Strict scans can be steered.** Under `Permissive`/`WarnOnly` an aligned MANIFEST header planted inside the payload of a segment appended after the real manifest wins the tail scan (both scanners agree; scanning cannot authenticate). Strict refuses it: the signed page names the manifest offset, length and payload hash. The legacy byte-granular scan is chosen from the file's layout (the first segment whose end is not 64-aligned decides between a padded and an unpadded successor header), not from the unhashed `checksum_algo` byte, which only breaks ties when the layout is undecidable.
- **Scan cost is bounded.** At most 256 manifest payloads are parsed per open, in place, with a total-size budget of `max(16 MiB, file size)`; windows are read in blocks of at most 8 MiB. With a signed page under Strict/Paranoid the scan stops at the page's manifest offset.
- **Signed stores authenticate whole segments.** The signed directory commits to a SHAKE-256-128 payload hash per segment and pins every header field (id, type and length from the directory; flags, compression, reserved words, timestamp, `uncompressed_len` and padding to the writer's constants).
- **Compaction never launders.** For a signing handle `compact()` carries only directory-listed segments, verifies each against the hash the handle committed to (not recomputed from file bytes), and refuses the whole compaction on any mismatch before changing state. Witness segments appended by `query_audited` and `compact` are listed in a new signed manifest, so a signed store has no unlisted trailing segments.
- **A `Permissive` handle will not cut a signature.** A plain `open()` of a store whose file ends in a valid signed page refuses writes (`ReadOnly`) unless the caller opts in with `allow_unsigning(true)`; `open_signed` keeps the signature.
- **No freshness yet.** An older validly signed file replays cleanly under runtime `Strict`; epoch pinning exists only for RVSD streams. COW children are `Permissive`; `Paranoid` is `Strict` plus reading every directory segment.
