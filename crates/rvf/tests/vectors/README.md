# Shared RVF byte fixtures (ADR-354 item 6)

Byte fixtures shared with the rvm repo (`rvm-rvf` must accept the first three;
see ADR-354 acceptance). Each is checked by a test in this repo; none is edited
by hand.

| File | Bytes | SHA-256 | Writer | Reproducible |
|---|---:|---|---|---|
| `runtime_wire_dialect.rvf` | 1792 | `e4341aebdd4c645aaf84739008e1e8ae5de2eebbaddc0fee4e5b829c1e209136` | `RvfStore::create` (wire dialect: 64-byte padded segments, `checksum_algo` 1) | structurally |
| `runtime_signed_wire.rvf` | 6912 | `238e2fb0d18d095031f14f1fcb927ef2e50a875f4164d2e9729f94767a268976` | `RvfStore::create_signed` (algo 2, per-segment hash trailer, signed `RVM0` page at EOF) | structurally |
| `forge_signed_rvm0.rvf` | 4928 | `26c3c90cd5cfa4266c1def8ab3a64922201f77c08a736d2039d723bf2b27ce72` | `rvf-forge-core` `ContainerBuilder` (algo 2, signed segments, 4096-byte `RVM0` root page) | byte-exact |
| `legacy/legacy_basic.rvf` | 1401 | `981c7718d62db8f4fa30d9468898f95ac787ba8df1085a5bbaac0d76f12348aa` | pre-ADR-354 `RvfStore` (algo 0 CRC-rotation, unpadded) | frozen |
| `legacy/legacy_multi.rvf` | 6221 | `360cee29083e84e377d83ede88f949d6f1446b6e6c56ee0bd6ead30100d7a624` | pre-ADR-354 `RvfStore` | frozen |
| `legacy/legacy_compacted.rvf` | 2448 | `fd7f6cb8af37f594128a7dc63b206a17a0cf18791d4c84a9c348dee519f6ebfa` | pre-ADR-354 `RvfStore` | frozen |

## How each was generated

**`runtime_wire_dialect.rvf`**: `RvfStore::create` with dimension 8, L2, all
witness options off (witness segments embed the wall clock), two ingest batches
of 10 vectors from a fixed LCG seed (`vecs(10, 8, 100 + b)`), `close()`.
Generator: `gen_wire` in `rvf-runtime/tests/vectors.rs`.

**`runtime_signed_wire.rvf`**: same data via `RvfStore::create_signed` with the
Ed25519 secret key `[0x42; 32]`, then `delete(&[3])`, `close()` (19 live
vectors). Trusted public key for `open_with_policy(.., Strict)`:
`Ed25519 public of [0x42; 32]` (`rvf_types::ed25519::Ed25519Keypair::from_secret`).
Generator: `gen_signed` in the same test.

**`forge_signed_rvm0.rvf`**: `ContainerBuilder::new()` with file id
`b"ofer-forge-fixt\0"`, capability `Network`, a `Wasm` segment
`\0asm\x01\0\0\0ofer fixture module`, a `Vec` segment of 200 bytes `0x5a`,
signed with Ed25519 secret key `[7; 32]`. Forge reads no clock, so the output is
byte-reproducible. Test: `crates/rvf-forge-core/tests/shared_vectors.rs`.

**`legacy/*.rvf`**: written by the writer at commit `b3fcc8520` (before any
ADR-354 change) using `legacy/gen_legacy.rs.txt` (an `examples/` program; kept
as `.txt` because it only produces legacy files when run at that commit).
`legacy/legacy_expected.txt` holds the status line and the top-5 hits (ids and
raw `f32` distance bits) for three fixed queries per file, recorded from the
same old code. The tests in `rvf-runtime/tests/legacy_dialect.rs` open copies
and require identical results.

## Determinism notes

`RvfStore` is not byte-deterministic: the file id mixes the path hash and the
clock, and it is embedded in the manifest (and hashed into later manifests'
hash trailers and the signed page). The regeneration tests therefore compare
structurally: segment sequence (type, id, `checksum_algo`, payload length,
padding), every non-manifest payload and header hash byte-for-byte, and
manifest payloads with the 68-byte FileIdentity and the hashes of earlier
manifest segments masked. Forge output is compared byte-for-byte.

Regenerate the two RvfStore fixtures and the Forge fixture with
`OFER_REGEN_VECTORS=1 cargo test` (in `crates/rvf` for `-p rvf-runtime --test
vectors`, in the repo root for `-p rvf-forge-core --test shared_vectors`).
Regenerating changes bytes and hashes above; update this table and the rvm
repo's copies.

## What rvm should expect

* `runtime_wire_dialect.rvf` has no root page and an unsigned manifest; the
  segments tile the file at 64-byte strides and verify under `rvm-rvf::walk`.
* `runtime_signed_wire.rvf` and `forge_signed_rvm0.rvf` end in a 4096-byte
  `RVM0` page. `rvm-rvf::walk` must split it off (ADR-354 item 3, rvm side).
  The runtime's page additionally carries `l1_content_hash` at `0xFD0` and the
  signer fingerprint at `0xFE0` (inside the signed region); Forge's pages leave
  both zero.
* Level-0 `sig_algo` is `1` for Ed25519 (the segment-footer enumeration uses
  `0`); the signature covers the page with `0x09C..0x0DC` and `0xFFC..`
  zeroed.
