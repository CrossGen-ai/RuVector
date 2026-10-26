//! Known-answer vectors for RVSD v1 (`crates/rvf/tests/vectors/rvsd-v1-vectors.json`).
//! `RVSD_REGEN=1 cargo test -p rvf-wire --test stream_vectors` rewrites the file;
//! without it the test fails if the committed file differs from a fresh run.
mod common;
use common::json::*;
use common::*;
use rvf_types::{ErrorCode, StreamEntry, StreamHeader};
use rvf_wire::stream::*;

const FILE_A: [u8; 16] = *b"rvsd-test-file01";
const FILE_B: [u8; 16] = *b"rvsd-other-file2";
const STAGES: [u8; 7] = [0, 0, 2, 1, 3, 1, 2];

fn vector_body(variant: u8) -> Vec<u8> {
    use rvf_types::SegmentFlags as F;
    use rvf_wire::writer::{write_segment as w, write_segment_with_algo as wa};
    let f = F::empty();
    let mut b = Vec::new();
    b.extend(w(0x07, &payload(11, 100), f, 1));
    b.extend(w(0x05, &payload(12, 64), f, 2));
    b.extend(wa(0x10, &payload(13, 5000), f, 3, 2));
    b.extend(wa(0x02, &payload(14, 4200), f, 4, 2));
    b.extend(w(0x01, &payload(15 + variant as u64, 6000), f, 5));
    b.extend(w(0x06, &payload(16, 300), f, 6));
    b.extend(root_page());
    b
}

#[derive(Clone)]
enum Edit {
    Xor { frame: usize, offset: usize, mask: u8 },
    Drop(usize),
    Dup(usize),
    Swap(usize, usize),
    Keep(usize),
    Insert { at: usize, bytes: Vec<u8> },
    Replace { frame: usize, bytes: Vec<u8> },
    CutTail { frame: usize, bytes: usize },
}

fn apply(frames: &[Vec<u8>], edits: &[Edit]) -> Vec<Vec<u8>> {
    let mut f = frames.to_vec();
    for e in edits {
        match e.clone() {
            Edit::Xor { frame, offset, mask } => f[frame][offset] ^= mask,
            Edit::Drop(i) => {
                f.remove(i);
            }
            Edit::Dup(i) => f.insert(i + 1, f[i].clone()),
            Edit::Swap(a, b) => f.swap(a, b),
            Edit::Keep(k) => f.truncate(k),
            Edit::Insert { at, bytes } => f.insert(at, bytes),
            Edit::Replace { frame, bytes } => f[frame] = bytes,
            Edit::CutTail { frame, bytes } => {
                let n = f[frame].len() - bytes;
                f[frame].truncate(n);
            }
        }
    }
    f
}

fn edit_json(e: &Edit) -> J {
    match e.clone() {
        Edit::Xor { frame, offset, mask } => J::Obj(vec![("op", s("xor")), ("frame", n(frame as u64)), ("offset", n(offset as u64)), ("mask", n(mask))]),
        Edit::Drop(i) => J::Obj(vec![("op", s("drop")), ("frame", n(i as u64))]),
        Edit::Dup(i) => J::Obj(vec![("op", s("dup")), ("frame", n(i as u64))]),
        Edit::Swap(a, b) => J::Obj(vec![("op", s("swap")), ("a", n(a as u64)), ("b", n(b as u64))]),
        Edit::Keep(k) => J::Obj(vec![("op", s("keep_first")), ("count", n(k as u64))]),
        Edit::Insert { at, bytes } => J::Obj(vec![("op", s("insert")), ("at", n(at as u64)), ("hex", h(&bytes))]),
        Edit::Replace { frame, bytes } => J::Obj(vec![("op", s("replace")), ("frame", n(frame as u64)), ("hex", h(&bytes))]),
        Edit::CutTail { frame, bytes } => J::Obj(vec![("op", s("cut_tail_bytes")), ("frame", n(frame as u64)), ("bytes", n(bytes as u64))]),
    }
}

#[derive(Clone)]
struct Pol {
    trusted: Vec<[u8; 32]>,
    budget: u32,
    pin: Option<EpochPin>,
    now: Option<u64>,
}
impl Pol {
    fn std(budget: u32) -> Self {
        Pol { trusted: vec![pk(0x2A)], budget, pin: None, now: None }
    }
    fn json(&self) -> J {
        let pin = match self.pin {
            None => J::Null,
            Some(p) => J::Obj(vec![("epoch", n(p.epoch)), ("descriptor_hash_hex", p.descriptor_hash.map_or(J::Null, |x| h(&x)))]),
        };
        J::Obj(vec![
            ("trusted_public_keys_hex", J::Arr(self.trusted.iter().map(|k| h(k)).collect())),
            ("fault_budget", n(self.budget)),
            ("epoch_pin", pin),
            ("now_ns", self.now.map_or(J::Null, J::Num)),
        ])
    }
}

struct Out {
    events: Vec<J>,
    errors: Vec<(usize, ErrorCode)>,
    state: String,
    chunks: u64,
    bytes: u64,
}

fn state_name(s: StreamState) -> String {
    match s {
        StreamState::Poisoned(e) => format!("Poisoned:{e:?}"),
        StreamState::Rejected(e) => format!("Rejected:{e:?}"),
        o => format!("{o:?}"),
    }
}

fn run(p: &Pol, frames: &[Vec<u8>]) -> Out {
    let pin = p.pin;
    let f = move |_: &[u8; 16]| pin;
    let mut pol = StreamPolicy::new(&p.trusted);
    pol.fault_budget = p.budget;
    pol.now_ns = p.now;
    if p.pin.is_some() {
        pol.epoch_pin = Some(&f);
    }
    let mut v = StreamVerifier::new(pol);
    let mut o = Out { events: vec![], errors: vec![], state: String::new(), chunks: 0, bytes: 0 };
    for (i, fr) in frames.iter().enumerate() {
        let ev = match v.push(fr) {
            Ok(Event::DescriptorVerified { descriptor_hash, seg_count, body_len }) => J::Obj(vec![("event", s("DescriptorVerified")), ("descriptor_hash_hex", h(&descriptor_hash)), ("seg_count", n(seg_count)), ("body_len", n(body_len))]),
            Ok(Event::TableVerified { seg_id }) => J::Obj(vec![("event", s("TableVerified")), ("seg_id", n(seg_id))]),
            Ok(Event::Chunk { chunk, segment_verified, stage_verified, complete }) => {
                o.chunks += 1;
                o.bytes += chunk.bytes.len() as u64;
                J::Obj(vec![
                    ("event", s("Chunk")),
                    ("seg_id", n(chunk.seg_id)),
                    ("chunk_index", n(chunk.chunk_index)),
                    ("body_offset", n(chunk.body_offset)),
                    ("len", n(chunk.bytes.len() as u64)),
                    ("segment_verified", J::Bool(segment_verified)),
                    ("stage_verified", stage_verified.map_or(J::Null, |x| n(x))),
                    ("complete", complete.map_or(J::Null, |c| s(format!("{c:?}")))),
                ])
            }
            Err(e) => {
                o.errors.push((i, e));
                J::Obj(vec![("event", s("Error")), ("reason", s(format!("{e:?}"))), ("code", s(format!("0x{:04x}", e as u16)))])
            }
        };
        o.events.push(ev);
    }
    o.state = state_name(v.state());
    o
}

struct Stream {
    name: &'static str,
    bundle: StreamBundle,
    frames: Vec<Vec<u8>>,
}

fn mk(name: &'static str, body: &[u8], file_id: [u8; 16], log2: u8, stage: bool) -> Stream {
    let mut b = StreamDescriptorBuilder::new(file_id, key(0x2A)).epoch(7).chunk_log2(log2).policy_hash([0xAB; 32]);
    if stage {
        b = b.stages(&STAGES).requires_lower_stages(true);
    }
    let bundle = b.build(body).unwrap();
    let frames = if stage { bundle.frames_stage_order(body).collect() } else { bundle.frames_file_order(body).collect() };
    Stream { name, bundle, frames }
}

fn entry_json(e: &StreamEntry, h_: &StreamHeader, body: &[u8]) -> J {
    use rvf_crypto::leaf_hash;
    let st = &body[e.body_offset as usize..(e.body_offset + e.stored_len) as usize];
    let n_ch = h_.chunk_count(e.stored_len).unwrap();
    let leaves: Vec<J> = (0..n_ch)
        .map(|i| {
            let a = (i as u64 * h_.chunk_size().unwrap_or(0)) as usize;
            let l = h_.chunk_len(e.stored_len, i).unwrap() as usize;
            h(&leaf_hash(&h_.file_id, e.seg_id, i, &st[a..a + l]))
        })
        .collect();
    J::Obj(vec![
        ("seg_id", n(e.seg_id)), ("seg_type", n(e.seg_type)), ("stage", n(e.stage)),
        ("requires_mask", n(e.requires_mask)), ("body_offset", n(e.body_offset)),
        ("stored_len", n(e.stored_len)), ("uncompressed_len", n(e.uncompressed_len)),
        ("chunk_count", n(n_ch)), ("leaf_hashes_hex", J::Arr(leaves)),
        ("chunk_table_hash_hex", h(&e.chunk_table_hash)),
    ])
}

fn stream_json(st: &Stream, body: &[u8]) -> J {
    let out = run(&Pol::std(4), &st.frames);
    assert!(out.errors.is_empty() && out.state == "Complete", "{} {:?}", st.name, out.errors);
    let hd = &st.bundle.header;
    let frames: Vec<J> = st.frames.iter().zip(out.events).map(|(f, e)| J::Obj(vec![("hex", h(f)), ("expect", e)])).collect();
    J::Obj(vec![
        ("name", s(st.name)), ("chunk_log2", n(hd.chunk_log2)), ("flags", n(hd.flags)),
        ("epoch", n(hd.epoch)), ("policy_hash_hex", h(&hd.policy_hash)),
        ("rvf_identity_hex", h(&hd.rvf_identity)),
        ("descriptor_hex", h(&st.bundle.descriptor)), ("descriptor_hash_hex", h(&st.bundle.descriptor_hash)),
        ("entries", J::Arr(st.bundle.entries.iter().map(|e| entry_json(e, hd, body)).collect())),
        ("frames", J::Arr(frames)), ("final_state", s(out.state)),
    ])
}

fn pos(fr: &[Vec<u8>], ty: u8, seg: u64, idx: Option<u32>) -> usize {
    fr.iter().position(|f| f[0] == ty && f[1..9] == seg.to_le_bytes() && idx.is_none_or(|i| f[9..13] == i.to_le_bytes())).unwrap()
}

struct Case { name: &'static str, stream: &'static str, what: &'static str, edits: Vec<Edit>, pol: Pol, reason: ErrorCode, state: &'static str }

fn cases(m1f: &Stream, m1s: &Stream, m2f: &Stream, other: &Stream) -> Vec<Case> {
    use ErrorCode as E;
    let fr = &m2f.frames;
    let (c0, c1) = (pos(fr, 1, 3, Some(0)), pos(fr, 1, 3, Some(1)));
    let t3 = pos(fr, 2, 3, None);
    let s5 = pos(fr, 1, 5, Some(0));
    let bundle = &m2f.bundle;
    let sk = key(0x2A);
    let df = |d: Vec<u8>| Edit::Replace { frame: 0, bytes: descriptor_frame(&d) };
    let c = |name, what, edits, pol, reason, state| Case { name, stream: "m2-file", what, edits, pol, reason, state };
    let p0 = Pol::std(0);
    let mut v = vec![
        c("descriptor-flip-epoch-bit", "bit flip in a signed header field", vec![Edit::Xor { frame: 0, offset: 5 + 0x20, mask: 1 }], p0.clone(), E::StreamBadSignature, "Rejected:StreamBadSignature"),
        c("descriptor-flip-table-bit", "bit flip inside the segment table", vec![Edit::Xor { frame: 0, offset: 5 + 0x90 + 45, mask: 0x80 }], p0.clone(), E::StreamBadSignature, "Rejected:StreamBadSignature"),
        c("descriptor-flip-signature-bit", "bit flip inside the signature", vec![Edit::Xor { frame: 0, offset: 5 + bundle.descriptor.len() - 3, mask: 4 }], p0.clone(), E::StreamBadSignature, "Rejected:StreamBadSignature"),
        c("descriptor-flip-magic", "magic changed", vec![Edit::Xor { frame: 0, offset: 5, mask: 1 }], p0.clone(), E::StreamBadMagic, "Rejected:StreamBadMagic"),
        c("descriptor-truncated", "last descriptor byte missing (frame len fixed up)", vec![df(bundle.descriptor[..bundle.descriptor.len() - 1].to_vec())], p0.clone(), E::StreamDescriptorTruncated, "Rejected:StreamDescriptorTruncated"),
        c("epoch-lower-than-pin", "replay of an older epoch", vec![], Pol { pin: Some(EpochPin { epoch: 8, descriptor_hash: None }), ..p0.clone() }, E::StreamStaleEpoch, "Rejected:StreamStaleEpoch"),
        c("epoch-equal-different-content", "equal epoch, pinned hash differs", vec![], Pol { pin: Some(EpochPin { epoch: 7, descriptor_hash: Some(other.bundle.descriptor_hash) }), ..p0.clone() }, E::StreamEpochConflict, "Rejected:StreamEpochConflict"),
        c("unknown-signer", "signer not in the trust set", vec![], Pol { trusted: vec![pk(0x2B)], ..p0.clone() }, E::UnknownSigner, "Rejected:UnknownSigner"),
        c("descriptor-expired", "expires_ns reached on a trusted clock", vec![df(resign(bundle, &sk, |_, h| h.expires_ns = 1000))], Pol { now: Some(1000), ..p0.clone() }, E::StreamExpired, "Rejected:StreamExpired"),
        c("size-lie-seg-count", "signed seg_count 16777215 with a short descriptor", vec![df(resign(bundle, &sk, |_, h| { h.seg_count = 0xFF_FFFF; h.table_len = 0xFF_FFFF * 72; }))], p0.clone(), E::StreamLimitExceeded, "Rejected:StreamLimitExceeded"),
        c("size-lie-uncompressed-len", "signed uncompressed_len u64::MAX", vec![df(resign(bundle, &sk, |es, _| es[0].uncompressed_len = u64::MAX))], p0.clone(), E::StreamLimitExceeded, "Rejected:StreamLimitExceeded"),
        c("size-lie-stored-len", "signed stored_len u64::MAX/2 (over the segment cap)", vec![df(resign(bundle, &sk, |es, _| es[2].stored_len = u64::MAX / 2))], p0.clone(), E::StreamLimitExceeded, "Rejected:StreamLimitExceeded"),
        c("size-lie-body-len", "signed body_len one byte short", vec![df(resign(bundle, &sk, |_, h| h.body_len -= 1))], p0.clone(), E::StreamTableInvalid, "Rejected:StreamTableInvalid"),
        c("table-gap", "signed gap between two segments", vec![df(resign(bundle, &sk, |es, _| es[3].body_offset += 64))], p0.clone(), E::StreamTableInvalid, "Rejected:StreamTableInvalid"),
        c("table-overlap", "signed overlap between two segments", vec![df(resign(bundle, &sk, |es, _| es[3].body_offset -= 64))], p0.clone(), E::StreamTableInvalid, "Rejected:StreamTableInvalid"),
        c("table-unsorted", "signed entries out of body order", vec![df(resign(bundle, &sk, |es, _| es.swap(1, 2)))], p0.clone(), E::StreamTableInvalid, "Rejected:StreamTableInvalid"),
        c("table-duplicate-seg-id", "signed duplicate seg_id", vec![df(resign(bundle, &sk, |es, _| es[2].seg_id = es[1].seg_id))], p0.clone(), E::StreamTableInvalid, "Rejected:StreamTableInvalid"),
        c("chunk-bit-flip", "bit flip in a chunk payload", vec![Edit::Xor { frame: c0, offset: 17 + 100, mask: 0x10 }], p0.clone(), E::StreamChunkHashMismatch, "Poisoned:StreamChunkHashMismatch"),
        c("table-bit-flip", "bit flip in a chunk table leaf", vec![Edit::Xor { frame: t3, offset: 13 + 40, mask: 1 }], p0.clone(), E::StreamTableHashMismatch, "Poisoned:StreamTableHashMismatch"),
        c("wrong-file-id-splice", "chunk validly hashed under another file_id (same signing key), spliced in", vec![Edit::Replace { frame: s5, bytes: other.frames[pos(&other.frames, 1, 5, Some(0))].clone() }], p0.clone(), E::StreamChunkHashMismatch, "Poisoned:StreamChunkHashMismatch"),
        c("frame-len-lie", "chunk frame declares one byte more than carried", vec![Edit::Xor { frame: c0, offset: 13, mask: 1 }], p0.clone(), E::StreamFrameSizeLie, "Poisoned:StreamFrameSizeLie"),
        c("frame-cut-tail", "last frame cut by one byte", vec![Edit::CutTail { frame: fr.len() - 1, bytes: 1 }], p0.clone(), E::StreamFrameSizeLie, "Poisoned:StreamFrameSizeLie"),
        c("truncate-frames", "stream ends after 10 frames", vec![Edit::Keep(10)], p0.clone(), E::StreamBadState, "Receiving"),
        c("duplicate-chunk", "chunk 0 delivered twice", vec![Edit::Dup(c0)], p0.clone(), E::StreamUnexpectedFrame, "Poisoned:StreamUnexpectedFrame"),
        c("out-of-order-chunk", "chunk 1 before chunk 0", vec![Edit::Swap(c0, c1)], p0.clone(), E::StreamUnexpectedFrame, "Poisoned:StreamUnexpectedFrame"),
        c("deleted-chunk", "chunk 0 missing", vec![Edit::Drop(c0)], p0.clone(), E::StreamUnexpectedFrame, "Poisoned:StreamUnexpectedFrame"),
        c("inserted-unknown-frame", "frame of unknown type inserted", vec![Edit::Insert { at: 2, bytes: vec![0x7F, 1, 2, 3] }], p0.clone(), E::StreamUnexpectedFrame, "Poisoned:StreamUnexpectedFrame"),
        c("chunk-before-table", "chunk 0 sent before its chunk table", vec![Edit::Swap(t3, c0)], p0.clone(), E::StreamUnexpectedFrame, "Poisoned:StreamUnexpectedFrame"),
        c("identity-mismatch", "signed rvf_identity differs from SHA-256 of the body", vec![df(resign(bundle, &sk, |_, h| h.rvf_identity[0] ^= 1))], p0.clone(), E::StreamIdentityMismatch, "Poisoned:StreamIdentityMismatch"),
    ];
    // Requires gate: stage order, but the stage-3 segment is delivered first.
    v.push(Case { name: "requires-unmet", stream: "m1-stage", what: "stage-3 segment delivered before stages 0..2 are verified", edits: vec![Edit::Swap(1, 7)], pol: p0, reason: E::StreamRequirementsUnmet, state: "Poisoned:StreamRequirementsUnmet" });
    let _ = (m1f, m1s);
    v
}

fn case_json(c: &Case, base: &Stream) -> J {
    let frames = apply(&base.frames, &c.edits);
    let o = run(&c.pol, &frames);
    let first = o.errors.first().copied();
    match (first, c.name) {
        (_, "truncate-frames") => assert_eq!(o.state, "Receiving"),
        (Some((_, r)), _) => assert_eq!(r, c.reason, "{}", c.name),
        (None, _) => panic!("{} produced no error", c.name),
    }
    assert_eq!(o.state, c.state, "{}", c.name);
    let reason = if c.name == "truncate-frames" { J::Null } else { s(format!("{:?}", c.reason)) };
    let code = if c.name == "truncate-frames" { J::Null } else { s(format!("0x{:04x}", c.reason as u16)) };
    J::Obj(vec![
        ("name", s(c.name)), ("stream", s(c.stream)), ("description", s(c.what)),
        ("policy", c.pol.json()), ("edits", J::Arr(c.edits.iter().map(edit_json).collect())),
        ("expect", J::Obj(vec![
            ("first_error_frame", first.map_or(J::Null, |(i, _)| n(i as u64))),
            ("reason", reason), ("code", code), ("final_state", s(o.state)),
            ("released_chunks", n(o.chunks)), ("released_bytes", n(o.bytes)),
            ("complete", J::Bool(false)),
        ])),
    ])
}

fn generate() -> String {
    let body = vector_body(0);
    let body_b = vector_body(1);
    let m1f = mk("m1-file", &body, FILE_A, 0xFF, false);
    let m1s = mk("m1-stage", &body, FILE_A, 0xFF, true);
    let m2f = mk("m2-file", &body, FILE_A, 12, false);
    let m2s = mk("m2-stage", &body, FILE_A, 12, true);
    let other = {
        let b = StreamDescriptorBuilder::new(FILE_B, key(0x2A)).epoch(7).chunk_log2(12).policy_hash([0xAB; 32]).build(&body_b).unwrap();
        let frames = b.frames_file_order(&body_b).collect();
        Stream { name: "other", bundle: b, frames }
    };
    let all = [&m1f, &m1s, &m2f, &m2s];
    let neg: Vec<J> = cases(&m1f, &m1s, &m2f, &other).iter().map(|c| case_json(c, all.iter().find(|s| s.name == c.stream).unwrap())).collect();
    let segs: Vec<J> = m2f.bundle.entries.iter().map(|e| J::Obj(vec![("seg_id", n(e.seg_id)), ("seg_type", n(e.seg_type)), ("body_offset", n(e.body_offset)), ("stored_len", n(e.stored_len))])).collect();
    // retry within budget: flipped chunk then the good one again.
    let c0 = pos(&m2f.frames, 1, 3, Some(0));
    let mut retry = m2f.frames.clone();
    let good = retry[c0].clone();
    retry[c0][24] ^= 1;
    retry.insert(c0 + 1, good.clone());
    let ro = run(&Pol::std(4), &retry);
    assert_eq!((ro.errors.len(), ro.state.as_str()), (1, "Complete"));
    let retry_j = J::Obj(vec![
        ("name", s("retry-within-budget")), ("stream", s("m2-file")),
        ("description", s("corrupted chunk then a clean resend; one counted fault, stream completes")),
        ("policy", Pol::std(4).json()),
        ("edits", J::Arr(vec![edit_json(&Edit::Xor { frame: c0, offset: 24, mask: 1 }), edit_json(&Edit::Insert { at: c0 + 1, bytes: good.clone() })])),
        ("expect", J::Obj(vec![("first_error_frame", n(c0 as u64)), ("reason", s("StreamChunkHashMismatch")), ("code", s("0x0810")), ("final_state", s(ro.state)), ("released_chunks", n(ro.chunks)), ("released_bytes", n(ro.bytes)), ("complete", J::Bool(true))])),
    ]);
    J::Obj(vec![
        ("format", s("rvsd-v1-vectors")), ("version", n(1u64)),
        ("generator", s("crates/rvf/rvf-wire/tests/stream_vectors.rs (RVSD_REGEN=1 rewrites this file)")),
        ("domain_tag_ascii_with_trailing_nul", s("RVF-v1-stream-descriptor\0")),
        ("test_only_key", J::Obj(vec![("seed_hex", h(&[0x2A; 32])), ("public_key_hex", h(&pk(0x2A))), ("signer_key_id_hex", h(&rvf_crypto::signer_key_id(&pk(0x2A)))), ("other_seed_hex", h(&[0x2B; 32])), ("other_public_key_hex", h(&pk(0x2B)))])),
        ("file_id_hex", h(&FILE_A)), ("other_file_id_hex", h(&FILE_B)),
        ("body", J::Obj(vec![("len", n(body.len() as u64)), ("sha256_hex", h(&rvf_types::sha256(&body))), ("segments", J::Arr(segs)), ("hex", h(&body))])),
        ("other_body_hex", h(&body_b)),
        ("stages_by_segment_order", J::Arr(STAGES.iter().map(|&x| n(x)).collect())),
        ("streams", J::Arr(vec![stream_json(&m1f, &body), stream_json(&m1s, &body), stream_json(&m2f, &body), stream_json(&m2s, &body)])),
        ("other_stream_m2_file_descriptor_hex", h(&other.bundle.descriptor)),
        ("positive_variants", J::Arr(vec![retry_j])),
        ("negative", J::Arr(neg)),
    ]).render()
}

#[test]
fn committed_vectors_match_a_fresh_deterministic_run() {
    let fresh = generate();
    assert_eq!(fresh, generate(), "generation is not deterministic");
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/vectors/rvsd-v1-vectors.json");
    if std::env::var("RVSD_REGEN").is_ok() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &fresh).unwrap();
        return;
    }
    let committed = std::fs::read_to_string(&path).expect("vectors file missing; run with RVSD_REGEN=1");
    assert!(committed == fresh, "rvsd-v1-vectors.json differs from the generated vectors; rerun with RVSD_REGEN=1 and review the diff");
}
