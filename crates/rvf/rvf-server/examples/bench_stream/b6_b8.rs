//! B6 (resume), B7 (verify-before-allocate), B8 (corrupted frames).

use crate::common::*;
use ed25519_dalek::SigningKey;
use rvf_runtime::progressive_stream::build::descriptor_for;
use rvf_runtime::progressive_stream::build::BuildParams;
use rvf_runtime::progressive_stream::client::{
    recover, ClientConfig, FetchEnd, MemTransport, RunEnd, StreamClient, Throttled, Transport,
};
use rvf_wire::stream::{StreamDescriptorBuilder, StreamPolicy, StreamVerifier};
use std::time::Instant;

// ---------------------------------------------------------------- B6
/// (cut, re-sent, secs, re-hashed, replay secs, network after replay, fetched)
type Row = (u64, u64, f64, u64, f64, u64, u64);

pub fn b6(f: &Fix) {
    println!("\n== B6 resume cost (kill at random offsets) ==");
    let (body, bundle) = descriptor_for(
        &f.path,
        f.file_id,
        SigningKey::from_bytes(&KEY),
        &BuildParams::default(),
    )
    .unwrap();
    let stream: Vec<u8> = bundle.frames_stage_order(&body).flatten().collect();
    let total = stream.len() as u64;
    let keys = [pubkey()];
    let mut x = 0x1234_5678_9abc_def0u64;
    let mut rows = Vec::new();
    let dir = tempfile::tempdir().unwrap();
    for trial in 0..40 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let cut = x % total;
        // (a) connection drop, process alive: resume from the consumer's offset.
        let mut t = MemTransport::new(stream.clone());
        t.cut_after = Some(cut as usize);
        let mut c = StreamClient::new(StreamPolicy::new(&keys), t, ClientConfig::default());
        let t0 = Instant::now();
        assert_eq!(c.run(&mut |_| {}), RunEnd::CompleteUnverified);
        let a_resent = c.stats.bytes_fetched - total;
        let a_s = t0.elapsed().as_secs_f64();
        // (b) process killed: replay the frame log (truncated at `cut`) through a fresh verifier.
        let log = dir.path().join(format!("log{trial}"));
        std::fs::write(&log, &stream[..cut as usize]).unwrap();
        let t1 = Instant::now();
        let (ing, rehashed) = recover(StreamPolicy::new(&keys), &log).unwrap();
        let rec_s = t1.elapsed().as_secs_f64();
        let resume_at = ing.offset();
        let mut c2 = StreamClient::with_ingest(
            ing,
            MemTransport::new(stream.clone()),
            ClientConfig::default(),
        );
        assert_eq!(c2.run(&mut |_| {}), RunEnd::CompleteUnverified);
        rows.push((
            cut,
            a_resent,
            a_s,
            rehashed,
            rec_s,
            total - resume_at,
            c2.stats.bytes_fetched,
        ));
    }
    let mean = |g: &dyn Fn(&Row) -> f64| rows.iter().map(g).sum::<f64>() / rows.len() as f64;
    println!("stream {total} bytes; 40 random kill offsets");
    println!("(a) connection drop, verifier kept: re-sent bytes mean {:.0} max {} (0 = resume at the exact byte); total time mean {:.3}s",
        mean(&|r| r.1 as f64), rows.iter().map(|r| r.1).max().unwrap(), mean(&|r| r.2));
    println!("(b) process kill, frame log replayed: re-hashed bytes mean {:.0} ({:.1}% of stream), replay time mean {:.3}s; re-sent from the network mean {:.0} ({:.1}%)",
        mean(&|r| r.3 as f64), 100.0 * mean(&|r| r.3 as f64) / total as f64, mean(&|r| r.4), mean(&|r| r.5 as f64), 100.0 * mean(&|r| r.5 as f64) / total as f64);
    println!(
        "    baseline restart-from-zero: re-sent 100% of the log prefix ({:.1}% on average)",
        100.0 * mean(&|r| r.0 as f64) / total as f64
    );
}

// ---------------------------------------------------------------- B7
pub fn b7() {
    println!("\n== B7 verify-before-allocate latency (budget 500 ms) ==");
    let key = SigningKey::from_bytes(&KEY);
    let keys = [key.verifying_key().to_bytes()];
    println!(
        "{:>10} {:>14} {:>14}",
        "seg_count", "descriptor B", "decode ms"
    );
    for count in [16usize, 256, 4096, 16384] {
        let mut body = Vec::new();
        for i in 0..count {
            body.extend(rvf_wire::writer::write_segment_with_algo(
                0x01,
                &[i as u8; 64],
                rvf_types::SegmentFlags::empty(),
                i as u64 + 1,
                2,
            ));
        }
        let bundle = StreamDescriptorBuilder::new([1; 16], key.clone())
            .chunk_log2(0xFF)
            .build(&body)
            .unwrap();
        let best = (0..5)
            .map(|_| {
                let t = Instant::now();
                let mut pol = StreamPolicy::new(&keys);
                pol.max_descriptor_bytes = 2 << 20; // the 1 MiB default caps seg_count near 14.5k
                let v = StreamVerifier::from_descriptor(&bundle.descriptor, pol).unwrap();
                std::hint::black_box(&v);
                t.elapsed().as_secs_f64() * 1e3
            })
            .fold(f64::MAX, f64::min);
        println!("{count:>10} {:>14} {:>14.3}", bundle.descriptor.len(), best);
    }
    println!(
        "{:>10} {:>14} {:>14} {:>16}",
        "module MiB", "chunks", "verify ms", "decode+verify ms"
    );
    for mib in [1usize, 4, 16, 64] {
        let payload = vec![0x5au8; mib << 20];
        let body = rvf_wire::writer::write_segment_with_algo(
            0x10,
            &payload,
            rvf_types::SegmentFlags::empty(),
            1,
            2,
        );
        let bundle = StreamDescriptorBuilder::new([2; 16], key.clone())
            .chunk_log2(16)
            .build(&body)
            .unwrap();
        let frames: Vec<Vec<u8>> = bundle.frames_file_order(&body).collect();
        let t0 = Instant::now();
        let mut v =
            StreamVerifier::from_descriptor(&bundle.descriptor, StreamPolicy::new(&keys)).unwrap();
        let dec = t0.elapsed().as_secs_f64() * 1e3;
        let mut done = false;
        for fr in &frames[1..] {
            if let Ok(rvf_wire::stream::Event::Chunk {
                segment_verified: true,
                ..
            }) = v.push(fr)
            {
                done = true;
            }
        }
        let all = t0.elapsed().as_secs_f64() * 1e3;
        assert!(done);
        println!(
            "{mib:>10} {:>14} {:>14.2} {:>16.2}",
            frames.len() - 1,
            all - dec,
            all
        );
    }
}

// ---------------------------------------------------------------- B8
/// Serves whole frames; each served frame is corrupted (one flipped bit) with
/// probability `p`, independently on every (re)fetch.
struct FrameCorrupt {
    frames: Vec<Vec<u8>>,
    starts: Vec<u64>,
    seed: u64,
    p: f64,
}

impl Transport for FrameCorrupt {
    fn fetch(
        &mut self,
        off: u64,
        sink: &mut dyn FnMut(&[u8]) -> bool,
    ) -> std::io::Result<FetchEnd> {
        let first = self.starts.partition_point(|&s| s < off);
        assert_eq!(
            self.starts.get(first).copied().unwrap_or(off),
            off,
            "frame-aligned resume"
        );
        for fr in &self.frames[first..] {
            self.seed = self
                .seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let hit = ((self.seed >> 11) as f64 / (1u64 << 53) as f64) < self.p;
            let ok = if hit && fr.len() > 40 {
                let mut m = fr.clone();
                m[20 + (self.seed >> 33) as usize % (fr.len() - 20)] ^= 0x08;
                sink(&m)
            } else {
                sink(fr)
            };
            if !ok {
                return Ok(FetchEnd::Aborted);
            }
        }
        Ok(FetchEnd::Eof)
    }
}

pub fn b8(f: &Fix) {
    println!("\n== B8 throughput with corrupted frames (each served frame corrupted with probability p; frame retry budget 8) ==");
    let (body, bundle) = descriptor_for(
        &f.path,
        f.file_id,
        SigningKey::from_bytes(&KEY),
        &BuildParams::default(),
    )
    .unwrap();
    let frames: Vec<Vec<u8>> = bundle.frames_stage_order(&body).collect();
    let mut starts = Vec::new();
    let mut at = 0u64;
    for fr in &frames {
        starts.push(at);
        at += fr.len() as u64;
    }
    let total = at;
    let keys = [pubkey()];
    println!("{} frames, {total} bytes", frames.len());
    println!(
        "{:>6} {:>9} | {:>9} {:>10} {:>10} {:>9} {:>10} {:>9}",
        "p", "link", "secs", "goodput", "fetched", "overhead", "refusals", "end"
    );
    for (link, mbps) in [("memory", 0.0), ("100 Mbps", 100.0)] {
        for p in [0.0, 0.05, 0.10] {
            let mut policy = StreamPolicy::new(&keys);
            policy.fault_budget = 1_000_000;
            let cfg = ClientConfig {
                max_stalls: 8,
                frame_retries: 8,
            };
            let fc = FrameCorrupt {
                frames: frames.clone(),
                starts: starts.clone(),
                seed: 5,
                p,
            };
            let t0 = Instant::now();
            let (end, stats) = if mbps > 0.0 {
                let mut c = StreamClient::new(policy, Throttled { inner: fc, mbps }, cfg);
                (c.run(&mut |_| {}), c.stats)
            } else {
                let mut c = StreamClient::new(policy, fc, cfg);
                (c.run(&mut |_| {}), c.stats)
            };
            let secs = t0.elapsed().as_secs_f64();
            println!(
                "{:>5.0}% {:>9} | {:>8.2}s {:>7.1} MB/s {:>10} {:>8.1}% {:>10} {:>9?}",
                p * 100.0,
                link,
                secs,
                total as f64 / secs / 1e6,
                stats.bytes_fetched,
                100.0 * (stats.bytes_fetched as f64 / total as f64 - 1.0),
                stats.refusals,
                end
            );
        }
    }
}
