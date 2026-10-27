//! Round 2 review: abuse of the stream endpoints.
mod support;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;
use support::*;

/// F2-05: `max_concurrent_streams` slots are held for as long as the client
/// keeps the connection open and does not read. No idle/write timeout: N
/// silent connections deny every other stream (HTTP and TCP 0x05 share the
/// pool) indefinitely.
#[test]
fn f2_05_slowloris_pins_every_stream_slot() {
    let f = fixture(60000, 32, 16, 5);
    let r = start(&f, |c| c.max_concurrent_streams = 2);
    assert!(r.server.stream_len() > 12 << 20, "need a body larger than socket buffers");
    let mut held = Vec::new();
    for _ in 0..2 {
        let mut s = TcpStream::connect(r.http).unwrap();
        s.write_all(b"GET /v1/stream/body HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").unwrap();
        held.push(s); // never read
    }
    std::thread::sleep(Duration::from_secs(4));
    let (st, _, _) = raw_get(r.http, "/v1/stream/body", Some("bytes=0-9"));
    assert_eq!(st, 429, "slots still pinned after 4 s of silence");
    // TCP 0x05 draws on the same pool.
    let mut t = TcpStream::connect(r.tcp).unwrap();
    let mut p = vec![1u8];
    p.extend_from_slice(&0u64.to_be_bytes());
    p.extend_from_slice(&100u32.to_be_bytes());
    let mut fr = (p.len() as u32).to_be_bytes().to_vec();
    fr.extend_from_slice(&[0x05, 0, 0, 1]);
    fr.extend_from_slice(&p);
    t.write_all(&fr).unwrap();
    let mut h = [0u8; 8];
    t.read_exact(&mut h).unwrap();
    assert_eq!(h[4], 0xFF, "TCP stream request refused while HTTP slots are pinned");
    drop(held);
}

/// Hostile TCP 0x05 frames and hostile HTTP Range headers never kill the
/// server (expected to pass).
#[test]
fn f2_oracle_hostile_inputs_do_not_panic() {
    let f = fixture(2000, 16, 8, 3);
    let r = start(&f, |_| {});
    let mut x = 0x9E3779B97F4A7C15u64;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for i in 0..300 {
        let mut t = TcpStream::connect(r.tcp).unwrap();
        t.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let plen = (rnd() % 20) as usize;
        let mut p: Vec<u8> = (0..plen).map(|_| rnd() as u8).collect();
        if i % 3 == 0 && p.len() == 13 {
            p[0] = (rnd() % 3) as u8;
        }
        let mut fr = (p.len() as u32).to_be_bytes().to_vec();
        fr.extend_from_slice(&[0x05, 1, 2, 3]);
        fr.extend_from_slice(&p);
        let _ = t.write_all(&fr);
        let mut h = [0u8; 8];
        let _ = t.read_exact(&mut h);
    }
    for rg in [
        "bytes=0-18446744073709551615",
        "bytes=18446744073709551615-",
        "bytes=0--1",
        "bytes= 0-1",
        "bytes=0-1 ",
        "bytes=+1-2",
        "bytes=00000000000000000000001-",
        "BYTES=0-1",
    ] {
        let (st, _, _) = raw_get(r.http, "/v1/stream/body", Some(rg));
        eprintln!("{rg:?} -> {st}");
    }
    // TCP offset u64::MAX, max_len u32::MAX
    let mut t = TcpStream::connect(r.tcp).unwrap();
    let mut p = vec![1u8];
    p.extend_from_slice(&(u64::MAX - 5).to_be_bytes());
    p.extend_from_slice(&u32::MAX.to_be_bytes());
    let mut fr = (p.len() as u32).to_be_bytes().to_vec();
    fr.extend_from_slice(&[0x05, 0, 0, 9]);
    fr.extend_from_slice(&p);
    t.write_all(&fr).unwrap();
    let mut h = [0u8; 8];
    t.read_exact(&mut h).unwrap();
    // server still alive
    assert_eq!(raw_get(r.http, "/v1/stream/descriptor", None).0, 200);
}
