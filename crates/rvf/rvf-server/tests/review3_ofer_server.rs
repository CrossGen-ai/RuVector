//! Round 3: slot pinning after idle_timeout / total_deadline.
mod support;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};
use support::*;

fn open_stream(addr: std::net::SocketAddr) -> TcpStream {
    let mut s = TcpStream::connect(addr).unwrap();
    s.write_all(b"GET /v1/stream/body HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .unwrap();
    s
}

/// Silent clients are now released after idle_timeout (expected to pass).
#[test]
fn f3_oracle_silent_clients_are_released() {
    let f = fixture(60000, 32, 16, 5);
    let r = start(&f, |c| {
        c.max_concurrent_streams = 2;
        c.idle_timeout = Duration::from_secs(2);
    });
    let _a = open_stream(r.http);
    let _b = open_stream(r.http);
    std::thread::sleep(Duration::from_secs(1));
    assert_eq!(raw_get(r.http, "/v1/stream/body", Some("bytes=0-9")).0, 429);
    std::thread::sleep(Duration::from_secs(5));
    assert_eq!(
        raw_get(r.http, "/v1/stream/body", Some("bytes=0-9")).0,
        206,
        "slots freed"
    );
}

/// F3-05 (fixed): a reader that trickles just fast enough to dodge idle_timeout
/// is cut off by the minimum-rate rule and its slot freed.
#[test]
fn f3_05_slow_reader_is_cut_by_min_rate() {
    let f = fixture(60000, 32, 16, 5);
    let r = start(&f, |c| {
        c.max_concurrent_streams = 2;
        c.max_streams_per_client = 2;
        c.idle_timeout = Duration::from_secs(2);
        c.min_rate_bytes_per_sec = 256 << 10;
        c.rate_grace = Duration::from_secs(2);
    });
    let mut readers = Vec::new();
    for _ in 0..2 {
        let s = open_stream(r.http);
        s.set_read_timeout(Some(Duration::from_millis(50))).unwrap();
        readers.push(s);
    }
    let t0 = Instant::now();
    let mut buf = [0u8; 4096];
    let mut st = 0;
    while t0.elapsed() < Duration::from_secs(14) {
        for s in &mut readers {
            let _ = s.read(&mut buf);
        }
        std::thread::sleep(Duration::from_millis(100));
        if t0.elapsed() > Duration::from_secs(3) {
            st = raw_get(r.http, "/v1/stream/body", Some("bytes=0-9")).0;
            if st == 206 {
                break;
            }
        }
    }
    assert_eq!(st, 206, "slow readers must lose their slots");
}

/// F3-05 (fixed): one client address cannot take the whole pool.
#[test]
fn f3_05_per_client_cap() {
    let f = fixture(60000, 32, 16, 5);
    let r = start(&f, |c| {
        c.max_concurrent_streams = 8;
        c.max_streams_per_client = 2;
        c.idle_timeout = Duration::from_secs(30);
    });
    let _a = open_stream(r.http);
    let _b = open_stream(r.http);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(
        raw_get(r.http, "/v1/stream/body", Some("bytes=0-9")).0,
        429,
        "third stream from the same address is refused although the pool has room"
    );
}
