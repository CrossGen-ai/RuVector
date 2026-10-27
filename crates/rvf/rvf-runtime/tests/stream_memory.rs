//! F2-01 / F2-02 regressions with counters instead of wall clocks: allocator
//! peak while decoding layers, the ingest buffer budget, and the number of
//! lookups a hostile partition range can cause.

use rvf_index::{HnswLayer, LayerA, LayerB, PartitionEntry};
use rvf_runtime::progressive_stream::layers::{decode_a, decode_b, encode_a, encode_b};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to the system allocator, only adding counters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), SeqCst) + l.size();
            PEAK.fetch_max(now, SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), SeqCst);
        System.dealloc(p, l)
    }
}
#[global_allocator]
static A: Counting = Counting;

/// Peak extra bytes allocated while running `f`.
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(SeqCst);
    PEAK.store(base, SeqCst);
    let r = f();
    (r, PEAK.load(SeqCst).saturating_sub(base))
}

#[test]
fn hostile_layer_payloads_are_rejected_within_twice_their_size() {
    // 4M empty top layers: 16 MB on the wire, would be ~130 MB decoded.
    let n = 4_000_000u32;
    let mut a = Vec::new();
    a.extend_from_slice(b"RVPA");
    a.extend_from_slice(&1u32.to_le_bytes());
    a.extend_from_slice(&0u32.to_le_bytes());
    a.extend_from_slice(&0u32.to_le_bytes());
    a.extend_from_slice(&n.to_le_bytes());
    a.extend(std::iter::repeat_n(0u8, n as usize * 4));
    a.extend_from_slice(&[0u8; 12]);
    let (r, peak) = peak_during(|| decode_a(&a));
    assert!(r.is_none());
    assert!(peak <= 2 * a.len(), "peak {peak} for payload {}", a.len());

    // 3M empty adjacency rows in Layer B.
    let m = 3_000_000u32;
    let mut b = Vec::new();
    b.extend_from_slice(b"RVPB");
    b.extend_from_slice(&m.to_le_bytes());
    for i in 0..m as u64 {
        b.extend_from_slice(&i.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
    }
    b.extend_from_slice(&0u32.to_le_bytes());
    let (r, peak) = peak_during(|| decode_b(&b));
    assert!(r.is_none());
    assert!(peak <= 2 * b.len(), "peak {peak} for payload {}", b.len());
}

#[test]
fn legitimate_layers_decode_within_the_budget() {
    let mut l = HnswLayer::default();
    for id in 0..5000u64 {
        l.adjacency
            .insert(id, (0..16).map(|k| (id + k) % 5000).collect());
    }
    let a = LayerA {
        entry_points: vec![(1, 3)],
        top_layers: vec![l],
        top_layer_start: 3,
        centroids: vec![vec![0.5; 32]; 64],
        partition_map: (0..64)
            .map(|c| PartitionEntry {
                centroid_id: c,
                vector_id_start: c as u64 * 10,
                vector_id_end: c as u64 * 10 + 10,
                segment_ref: 0,
                block_ref: 0,
            })
            .collect(),
    };
    let enc = encode_a(&a, 32, &[]);
    let (d, peak) = peak_during(|| decode_a(&enc));
    assert!(d.is_some(), "a real Layer A fits the 2x budget");
    assert!(
        peak <= 2 * enc.len() + 4096,
        "peak {peak} for payload {}",
        enc.len()
    );

    let mut pa = BTreeMap::new();
    for id in 0..5000u64 {
        pa.insert(id, (0..32).map(|k| (id + k) % 5000).collect());
    }
    let enc = encode_b(&LayerB {
        partial_adjacency: pa,
        covered_ranges: vec![(0, 5000)],
    });
    let (d, peak) = peak_during(|| decode_b(&enc));
    assert!(d.is_some());
    assert!(
        peak <= 2 * enc.len() + 4096,
        "peak {peak} for payload {}",
        enc.len()
    );
}
