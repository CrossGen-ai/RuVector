//! A tiny beam-search harness that exercises each collector against a shared
//! stream of (id, distance) events. Used by the demo and the bench to
//! confirm that the collectors are interchangeable at the trait boundary.

use crate::{Candidate, TopKCollector};

/// Stream `events` through `collector`, pruning anything above the running
/// upper bound. Returns the final top-K and the number of candidates that
/// actually reached the collector (post-prune).
pub fn run_beam<C: TopKCollector>(mut collector: C, events: &[Candidate]) -> (Vec<Candidate>, usize) {
    let mut admitted = 0usize;
    for ev in events {
        let bound = collector.kth_upper_bound();
        if ev.dist > bound {
            continue;
        }
        collector.push(*ev);
        admitted += 1;
    }
    (collector.finalize(), admitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinaryHeapTopK, RadixBucketTopK};

    #[test]
    fn beam_heap_and_radix_agree_on_exact_distances() {
        let events: Vec<Candidate> = (0..2000u32)
            .map(|i| Candidate::new(i, (i as f32) * 1e-3 + 1e-6))
            .collect();

        let (heap_out, _) = run_beam(BinaryHeapTopK::new(10), &events);
        let (radix_out, _) = run_beam(RadixBucketTopK::new(10), &events);

        assert_eq!(heap_out.len(), 10);
        assert_eq!(radix_out.len(), 10);
        // The 10 smallest distances in the stream are i=0..=9. Both
        // collectors must return the same ids in the same order.
        for i in 0..10 {
            assert_eq!(heap_out[i].id, radix_out[i].id);
        }
    }
}
