//! Plain `Vec<u32>` baseline — one allocation per node plus an offsets array,
//! stored as a flat CSR-style buffer so the comparison is honest (no
//! per-node allocation overhead polluting the numbers).

use crate::Adjacency;

#[derive(Debug, Default)]
pub struct PlainAdjacency {
    offsets: Vec<u32>, // length n+1
    neighbors: Vec<u32>,
}

impl PlainAdjacency {
    pub fn new() -> Self {
        Self {
            offsets: vec![0],
            neighbors: Vec::new(),
        }
    }
}

impl Adjacency for PlainAdjacency {
    fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn push(&mut self, neighbors: &[u32]) {
        // Store sorted so the roundtrip test is deterministic across backends.
        let mut tmp = neighbors.to_vec();
        tmp.sort_unstable();
        self.neighbors.extend_from_slice(&tmp);
        self.offsets.push(self.neighbors.len() as u32);
    }

    fn bytes(&self) -> usize {
        self.offsets.len() * std::mem::size_of::<u32>()
            + self.neighbors.len() * std::mem::size_of::<u32>()
    }

    fn decode_into(&self, node: usize, out: &mut Vec<u32>) {
        out.clear();
        let start = self.offsets[node] as usize;
        let end = self.offsets[node + 1] as usize;
        out.extend_from_slice(&self.neighbors[start..end]);
    }
}
