//! Bidirectional graph representation.

/// Flat, cache-friendly graph. Neighbors stored contiguously.
#[derive(Debug, Clone)]
pub struct BidirGraph {
    pub n: usize,
    pub m_max: usize,
    /// Forward adjacency, length n * m_max. Entries == usize::MAX are unused.
    pub adj: Vec<usize>,
    /// Reverse-neighbor sidecar, at most `r_max` entries per node.
    pub rev: Option<RevAdj>,
    /// Entry point for beam search.
    pub entry: usize,
}

#[derive(Debug, Clone)]
pub struct RevAdj {
    pub r_max: usize,
    /// length n * r_max. Entries == usize::MAX are unused.
    pub rev: Vec<usize>,
}

impl BidirGraph {
    pub fn empty(n: usize, m_max: usize) -> Self {
        Self {
            n,
            m_max,
            adj: vec![usize::MAX; n * m_max],
            rev: None,
            entry: 0,
        }
    }

    #[inline]
    pub fn neighbors(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        let base = u * self.m_max;
        self.adj[base..base + self.m_max]
            .iter()
            .copied()
            .take_while(|&x| x != usize::MAX)
    }

    #[inline]
    pub fn rev_neighbors(&self, u: usize) -> impl Iterator<Item = usize> + '_ {
        let it: Box<dyn Iterator<Item = usize>> = match &self.rev {
            Some(r) => {
                let base = u * r.r_max;
                Box::new(
                    r.rev[base..base + r.r_max]
                        .iter()
                        .copied()
                        .take_while(|&x| x != usize::MAX),
                )
            }
            None => Box::new(std::iter::empty()),
        };
        it
    }

    /// Push a neighbor at position `slot` (must be < m_max).
    pub fn set_neighbor(&mut self, u: usize, slot: usize, v: usize) {
        self.adj[u * self.m_max + slot] = v;
    }

    /// Compute in-degrees over the current forward adjacency.
    pub fn indegrees(&self) -> Vec<u32> {
        let mut deg = vec![0u32; self.n];
        for u in 0..self.n {
            for v in self.neighbors(u) {
                deg[v] += 1;
            }
        }
        deg
    }

    /// Materialise reverse adjacency: for each node `v`, record the top-r_max
    /// in-neighbors (by their forward slot order — cheap deterministic pick).
    pub fn build_reverse(&mut self, r_max: usize) {
        let mut rev = vec![usize::MAX; self.n * r_max];
        let mut fill = vec![0usize; self.n];
        for u in 0..self.n {
            for v in self.neighbors(u) {
                let idx = fill[v];
                if idx < r_max {
                    rev[v * r_max + idx] = u;
                    fill[v] = idx + 1;
                }
            }
        }
        self.rev = Some(RevAdj { r_max, rev });
    }

    /// Symmetrise the forward adjacency: for every edge u->v, ensure v->u is
    /// present, then re-apply HNSW diversity pruning to `Adj[v]` so we don't
    /// destroy long-range structure by flooding v with clique-mates.
    pub fn symmetrize(&mut self, vectors: &[Vec<f32>]) {
        let n = self.n;
        let m_max = self.m_max;
        // Collect reverse-adjacency requests.
        let mut want: Vec<Vec<usize>> = vec![Vec::new(); n];
        for u in 0..n {
            for v in self.neighbors(u) {
                want[v].push(u);
            }
        }
        for v in 0..n {
            let mut union: Vec<usize> = self.neighbors(v).collect();
            for &u in &want[v] {
                if !union.contains(&u) && u != v {
                    union.push(u);
                }
            }
            let mut cand: Vec<(f32, usize)> =
                union.iter().map(|&w| (l2_sq(&vectors[v], &vectors[w]), w)).collect();
            cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            // Diversity heuristic.
            let mut selected: Vec<(f32, usize)> = Vec::with_capacity(m_max);
            for &(d_vw, w) in &cand {
                if selected.len() >= m_max {
                    break;
                }
                let keep = selected.iter().all(|&(_, x)| l2_sq(&vectors[w], &vectors[x]) >= d_vw);
                if keep {
                    selected.push((d_vw, w));
                }
            }
            if selected.len() < m_max {
                for &(d, w) in &cand {
                    if selected.len() >= m_max {
                        break;
                    }
                    if !selected.iter().any(|(_, x)| *x == w) {
                        selected.push((d, w));
                    }
                }
            }
            let base = v * m_max;
            for slot in 0..m_max {
                self.adj[base + slot] = selected.get(slot).map(|x| x.1).unwrap_or(usize::MAX);
            }
        }
    }
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}
