use std::cmp::Ordering;

#[derive(Copy, Clone, Debug)]
pub struct Hit {
    pub id: u32,
    pub dist: f32,
}

/// Fixed-size max-heap over (dist, id) — root is the current worst (largest dist).
pub struct TopK {
    k: usize,
    heap: Vec<Hit>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self { k, heap: Vec::with_capacity(k + 1) }
    }

    #[inline]
    pub fn len(&self) -> usize { self.heap.len() }

    #[inline]
    pub fn is_full(&self) -> bool { self.heap.len() >= self.k }

    /// Current worst distance (K-th), or +inf if not full.
    #[inline]
    pub fn threshold(&self) -> f32 {
        if self.heap.len() < self.k {
            f32::INFINITY
        } else {
            self.heap[0].dist
        }
    }

    #[inline]
    pub fn push(&mut self, h: Hit) {
        if self.heap.len() < self.k {
            self.heap.push(h);
            self.sift_up(self.heap.len() - 1);
        } else if h.dist < self.heap[0].dist {
            self.heap[0] = h;
            self.sift_down(0);
        }
    }

    pub fn into_sorted_vec(mut self) -> Vec<Hit> {
        self.heap.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        self.heap
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if self.heap[i].dist > self.heap[p].dist {
                self.heap.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.heap.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut m = i;
            if l < n && self.heap[l].dist > self.heap[m].dist { m = l; }
            if r < n && self.heap[r].dist > self.heap[m].dist { m = r; }
            if m == i { break; }
            self.heap.swap(i, m);
            i = m;
        }
    }
}
