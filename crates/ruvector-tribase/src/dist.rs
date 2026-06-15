//! L2 distance utilities. Squared distances used internally; sqrt only for bounds.

#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    let mut i = 0;
    let n = a.len();
    // Unroll 4-way
    while i + 4 <= n {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        s += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        i += 4;
    }
    while i < n {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    s
}

#[inline]
pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    sq_l2(a, b).sqrt()
}

/// Top-k min-heap (largest-on-top) over (sq_distance, id).
/// `peek` returns the current k-th-best (largest of the kept) so callers can
/// prune via `d > heap.peek()`.
#[derive(Clone)]
pub struct TopK {
    k: usize,
    /// Max-heap stored as Vec; we maintain it manually since std BinaryHeap
    /// is max-heap on Ord but we want a max-heap keyed on f32.
    buf: Vec<(f32, u32)>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            buf: Vec::with_capacity(k + 1),
        }
    }

    #[inline]
    pub fn worst(&self) -> f32 {
        if self.buf.len() < self.k {
            f32::INFINITY
        } else {
            self.buf[0].0
        }
    }

    #[inline]
    pub fn push(&mut self, d: f32, id: u32) {
        if self.buf.len() < self.k {
            self.buf.push((d, id));
            self.sift_up(self.buf.len() - 1);
        } else if d < self.buf[0].0 {
            self.buf[0] = (d, id);
            self.sift_down(0);
        }
    }

    fn sift_up(&mut self, mut i: usize) {
        while i > 0 {
            let p = (i - 1) / 2;
            if self.buf[i].0 > self.buf[p].0 {
                self.buf.swap(i, p);
                i = p;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.buf.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut m = i;
            if l < n && self.buf[l].0 > self.buf[m].0 {
                m = l;
            }
            if r < n && self.buf[r].0 > self.buf[m].0 {
                m = r;
            }
            if m == i {
                break;
            }
            self.buf.swap(i, m);
            i = m;
        }
    }

    pub fn into_sorted(mut self) -> Vec<(f32, u32)> {
        self.buf.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        self.buf
    }
}
