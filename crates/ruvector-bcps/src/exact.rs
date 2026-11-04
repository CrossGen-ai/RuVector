use crate::ProbeSelector;

pub struct ExactL2Selector {
    centroids: Vec<Vec<f32>>,
    d: usize,
}

impl ExactL2Selector {
    pub fn new(centroids: Vec<Vec<f32>>) -> Self {
        let d = centroids[0].len();
        Self { centroids, d }
    }

    #[inline]
    fn l2(&self, q: &[f32], c: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..self.d {
            let dv = q[i] - c[i];
            s += dv * dv;
        }
        s
    }
}

impl ProbeSelector for ExactL2Selector {
    fn name(&self) -> &'static str { "exact-l2" }

    fn select(&self, query: &[f32], n: usize) -> Vec<(u32, f32)> {
        let mut dists: Vec<(u32, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i as u32, self.l2(query, c)))
            .collect();
        // partial sort via select_nth; stable top-n then sort.
        let n = n.min(dists.len());
        dists.select_nth_unstable_by(n.saturating_sub(1).max(0), |a, b| {
            a.1.partial_cmp(&b.1).unwrap()
        });
        dists.truncate(n);
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists
    }
}
