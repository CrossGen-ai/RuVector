//! Closure policies: how many posting lists each base vector gets replicated into.
//!
//! Given a sorted list of (centroid_id, distance) pairs (ascending by distance),
//! return the centroid ids the vector should be assigned to.

#[derive(Debug, Clone, Copy)]
pub enum PolicyKind {
    Single,
    FixedMulti(usize),
    Spann { epsilon: f32, cap: usize },
}

pub trait ClosurePolicy: Send + Sync {
    fn assign(&self, sorted: &[(usize, f32)]) -> Vec<usize>;
    fn kind(&self) -> PolicyKind;
}

pub struct SingleAssign;

impl ClosurePolicy for SingleAssign {
    fn assign(&self, sorted: &[(usize, f32)]) -> Vec<usize> {
        if sorted.is_empty() {
            return Vec::new();
        }
        vec![sorted[0].0]
    }
    fn kind(&self) -> PolicyKind {
        PolicyKind::Single
    }
}

pub struct FixedMultiAssign {
    pub k: usize,
}

impl ClosurePolicy for FixedMultiAssign {
    fn assign(&self, sorted: &[(usize, f32)]) -> Vec<usize> {
        sorted.iter().take(self.k).map(|(c, _)| *c).collect()
    }
    fn kind(&self) -> PolicyKind {
        PolicyKind::FixedMulti(self.k)
    }
}

/// SPANN boundary closure (Chen et al., NeurIPS 2021).
///
/// Replicate x into the i-th nearest centroid iff
///     dist(x, c_i) <= (1 + epsilon) * dist(x, c_1)
/// up to a hard cap of `cap` replicas.
///
/// This is the cheap, training-free flavor of SPANN's RNG-style closure.
/// It captures the boundary insight: points near a Voronoi cell boundary
/// are replicated; interior points are not.
pub struct SpannClosure {
    pub epsilon: f32,
    pub cap: usize,
}

impl ClosurePolicy for SpannClosure {
    fn assign(&self, sorted: &[(usize, f32)]) -> Vec<usize> {
        if sorted.is_empty() {
            return Vec::new();
        }
        let d1 = sorted[0].1.max(1e-12);
        let thresh = (1.0 + self.epsilon) * d1;
        let mut out = Vec::with_capacity(self.cap.min(sorted.len()));
        for (c, d) in sorted.iter().take(self.cap) {
            if *d <= thresh {
                out.push(*c);
            } else {
                break;
            }
        }
        out
    }
    fn kind(&self) -> PolicyKind {
        PolicyKind::Spann {
            epsilon: self.epsilon,
            cap: self.cap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(v: &[f32]) -> Vec<(usize, f32)> {
        v.iter().enumerate().map(|(i, d)| (i, *d)).collect()
    }

    #[test]
    fn single_returns_one() {
        let p = SingleAssign;
        assert_eq!(p.assign(&sorted(&[1.0, 2.0, 3.0])), vec![0]);
    }

    #[test]
    fn fixed_multi_takes_k() {
        let p = FixedMultiAssign { k: 2 };
        assert_eq!(p.assign(&sorted(&[1.0, 2.0, 3.0])), vec![0, 1]);
    }

    #[test]
    fn spann_replicates_near_boundary() {
        // d1=1.0, d2=1.05 → within (1+0.1)*1.0=1.1, included.
        // d3=1.5 → outside threshold, excluded.
        let p = SpannClosure { epsilon: 0.1, cap: 8 };
        assert_eq!(p.assign(&sorted(&[1.0, 1.05, 1.5])), vec![0, 1]);
    }

    #[test]
    fn spann_skips_interior() {
        // Only nearest is within epsilon — interior point.
        let p = SpannClosure { epsilon: 0.1, cap: 8 };
        assert_eq!(p.assign(&sorted(&[1.0, 5.0, 9.0])), vec![0]);
    }

    #[test]
    fn spann_respects_cap() {
        // All centroids equidistant → cap kicks in.
        let p = SpannClosure { epsilon: 1.0, cap: 2 };
        assert_eq!(p.assign(&sorted(&[1.0, 1.0, 1.0, 1.0])), vec![0, 1]);
    }
}
