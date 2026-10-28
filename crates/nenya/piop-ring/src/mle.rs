//! Multilinear extension over `R₂ = GR(2^n, 128)`, generic over the base-limb
//! width `L`.

use galois_ring::{Gr128, Limb};
use rayon::prelude::*;

/// `L` defaults to `u32` so the MIMD/general checks (still `u32`-pinned) can
/// keep writing the bare `Mle` while the genericized SIMD path uses `Mle<L>`.
#[derive(Clone, Debug)]
pub struct Mle<L: Limb = u32> {
    pub(crate) evals: Vec<Gr128<L>>,
}

impl<L: Limb> Mle<L> {
    /// Build an MLE from its `2^n` Boolean-hypercube evaluations.
    pub fn new(evals: Vec<Gr128<L>>) -> Self {
        assert!(
            evals.len().is_power_of_two() && !evals.is_empty(),
            "MLE length must be a positive power of two"
        );
        Self { evals }
    }

    pub fn num_vars(&self) -> usize {
        self.evals.len().trailing_zeros() as usize
    }

    pub fn len(&self) -> usize {
        self.evals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.evals.is_empty()
    }

    pub fn evals(&self) -> &[Gr128<L>] {
        &self.evals
    }

    /// Fold the highest-indexed variable at `r`, in place.
    /// `(lo, hi) ↦ lo + r·(hi − lo)`. Halves the table.
    pub fn fold_in_place(&mut self, r: Gr128<L>) {
        let m = self.evals.len() / 2;
        assert!(m > 0, "cannot fold a length-1 MLE");
        // `lo[i] ← lo[i] + r·(hi[i] − lo[i])`: each output index is independent,
        // so this parallelizes by disjoint write (bit-identical to the serial fold).
        let (lo, hi) = self.evals.split_at_mut(m);
        lo.par_iter_mut()
            .zip(hi.par_iter())
            .for_each(|(lo_i, &hi_i)| {
                *lo_i = *lo_i + r * (hi_i - *lo_i);
            });
        self.evals.truncate(m);
    }

    pub fn final_eval(&self) -> Gr128<L> {
        assert_eq!(self.evals.len(), 1, "MLE not fully folded");
        self.evals[0]
    }
}
