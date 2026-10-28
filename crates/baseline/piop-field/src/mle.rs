//! Multilinear extension over BB4, mirroring `piop_ring::Mle`.

use field::BB4;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct Mle {
    pub(crate) evals: Vec<BB4>,
}

impl Mle {
    pub fn new(evals: Vec<BB4>) -> Self {
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

    pub fn evals(&self) -> &[BB4] {
        &self.evals
    }

    /// Fold the highest-indexed variable at `r`. Halves the table.
    pub fn fold_in_place(&mut self, r: BB4) {
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

    pub fn final_eval(&self) -> BB4 {
        assert_eq!(self.evals.len(), 1, "MLE not fully folded");
        self.evals[0]
    }
}
