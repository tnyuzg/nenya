//! Multilinear extension over BB4, evaluation-form.
//!
//! Mirror of `sumcheck-ring`'s `mle.rs`, with `GR32_128` swapped for `BB4`.
//! See that file for the convention/indexing notes.

use field::{AbstractField, BB4};

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

    /// Fold the highest-indexed variable at `r`, in place.
    /// `(lo, hi) ↦ lo + r·(hi − lo)`. Halves the table.
    pub fn fold_in_place(&mut self, r: BB4) {
        let m = self.evals.len() / 2;
        assert!(m > 0, "cannot fold a length-1 MLE");
        for i in 0..m {
            let lo = self.evals[i];
            let hi = self.evals[i + m];
            self.evals[i] = lo + r * (hi - lo);
        }
        self.evals.truncate(m);
    }

    pub fn final_eval(&self) -> BB4 {
        assert_eq!(self.evals.len(), 1, "MLE not fully folded");
        self.evals[0]
    }

    /// Convenience: zero-vector MLE of the given num_vars.
    pub fn zeros(num_vars: usize) -> Self {
        Self::new(vec![BB4::zero(); 1 << num_vars])
    }
}
