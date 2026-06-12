//! Multilinear extension over `GR32_128`, evaluation-form.
//!
//! An MLE in `n` variables is stored as a `Vec<GR32_128>` of length `2^n`,
//! indexed by Boolean assignments `(x_1, …, x_n)` interpreted as `x_1·2^{n-1}
//! + … + x_n·2^0` (so the *last* variable is the *low* bit). This matches the
//! convention used by `slop-multilinear::Mle` and most sumcheck literature.
//!
//! The protocol's per-round prover hot path is [`Mle::fold_in_place`]: given a
//! challenge `r`, replace each pair `(lo, hi)` of consecutive entries with
//! `lo + r·(hi − lo)`, halving the table. After `n` folds the table has length
//! 1 and holds `f(r_n, …, r_1)`.

use galois_ring::GR32_128;

#[derive(Clone, Debug)]
pub struct Mle {
    /// Length is always a power of two; entries are evaluations on the
    /// Boolean hypercube of size `2^num_vars`.
    pub(crate) evals: Vec<GR32_128>,
}

impl Mle {
    /// Build an MLE from its 2ⁿ Boolean-hypercube evaluations.
    pub fn new(evals: Vec<GR32_128>) -> Self {
        assert!(
            evals.len().is_power_of_two() && !evals.is_empty(),
            "MLE length must be a positive power of two"
        );
        Self { evals }
    }

    /// Number of variables `n`. Equal to `log2(self.len())`.
    pub fn num_vars(&self) -> usize {
        self.evals.len().trailing_zeros() as usize
    }

    /// Number of stored evaluations (= 2^num_vars).
    pub fn len(&self) -> usize {
        self.evals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.evals.is_empty()
    }

    /// Read-only view of the underlying eval table.
    pub fn evals(&self) -> &[GR32_128] {
        &self.evals
    }

    /// Fold the highest-indexed variable at `r`, in place.
    ///
    /// On entry the table has length `2m`; on exit it has length `m`. The
    /// pair `(evals[i], evals[i + m])` becomes `evals[i] + r·(evals[i+m] −
    /// evals[i])`. Geometrically, we are restricting `f(x_1, …, x_n)` to the
    /// affine line `x_n = r`, leaving an MLE in `n−1` variables.
    pub fn fold_in_place(&mut self, r: GR32_128) {
        let m = self.evals.len() / 2;
        assert!(m > 0, "cannot fold a length-1 MLE");
        for i in 0..m {
            let lo = self.evals[i];
            let hi = self.evals[i + m];
            self.evals[i] = lo + r * (hi - lo);
        }
        self.evals.truncate(m);
    }

    /// After all folds the table is length 1; return that final evaluation.
    pub fn final_eval(&self) -> GR32_128 {
        assert_eq!(self.evals.len(), 1, "MLE not fully folded");
        self.evals[0]
    }
}
