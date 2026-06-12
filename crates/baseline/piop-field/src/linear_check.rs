//! Field-side linear-check sumcheck. Mirror of [`piop_ring::linear_check`].
//!
//! Differences from the ring side:
//!
//! - Matrix entries are full BB4 elements, not u32-lifted "small" elements.
//!   The matvec uses the generic `*` rather than a `mul_by_*` specialization,
//!   because none exists at the field level.
//! - The deg-2 round message *still* uses the `{s(0), s(1), s(∞)}` form
//!   with `c_prev`-based recovery of `a1` (2 muls per fold-pair), since
//!   that trick is purely algebraic and works in any commutative ring/field.

use field::{AbstractField, BB4};
use rayon::prelude::*;

use crate::eq::build_eq_table;
use crate::fs::Challenger;
use crate::mle::Mle;
use crate::proof::LinearCheckProof;
use crate::sparse_matrix::SparseMatrix;

pub(crate) fn build_a_row(matrix: &SparseMatrix, eq_row: &[BB4]) -> Mle {
    assert_eq!(
        eq_row.len(),
        1usize << matrix.log_num_rows,
        "eq_row length must match matrix row count"
    );
    let num_cols = matrix.num_cols();
    let mut a_row = vec![BB4::zero(); num_cols];
    for &(row, col, val) in matrix.entries.iter() {
        a_row[col] += eq_row[row] * val;
    }
    Mle::new(a_row)
}

pub fn prove_linear_check(
    matrix: &SparseMatrix,
    w: &Mle,
    r_target: &[BB4],
    claim: BB4,
    challenger: &mut impl Challenger,
) -> (LinearCheckProof, Vec<BB4>) {
    let n = w.num_vars();
    assert_eq!(
        matrix.log_num_cols, n,
        "matrix column count must match witness MLE size"
    );
    assert_eq!(
        r_target.len(),
        matrix.log_num_rows,
        "r_target must have one entry per matrix-row variable"
    );
    assert!(n > 0, "linear check on a 0-variable witness is meaningless");

    let eq_row = build_eq_table(r_target);
    let a_row = build_a_row(matrix, eq_row.evals());
    prove_linear_check_sumcheck_only(a_row, w.clone(), claim, challenger)
}

/// Sumcheck-only entry: the caller supplies a pre-built row MLE `a_row`
/// (e.g. the γ-combined `A_row(s1) + γ·B_row(s1) + γ²·C_row(r)` used by the
/// SNARK). Same `{0,1,∞}`-form deg-2 prover. Verifier-side is the existing
/// [`crate::verifier::verify_linear_check`] (already matrix-free).
pub fn prove_linear_check_sumcheck_only(
    mut a_row: Mle,
    mut w: Mle,
    claim: BB4,
    challenger: &mut impl Challenger,
) -> (LinearCheckProof, Vec<BB4>) {
    let n = w.num_vars();
    assert_eq!(a_row.num_vars(), n, "a_row / w num_vars mismatch");
    assert!(n > 0, "linear check on a 0-variable witness is meaningless");

    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);
    let mut c = claim;

    for _ in 0..n {
        let poly = compute_round_poly(&a_row, &w, c);
        // Per-round Fiat–Shamir: bind the round message before its challenge.
        challenger.observe(&poly);
        round_polys.push(poly);

        let r = challenger.sample();
        challenges.push(r);

        let [a0, a1, a2] = poly;
        c = a0 + r * (a1 + r * a2);

        a_row.fold_in_place(r);
        w.fold_in_place(r);
    }

    let proof = LinearCheckProof {
        claimed_sum: claim,
        round_polys,
        final_a_row: a_row.final_eval(),
        final_w: w.final_eval(),
    };
    (proof, challenges)
}

#[inline]
fn compute_round_poly(f: &Mle, g: &Mle, c_prev: BB4) -> [BB4; 3] {
    let m = f.len() / 2;
    debug_assert_eq!(g.len(), f.len());

    let f_evals = f.evals();
    let g_evals = g.evals();

    // Per-pair contributions are independent; field addition is associative+
    // commutative, so the parallel 2-tuple reduction is bit-identical to serial.
    let zero = BB4::zero();
    let (s0, sinf) = (0..m)
        .into_par_iter()
        .map(|i| {
            let f_lo = f_evals[i];
            let f_hi = f_evals[i + m];
            let g_lo = g_evals[i];
            let g_hi = g_evals[i + m];

            let df = f_hi - f_lo;
            let dg = g_hi - g_lo;

            (f_lo * g_lo, df * dg)
        })
        .reduce(|| (zero, zero), |x, y| (x.0 + y.0, x.1 + y.1));

    let two_s0 = s0 + s0;
    let a1 = c_prev - two_s0 - sinf;
    [s0, a1, sinf]
}

#[cfg(test)]
mod tests {
    use super::*;
    use field::{AbstractExtensionField, BB, BB4UniformSampler};

    fn rand_bb4(state: &mut u64) -> BB4 {
        let mut next = || {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*state >> 32) as u32
        };
        let bases: [BB; 4] = [
            BB::from_canonical_u32(next() % 0x78000001),
            BB::from_canonical_u32(next() % 0x78000001),
            BB::from_canonical_u32(next() % 0x78000001),
            BB::from_canonical_u32(next() % 0x78000001),
        ];
        <BB4 as AbstractExtensionField<BB>>::from_base_slice(&bases)
    }

    fn rand_mle(num_vars: usize, state: &mut u64) -> Mle {
        let len = 1usize << num_vars;
        let mut evals = Vec::with_capacity(len);
        for _ in 0..len {
            evals.push(rand_bb4(state));
        }
        Mle::new(evals)
    }

    fn rand_target(num_vars: usize, state: &mut u64) -> Vec<BB4> {
        (0..num_vars).map(|_| rand_bb4(state)).collect()
    }

    fn rand_sparse(
        log_rows: usize,
        log_cols: usize,
        nnz_per_row: usize,
        state: &mut u64,
    ) -> SparseMatrix {
        let nrows = 1usize << log_rows;
        let ncols = 1usize << log_cols;
        let mut m = SparseMatrix::new(log_rows, log_cols);
        for r in 0..nrows {
            for _ in 0..nnz_per_row {
                let mut next = || {
                    *state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (*state >> 32) as u32
                };
                let c = (next() as usize) % ncols;
                let v = rand_bb4(state);
                m.push(r, c, v);
            }
        }
        m
    }

    fn direct_claim(matrix: &SparseMatrix, w: &Mle, r_target: &[BB4]) -> BB4 {
        let eq = build_eq_table(r_target);
        let a_row = build_a_row(matrix, eq.evals());
        let mut acc = BB4::zero();
        for i in 0..w.len() {
            acc += a_row.evals()[i] * w.evals()[i];
        }
        acc
    }

    #[test]
    fn prove_emits_consistent_round_polys() {
        let log_rows = 4;
        let log_cols = 4;
        let mut state = 0xCAFE_u64;
        let m = rand_sparse(log_rows, log_cols, 3, &mut state);
        let w = rand_mle(log_cols, &mut state);
        let r_target = rand_target(log_rows, &mut state);
        let claim = direct_claim(&m, &w, &r_target);

        let mut sampler = BB4UniformSampler::new(0xFEEDBEEF);
        let (proof, challenges) = prove_linear_check(&m, &w, &r_target, claim, &mut sampler);

        let mut c = proof.claimed_sum;
        for (round, [a0, a1, a2]) in proof.round_polys.iter().copied().enumerate() {
            let s0 = a0;
            let s1 = a0 + a1 + a2;
            assert_eq!(s0 + s1, c, "round {round} consistency");
            let r = challenges[round];
            c = a0 + r * (a1 + r * a2);
        }
        assert_eq!(c, proof.final_a_row * proof.final_w);
    }
}
