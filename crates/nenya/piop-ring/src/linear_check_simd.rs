//! Degree-2 sumcheck for `∑_h A_row[h] · w(h)` where
//! `A_row[h] = ∑_y eq(r_target, y) · A[y, h]`. The matrix `A` is sparse
//! and `R₀ = u32`-valued (SIMD reduction, paper §3.2.1).
//!
//! This is the **linear check** for the SIMD path. It has three structural
//! ingredients:
//!
//! 1. Build `eq_row = eq(r_target, ·)` as a length-`2^{log_rows}` table over
//!    `R₂`. (One call to [`crate::build_eq_table`].)
//! 2. **Sparse matvec** `A_row[col] = ∑_y eq_row[y] · A[y, col]` over the
//!    triple list of `A`. This is the only place the u32-ness of `A` shows
//!    up: each contribution is `eq_row[y].mul_by_gr1(A[y, col])`, which
//!    is ~15× faster than the generic schoolbook
//!    `eq_row[y] * GR32_128::from_gr1(A[y, col])` — see
//!    [`galois_ring::GR32_128::mul_by_gr1`].
//! 3. **Deg-2 sumcheck** on `∑_h A_row[h] · w(h)`, identical in structure
//!    to [`sumcheck_ring::prove`] **but with the {s(0), s(1), s(∞)} round
//!    message form**, which trades 1 mul/pair for 2 adds/pair (~25% faster
//!    on this hot loop, since adds are ~30× cheaper than muls on `R₂`).
//!
//! ## Per-pair inner-loop arithmetic
//!
//! Only `s(0)` and `s(∞)` are accumulated per pair; `s(1)` is never
//! computed explicitly — the linear coefficient is recovered once per
//! round from the running claim `c_prev = s(0) + s(1)`.
//!
//! ```text
//! da  = a_hi − a_lo                 // 1 sub
//! db  = b_hi − b_lo                 // 1 sub
//! s0   += a_lo · b_lo               // 1 mul (s evaluated at Z = 0)
//! sinf += da   · db                 // 1 mul (leading coefficient = s at ∞)
//! ```
//!
//! Total per pair: **2 muls** + 1 add + 2 subs. (Compare to the 4 muls/pair
//! of `sumcheck-ring::prover::compute_round_poly`.)
//!
//! ## Per-round conversion to coefficient form
//!
//! Once per round (not per pair), recover `a1` from `c_prev = s(0) + s(1)`:
//!
//! ```text
//! a0 = s0
//! a2 = sinf
//! a1 = c_prev − 2·s0 − sinf          // 2·x is just x + x, no division
//! ```
//!
//! All add/sub. No division by 2 — the `{0, 1, ∞}` change-of-basis matrix
//! has determinant `−1`, not `2`, so it works in `Z/2³²` where the
//! `{0, 1, 2}` basis would not.
//!
//! ## What the prover sends
//!
//! `[a0, a1, a2]` per round, plus `final_a_row` and `final_w`. The verifier
//! reconstructs `A_row` itself by doing the same matvec, then checks
//! `c_n == final_a_row · final_w` and the per-round consistency.

use galois_ring::{Gr128, Limb, GR32_128};
use rayon::prelude::*;

use crate::eq::build_eq_table;
use crate::error::VerifyError;
use crate::fs::Challenger;
use crate::mle::Mle;
use crate::sparse_matrix::SparseMatrix;

/// Proof of `sum = ∑_h A_row[h] · w(h)` where
/// `A_row[h] = ∑_y eq(r_target, y) · A[y, h]`.
///
/// `L` defaults to `u32` so the matrix-based [`prove_linear_check_simd`] entry
/// keeps its concrete signature; the genericized sumcheck path uses `<L>`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCheckSimdProof<L: Limb = u32> {
    pub claimed_sum: Gr128<L>,
    /// Round-`i` univariate `s_i(Z) = a0 + a1·Z + a2·Z²`, in coefficient form.
    /// Length = `num_vars`.
    pub round_polys: Vec<[Gr128<L>; 3]>,
    /// `A_row(challenges)`.
    pub final_a_row: Gr128<L>,
    /// `w(challenges)`.
    pub final_w: Gr128<L>,
}

impl<L: Limb> LinearCheckSimdProof<L> {
    pub fn num_rounds(&self) -> usize {
        self.round_polys.len()
    }

    /// Serialize proof to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let limb_bytes = (L::BITS / 8) as usize;

        // claimed_sum: Gr128<L>
        for c in self.claimed_sum.0.iter() {
            for w in c.0.iter() {
                buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
            }
        }

        // round_polys: Vec<[Gr128<L>; 3]>
        buf.extend_from_slice(&(self.round_polys.len() as u64).to_le_bytes());
        for poly in &self.round_polys {
            for gr in poly.iter() {
                for c in gr.0.iter() {
                    for w in c.0.iter() {
                        buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
                    }
                }
            }
        }

        // final_a_row: Gr128<L>
        for c in self.final_a_row.0.iter() {
            for w in c.0.iter() {
                buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
            }
        }

        // final_w: Gr128<L>
        for c in self.final_w.0.iter() {
            for w in c.0.iter() {
                buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
            }
        }

        buf
    }

    /// Actual serialized size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.serialize().len()
    }
}

/// Compute `A_row[col] = ∑_y eq_row[y] · A[y, col]` via the u32-aware
/// `mul_by_gr1` helper. The output is a length-`2^{log_num_cols}` MLE.
///
/// Exposed publicly so callers (verifier, benches, snark-ring) can run
/// the same R_0-aware matvec.
pub fn build_a_row(matrix: &SparseMatrix, eq_row: &[GR32_128]) -> Mle {
    assert_eq!(
        eq_row.len(),
        1usize << matrix.log_num_rows,
        "eq_row length must match matrix row count"
    );
    let num_cols = matrix.num_cols();
    let mut a_row = vec![GR32_128::ZERO; num_cols];
    for &(row, col, val) in matrix.entries.iter() {
        a_row[col] = a_row[col] + eq_row[row].mul_by_gr1(val);
    }
    Mle::new(a_row)
}

/// Run the deg-2 linear-check sumcheck prover.
///
/// Inputs:
/// - `matrix`: sparse `R₀`-valued matrix `A` (rows = "y", cols = "h"). Must
///   satisfy `matrix.log_num_cols == w.num_vars()`.
/// - `w`: witness MLE in `n = matrix.log_num_cols` variables.
/// - `r_target`: row-side challenge `r_target ∈ T_2^{log_num_rows}`.
/// - `claim`: the claimed value `∑_h A_row[h] · w[h]`.
///
/// Returns the proof and the challenge sequence (`r_1, …, r_n`, `r_1` first
/// folded — same convention as [`crate::Mle::fold_in_place`]).
pub fn prove_linear_check_simd(
    matrix: &SparseMatrix,
    w: &Mle,
    r_target: &[GR32_128],
    claim: GR32_128,
    challenger: &mut impl Challenger<u32>,
) -> (LinearCheckSimdProof, Vec<GR32_128>) {
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

    // Step 1+2: build eq_row over R₂, then matvec A_row[col] = Σ_y eq_row[y] · A[y, col].
    let eq_row = build_eq_table(r_target);
    let a_row = build_a_row(matrix, eq_row.evals());

    // Step 3: deg-2 sumcheck on A_row · w using {s(0), s(1), s(∞)} form.
    prove_linear_check_simd_sumcheck_only(a_row, w.clone(), claim, challenger)
}

/// Sumcheck-only entry point: skips `build_eq_table` and the sparse matvec.
/// The caller passes a pre-built `a_row = ∑_y eq(r_target, y) · A[y, ·]`
/// (the verifier independently recomputes the same `a_row` from the public
/// matrix and `r_target` during verification, so this prep is symmetric
/// preprocessing — exclude it when comparing pure PIOP-prover cost).
pub fn prove_linear_check_simd_sumcheck_only<L: Limb>(
    mut a_row: Mle<L>,
    mut w: Mle<L>,
    claim: Gr128<L>,
    challenger: &mut impl Challenger<L>,
) -> (LinearCheckSimdProof<L>, Vec<Gr128<L>>) {
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

        // Advance c via Horner: c_i = a0 + a1·r + a2·r².
        let [a0, a1, a2] = poly;
        c = a0 + r * (a1 + r * a2);

        a_row.fold_in_place(r);
        w.fold_in_place(r);
    }

    let proof = LinearCheckSimdProof {
        claimed_sum: claim,
        round_polys,
        final_a_row: a_row.final_eval(),
        final_w: w.final_eval(),
    };
    (proof, challenges)
}

/// Build the deg-2 round message in coefficient form via the
/// `{s(0), s(1), s(∞)}` basis, saving one mul per fold-pair vs. the
/// `sumcheck-ring` coefficient-form prover.
///
/// `c_prev` is the running sumcheck claim entering this round, equal to
/// `s(0) + s(1)` if the prover has been honest. We use it to recover `a1`
/// without ever computing `s(1)` explicitly:
/// `a1 = s(1) − s(0) − s(∞) = (c_prev − s(0)) − s(0) − s(∞) = c_prev − 2·s(0) − s(∞)`.
/// Skipping `s(1)` is exactly what drops per-pair cost from 3 muls to 2.
///
/// Soundness note: the `c_prev` shortcut *trusts* the verifier-side
/// `c_prev`. Since the verifier independently computes `c_prev` and checks
/// the round message against it, this is fine — the prover cannot game
/// `c_prev` without breaking the round-`i−1` check.
#[inline]
fn compute_round_poly<L: Limb>(f: &Mle<L>, g: &Mle<L>, c_prev: Gr128<L>) -> [Gr128<L>; 3] {
    let m = f.len() / 2;
    debug_assert_eq!(g.len(), f.len());

    let f_evals = f.evals();
    let g_evals = g.evals();

    // Per-pair contributions are independent; ring addition is associative+
    // commutative, so the parallel 2-tuple reduction is bit-identical to the
    // serial accumulation regardless of thread count.
    let zero = Gr128::<L>::ZERO;
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

    // Coefficient form:
    //   a0 = s(0)            = s0
    //   a2 = leading coef    = sinf
    //   a1 = s(1) − s(0) − sinf
    //      = (c_prev − s(0)) − s(0) − sinf      // because s(0)+s(1)=c_prev
    //      = c_prev − 2·s(0) − sinf
    let two_s0 = s0 + s0;
    let a1 = c_prev - two_s0 - sinf;
    [s0, a1, sinf]
}

/// Verify a SIMD linear-check proof.
///
/// On success returns the binding point `(r_1, …, r_n)` (sample order, MSB
/// first — same convention as the prover's challenge sequence).
///
/// The verifier accepts `proof.final_a_row` and `proof.final_w` at face
/// value; the caller is expected to plug in independent checks for these
/// (e.g. PCS openings) on top.
pub fn verify_linear_check_simd<L: Limb>(
    proof: &LinearCheckSimdProof<L>,
    challenger: &mut impl Challenger<L>,
) -> Result<Vec<Gr128<L>>, VerifyError> {
    let n = proof.round_polys.len();
    assert!(n > 0, "linear-check proof has zero rounds");

    let mut c = proof.claimed_sum;
    let mut challenges = Vec::with_capacity(n);

    for (round, [a0, a1, a2]) in proof.round_polys.iter().copied().enumerate() {
        // s_i(0) + s_i(1) = 2·a0 + a1 + a2.
        let two_a0 = a0 + a0;
        let lhs = two_a0 + a1 + a2;
        if lhs != c {
            return Err(VerifyError::RoundCheckFailed { round });
        }

        // Per-round Fiat–Shamir: observe this round's message (same order as
        // the prover) before sampling its challenge.
        challenger.observe(&[a0, a1, a2]);
        let r = challenger.sample();
        challenges.push(r);

        // c_i = a0 + a1·r + a2·r² — Horner.
        c = a0 + r * (a1 + r * a2);
    }

    if c != proof.final_a_row * proof.final_w {
        return Err(VerifyError::FinalCheckFailed);
    }
    Ok(challenges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use galois_ring::{R2TeichmullerSampler, GR32_1, GR32_4};

    fn rand_gr128(seed: u64) -> GR32_128 {
        let mut s: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 32) as u32
        };
        let mut arr = [GR32_4::ZERO; 32];
        for slot in arr.iter_mut() {
            *slot = GR32_4([next(), next(), next(), next()]);
        }
        GR32_128(arr)
    }

    fn rand_mle(num_vars: usize, seed: u64) -> Mle {
        let len = 1usize << num_vars;
        let mut evals = Vec::with_capacity(len);
        for i in 0..len {
            evals.push(rand_gr128(seed.wrapping_add(i as u64)));
        }
        Mle::new(evals)
    }

    fn rand_target(num_vars: usize, seed: u64) -> Vec<GR32_128> {
        (0..num_vars)
            .map(|i| rand_gr128(seed.wrapping_add(0xCAFE * i as u64 + 0xBEEF)))
            .collect()
    }

    fn rand_sparse_matrix(log_rows: usize, log_cols: usize, nnz: usize, seed: u64) -> SparseMatrix {
        let mut s = seed | 1;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 32) as u32
        };
        let nrows = 1usize << log_rows;
        let ncols = 1usize << log_cols;
        let mut m = SparseMatrix::new(log_rows, log_cols);
        for _ in 0..nnz {
            let r = (next() as usize) % nrows;
            let c = (next() as usize) % ncols;
            let v = GR32_1(next());
            m.push(r, c, v);
        }
        m
    }

    /// Direct: ∑_h A_row[h] · w[h].
    fn direct_claim(matrix: &SparseMatrix, w: &Mle, r_target: &[GR32_128]) -> GR32_128 {
        let eq = build_eq_table(r_target);
        let a_row = build_a_row(matrix, eq.evals());
        let mut acc = GR32_128::ZERO;
        for i in 0..w.len() {
            acc = acc + a_row.evals()[i] * w.evals()[i];
        }
        acc
    }

    #[test]
    fn build_a_row_matches_naive() {
        // Cross-check the matvec against a naive R₂-only implementation that
        // lifts each u32 entry to GR32_128 and uses the generic schoolbook.
        let log_rows = 3;
        let log_cols = 3;
        let m = rand_sparse_matrix(log_rows, log_cols, 12, 0xABCD);
        let r_target = rand_target(log_rows, 0xDEAD);
        let eq = build_eq_table(&r_target);

        let fast = build_a_row(&m, eq.evals());

        let mut naive = vec![GR32_128::ZERO; 1 << log_cols];
        for &(r, c, v) in &m.entries {
            naive[c] = naive[c] + eq.evals()[r] * GR32_128::from_gr1(v);
        }
        assert_eq!(fast.evals(), naive.as_slice());
    }

    #[test]
    fn round_poly_consistency_first_round() {
        // s(0) + s(1) must equal the claim entering the round.
        let log_rows = 3;
        let log_cols = 3;
        let m = rand_sparse_matrix(log_rows, log_cols, 12, 1);
        let w = rand_mle(log_cols, 2);
        let r_target = rand_target(log_rows, 3);
        let claim = direct_claim(&m, &w, &r_target);

        let eq = build_eq_table(&r_target);
        let a_row = build_a_row(&m, eq.evals());
        let [a0, a1, a2] = compute_round_poly(&a_row, &w, claim);

        let s0 = a0;
        let s1 = a0 + a1 + a2;
        assert_eq!(s0 + s1, claim);
    }

    #[test]
    fn prove_emits_consistent_round_polys() {
        let log_rows = 4;
        let log_cols = 4;
        let m = rand_sparse_matrix(log_rows, log_cols, 24, 11);
        let w = rand_mle(log_cols, 22);
        let r_target = rand_target(log_rows, 33);
        let claim = direct_claim(&m, &w, &r_target);

        let mut sampler = R2TeichmullerSampler::new(0xFEEDBEEF);
        let (proof, challenges) = prove_linear_check_simd(&m, &w, &r_target, claim, &mut sampler);

        assert_eq!(proof.round_polys.len(), log_cols);

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
