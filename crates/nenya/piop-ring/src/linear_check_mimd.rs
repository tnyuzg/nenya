//! Degree-2 sumcheck for `∑_h M_row[h] · w(h)` where
//! `M_row[h] = ∑_y eq(r_target, y) · M[y, h]` and `M` is sparse and
//! `R_2 = GR(2^32, 128)`-valued (MIMD reduction, paper §3.2.2).
//!
//! Structurally identical to [`crate::linear_check_simd`]; the only
//! difference is the matvec step at the front, which uses full ring
//! multiplication on each `(eq_row[y], M[y, col])` pair instead of the
//! `mul_by_gr1` fast path. Once `M_row` is built the sumcheck loop is
//! the same `{s(0), s(1), s(∞)}`-form deg-2 prover (2 muls/pair).

use galois_ring::{Gr128, Limb};
use rayon::prelude::*;
use reduction::SparseMatrix;

use crate::eq::build_eq_table;
use crate::error::VerifyError;
use crate::fs::Challenger;
use crate::mle::Mle;

/// Proof of `sum = ∑_h M_row[h] · w(h)` where
/// `M_row[h] = ∑_y eq(r_target, y) · M[y, h]` and `M` is `R_2`-valued.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCheckMimdProof<L: Limb = u32> {
    pub claimed_sum: Gr128<L>,
    /// Round-`i` univariate `s_i(Z) = a0 + a1·Z + a2·Z²`, in coefficient form.
    /// Length = `num_vars`.
    pub round_polys: Vec<[Gr128<L>; 3]>,
    /// `M_row(challenges)`.
    pub final_m_row: Gr128<L>,
    /// `w(challenges)`.
    pub final_w: Gr128<L>,
}

impl<L: Limb> LinearCheckMimdProof<L> {
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

        // final_m_row: Gr128<L>
        for c in self.final_m_row.0.iter() {
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

/// Compute `M_row[col] = ∑_y eq_row[y] · M[y, col]` via full ring
/// multiplication. The output is a length-`m.cols` MLE.
///
/// Exposed publicly so callers (verifier, benches, snark-ring) can run
/// the same R_2 matvec.
pub fn build_m_row<L: Limb>(matrix: &SparseMatrix<Gr128<L>>, eq_row: &[Gr128<L>]) -> Mle<L> {
    assert_eq!(
        eq_row.len(),
        matrix.rows,
        "eq_row length must match matrix row count"
    );
    let mut m_row = vec![Gr128::<L>::ZERO; matrix.cols];
    for &(row, col, val) in matrix.triples.iter() {
        m_row[col] = m_row[col] + eq_row[row] * val;
    }
    Mle::new(m_row)
}

/// Run the deg-2 MIMD linear-check sumcheck prover.
///
/// Inputs:
/// - `matrix`: sparse `R_2`-valued matrix `M` (rows = "y", cols = "h"). Must
///   satisfy `matrix.cols == 1 << w.num_vars()` and
///   `matrix.rows == 1 << r_target.len()`.
/// - `w`: witness MLE in `n = log_2(matrix.cols)` variables.
/// - `r_target`: row-side challenge `r_target ∈ T_2^{log_num_rows}`.
/// - `claim`: the claimed value `∑_h M_row[h] · w[h]`.
///
/// Returns the proof and the challenge sequence (`r_1, …, r_n`, `r_1` first
/// folded — same convention as [`crate::Mle::fold_in_place`]).
pub fn prove_linear_check_mimd<L: Limb>(
    matrix: &SparseMatrix<Gr128<L>>,
    w: &Mle<L>,
    r_target: &[Gr128<L>],
    claim: Gr128<L>,
    challenger: &mut impl Challenger<L>,
) -> (LinearCheckMimdProof<L>, Vec<Gr128<L>>) {
    let n = w.num_vars();
    assert_eq!(
        matrix.cols,
        1usize << n,
        "matrix column count must match witness MLE size"
    );
    assert_eq!(
        matrix.rows,
        1usize << r_target.len(),
        "matrix row count must match r_target"
    );
    assert!(n > 0, "linear check on a 0-variable witness is meaningless");

    // Matvec: build M_row over R_2 directly (no u32 fast path).
    let eq_row = build_eq_table(r_target);
    let m_row = build_m_row(matrix, eq_row.evals());

    // Deg-2 sumcheck on M_row · w using {s(0), s(1), s(∞)} form.
    prove_linear_check_mimd_sumcheck_only(m_row, w.clone(), claim, challenger)
}

/// Sumcheck-only entry point: skips `build_eq_table` and the sparse matvec.
/// The caller passes a pre-built `m_row = ∑_y eq(r_target, y) · M[y, ·]`
/// (verifier-recomputable preprocessing — exclude when comparing pure
/// PIOP-prover cost).
pub fn prove_linear_check_mimd_sumcheck_only<L: Limb>(
    mut m_row: Mle<L>,
    mut w: Mle<L>,
    claim: Gr128<L>,
    challenger: &mut impl Challenger<L>,
) -> (LinearCheckMimdProof<L>, Vec<Gr128<L>>) {
    let n = w.num_vars();
    assert_eq!(m_row.num_vars(), n, "m_row / w num_vars mismatch");
    assert!(n > 0, "linear check on a 0-variable witness is meaningless");

    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);
    let mut c = claim;

    for _ in 0..n {
        let poly = compute_round_poly(&m_row, &w, c);
        // Per-round Fiat–Shamir: bind the round message before its challenge.
        challenger.observe(&poly);
        round_polys.push(poly);

        let r = challenger.sample();
        challenges.push(r);

        let [a0, a1, a2] = poly;
        c = a0 + r * (a1 + r * a2);

        m_row.fold_in_place(r);
        w.fold_in_place(r);
    }

    let proof = LinearCheckMimdProof {
        claimed_sum: claim,
        round_polys,
        final_m_row: m_row.final_eval(),
        final_w: w.final_eval(),
    };
    (proof, challenges)
}

/// Build the deg-2 round message in coefficient form via the
/// `{s(0), s(1), s(∞)}` basis. Identical to
/// [`crate::linear_check_simd::compute_round_poly`].
#[inline]
fn compute_round_poly<L: Limb>(f: &Mle<L>, g: &Mle<L>, c_prev: Gr128<L>) -> [Gr128<L>; 3] {
    let m = f.len() / 2;
    debug_assert_eq!(g.len(), f.len());

    let f_evals = f.evals();
    let g_evals = g.evals();

    // Per-pair contributions are independent; ring addition is associative+
    // commutative, so the parallel 2-tuple reduction is bit-identical to serial.
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

    let two_s0 = s0 + s0;
    let a1 = c_prev - two_s0 - sinf;
    [s0, a1, sinf]
}

/// Verify a MIMD linear-check proof.
///
/// On success returns the binding point `(r_1, …, r_n)` (sample order, MSB
/// first — same convention as the prover's challenge sequence).
///
/// The verifier accepts `proof.final_m_row` and `proof.final_w` at face
/// value; the caller is expected to plug in independent checks for these
/// (e.g. PCS openings) on top.
pub fn verify_linear_check_mimd<L: Limb>(
    proof: &LinearCheckMimdProof<L>,
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

        // Per-round Fiat–Shamir: observe before sampling (prover's order).
        challenger.observe(&[a0, a1, a2]);
        let r = challenger.sample();
        challenges.push(r);

        c = a0 + r * (a1 + r * a2);
    }

    if c != proof.final_m_row * proof.final_w {
        return Err(VerifyError::FinalCheckFailed);
    }
    Ok(challenges)
}
