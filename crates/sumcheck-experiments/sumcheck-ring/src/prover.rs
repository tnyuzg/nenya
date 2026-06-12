//! Sumcheck provers over R₂ (deg-2 and deg-3 single product).
//!
//! Both use **monomial-coefficient form** round messages and the standard
//! Boolean-hypercube sumcheck round structure. These are reference
//! implementations — not the fastest possible. See
//! [`piop-ring`](../piop_ring/index.html) for production-grade specialized
//! provers (`{0, 1, ∞}` form for deg-2; Gruen's eq-MLE specialization for
//! deg-3).

use galois_ring::{R2TeichmullerSampler, GR32_128};

use crate::mle::Mle;
use crate::proof::{SumcheckProof, SumcheckProofDeg2};

// -----------------------------------------------------------------------
// Deg-3 single product: ∑ f1(h)·f2(h)·f3(h)
// -----------------------------------------------------------------------

/// Run the deg-3 sumcheck prover for `claim = ∑_h f1(h)·f2(h)·f3(h)`.
///
/// All three MLEs must share `num_vars > 0`. Returns the proof and the
/// random challenge sequence in sample order (`(r_1, …, r_n)`).
///
/// **Lock-step sampler.** Verification requires an `R2TeichmullerSampler`
/// with the same seed.
pub fn prove(
    f1: &Mle,
    f2: &Mle,
    f3: &Mle,
    claim: GR32_128,
    sampler: &mut R2TeichmullerSampler,
) -> (SumcheckProof, Vec<GR32_128>) {
    let n = f1.num_vars();
    assert_eq!(f2.num_vars(), n, "f1, f2 must have the same num_vars");
    assert_eq!(f3.num_vars(), n, "f1, f3 must have the same num_vars");
    assert!(n > 0, "sumcheck on a 0-variable MLE is meaningless");

    let mut f1 = f1.clone();
    let mut f2 = f2.clone();
    let mut f3 = f3.clone();

    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for _ in 0..n {
        let poly = compute_round_poly_deg3(&f1, &f2, &f3);
        round_polys.push(poly);

        let r = sampler.sample();
        challenges.push(r);

        f1.fold_in_place(r);
        f2.fold_in_place(r);
        f3.fold_in_place(r);
    }

    (
        SumcheckProof {
            claimed_sum: claim,
            round_polys,
            final_f1: f1.final_eval(),
            final_f2: f2.final_eval(),
            final_f3: f3.final_eval(),
        },
        challenges,
    )
}

/// Build the deg-3 round univariate `[a0, a1, a2, a3]` using shared
/// sub-products. Per fold-pair: **10 muls + ~6 adds + 3 subs**.
///
/// Exactly the inner-loop shape of
/// `piop-ring::mult_check::compute_round_poly` — that's the point of
/// keeping the two in lock-step.
#[inline]
fn compute_round_poly_deg3(f1: &Mle, f2: &Mle, f3: &Mle) -> [GR32_128; 4] {
    let m = f1.len() / 2;
    debug_assert_eq!(f2.len(), f1.len());
    debug_assert_eq!(f3.len(), f1.len());

    let e1 = f1.evals();
    let e2 = f2.evals();
    let e3 = f3.evals();

    let mut a0 = GR32_128::ZERO;
    let mut a1 = GR32_128::ZERO;
    let mut a2 = GR32_128::ZERO;
    let mut a3 = GR32_128::ZERO;

    for x in 0..m {
        let f1_lo = e1[x];
        let f1_hi = e1[x + m];
        let f2_lo = e2[x];
        let f2_hi = e2[x + m];
        let f3_lo = e3[x];
        let f3_hi = e3[x + m];

        let df1 = f1_hi - f1_lo;
        let df2 = f2_hi - f2_lo;
        let df3 = f3_hi - f3_lo;

        let p_lolo = f1_lo * f2_lo;
        let p_lohi = f1_lo * df2;
        let p_hilo = df1 * f2_lo;
        let p_hihi = df1 * df2;

        a0 = a0 + p_lolo * f3_lo;
        a1 = a1 + (p_lohi + p_hilo) * f3_lo + p_lolo * df3;
        a2 = a2 + p_hihi * f3_lo + (p_lohi + p_hilo) * df3;
        a3 = a3 + p_hihi * df3;
    }

    [a0, a1, a2, a3]
}

// -----------------------------------------------------------------------
// Deg-2 single product: ∑ f1(h)·f2(h)
// -----------------------------------------------------------------------

/// Run the deg-2 sumcheck prover for `claim = ∑_h f1(h)·f2(h)`.
///
/// Reference baseline for `piop-ring::linear_check`'s inner sumcheck
/// (which uses `{0, 1, ∞}` form and recovers `a1` from the running claim;
/// that's 2 muls/pair vs the 4 muls/pair here). Emits round messages in
/// the same coefficient form, so the two should agree round-by-round.
pub fn prove_deg2(
    f1: &Mle,
    f2: &Mle,
    claim: GR32_128,
    sampler: &mut R2TeichmullerSampler,
) -> (SumcheckProofDeg2, Vec<GR32_128>) {
    let n = f1.num_vars();
    assert_eq!(f2.num_vars(), n, "f1, f2 must have the same num_vars");
    assert!(n > 0, "sumcheck on a 0-variable MLE is meaningless");

    let mut f1 = f1.clone();
    let mut f2 = f2.clone();

    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for _ in 0..n {
        let poly = compute_round_poly_deg2(&f1, &f2);
        round_polys.push(poly);

        let r = sampler.sample();
        challenges.push(r);

        f1.fold_in_place(r);
        f2.fold_in_place(r);
    }

    (
        SumcheckProofDeg2 {
            claimed_sum: claim,
            round_polys,
            final_f1: f1.final_eval(),
            final_f2: f2.final_eval(),
        },
        challenges,
    )
}

/// Build the deg-2 round univariate `[a0, a1, a2]` directly. Per fold-pair:
/// **4 muls + 4 adds + 2 subs**. The `piop-ring::linear_check` specialization
/// shaves this to 2 muls/pair via `{0, 1, ∞}` form.
#[inline]
fn compute_round_poly_deg2(f1: &Mle, f2: &Mle) -> [GR32_128; 3] {
    let m = f1.len() / 2;
    debug_assert_eq!(f2.len(), f1.len());

    let e1 = f1.evals();
    let e2 = f2.evals();

    let mut a0 = GR32_128::ZERO;
    let mut a1 = GR32_128::ZERO;
    let mut a2 = GR32_128::ZERO;

    for x in 0..m {
        let f1_lo = e1[x];
        let f1_hi = e1[x + m];
        let f2_lo = e2[x];
        let f2_hi = e2[x + m];

        let df1 = f1_hi - f1_lo;
        let df2 = f2_hi - f2_lo;

        a0 = a0 + f1_lo * f2_lo;
        a1 = a1 + (f1_lo * df2 + f2_lo * df1);
        a2 = a2 + df1 * df2;
    }

    [a0, a1, a2]
}
