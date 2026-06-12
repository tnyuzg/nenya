//! Sumcheck prover for `sum = ∑_h f(h) · g(h)` over BB4.
//!
//! Mirror of `sumcheck-ring`'s `prover.rs`. See that file for the
//! per-round-poly derivation; the algebra is identical, only the element type
//! is different.

use field::{AbstractField, BB4UniformSampler, BB4};

use crate::mle::Mle;
use crate::proof::SumcheckProof;

pub fn prove(
    f: &Mle,
    g: &Mle,
    claim: BB4,
    sampler: &mut BB4UniformSampler,
) -> (SumcheckProof, Vec<BB4>) {
    assert_eq!(
        f.num_vars(),
        g.num_vars(),
        "f and g must have the same number of variables"
    );
    let n = f.num_vars();
    assert!(n > 0, "sumcheck on a 0-variable MLE is meaningless");

    let mut f = f.clone();
    let mut g = g.clone();

    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for _ in 0..n {
        let poly = compute_round_poly(&f, &g);
        round_polys.push(poly);

        let r = sampler.sample();
        challenges.push(r);

        f.fold_in_place(r);
        g.fold_in_place(r);
    }

    (
        SumcheckProof {
            claimed_sum: claim,
            round_polys,
            final_f: f.final_eval(),
            final_g: g.final_eval(),
        },
        challenges,
    )
}

#[inline]
fn compute_round_poly(f: &Mle, g: &Mle) -> [BB4; 3] {
    let m = f.len() / 2;
    debug_assert_eq!(g.len(), f.len());

    let f_evals = f.evals();
    let g_evals = g.evals();

    let mut a0 = BB4::zero();
    let mut a1 = BB4::zero();
    let mut a2 = BB4::zero();

    for x in 0..m {
        let f_lo = f_evals[x];
        let f_hi = f_evals[x + m];
        let g_lo = g_evals[x];
        let g_hi = g_evals[x + m];

        let df = f_hi - f_lo;
        let dg = g_hi - g_lo;

        a0 += f_lo * g_lo;
        a1 += f_lo * dg + g_lo * df;
        a2 += df * dg;
    }

    [a0, a1, a2]
}
