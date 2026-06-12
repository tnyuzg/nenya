//! Isolated comparison of the two ways to phrase the mult-check / zero-check
//! for an R1CS multiplication gate with a kernel error term:
//!
//! - **Form 1 (zero-check, everything inside):**
//!   `∑_h eq(r,h) · (a(h)·b(h) − c(h) − e(h)) = 0`.
//!   The summand is degree-3 in each variable (dominated by `eq·a·b`); the
//!   `−c−e` terms are degree-2 and fold into the same round message. The
//!   prover folds **five** tables (`eq, a, b, c, e`) every round.
//!
//! - **Form 2 (linear terms on the RHS):**
//!   `∑_h eq(r,h) · a(h)·b(h) = c(r) + e(r)`.
//!   `c(r) = ⟨eq, c⟩` and `e(r) = ⟨eq, e⟩` are two one-shot inner products
//!   computed *before* the sumcheck; the sumcheck itself folds only **three**
//!   tables (`eq, a, b`) — it is exactly [`crate::prove`].
//!
//! The two are algebraically identical (`∑_h eq(r,h)·c(h) = c̃(r)`), so this
//! is purely a prover-cost question. Both provers here use the same non-Gruen
//! reference round structure and run serially, so the *only* variable is where
//! `c` and `e` live: folded through the sumcheck (form 1) vs. two dot products
//! outside it (form 2). The Gruen eq-specialization would lower the absolute
//! time of *both* by the same eq-handling savings, so this delta is what you'd
//! see on top of either baseline.

use galois_ring::{R2TeichmullerSampler, GR32_128};

use crate::mle::Mle;

/// Form-1 proof: degree-3 round messages for `∑ eq·(ab − c − e)`, plus the
/// four folded finals the verifier would need at the binding point (`eq` is
/// verifier-reconstructible, so it is not carried — but the prover still folds
/// it, which is what we are timing).
#[derive(Clone, Debug)]
pub struct Form1Proof {
    pub claimed_sum: GR32_128,
    pub round_polys: Vec<[GR32_128; 4]>,
    pub final_a: GR32_128,
    pub final_b: GR32_128,
    pub final_c: GR32_128,
    pub final_e: GR32_128,
}

/// `⟨x, y⟩ = ∑_i x_i·y_i` over the eval tables — the one-shot inner product
/// form 2 uses to pull `c(r)` and `e(r)` to the RHS. Serial, to match the
/// reference prover's single-threaded inner loop.
#[inline]
pub fn dot(x: &Mle, y: &Mle) -> GR32_128 {
    let xe = x.evals();
    let ye = y.evals();
    debug_assert_eq!(xe.len(), ye.len());
    let mut acc = GR32_128::ZERO;
    for i in 0..xe.len() {
        acc = acc + xe[i] * ye[i];
    }
    acc
}

/// **Form 1.** Run the zero-check sumcheck `∑_h eq(h)·(a(h)·b(h) − c(h) − e(h))`,
/// folding all five tables each round. Returns the proof and the challenge
/// sequence (sample order).
pub fn prove_form1(
    eq: &Mle,
    a: &Mle,
    b: &Mle,
    c: &Mle,
    e: &Mle,
    claim: GR32_128,
    sampler: &mut R2TeichmullerSampler,
) -> (Form1Proof, Vec<GR32_128>) {
    let n = eq.num_vars();
    assert_eq!(a.num_vars(), n);
    assert_eq!(b.num_vars(), n);
    assert_eq!(c.num_vars(), n);
    assert_eq!(e.num_vars(), n);
    assert!(n > 0);

    let mut eq = eq.clone();
    let mut a = a.clone();
    let mut b = b.clone();
    let mut c = c.clone();
    let mut e = e.clone();

    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for _ in 0..n {
        round_polys.push(compute_round_poly_form1(&eq, &a, &b, &c, &e));

        let r = sampler.sample();
        challenges.push(r);

        eq.fold_in_place(r);
        a.fold_in_place(r);
        b.fold_in_place(r);
        c.fold_in_place(r);
        e.fold_in_place(r);
    }

    (
        Form1Proof {
            claimed_sum: claim,
            round_polys,
            final_a: a.final_eval(),
            final_b: b.final_eval(),
            final_c: c.final_eval(),
            final_e: e.final_eval(),
        },
        challenges,
    )
}

/// Degree-3 round univariate `[g0, g1, g2, g3]` for `eq·(ab − c − e)`.
///
/// Per fold-pair: inner factor `p(Z) = a(Z)b(Z) − c(Z) − e(Z)` (deg-2, 4 muls)
/// times `eq(Z)` (deg-1, 6 muls) = **10 muls/pair** — the same product count
/// as the deg-3 `eq·a·b` baseline ([`crate::prover`]'s `compute_round_poly_deg3`),
/// plus four extra subtractions for the `−c−e`.
#[inline]
fn compute_round_poly_form1(
    eq: &Mle,
    a: &Mle,
    b: &Mle,
    c: &Mle,
    e: &Mle,
) -> [GR32_128; 4] {
    let m = eq.len() / 2;
    let eqe = eq.evals();
    let ae = a.evals();
    let be = b.evals();
    let ce = c.evals();
    let ee = e.evals();

    let mut g0 = GR32_128::ZERO;
    let mut g1 = GR32_128::ZERO;
    let mut g2 = GR32_128::ZERO;
    let mut g3 = GR32_128::ZERO;

    for x in 0..m {
        let eq_lo = eqe[x];
        let deq = eqe[x + m] - eq_lo;
        let a_lo = ae[x];
        let da = ae[x + m] - a_lo;
        let b_lo = be[x];
        let db = be[x + m] - b_lo;
        let c_lo = ce[x];
        let dc = ce[x + m] - c_lo;
        let e_lo = ee[x];
        let de = ee[x + m] - e_lo;

        // Inner deg-2 factor p(Z) = a(Z)·b(Z) − c(Z) − e(Z).
        let p0 = a_lo * b_lo - c_lo - e_lo;
        let p1 = a_lo * db + da * b_lo - dc - de;
        let p2 = da * db;

        // g(Z) = eq(Z)·p(Z), eq(Z) = eq_lo + Z·deq.
        g0 = g0 + eq_lo * p0;
        g1 = g1 + eq_lo * p1 + deq * p0;
        g2 = g2 + eq_lo * p2 + deq * p1;
        g3 = g3 + deq * p2;
    }

    [g0, g1, g2, g3]
}

/// **Form 2.** The two RHS inner products `c(r)`, `e(r)` plus the deg-3
/// sumcheck on `eq·a·b` ([`crate::prove`]). Returned as a tuple so the bench
/// can time the whole thing; the sumcheck proof is the same `SumcheckProof`.
pub fn prove_form2(
    eq: &Mle,
    a: &Mle,
    b: &Mle,
    c: &Mle,
    e: &Mle,
    sampler: &mut R2TeichmullerSampler,
) -> (GR32_128, crate::proof::SumcheckProof, Vec<GR32_128>) {
    // RHS: pull the linear terms out as evaluations at r.
    let c_at_r = dot(eq, c);
    let e_at_r = dot(eq, e);
    let claim = c_at_r + e_at_r;

    // Sumcheck folds only eq, a, b.
    let (proof, challenges) = crate::prove(eq, a, b, claim, sampler);
    (claim, proof, challenges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use galois_ring::GR32_4;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn rand_r2(rng: &mut StdRng) -> GR32_128 {
        let mut arr = [GR32_4::ZERO; 32];
        for slot in arr.iter_mut() {
            *slot = GR32_4([rng.gen(), rng.gen(), rng.gen(), rng.gen()]);
        }
        GR32_128(arr)
    }

    fn rand_mle(nv: usize, rng: &mut StdRng) -> Mle {
        Mle::new((0..(1usize << nv)).map(|_| rand_r2(rng)).collect())
    }

    /// Both forms must produce identical round-0 messages *as a statement about
    /// the same polynomial*: form 1's round poly evaluated at {0,1} summed must
    /// equal form 2's claim (`c(r)+e(r)`) iff the gate holds. Here we instead
    /// check the weaker invariant that both run to completion with matching
    /// challenge sequences — the algebraic equivalence is argued in the docs.
    #[test]
    fn both_forms_run_with_matching_challenges() {
        let nv = 6;
        let mut rng = StdRng::seed_from_u64(1);
        let eq = rand_mle(nv, &mut rng);
        let a = rand_mle(nv, &mut rng);
        let b = rand_mle(nv, &mut rng);
        let c = rand_mle(nv, &mut rng);
        let e = rand_mle(nv, &mut rng);

        let mut s1 = R2TeichmullerSampler::new(42);
        let (_p1, ch1) = prove_form1(&eq, &a, &b, &c, &e, GR32_128::ZERO, &mut s1);

        let mut s2 = R2TeichmullerSampler::new(42);
        let (_claim, _p2, ch2) = prove_form2(&eq, &a, &b, &c, &e, &mut s2);

        assert_eq!(ch1, ch2, "same sampler seed → same challenges");
        assert_eq!(ch1.len(), nv);
    }

    /// Sanity: for a gate that genuinely holds (`e = a∘b − c`), form 1's first
    /// round message satisfies `g(0)+g(1) = 0` (the zero-check claim), and
    /// form 2's RHS claim equals `∑ eq·a·b`.
    #[test]
    fn satisfied_gate_consistency() {
        let nv = 5;
        let mut rng = StdRng::seed_from_u64(7);
        let eq = rand_mle(nv, &mut rng);
        let a = rand_mle(nv, &mut rng);
        let b = rand_mle(nv, &mut rng);
        let c = rand_mle(nv, &mut rng);
        // Make the gate hold: e[h] = a[h]·b[h] − c[h].
        let e_vec: Vec<GR32_128> = (0..a.len())
            .map(|i| a.evals()[i] * b.evals()[i] - c.evals()[i])
            .collect();
        let e = Mle::new(e_vec);

        // Form 1: g(0)+g(1) over round 0 = ∑_h eq·(ab−c−e) = 0.
        let [g0, g1, g2, g3] = compute_round_poly_form1(&eq, &a, &b, &c, &e);
        // s(Z) = g0 + g1 Z + g2 Z² + g3 Z³; s(0)+s(1) = 2 g0 + g1 + g2 + g3.
        let s0 = g0;
        let s1 = g0 + g1 + g2 + g3;
        assert_eq!(s0 + s1, GR32_128::ZERO, "zero-check claim must be 0");

        // Form 2: claim = c(r)+e(r) must equal ∑ eq·a·b.
        let mut ab = GR32_128::ZERO;
        for i in 0..a.len() {
            ab = ab + eq.evals()[i] * a.evals()[i] * b.evals()[i];
        }
        let claim = dot(&eq, &c) + dot(&eq, &e);
        assert_eq!(claim, ab, "form-2 RHS must equal ∑ eq·a·b");
    }
}
