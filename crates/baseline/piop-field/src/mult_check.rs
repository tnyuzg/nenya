//! Field-side mult-check sumcheck. Mirror of [`piop_ring::mult_check`].
//! Same naive 3-table deg-3 sumcheck, same per-pair shared-sub-product
//! shape (10 muls per pair in the round-poly construction).

use field::{AbstractField, BB4};
use rayon::prelude::*;

use crate::eq::build_eq_table;
use crate::fs::Challenger;
use crate::mle::Mle;
use crate::proof::MultCheckProof;

pub fn prove_mult_check(
    a: &Mle,
    b: &Mle,
    r_target: &[BB4],
    claim: BB4,
    challenger: &mut impl Challenger,
) -> (MultCheckProof, Vec<BB4>) {
    let n = a.num_vars();
    assert_eq!(b.num_vars(), n, "a and b must agree on num_vars");
    assert_eq!(
        r_target.len(),
        n,
        "r_target must have one entry per variable"
    );
    assert!(n > 0, "mult check on a 0-variable MLE is meaningless");

    let mut eq = build_eq_table(r_target);
    let mut a = a.clone();
    let mut b = b.clone();

    let mut round_polys = Vec::with_capacity(n);
    let mut challenges = Vec::with_capacity(n);

    for _ in 0..n {
        let poly = compute_round_poly(&eq, &a, &b);
        // Per-round Fiat–Shamir: bind the round message before its challenge.
        challenger.observe(&poly);
        round_polys.push(poly);

        let r = challenger.sample();
        challenges.push(r);

        eq.fold_in_place(r);
        a.fold_in_place(r);
        b.fold_in_place(r);
    }

    let proof = MultCheckProof {
        claimed_sum: claim,
        round_polys,
        final_a: a.final_eval(),
        final_b: b.final_eval(),
    };
    (proof, challenges)
}

#[inline]
fn compute_round_poly(eq: &Mle, a: &Mle, b: &Mle) -> [BB4; 4] {
    let m = a.len() / 2;
    debug_assert_eq!(b.len(), a.len());
    debug_assert_eq!(eq.len(), a.len());

    let eq_evals = eq.evals();
    let a_evals = a.evals();
    let b_evals = b.evals();

    // Per-pair contributions are independent; field addition is associative+
    // commutative, so the parallel 4-tuple reduction is bit-identical to the
    // serial accumulation regardless of thread count.
    let zero = BB4::zero();
    let (a0, a1, a2, a3) = (0..m)
        .into_par_iter()
        .map(|i| {
            let eq_lo = eq_evals[i];
            let eq_hi = eq_evals[i + m];
            let a_lo = a_evals[i];
            let a_hi = a_evals[i + m];
            let b_lo = b_evals[i];
            let b_hi = b_evals[i + m];

            let de = eq_hi - eq_lo;
            let da = a_hi - a_lo;
            let db = b_hi - b_lo;

            let p_lolo = a_lo * b_lo;
            let p_lohi = a_lo * db;
            let p_hilo = da * b_lo;
            let p_hihi = da * db;

            let mid = p_lohi + p_hilo;

            (
                eq_lo * p_lolo,
                eq_lo * mid + de * p_lolo,
                eq_lo * p_hihi + de * mid,
                de * p_hihi,
            )
        })
        .reduce(
            || (zero, zero, zero, zero),
            |x, y| (x.0 + y.0, x.1 + y.1, x.2 + y.2, x.3 + y.3),
        );

    [a0, a1, a2, a3]
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

    fn direct_claim(a: &Mle, b: &Mle, r_target: &[BB4]) -> BB4 {
        let eq = build_eq_table(r_target);
        let mut acc = BB4::zero();
        for i in 0..a.len() {
            acc += eq.evals()[i] * a.evals()[i] * b.evals()[i];
        }
        acc
    }

    #[test]
    fn prove_emits_consistent_round_polys() {
        let n = 4;
        let mut state = 0xC0FFEE_u64;
        let a = rand_mle(n, &mut state);
        let b = rand_mle(n, &mut state);
        let r_target = rand_target(n, &mut state);
        let claim = direct_claim(&a, &b, &r_target);

        let mut sampler = BB4UniformSampler::new(0xBEEF);
        let (proof, challenges) = prove_mult_check(&a, &b, &r_target, claim, &mut sampler);

        let mut c = proof.claimed_sum;
        for (round, [a0, a1, a2, a3]) in proof.round_polys.iter().copied().enumerate() {
            let s0 = a0;
            let s1 = a0 + a1 + a2 + a3;
            assert_eq!(s0 + s1, c, "round {round} consistency");
            let r = challenges[round];
            c = a0 + r * (a1 + r * (a2 + r * a3));
        }

        let mut eq_final = BB4::one();
        for (rt, &rc) in r_target.iter().zip(challenges.iter()) {
            eq_final *= *rt * rc + (BB4::one() - *rt) * (BB4::one() - rc);
        }
        assert_eq!(c, eq_final * proof.final_a * proof.final_b);
    }
}
