//! Soundness sanity: tampering with a round message must cause rejection.
//!
//! Not a full soundness proof — just a check that the round-by-round
//! consistency check actually fires when an adversary perturbs the proof.

use galois_ring::{R2TeichmullerSampler, GR32_128, GR32_4};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use sumcheck_ring::{prove, verifier::VerifyError, verify, Mle};

fn rand_r2(rng: &mut StdRng) -> GR32_128 {
    let mut arr = [GR32_4::ZERO; 32];
    for slot in arr.iter_mut() {
        *slot = GR32_4([rng.gen(), rng.gen(), rng.gen(), rng.gen()]);
    }
    GR32_128(arr)
}

fn hypercube_triple(f1: &Mle, f2: &Mle, f3: &Mle) -> GR32_128 {
    let mut acc = GR32_128::ZERO;
    for i in 0..f1.len() {
        acc = acc + f1.evals()[i] * f2.evals()[i] * f3.evals()[i];
    }
    acc
}

#[test]
fn tampered_round_message_is_rejected() {
    let num_vars = 6;
    let n = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(0xBADC0DE);
    let f1 = Mle::new((0..n).map(|_| rand_r2(&mut rng)).collect());
    let f2 = Mle::new((0..n).map(|_| rand_r2(&mut rng)).collect());
    let f3 = Mle::new((0..n).map(|_| rand_r2(&mut rng)).collect());
    let claim = hypercube_triple(&f1, &f2, &f3);

    let mut prover_sampler = R2TeichmullerSampler::new(1);
    let (mut proof, _) = prove(&f1, &f2, &f3, claim, &mut prover_sampler);

    proof.round_polys[2][1] = proof.round_polys[2][1] + GR32_128::ONE;

    let mut verifier_sampler = R2TeichmullerSampler::new(1);
    let result = verify(&proof, num_vars, &mut verifier_sampler);

    match result {
        Err(VerifyError::RoundCheckFailed { .. }) => {}
        Err(VerifyError::FinalCheckFailed) => {}
        Ok(_) => panic!("tampered proof was accepted"),
    }
}

#[test]
fn tampered_claim_is_rejected() {
    let num_vars = 5;
    let n = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(0xFEED);
    let f1 = Mle::new((0..n).map(|_| rand_r2(&mut rng)).collect());
    let f2 = Mle::new((0..n).map(|_| rand_r2(&mut rng)).collect());
    let f3 = Mle::new((0..n).map(|_| rand_r2(&mut rng)).collect());
    let claim = hypercube_triple(&f1, &f2, &f3);

    let mut prover_sampler = R2TeichmullerSampler::new(2);
    let (mut proof, _) = prove(&f1, &f2, &f3, claim, &mut prover_sampler);

    proof.claimed_sum = proof.claimed_sum + GR32_128::ONE;

    let mut verifier_sampler = R2TeichmullerSampler::new(2);
    assert_eq!(
        verify(&proof, num_vars, &mut verifier_sampler),
        Err(VerifyError::RoundCheckFailed { round: 0 }),
        "tampered claim must fail round 0"
    );
}
