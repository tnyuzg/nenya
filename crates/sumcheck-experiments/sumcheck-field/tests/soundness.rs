//! Soundness sanity: tampering with a round message must cause rejection.
//! Mirror of sumcheck-ring's soundness tests.

use field::{AbstractExtensionField, AbstractField, BB};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use sumcheck_field::{prove, verifier::VerifyError, verify, BB4UniformSampler, Mle, BB4};

fn rand_bb4(rng: &mut StdRng) -> BB4 {
    let bases: [BB; 4] = [rng.gen(), rng.gen(), rng.gen(), rng.gen()];
    <BB4 as AbstractExtensionField<BB>>::from_base_slice(&bases)
}

fn hypercube_sum_product(f: &Mle, g: &Mle) -> BB4 {
    let mut acc = BB4::zero();
    for (a, b) in f.evals().iter().zip(g.evals().iter()) {
        acc += (*a) * (*b);
    }
    acc
}

#[test]
fn tampered_round_message_is_rejected() {
    let num_vars = 6;
    let n = 1usize << num_vars;
    let mut rng = StdRng::seed_from_u64(0xBADC0DE);
    let f = Mle::new((0..n).map(|_| rand_bb4(&mut rng)).collect());
    let g = Mle::new((0..n).map(|_| rand_bb4(&mut rng)).collect());
    let claim = hypercube_sum_product(&f, &g);

    let mut prover_sampler = BB4UniformSampler::new(1);
    let (mut proof, _) = prove(&f, &g, claim, &mut prover_sampler);

    proof.round_polys[2][1] += BB4::one();

    let mut verifier_sampler = BB4UniformSampler::new(1);
    match verify(&proof, num_vars, &mut verifier_sampler) {
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
    let f = Mle::new((0..n).map(|_| rand_bb4(&mut rng)).collect());
    let g = Mle::new((0..n).map(|_| rand_bb4(&mut rng)).collect());
    let claim = hypercube_sum_product(&f, &g);

    let mut prover_sampler = BB4UniformSampler::new(2);
    let (mut proof, _) = prove(&f, &g, claim, &mut prover_sampler);

    proof.claimed_sum += BB4::one();

    let mut verifier_sampler = BB4UniformSampler::new(2);
    assert_eq!(
        verify(&proof, num_vars, &mut verifier_sampler),
        Err(VerifyError::RoundCheckFailed { round: 0 }),
    );
}
