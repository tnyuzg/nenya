//! Prover-throughput bench for the BB4 deg-2 sumcheck.
//! Mirror of sumcheck-ring's bench. Same num_vars, same protocol, same loop
//! shape — only the element type differs.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use field::{AbstractExtensionField, AbstractField, BB};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use sumcheck_field::{prove, BB4UniformSampler, Mle, BB4};

fn rand_bb4(rng: &mut StdRng) -> BB4 {
    let bases: [BB; 4] = [rng.gen(), rng.gen(), rng.gen(), rng.gen()];
    <BB4 as AbstractExtensionField<BB>>::from_base_slice(&bases)
}

fn build_random_pair(num_vars: u32, seed: u64) -> (Mle, Mle, BB4) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = 1usize << num_vars;
    let f_vec: Vec<BB4> = (0..n).map(|_| rand_bb4(&mut rng)).collect();
    let g_vec: Vec<BB4> = (0..n).map(|_| rand_bb4(&mut rng)).collect();
    let mut claim = BB4::zero();
    for (a, b) in f_vec.iter().zip(g_vec.iter()) {
        claim += (*a) * (*b);
    }
    (Mle::new(f_vec), Mle::new(g_vec), claim)
}

fn bench_prover(c: &mut Criterion) {
    let var_counts: &[u32] = &[10, 14, 18];

    let mut g = c.benchmark_group("sumcheck_field/prover");
    for &nv in var_counts {
        let (f, gpoly, claim) = build_random_pair(nv, 0xC0FFEE_u64 ^ nv as u64);
        g.throughput(Throughput::Elements(1u64 << nv));
        g.bench_function(BenchmarkId::from_parameter(nv), |bench| {
            bench.iter(|| {
                let mut sampler = BB4UniformSampler::new(0);
                let (proof, _) = prove(&f, &gpoly, claim, &mut sampler);
                black_box(proof)
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_prover);
criterion_main!(benches);
