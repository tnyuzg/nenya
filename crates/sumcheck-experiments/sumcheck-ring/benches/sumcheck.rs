//! Prover-throughput bench for the R₂ sumcheck — both deg-2 and deg-3
//! single-product variants.
//!
//! Pair with `cargo bench -p sumcheck-field` to read off the per-element
//! overhead vs a BabyBear^4 reference using identical protocol shape.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use galois_ring::{R2TeichmullerSampler, GR32_128, GR32_4};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use sumcheck_ring::{prove, prove_deg2, Mle};

fn rand_r2(rng: &mut StdRng) -> GR32_128 {
    let mut arr = [GR32_4::ZERO; 32];
    for slot in arr.iter_mut() {
        *slot = GR32_4([rng.gen(), rng.gen(), rng.gen(), rng.gen()]);
    }
    GR32_128(arr)
}

fn build_pair(num_vars: u32, seed: u64) -> (Mle, Mle, GR32_128) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = 1usize << num_vars;
    let f1: Vec<GR32_128> = (0..n).map(|_| rand_r2(&mut rng)).collect();
    let f2: Vec<GR32_128> = (0..n).map(|_| rand_r2(&mut rng)).collect();
    let mut claim = GR32_128::ZERO;
    for i in 0..n {
        claim = claim + f1[i] * f2[i];
    }
    (Mle::new(f1), Mle::new(f2), claim)
}

fn build_triple(num_vars: u32, seed: u64) -> (Mle, Mle, Mle, GR32_128) {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = 1usize << num_vars;
    let f1: Vec<GR32_128> = (0..n).map(|_| rand_r2(&mut rng)).collect();
    let f2: Vec<GR32_128> = (0..n).map(|_| rand_r2(&mut rng)).collect();
    let f3: Vec<GR32_128> = (0..n).map(|_| rand_r2(&mut rng)).collect();
    let mut claim = GR32_128::ZERO;
    for i in 0..n {
        claim = claim + f1[i] * f2[i] * f3[i];
    }
    (Mle::new(f1), Mle::new(f2), Mle::new(f3), claim)
}

fn bench_deg2(c: &mut Criterion) {
    let var_counts: &[u32] = &[10, 13, 15];

    let mut g = c.benchmark_group("sumcheck_ring/prover_deg2");
    for &nv in var_counts {
        let (f1, f2, claim) = build_pair(nv, 0xC0FFEE_u64 ^ nv as u64);
        g.throughput(Throughput::Elements(1u64 << nv));
        g.bench_function(BenchmarkId::from_parameter(nv), |bench| {
            bench.iter(|| {
                let mut sampler = R2TeichmullerSampler::new(0);
                let (proof, _) = prove_deg2(&f1, &f2, claim, &mut sampler);
                black_box(proof)
            });
        });
    }
    g.finish();
}

fn bench_deg3(c: &mut Criterion) {
    let var_counts: &[u32] = &[10, 13, 15];

    let mut g = c.benchmark_group("sumcheck_ring/prover_deg3");
    for &nv in var_counts {
        let (f1, f2, f3, claim) = build_triple(nv, 0xBEEF_u64 ^ nv as u64);
        g.throughput(Throughput::Elements(1u64 << nv));
        g.bench_function(BenchmarkId::from_parameter(nv), |bench| {
            bench.iter(|| {
                let mut sampler = R2TeichmullerSampler::new(0);
                let (proof, _) = prove(&f1, &f2, &f3, claim, &mut sampler);
                black_box(proof)
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench_deg2, bench_deg3);
criterion_main!(benches);
