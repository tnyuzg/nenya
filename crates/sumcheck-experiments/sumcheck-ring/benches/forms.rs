//! Isolated head-to-head: **form 1** (`∑ eq·(ab − c − e)`, fold 5 tables) vs
//! **form 2** (`∑ eq·a·b = c(r)+e(r)`, fold 3 tables + 2 dot products).
//!
//! Same field (R₂), same reference (non-Gruen) round structure, same serial
//! inner loop — the *only* difference is where the linear terms `c`, `e` live.
//! `eq, a, b, c, e` are precomputed inputs (not timed); the eq table is common
//! to both forms.
//!
//! Run: `cargo bench -p sumcheck-ring --bench forms`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use galois_ring::{R2TeichmullerSampler, GR32_128, GR32_4};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use sumcheck_ring::{prove_form1, prove_form2, Mle};

fn rand_r2(rng: &mut StdRng) -> GR32_128 {
    let mut arr = [GR32_4::ZERO; 32];
    for slot in arr.iter_mut() {
        *slot = GR32_4([rng.gen(), rng.gen(), rng.gen(), rng.gen()]);
    }
    GR32_128(arr)
}

fn rand_mle(num_vars: u32, rng: &mut StdRng) -> Mle {
    let n = 1usize << num_vars;
    Mle::new((0..n).map(|_| rand_r2(rng)).collect())
}

/// Build the five input tables for a given size.
fn inputs(num_vars: u32, seed: u64) -> (Mle, Mle, Mle, Mle, Mle) {
    let mut rng = StdRng::seed_from_u64(seed);
    (
        rand_mle(num_vars, &mut rng), // eq
        rand_mle(num_vars, &mut rng), // a
        rand_mle(num_vars, &mut rng), // b
        rand_mle(num_vars, &mut rng), // c
        rand_mle(num_vars, &mut rng), // e
    )
}

const VAR_COUNTS: &[u32] = &[12, 15, 18];

fn bench_form1(c: &mut Criterion) {
    let mut g = c.benchmark_group("mult_check_form/form1_zerocheck_fold5");
    for &nv in VAR_COUNTS {
        let (eq, a, b, cc, e) = inputs(nv, 0xF0F0 ^ nv as u64);
        g.throughput(Throughput::Elements(1u64 << nv));
        g.bench_function(BenchmarkId::from_parameter(nv), |bench| {
            bench.iter(|| {
                let mut sampler = R2TeichmullerSampler::new(0);
                let (proof, _) =
                    prove_form1(&eq, &a, &b, &cc, &e, GR32_128::ZERO, &mut sampler);
                black_box(proof)
            });
        });
    }
    g.finish();
}

fn bench_form2(c: &mut Criterion) {
    let mut g = c.benchmark_group("mult_check_form/form2_rhs_fold3");
    for &nv in VAR_COUNTS {
        let (eq, a, b, cc, e) = inputs(nv, 0xF0F0 ^ nv as u64);
        g.throughput(Throughput::Elements(1u64 << nv));
        g.bench_function(BenchmarkId::from_parameter(nv), |bench| {
            bench.iter(|| {
                let mut sampler = R2TeichmullerSampler::new(0);
                let (claim, proof, _) = prove_form2(&eq, &a, &b, &cc, &e, &mut sampler);
                black_box((claim, proof))
            });
        });
    }
    g.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_form1, bench_form2
}
criterion_main!(benches);
