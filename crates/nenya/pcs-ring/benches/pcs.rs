//! Criterion benchmarks for the tensor PCS (packed u32 commit).
//!
//! Reports commit, open, verify wall-clock time and proof size at several
//! `(k1, k2)` shapes under both protocol parameter sets:
//!
//! - **Optimistic**: `q = 4` (rate 1/4), `δ = 0.5`, `t = 487`.
//! - **Conservative**: `q = 6` (rate 1/6), `δ = 0.3`, `t = 842`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use galois_ring::{GR32_1, GR32_128};
use pcs_ring::Pcs;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn random_packed_evals(n: usize, seed: u64) -> Vec<[u32; 32]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut a = [0u32; 32];
            for s in 0..32 {
                a[s] = rng.gen();
            }
            a
        })
        .collect()
}

fn random_point(num_vars: usize, seed: u64) -> Vec<GR32_128> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..num_vars)
        .map(|_| GR32_128::from_gr1(GR32_1(rng.gen())))
        .collect()
}

#[derive(Copy, Clone)]
struct Shape {
    k1: usize,
    k2: usize,
    label: &'static str,
}

fn bench_shape(c: &mut Criterion, group_name: &str, pcs: &Pcs, shape: &Shape) {
    let n_poly = shape.k1 * shape.k2;
    let evals = random_packed_evals(n_poly, 1);
    let z = random_point(pcs.num_vars(), 2);

    let comm = pcs.commit(&evals);
    let (_y, proof) = pcs.open(&comm, &z);
    eprintln!(
        "[{}] N={}, k1={}, k2={}, n={}, t={} -> proof = {:.2} MB",
        shape.label,
        n_poly,
        shape.k1,
        shape.k2,
        pcs.n(),
        pcs.ell,
        proof.size_bytes() as f64 / (1024.0 * 1024.0),
    );

    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);

    group.bench_with_input(BenchmarkId::new("commit", shape.label), shape, |b, _| {
        b.iter(|| pcs.commit(&evals))
    });
    group.bench_with_input(BenchmarkId::new("open", shape.label), shape, |b, _| {
        b.iter(|| pcs.open(&comm, &z))
    });
    group.bench_with_input(BenchmarkId::new("verify", shape.label), shape, |b, _| {
        let (y, proof) = pcs.open(&comm, &z);
        b.iter(|| pcs.verify(comm.root, &z, y, &proof))
    });

    group.finish();
}

fn shapes_for_sweep() -> &'static [Shape] {
    &[
        Shape {
            k1: 16,
            k2: 1024,
            label: "N=2^14 rect (k1=16,  k2=1024)",
        },
        Shape {
            k1: 128,
            k2: 128,
            label: "N=2^14 sq   (k1=128, k2=128)",
        },
        Shape {
            k1: 16,
            k2: 2048,
            label: "N=2^15 rect (k1=16,  k2=2048)",
        },
        Shape {
            k1: 128,
            k2: 256,
            label: "N=2^15 sq   (k1=128, k2=256)",
        },
        Shape {
            k1: 32,
            k2: 2048,
            label: "N=2^16 rect (k1=32,  k2=2048)",
        },
        Shape {
            k1: 256,
            k2: 256,
            label: "N=2^16 sq   (k1=256, k2=256)",
        },
        Shape {
            k1: 64,
            k2: 4096,
            label: "N=2^18 rect (k1=64,  k2=4096)",
        },
        Shape {
            k1: 512,
            k2: 512,
            label: "N=2^18 sq   (k1=512, k2=512)",
        },
        Shape {
            k1: 64,
            k2: 16384,
            label: "N=2^20 rect (k1=64,  k2=16384)",
        },
        Shape {
            k1: 1024,
            k2: 1024,
            label: "N=2^20 sq   (k1=1024,k2=1024)",
        },
    ]
}

fn bench_packed_opt(c: &mut Criterion) {
    let q = 4;
    let t = 487;
    for shape in shapes_for_sweep() {
        let pcs = Pcs::new(shape.k1, shape.k2, q, t, 0xC0DE);
        bench_shape(c, "packed-opt", &pcs, shape);
    }
}

fn bench_packed_con(c: &mut Criterion) {
    let q = 6;
    let t = 842;
    for shape in shapes_for_sweep() {
        let pcs = Pcs::new(shape.k1, shape.k2, q, t, 0xC0DE);
        bench_shape(c, "packed-con", &pcs, shape);
    }
}

criterion_group!(benches, bench_packed_opt, bench_packed_con);
criterion_main!(benches);
