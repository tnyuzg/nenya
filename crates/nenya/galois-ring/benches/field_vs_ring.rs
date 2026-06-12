//! Full add/mul benchmark: Galois ring at every base width vs the field.
//!
//! Widths: `u8`/`u16`/`u32`/`u64` (base rings `Z/2^8 … Z/2^64`). Field
//! baselines: `BabyBear` (~31-bit prime field) and `BabyBear⁴` (its degree-4
//! extension, ~124 bits).
//!
//! Three ring levels are measured per width:
//!   - `R0 = Z/2^n`            (one limb)        — paired with `BabyBear`
//!   - `R1 = GR(2^n, 4)`       (4 limbs)
//!   - `R2 = GR(2^n, 128)`     (128 limbs)       — paired with `BabyBear⁴`
//!
//! Why pair `R2` with `BabyBear⁴`: both deliver ~128-bit soundness per
//! challenge (`|T_2| = 2^128` for *every* width, independent of `n`; `BabyBear⁴`
//! is ~2^124). So at equal security the ring element storage shrinks 4× going
//! `u32 → u8`, while the field element is fixed — that's the comparison the
//! paper's "ring vs field" numbers want.
//!
//! One op per criterion iter, both operands `black_box`'d ⇒ per-op throughput
//! (not single-op latency); consecutive iters are data-independent.
//!
//!   cargo bench -p galois-ring --bench field_vs_ring

use std::ops::{Add, Mul};
use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
use field::{AbstractField, BB4};
use galois_ring::{Gr1, Gr128, Gr4, Limb};

/// Bench `add` and `mul` for one element type, as `"<label>/add"` and
/// `"<label>/mul"` under the current group.
fn add_mul<T>(g: &mut BenchmarkGroup<WallTime>, label: &str, a: T, b: T)
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    g.bench_function(format!("{label}/add"), |bn| {
        bn.iter(|| black_box(a) + black_box(b))
    });
    g.bench_function(format!("{label}/mul"), |bn| {
        bn.iter(|| black_box(a) * black_box(b))
    });
}

// --- generic nontrivial operand builders (values are irrelevant to op cost) --

fn gr1_pair<L: Limb>() -> (Gr1<L>, Gr1<L>) {
    (Gr1(L::from_u64(0xDEAD_BEEF)), Gr1(L::from_u64(0x1234_5678)))
}

fn gr4_pair<L: Limb>() -> (Gr4<L>, Gr4<L>) {
    (
        Gr4([
            L::from_u64(0xDEAD_BEEF),
            L::from_u64(7),
            L::from_u64(0x1234),
            L::from_u64(99),
        ]),
        Gr4([
            L::from_u64(0x1234_5678),
            L::from_u64(3),
            L::from_u64(0xBEEF),
            L::from_u64(5),
        ]),
    )
}

fn gr128_pair<L: Limb>() -> (Gr128<L>, Gr128<L>) {
    // Nontrivial elements via β = Y so the schoolbook + reduction both fire.
    let mut beta_arr = [Gr4::<L>::ZERO; 32];
    beta_arr[1] = Gr4::<L>::ONE;
    let beta = Gr128(beta_arr);
    let a = beta * beta + Gr128::<L>::ONE;
    let b = beta + Gr128::from_gr4(Gr4([L::from_u64(3), L::ONE, L::ZERO, L::ZERO]));
    (a, b)
}

fn bench_r0(c: &mut Criterion) {
    // Base ring Z/2^n — one machine word. (No field pairing here: the field
    // baseline is the ~128-bit extension, measured in the R2 group below.)
    let mut g = c.benchmark_group("R0 = Z|2^n  (base ring, one limb)");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    let (a, b) = gr1_pair::<u8>();
    add_mul(&mut g, "GR_u8", a, b);
    let (a, b) = gr1_pair::<u16>();
    add_mul(&mut g, "GR_u16", a, b);
    let (a, b) = gr1_pair::<u32>();
    add_mul(&mut g, "GR_u32", a, b);
    let (a, b) = gr1_pair::<u64>();
    add_mul(&mut g, "GR_u64", a, b);
    g.finish();
}

fn bench_r1(c: &mut Criterion) {
    let mut g = c.benchmark_group("R1 = GR(2^n, 4)");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    let (a, b) = gr4_pair::<u8>();
    add_mul(&mut g, "GR_u8", a, b);
    let (a, b) = gr4_pair::<u16>();
    add_mul(&mut g, "GR_u16", a, b);
    let (a, b) = gr4_pair::<u32>();
    add_mul(&mut g, "GR_u32", a, b);
    let (a, b) = gr4_pair::<u64>();
    add_mul(&mut g, "GR_u64", a, b);
    g.finish();
}

fn bench_r2_vs_field(c: &mut Criterion) {
    let mut g = c.benchmark_group("R2 = GR(2^n, 128)  (vs BabyBear^4, equal ~128-bit security)");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    add_mul(
        &mut g,
        "BabyBear4",
        BB4::from_canonical_u32(0xDEAD_BEEF),
        BB4::from_canonical_u32(0x1234_5678),
    );
    let (a, b) = gr128_pair::<u8>();
    add_mul(&mut g, "GR_u8", a, b);
    let (a, b) = gr128_pair::<u16>();
    add_mul(&mut g, "GR_u16", a, b);
    let (a, b) = gr128_pair::<u32>();
    add_mul(&mut g, "GR_u32", a, b);
    let (a, b) = gr128_pair::<u64>();
    add_mul(&mut g, "GR_u64", a, b);
    g.finish();
}

/// Add *throughput*: accumulate-sum of `N` elements in a tight loop. The
/// single-op `add` microbench above is dominated by `black_box`'ing a 128-byte
/// struct in/out of memory each iter; this amortizes that and exposes the real
/// vectorized per-add cost the SNARK prover sees in matvec / sumcheck folds.
/// Reported time is for the whole sum — divide by N for per-add.
fn bench_r2_add_throughput(c: &mut Criterion) {
    const N: usize = 1024;
    let mut g = c.benchmark_group("R2 add throughput (sum of 1024, divide by 1024 for per-add)");
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(2));

    macro_rules! sum_bench {
        ($label:expr, $ty:ty, $mk:expr) => {{
            let data: Vec<$ty> = (0..N).map(|i| $mk(i as u64)).collect();
            g.bench_function($label, |bn| {
                bn.iter(|| {
                    let mut acc = <$ty>::ZERO;
                    for x in &data {
                        acc = acc + *x;
                    }
                    black_box(acc)
                })
            });
        }};
    }

    // BabyBear4 has no `ZERO` assoc const in this facade; fold from zero().
    {
        let data: Vec<BB4> = (0..N)
            .map(|i| BB4::from_canonical_u32(i as u32 | 1))
            .collect();
        g.bench_function("BabyBear4", |bn| {
            bn.iter(|| {
                let mut acc = BB4::zero();
                for x in &data {
                    acc = acc + *x;
                }
                black_box(acc)
            })
        });
    }
    sum_bench!("GR_u8", Gr128<u8>, |i| gr128_seeded::<u8>(i));
    sum_bench!("GR_u16", Gr128<u16>, |i| gr128_seeded::<u16>(i));
    sum_bench!("GR_u32", Gr128<u32>, |i| gr128_seeded::<u32>(i));
    sum_bench!("GR_u64", Gr128<u64>, |i| gr128_seeded::<u64>(i));
    g.finish();
}

/// A distinct nontrivial `Gr128<L>` per seed (so the sum isn't a trivial scale).
fn gr128_seeded<L: Limb>(seed: u64) -> Gr128<L> {
    let mut arr = [Gr4::<L>::ZERO; 32];
    let mut s = seed | 1;
    for slot in arr.iter_mut() {
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            L::from_u64(s >> 11)
        };
        *slot = Gr4([next(), next(), next(), next()]);
    }
    Gr128(arr)
}

criterion_group!(
    benches,
    bench_r0,
    bench_r1,
    bench_r2_vs_field,
    bench_r2_add_throughput
);
criterion_main!(benches);
