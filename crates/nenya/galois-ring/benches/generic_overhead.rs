//! Does genericizing the ring tower over the limb type cost anything?
//!
//! This bench answers that directly: it contains a **frozen, verbatim copy**
//! of the original hand-written concrete `u32` arithmetic (`Leg*`, "legacy")
//! exactly as it stood before the `Limb`-generic refactor, and benches it
//! *side by side, in the same binary* against the generic `Gr128<u32>` (= the
//! `GR32_128` alias). Same machine, same run — immune to the cross-run noise
//! you get from comparing separate `cargo bench` invocations.
//!
//! Expectation: the two columns are within measurement noise, because Rust
//! monomorphizes `Gr128<u32>` to the same code the legacy concrete type
//! compiles to (the `Limb` ops are `#[inline(always)]` wrappers over the same
//! `wrapping_*` intrinsics). Run:
//!
//!   cargo bench -p galois-ring --bench generic_overhead

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use galois_ring::{Gr1, Gr128, Gr4};

// ===========================================================================
// FROZEN LEGACY: the original concrete u32 implementation, copied verbatim
// from git HEAD (pre-Limb-refactor). Do not "improve" it — its whole job is
// to be the unchanged baseline.
// ===========================================================================
mod legacy {
    use std::ops::{Add, Mul, Sub};

    #[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
    #[repr(transparent)]
    pub struct LegGr1(pub u32);

    impl Mul for LegGr1 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            Self(self.0.wrapping_mul(rhs.0))
        }
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    #[repr(C, align(16))]
    pub struct LegGr4(pub [u32; 4]);

    impl LegGr4 {
        pub const ZERO: Self = Self([0, 0, 0, 0]);
    }

    impl Add for LegGr4 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            LegGr4([
                self.0[0].wrapping_add(rhs.0[0]),
                self.0[1].wrapping_add(rhs.0[1]),
                self.0[2].wrapping_add(rhs.0[2]),
                self.0[3].wrapping_add(rhs.0[3]),
            ])
        }
    }

    impl Sub for LegGr4 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            LegGr4([
                self.0[0].wrapping_sub(rhs.0[0]),
                self.0[1].wrapping_sub(rhs.0[1]),
                self.0[2].wrapping_sub(rhs.0[2]),
                self.0[3].wrapping_sub(rhs.0[3]),
            ])
        }
    }

    impl Mul for LegGr4 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            let a = self.0;
            let b = rhs.0;
            let mut c = [0u32; 8];
            for i in 0..4 {
                for j in 0..4 {
                    c[i + j] = c[i + j].wrapping_add(a[i].wrapping_mul(b[j]));
                }
            }
            let c4 = c[4];
            let c5 = c[5];
            let c6 = c[6];
            LegGr4([
                c[0].wrapping_sub(c4),
                c[1].wrapping_sub(c4).wrapping_sub(c5),
                c[2].wrapping_sub(c5).wrapping_sub(c6),
                c[3].wrapping_sub(c6),
            ])
        }
    }

    #[inline(always)]
    fn mul_by_alpha(h: LegGr4) -> LegGr4 {
        let [h0, h1, h2, h3] = h.0;
        LegGr4([h3.wrapping_neg(), h0.wrapping_sub(h3), h1, h2])
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    #[repr(C, align(16))]
    pub struct LegGr128(pub [LegGr4; 32]);

    impl LegGr128 {
        pub const ZERO: Self = Self([LegGr4::ZERO; 32]);

        #[inline]
        pub fn mul_by_gr1(self, small: LegGr1) -> Self {
            let s = small.0;
            let mut out = Self::ZERO;
            for i in 0..32 {
                let f = self.0[i].0;
                out.0[i] = LegGr4([
                    s.wrapping_mul(f[0]),
                    s.wrapping_mul(f[1]),
                    s.wrapping_mul(f[2]),
                    s.wrapping_mul(f[3]),
                ]);
            }
            out
        }
    }

    impl Add for LegGr128 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            let mut out = Self::ZERO;
            for i in 0..32 {
                out.0[i] = self.0[i] + rhs.0[i];
            }
            out
        }
    }

    impl Mul for LegGr128 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            let mut c = [LegGr4::ZERO; 63];
            for i in 0..32 {
                for j in 0..32 {
                    c[i + j] = c[i + j] + self.0[i] * rhs.0[j];
                }
            }
            for i in (0..31).rev() {
                let hi = c[32 + i];
                c[3 + i] = c[3 + i] - hi;
                c[1 + i] = c[1 + i] - hi;
                c[0 + i] = c[0 + i] - mul_by_alpha(hi);
            }
            let mut out = Self::ZERO;
            for i in 0..32 {
                out.0[i] = c[i];
            }
            out
        }
    }
}

use legacy::{LegGr1, LegGr128, LegGr4};

// ---------------------------------------------------------------------------
// Deterministic fill (no rand dep), identical bit-pattern for both columns.
// ---------------------------------------------------------------------------
fn words(seed: u64) -> impl FnMut() -> u32 {
    let mut s = seed | 1;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 32) as u32
    }
}

fn leg128(seed: u64) -> LegGr128 {
    let mut next = words(seed);
    let mut a = [LegGr4::ZERO; 32];
    for s in a.iter_mut() {
        *s = LegGr4([next(), next(), next(), next()]);
    }
    LegGr128(a)
}
fn gen128(seed: u64) -> Gr128<u32> {
    let mut next = words(seed);
    let mut a = [Gr4::<u32>::ZERO; 32];
    for s in a.iter_mut() {
        *s = Gr4([next(), next(), next(), next()]);
    }
    Gr128(a)
}

fn bench(c: &mut Criterion) {
    // gr4 mul
    {
        let mut g = c.benchmark_group("overhead/gr4_mul");
        let (la, lb) = (LegGr4([1, 2, 3, 4]), LegGr4([5, 6, 7, 8]));
        let (ga, gb) = (Gr4::<u32>([1, 2, 3, 4]), Gr4::<u32>([5, 6, 7, 8]));
        g.bench_function(BenchmarkId::new("legacy", ""), |b| {
            b.iter(|| black_box(black_box(la) * black_box(lb)))
        });
        g.bench_function(BenchmarkId::new("generic", ""), |b| {
            b.iter(|| black_box(black_box(ga) * black_box(gb)))
        });
        g.finish();
    }
    // gr128 mul (the 1024-mul schoolbook — the load-bearing hot path)
    {
        let mut g = c.benchmark_group("overhead/gr128_mul");
        let (la, lb) = (leg128(1), leg128(2));
        let (ga, gb) = (gen128(1), gen128(2));
        g.bench_function(BenchmarkId::new("legacy", ""), |b| {
            b.iter(|| black_box(black_box(la) * black_box(lb)))
        });
        g.bench_function(BenchmarkId::new("generic", ""), |b| {
            b.iter(|| black_box(black_box(ga) * black_box(gb)))
        });
        g.finish();
    }
    // gr128 add
    {
        let mut g = c.benchmark_group("overhead/gr128_add");
        let (la, lb) = (leg128(1), leg128(2));
        let (ga, gb) = (gen128(1), gen128(2));
        g.bench_function(BenchmarkId::new("legacy", ""), |b| {
            b.iter(|| black_box(black_box(la) + black_box(lb)))
        });
        g.bench_function(BenchmarkId::new("generic", ""), |b| {
            b.iter(|| black_box(black_box(ga) + black_box(gb)))
        });
        g.finish();
    }
    // gr128 mul_by_gr1 (R0 fast path)
    {
        let mut g = c.benchmark_group("overhead/gr128_mul_by_base");
        let lf = leg128(3);
        let gf = gen128(3);
        let (ls, gs) = (LegGr1(0x9E37_79B9), Gr1::<u32>(0x9E37_79B9));
        g.bench_function(BenchmarkId::new("legacy", ""), |b| {
            b.iter(|| black_box(black_box(lf).mul_by_gr1(black_box(ls))))
        });
        g.bench_function(BenchmarkId::new("generic", ""), |b| {
            b.iter(|| black_box(black_box(gf).mul_by_gr1(black_box(gs))))
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
