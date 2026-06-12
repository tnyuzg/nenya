use galois_ring::teichmuller::compute_t1;
use galois_ring::{GR32_1, GR32_128, GR32_4, T1};

// ---------------------------------------------------------------------------
// GR32_1 tests
// ---------------------------------------------------------------------------

#[test]
fn gr1_add_zero() {
    let a = GR32_1(42);
    assert_eq!(a + GR32_1::ZERO, a);
}

#[test]
fn gr1_wrapping() {
    assert_eq!(GR32_1(u32::MAX) + GR32_1(1), GR32_1(0));
    assert_eq!(GR32_1(0) - GR32_1(1), GR32_1(u32::MAX));
}

#[test]
fn gr1_neg() {
    let a = GR32_1(7);
    assert_eq!(a + (-a), GR32_1::ZERO);
}

#[test]
fn gr1_mul_one() {
    let a = GR32_1(123456789);
    assert_eq!(a * GR32_1::ONE, a);
    assert_eq!(a * GR32_1::ZERO, GR32_1::ZERO);
}

#[test]
fn gr1_is_unit() {
    assert!(GR32_1(1).is_unit());
    assert!(GR32_1(3).is_unit());
    assert!(!GR32_1(0).is_unit());
    assert!(!GR32_1(2).is_unit());
    assert!(!GR32_1(4).is_unit());
}

// ---------------------------------------------------------------------------
// GR32_4 tests
// ---------------------------------------------------------------------------

#[test]
fn gr4_is_unit() {
    assert!(GR32_4::ONE.is_unit());
    assert!(!GR32_4::ZERO.is_unit());
    assert!(GR32_4([1, 0, 0, 0]).is_unit()); // constant 1
    assert!(GR32_4([0, 1, 0, 0]).is_unit()); // X — odd coeff
    assert!(!GR32_4([2, 4, 6, 8]).is_unit()); // all even
}

// ---------------------------------------------------------------------------
// Teichmüller tests
// ---------------------------------------------------------------------------

/// Print computed T1 — run with `cargo test print_t1 -- --nocapture`
#[test]
fn print_t1() {
    let t1 = compute_t1();
    println!("\nComputed T1 table:");
    for (i, e) in t1.iter().enumerate() {
        println!(
            "  T1[{:2}] = GR32_4([0x{:08x}, 0x{:08x}, 0x{:08x}, 0x{:08x}]),",
            i, e.0[0], e.0[1], e.0[2], e.0[3]
        );
    }
}

/// The hardcoded const T1 must match what `compute_t1()` produces — otherwise
/// any consumer of T1 sees an element that does NOT satisfy x^16 = x.
#[test]
fn t1_const_matches_computed() {
    let computed = compute_t1();
    for i in 0..16 {
        assert_eq!(
            T1[i], computed[i],
            "T1[{i}] const drifted from compute_t1()"
        );
    }
}

// ---------------------------------------------------------------------------
// GR32_128 tests
// ---------------------------------------------------------------------------

#[test]
fn gr128_is_unit() {
    assert!(GR32_128::ONE.is_unit());
    assert!(!GR32_128::ZERO.is_unit());
    assert!(GR32_128::from_gr1(GR32_1(3)).is_unit());
}

// ===========================================================================
// Width-generic coverage (u8 / u16 / u32 / u64).
//
// This is the primary suite: the `width_tests!` macro instantiates a per-width
// module (`w8`/`w16`/`w32`/`w64`) of the generic core fns below — ring laws,
// defining relations, `mul_by_*` agreement, and Teichmüller lifts — so every
// invariant is checked at each width (the extension structure d_1 = 4,
// d_2 = 128 is width-independent; only the base ring Z/2^n changes).
//
// The concrete `u32` tests above are only the bits this suite does NOT cover:
// the `Gr1` newtype operators, `is_unit` edge cases (incl. the negative
// all-even case), and validation of the hardcoded `u32` `T1` const.
//
// `inv_2adic` is the one u64-only code path with no direct in-tree assertion,
// so it gets dedicated edge-case coverage below.
// ===========================================================================

use galois_ring::teichmuller::{compute_t1_generic, teichmuller_lift_r2};
use galois_ring::{Gr128, Gr4, Limb};

/// Tiny SplitMix64-flavored LCG so the generic tests get pseudo-random
/// full-width limbs without depending on `rand` semantics. Returns a closure
/// producing one `L` per call (low/high bits both exercised via `from_u64`).
fn limb_lcg<L: Limb>(seed: u64) -> impl FnMut() -> L {
    let mut s: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Mix high and low halves before narrowing, so widths < 64 still see
        // entropy from the high bits of the LCG state.
        L::from_u64(s ^ (s >> 32))
    }
}

fn rand_gr4<L: Limb>(seed: u64) -> Gr4<L> {
    let mut next = limb_lcg::<L>(seed);
    Gr4([next(), next(), next(), next()])
}

fn rand_gr128<L: Limb>(seed: u64) -> Gr128<L> {
    let mut next = limb_lcg::<L>(seed);
    let mut arr = [Gr4::<L>::ZERO; 32];
    for slot in arr.iter_mut() {
        *slot = Gr4([next(), next(), next(), next()]);
    }
    Gr128(arr)
}

// ---------------------------------------------------------------------------
// 1. inv_2adic round-trip (the sole un-asserted u64 path).
// ---------------------------------------------------------------------------

/// For many odd values `a` (including width-specific edge cases),
/// `inv_2adic(a) * a ≡ 1 (mod 2^BITS)`.
fn check_inv_2adic<L: Limb>() {
    // Width-independent small odds + a spread of "interesting" 64-bit patterns
    // that fold down correctly via `from_u64` for narrower widths. Any even
    // value is skipped (units of Z/2^n are exactly the odd elements).
    let seeds: [u64; 12] = [
        1,
        3,
        5,
        15,
        0x8000_0000_0000_0001, // high bit + low bit set
        0xFFFF_FFFF_0000_0001, // Goldilocks-ish pattern
        u64::MAX,              // all ones (odd at every width)
        0xDEAD_BEEF_DEAD_BEEF,
        0x0123_4567_89AB_CDEF,
        0xEEEE_EEEF_EEEE_EEEF,
        0x7FFF_FFFF_FFFF_FFFF,
        0xCAFE_BABE_CAFE_BABD,
    ];
    for &seed in &seeds {
        let a = L::from_u64(seed);
        if !a.is_odd() {
            // `seed` folded to an even residue at this width — not a unit.
            continue;
        }
        let inv = a.inv_2adic();
        assert_eq!(
            a.wrapping_mul(inv),
            L::ONE,
            "inv_2adic round-trip failed for seed=0x{seed:016x} (BITS={})",
            L::BITS
        );
        // Inverse is symmetric: inv is itself a unit and inverts back to `a`.
        assert!(inv.is_odd(), "inv of a unit must be a unit (seed=0x{seed:016x})");
    }
    // Exhaustive sweep of every odd element for the narrow widths, where it is
    // cheap (256 / 65536 values). This pins the full inverse table at u8/u16.
    if L::BITS <= 16 {
        let n: u64 = 1u64 << L::BITS;
        let mut a_raw: u64 = 1;
        while a_raw < n {
            let a = L::from_u64(a_raw);
            assert_eq!(
                a.wrapping_mul(a.inv_2adic()),
                L::ONE,
                "inv_2adic exhaustive failed for a={a_raw} (BITS={})",
                L::BITS
            );
            a_raw += 2;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Gr4<L> ring laws + defining relation X^4 = -X - 1.
// ---------------------------------------------------------------------------

fn check_gr4_ring_laws<L: Limb>() {
    let a = rand_gr4::<L>(1);
    let b = rand_gr4::<L>(2);
    let c = rand_gr4::<L>(3);

    // Additive identity / inverse.
    assert_eq!(a + Gr4::<L>::ZERO, a, "Gr4 add zero");
    assert_eq!(a + (-a), Gr4::<L>::ZERO, "Gr4 add inverse");
    assert_eq!(a - a, Gr4::<L>::ZERO, "Gr4 sub self");

    // Add commutative / associative.
    assert_eq!(a + b, b + a, "Gr4 add comm");
    assert_eq!((a + b) + c, a + (b + c), "Gr4 add assoc");

    // Multiplicative identity / annihilator.
    assert_eq!(a * Gr4::<L>::ONE, a, "Gr4 mul one");
    assert_eq!(a * Gr4::<L>::ZERO, Gr4::<L>::ZERO, "Gr4 mul zero");

    // Mul commutative / associative / distributive.
    assert_eq!(a * b, b * a, "Gr4 mul comm");
    assert_eq!((a * b) * c, a * (b * c), "Gr4 mul assoc");
    assert_eq!(a * (b + c), a * b + a * c, "Gr4 distrib");
}

/// X^4 + X + 1 = 0  ⟹  alpha^4 = -alpha - 1, at every width.
fn check_gr4_relation<L: Limb>() {
    let alpha = Gr4([L::ZERO, L::ONE, L::ZERO, L::ZERO]);
    let alpha4 = alpha * alpha * alpha * alpha;
    let expected = -alpha - Gr4::<L>::ONE;
    assert_eq!(alpha4, expected, "ring relation X^4 = -X-1 failed");
}

// ---------------------------------------------------------------------------
// 3. Gr128<L> ring laws + defining relation f2(β)=0, plus mul_by_* agreement.
// ---------------------------------------------------------------------------

fn gr128_gen<L: Limb>() -> Gr128<L> {
    // β = (0, 1, 0, ..., 0): the generator Y of Gr4<L>[Y]/(f2).
    let mut arr = [Gr4::<L>::ZERO; 32];
    arr[1] = Gr4::<L>::ONE;
    Gr128(arr)
}

fn check_gr128_ring_laws<L: Limb>() {
    let beta = gr128_gen::<L>();
    let a = beta * beta + Gr128::<L>::ONE;
    let b = beta + Gr128::from_gr4(Gr4([L::from_u64(3), L::ONE, L::ZERO, L::ZERO]));
    let c = beta * beta * beta + beta;

    // Additive identity / inverse.
    assert_eq!(a + Gr128::<L>::ZERO, a, "Gr128 add zero");
    assert_eq!(a + (-a), Gr128::<L>::ZERO, "Gr128 add inverse");
    assert_eq!(a - a, Gr128::<L>::ZERO, "Gr128 sub self");

    // Multiplicative identity / annihilator.
    assert_eq!(a * Gr128::<L>::ONE, a, "Gr128 mul one");
    assert_eq!(a * Gr128::<L>::ZERO, Gr128::<L>::ZERO, "Gr128 mul zero");

    // Mul commutative / associative / distributive.
    assert_eq!(a * b, b * a, "Gr128 mul comm");
    assert_eq!((a * b) * c, a * (b * c), "Gr128 mul assoc");
    assert_eq!(a * (b + c), a * b + a * c, "Gr128 distrib");

    // Embedding Gr4 ↪ Gr128 is a ring hom.
    let x = rand_gr4::<L>(10);
    let y = rand_gr4::<L>(11);
    assert_eq!(
        Gr128::from_gr4(x) * Gr128::from_gr4(y),
        Gr128::from_gr4(x * y),
        "Gr128 embed: mul not compatible"
    );
    assert_eq!(
        Gr128::from_gr4(x) + Gr128::from_gr4(y),
        Gr128::from_gr4(x + y),
        "Gr128 embed: add not compatible"
    );
}

/// f2(β) = β^32 + β^3 + β + α = 0  in Gr128<L>, at every width.
fn check_gr128_relation<L: Limb>() {
    let beta = gr128_gen::<L>();
    let alpha = Gr128::from_gr4(Gr4([L::ZERO, L::ONE, L::ZERO, L::ZERO]));

    let beta3 = beta * beta * beta;
    let beta32 = beta.pow(32);

    let result = beta32 + beta3 + beta + alpha;
    assert_eq!(result, Gr128::<L>::ZERO, "ring relation f2(β)=0 failed");
}

/// `mul_by_gr1` and `mul_by_gr4` are specialized fast paths; they must
/// agree with the generic schoolbook `mul` against the corresponding lift.
fn check_gr128_mul_by_small<L: Limb>() {
    for seed in 0u64..8 {
        let f = rand_gr128::<L>(seed);

        // R_0 scalar: f.mul_by_gr1(s) == f * lift(s).
        let mut next = limb_lcg::<L>(seed.wrapping_add(50));
        let small1 = galois_ring::Gr1(next());
        assert_eq!(
            f.mul_by_gr1(small1),
            f * Gr128::from_gr1(small1),
            "mul_by_gr1 ≠ schoolbook (seed={seed}, BITS={})",
            L::BITS
        );

        // R_1 scalar: f.mul_by_gr4(s) == f * lift(s).
        let small4 = rand_gr4::<L>(seed.wrapping_add(100));
        assert_eq!(
            f.mul_by_gr4(small4),
            f * Gr128::from_gr4(small4),
            "mul_by_gr4 ≠ schoolbook (seed={seed}, BITS={})",
            L::BITS
        );
    }
    // Edge scalars: 0 (annihilates) and 1 (identity), both paths.
    let f = rand_gr128::<L>(123);
    assert_eq!(f.mul_by_gr1(galois_ring::Gr1(L::ZERO)), Gr128::<L>::ZERO);
    assert_eq!(f.mul_by_gr1(galois_ring::Gr1(L::ONE)), f);
    assert_eq!(f.mul_by_gr4(Gr4::<L>::ZERO), Gr128::<L>::ZERO);
    assert_eq!(f.mul_by_gr4(Gr4::<L>::ONE), f);
}

// ---------------------------------------------------------------------------
// 4. Teichmüller lifts at every width (mirrors the u32 `tests_r2` + T1 tests).
// ---------------------------------------------------------------------------

/// Apply x ↦ x^{2^128} once (128 squarings). Checks the fixed-point property
/// without needing a u128 exponent.
fn frob_q<L: Limb>(mut x: Gr128<L>) -> Gr128<L> {
    for _ in 0..128 {
        x = x * x;
    }
    x
}

/// `teichmuller_lift_r2` produces a fixed point of x ↦ x^{2^128}, and is the
/// identity on 0, on 1, and on the embedded T1 elements (already Teichmüller).
fn check_teichmuller_lift_r2<L: Limb>() {
    assert_eq!(
        teichmuller_lift_r2(Gr128::<L>::ZERO),
        Gr128::<L>::ZERO,
        "lift(0) ≠ 0"
    );
    assert_eq!(
        teichmuller_lift_r2(Gr128::<L>::ONE),
        Gr128::<L>::ONE,
        "lift(1) ≠ 1"
    );

    // Several "random" natural lifts (low bits set ⇒ nontrivial F_{2^128} image).
    let natural_seeds: [[u64; 4]; 3] = [
        [0xdead_beef, 0, 0, 0],
        [0x1234_5678, 0x9abc_def0, 0, 0],
        [0x0000_0001, 0x0000_0010, 0x0000_0100, 0x0000_1000],
    ];
    for seed in natural_seeds {
        let mut arr = [Gr4::<L>::ZERO; 32];
        arr[0] = Gr4([
            L::from_u64(seed[0]),
            L::from_u64(seed[1]),
            L::from_u64(seed[2]),
            L::from_u64(seed[3]),
        ]);
        let natural = Gr128(arr);
        let t = teichmuller_lift_r2(natural);
        assert_eq!(frob_q(t), t, "lift not fixed under x ↦ x^{{2^128}}");
    }

    // T1[i] embedded as a constant polynomial is already Teichmüller in R_2,
    // so the lift fixes it.
    let t1 = compute_t1_generic::<L>();
    for (i, &t1i) in t1.iter().enumerate() {
        let embedded = Gr128::from_gr4(t1i);
        assert_eq!(
            teichmuller_lift_r2(embedded),
            embedded,
            "T1[{i}] embedding not fixed under R_2 Teichmüller lift"
        );
    }
}

/// `compute_t1::<L>` gives 16 Teichmüller elements of GR(2^n, 4):
/// each satisfies x^16 = x, the 15 nonzero ones satisfy x^15 = 1, and all
/// pairwise differences are units (unit-separated).
fn check_compute_t1<L: Limb>() {
    let t1 = compute_t1_generic::<L>();

    // x^16 = x for all 16 elements.
    for (i, &t) in t1.iter().enumerate() {
        assert_eq!(t.pow(16), t, "T1[{i}]^16 ≠ T1[{i}] (BITS={})", L::BITS);
    }
    // x^15 = 1 for the 15 nonzero elements.
    for (i, &t) in t1.iter().enumerate().skip(1) {
        assert_eq!(
            t.pow(15),
            Gr4::<L>::ONE,
            "T1[{i}]^15 ≠ 1 (BITS={})",
            L::BITS
        );
    }
    // Pairwise unit-separation.
    for i in 0..16 {
        for j in 0..16 {
            if i != j {
                assert!(
                    (t1[i] - t1[j]).is_unit(),
                    "T1[{i}] - T1[{j}] is not a unit (BITS={})",
                    L::BITS
                );
            }
        }
    }
}

macro_rules! width_tests {
    ($modname:ident, $t:ty) => {
        mod $modname {
            use super::*;

            #[test]
            fn inv_2adic_roundtrip() {
                check_inv_2adic::<$t>();
            }
            #[test]
            fn gr4_ring_laws() {
                check_gr4_ring_laws::<$t>();
            }
            #[test]
            fn gr4_relation() {
                check_gr4_relation::<$t>();
            }
            #[test]
            fn gr128_ring_laws() {
                check_gr128_ring_laws::<$t>();
            }
            #[test]
            fn gr128_relation() {
                check_gr128_relation::<$t>();
            }
            #[test]
            fn gr128_mul_by_small() {
                check_gr128_mul_by_small::<$t>();
            }
            #[test]
            fn teichmuller_lift_r2_fixed_point() {
                check_teichmuller_lift_r2::<$t>();
            }
            #[test]
            fn teichmuller_t1() {
                check_compute_t1::<$t>();
            }
        }
    };
}

width_tests!(w8, u8);
width_tests!(w16, u16);
width_tests!(w32, u32);
width_tests!(w64, u64);
