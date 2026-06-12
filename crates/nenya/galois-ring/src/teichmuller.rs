//! Precomputed Teichmüller set T1 of GR(2^32, 4).
//!
//! T1 = { x ∈ GR(2^32,4) : x^16 = x }   (16 elements)
//! T1 = { 0, 1, ω, ω², …, ω^14 }  where ω = Teichmüller lift of α ∈ F_16.
//!
//! Ordering matches the paper §2.3.1:
//!   t_0 = 0 = T1[0]
//!   t_1 = 1 = T1[1]
//!   t_i = ω^{i-1} = T1[i]   for i = 2..=15
//!
//! The natural 0/1 lift of F_16 elements is NOT a Teichmüller element in
//! general. Values below were computed by iterating x ↦ x^16 until convergence
//! (32 iterations from the natural lift). See `compute_t1()`.

use crate::limb::{ceil_log2, Limb};
use crate::{Gr128, Gr4, GR32_4};

/// T1[0..16]: the 16 Teichmüller elements of GR(2^32, 4) (the `u32` tower).
/// Kept hardcoded as a backwards-compatible constant and as a known-good
/// reference that the generic [`compute_t1_generic`] is checked against.
pub const T1: [GR32_4; 16] = [
    GR32_4([0x00000000, 0x00000000, 0x00000000, 0x00000000]), // t_0  = 0
    GR32_4([0x00000001, 0x00000000, 0x00000000, 0x00000000]), // t_1  = 1
    GR32_4([0x332608c0, 0x65f5b943, 0x9eff0726, 0xae851724]), // t_2  = ω
    GR32_4([0x8b615184, 0xcfa3ff54, 0x1296acc9, 0xced422d4]), // t_3  = ω²
    GR32_4([0xd89a58de, 0xafdf9bae, 0xce5fd6c8, 0x20cdcbd3]), // t_4  = ω³
    GR32_4([0x7940e31f, 0x6742fcc7, 0x5a734460, 0xb6a8e4f8]), // t_5  = ω⁴
    GR32_4([0xf73af71c, 0x50bc4ae9, 0x9e74517f, 0x49a3f426]), // t_6  = ω⁵
    GR32_4([0x4d21cf70, 0x2674d304, 0x372e9f8b, 0x118269eb]), // t_7  = ω⁶
    GR32_4([0xacaf044d, 0x96867b2f, 0x6758e472, 0xd141a4ed]), // t_8  = ω⁷
    GR32_4([0x8940a031, 0x63234aa2, 0xf3f707b1, 0xcbfde110]), // t_9  = ω⁸
    GR32_4([0x12542d82, 0x8fd60bd1, 0x74611ae8, 0x6dc59203]), // t_10 = ω⁹
    GR32_4([0x08c508e3, 0xaf43b517, 0x618bae81, 0xb65c0bda]), // t_11 = ω^10
    GR32_4([0x7fb827d0, 0x739573b9, 0x29d2594f, 0x3ff87ef1]), // t_12 = ω^11
    GR32_4([0xc7efaa2f, 0x99d5857d, 0x86106ec5, 0x5fea383f]), // t_13 = ω^12
    GR32_4([0x9e2506f1, 0xecfd19b0, 0x3da92187, 0x1334531d]), // t_14 = ω^13
    GR32_4([0x746aef5f, 0x08e6f768, 0x312ba0b8, 0xdb918905]), // t_15 = ω^14
];

/// Generator of T1* = T1 \ {0}, multiplicative order 15.
/// ω = Teichmüller lift of α ∈ F_16 (primitive element, root of X^4+X+1 mod 2).
pub const OMEGA: GR32_4 = T1[2];

// ---------------------------------------------------------------------------
// Runtime computation (for verification / re-derivation)
// ---------------------------------------------------------------------------

/// Compute the Teichmüller lift of a natural F_16 element (0/1 coefficients)
/// by iterating `x ↦ x^16`. For this `q = 16` map the residue degree (`m = 4`)
/// is small relative to the base width, so convergence is roughly linear
/// (≥ 1 bit of 2-adic precision per step, then a fixed point). `BITS`
/// iterations therefore always suffice — and reproduce the historical
/// hardcoded `T1` for `u32` (which used 32). Cheap: a handful of `Gr4` muls.
pub fn compute_teichmuller_lift<L: Limb>(natural_lift: Gr4<L>) -> Gr4<L> {
    let mut x = natural_lift;
    for _ in 0..L::BITS {
        x = x.pow(16);
    }
    x
}

/// Recompute all 16 Teichmüller elements of `GR(2^n, 4)` for limb width `L`.
/// `T1[0] = 0`, `T1[i] = Teichmüller(α^{i-1})` for `i = 1..=15`, where
/// `α = (0,1,0,0)` is the natural lift of the primitive element of `F_16`.
pub fn compute_t1_generic<L: Limb>() -> [Gr4<L>; 16] {
    let mut result = [Gr4::<L>::ZERO; 16];
    result[0] = Gr4::<L>::ZERO;

    let alpha_lift = Gr4([L::ZERO, L::ONE, L::ZERO, L::ZERO]); // natural lift of α ∈ F_16
    let mut pow = Gr4::<L>::ONE; // starts at α^0 = 1
    for i in 1..16 {
        result[i] = compute_teichmuller_lift(pow);
        pow = pow * alpha_lift;
    }
    result
}

/// The `u32` Teichmüller set (historical no-type-parameter form).
pub fn compute_t1() -> [GR32_4; 16] {
    compute_t1_generic::<u32>()
}

// ---------------------------------------------------------------------------
// R₂ = GR(2³², 128) Teichmüller lift
// ---------------------------------------------------------------------------

/// Hensel-lift x ∈ R₂ to its Teichmüller representative.
///
/// The Teichmüller set T_2 ⊂ R₂ is { x : x^q = x } for q = |F_{2¹²⁸}| = 2¹²⁸.
/// It is unit-separated (any two distinct elements differ by a unit) and has
/// size 2¹²⁸ — exactly the challenge set the paper §3.1.1 prescribes for
/// sumcheck over R₂.
///
/// **Iteration.** The map φ(x) = x^q is the q-th power Frobenius lift; its
/// fixed points in R₂ are precisely the Teichmüller elements. Starting from
/// any natural lift `x` of α ∈ F_q, repeated application of φ converges to
/// `Teich(α)`. Each application doubles the 2-adic precision of the result, so
/// for the 32-bit modulus 2³² we need ⌈log₂(32)⌉ = 5 outer iterations.
///
/// Each outer iteration is q = 2¹²⁸ → 128 successive squarings → ~640 GR32_128
/// muls per lift. At ~910 ns/mul on Apple silicon that's ~580 µs per lift.
/// Sumcheck samples this O(log n) times per proof, so the total sampler cost
/// is small relative to per-round prover work for any nontrivial num_vars.
pub fn teichmuller_lift_r2<L: Limb>(natural: Gr128<L>) -> Gr128<L> {
    let mut x = natural;
    // ceil(log2(BITS)) + 1 outer iterations × 128 squarings.
    // (For u32: 6 × 128; precision doubles per outer step, the +1 is margin.)
    for _ in 0..ceil_log2(L::BITS) + 1 {
        for _ in 0..128 {
            x = x * x;
        }
    }
    x
}

#[cfg(test)]
mod tests_r2 {
    use super::*;
    use crate::GR32_128;

    /// Apply x ↦ x^{2¹²⁸} once (128 squarings). Used to check the
    /// fixed-point property without needing a u128 exponent.
    fn frob_q(mut x: GR32_128) -> GR32_128 {
        for _ in 0..128 {
            x = x * x;
        }
        x
    }

    #[test]
    fn lift_zero_is_zero() {
        assert_eq!(teichmuller_lift_r2(GR32_128::ZERO), GR32_128::ZERO);
    }

    #[test]
    fn lift_one_is_one() {
        assert_eq!(teichmuller_lift_r2(GR32_128::ONE), GR32_128::ONE);
    }

    #[test]
    fn lift_is_fixed_point_of_frobenius() {
        // Pick three "random" natural lifts (low bits set so they have nontrivial
        // image in F_{2¹²⁸}). Each lift must satisfy t == t^{2¹²⁸}.
        let natural_seeds: [[u32; 4]; 3] = [
            [0xdead_beef, 0, 0, 0],
            [0x1234_5678, 0x9abc_def0, 0, 0],
            [0x0000_0001, 0x0000_0010, 0x0000_0100, 0x0000_1000],
        ];
        for seed in natural_seeds {
            let mut arr = [GR32_4::ZERO; 32];
            arr[0] = GR32_4(seed);
            let natural = GR32_128(arr);
            let t = teichmuller_lift_r2(natural);
            assert_eq!(frob_q(t), t, "lift not fixed under x ↦ x^{{2¹²⁸}}");
        }
    }

    #[test]
    fn lift_agrees_with_t1_via_constant_embedding() {
        // T1[i] is already a Teichmüller element of GR32_4. Embedded as a
        // constant polynomial in GR32_128, it must remain a Teichmüller element
        // of GR32_128 — i.e. teichmuller_lift_r2 is the identity on its image.
        for i in 0..16 {
            let t1 = T1[i];
            let embedded = GR32_128::from_gr4(t1);
            assert_eq!(
                teichmuller_lift_r2(embedded),
                embedded,
                "T1[{}] embedding not fixed under R₂ Teichmüller lift",
                i
            );
        }
    }

    // -----------------------------------------------------------------------
    // Width-generic versions of the above (u8 / u16 / u32 / u64).
    //
    // The lift's iteration count is keyed off `L::BITS` (precision doubles per
    // outer Frobenius step), so its correctness is per-width; the macro below
    // pins all four widths. The concrete `u32` tests above are preserved.
    // -----------------------------------------------------------------------

    fn check_lift_fixed_point<L: Limb>() {
        // 0 and 1 are Teichmüller elements at every width.
        assert_eq!(teichmuller_lift_r2(Gr128::<L>::ZERO), Gr128::<L>::ZERO);
        assert_eq!(teichmuller_lift_r2(Gr128::<L>::ONE), Gr128::<L>::ONE);

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
            // One Frobenius application x ↦ x^{2¹²⁸} = 128 squarings.
            let mut frob = t;
            for _ in 0..128 {
                frob = frob * frob;
            }
            assert_eq!(frob, t, "lift not fixed under x ↦ x^{{2¹²⁸}} (BITS={})", L::BITS);
        }
    }

    fn check_lift_fixes_t1_embedding<L: Limb>() {
        let t1 = compute_t1_generic::<L>();
        for (i, &t1i) in t1.iter().enumerate() {
            let embedded = Gr128::from_gr4(t1i);
            assert_eq!(
                teichmuller_lift_r2(embedded),
                embedded,
                "T1[{i}] embedding not fixed under R₂ Teichmüller lift (BITS={})",
                L::BITS
            );
        }
    }

    macro_rules! width_tests {
        ($modname:ident, $t:ty) => {
            mod $modname {
                use super::*;

                #[test]
                fn lift_fixed_point() {
                    check_lift_fixed_point::<$t>();
                }
                #[test]
                fn lift_fixes_t1_embedding() {
                    check_lift_fixes_t1_embedding::<$t>();
                }
            }
        };
    }

    width_tests!(w8, u8);
    width_tests!(w16, u16);
    width_tests!(w32, u32);
    width_tests!(w64, u64);
}
