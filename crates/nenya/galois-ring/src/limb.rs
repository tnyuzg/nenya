//! The base ring `Z/2^n` as a trait, so the whole Galois-ring tower can be
//! instantiated over any machine-word width without code duplication.
//!
//! `Limb` is implemented for `u8` / `u16` / `u32` / `u64`, giving base rings
//! `Z/2^8`, `Z/2^16`, `Z/2^32`, `Z/2^64`. The extension structure
//! (`d_1 = 4`, `d_2 = 128`, packing size `κ = 32`) is *independent* of the
//! base width — only the limb type and the precomputed packing constants
//! change. Soundness comes from `|T_2| = 2^128`, which is width-independent.
//!
//! The trait exposes only **wrapping** arithmetic (never the std `+`, which
//! panics on overflow for `u8`/`u16` in debug). Every method is
//! `#[inline(always)]`, so each monomorphization (`Gr128<u8>`, `Gr128<u32>`,
//! …) compiles to the same machine code a hand-written concrete type would —
//! no dynamic dispatch, and LLVM autovectorization of the fixed-length loops
//! is preserved.

use rand::Rng;

/// A base ring `Z/2^BITS`, realized by an unsigned machine word.
pub trait Limb:
    Copy + Clone + PartialEq + Eq + core::fmt::Debug + Default + Send + Sync + 'static
{
    const ZERO: Self;
    const ONE: Self;
    /// Bit width `n`; the base ring is `Z/2^n`. Drives Teichmüller iteration
    /// counts (precision doubles per step, so `ceil(log2(BITS))` suffice).
    const BITS: u32;

    fn wrapping_add(self, rhs: Self) -> Self;
    fn wrapping_sub(self, rhs: Self) -> Self;
    fn wrapping_mul(self, rhs: Self) -> Self;
    fn wrapping_neg(self) -> Self;

    /// True iff odd. In `Z/2^n` the units are exactly the odd elements, so
    /// this is the unit test for the base ring.
    fn is_odd(self) -> bool;

    /// 2-adic inverse of an odd element (Newton iteration). `self` must be a
    /// unit (odd); the result satisfies `self * inv ≡ 1 (mod 2^BITS)`.
    /// Six doublings give ≥ 64 correct bits — enough for every supported width.
    fn inv_2adic(self) -> Self;

    /// Inject a small integer literal, reduced mod `2^BITS`.
    fn from_u64(x: u64) -> Self;

    /// Widen to `u64` (zero-extend). Used for width-correct byte serialization
    /// (Merkle leaves, Fiat–Shamir): hash the low `BITS/8` bytes.
    fn to_u64(self) -> u64;

    /// Uniform sample over `Z/2^BITS`. Kept as a trait method so the `rand`
    /// `Distribution` bound does not leak onto every downstream generic.
    fn random<R: Rng>(rng: &mut R) -> Self;
}

macro_rules! impl_limb {
    ($t:ty) => {
        impl Limb for $t {
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const BITS: u32 = <$t>::BITS;

            #[inline(always)]
            fn wrapping_add(self, rhs: Self) -> Self {
                <$t>::wrapping_add(self, rhs)
            }
            #[inline(always)]
            fn wrapping_sub(self, rhs: Self) -> Self {
                <$t>::wrapping_sub(self, rhs)
            }
            #[inline(always)]
            fn wrapping_mul(self, rhs: Self) -> Self {
                <$t>::wrapping_mul(self, rhs)
            }
            #[inline(always)]
            fn wrapping_neg(self) -> Self {
                <$t>::wrapping_neg(self)
            }
            #[inline(always)]
            fn is_odd(self) -> bool {
                self & 1 == 1
            }
            #[inline(always)]
            fn from_u64(x: u64) -> Self {
                x as $t
            }
            #[inline(always)]
            fn to_u64(self) -> u64 {
                self as u64
            }
            #[inline]
            fn inv_2adic(self) -> Self {
                debug_assert!(self & 1 == 1, "inv_2adic requires an odd (unit) element");
                // Newton: x_{k+1} = x_k * (2 - a * x_k). Doubles correct bits
                // each step; 6 steps ⇒ ≥ 64 bits, covers u8..u64.
                let mut x: $t = 1;
                for _ in 0..6 {
                    x = x.wrapping_mul((2 as $t).wrapping_sub(self.wrapping_mul(x)));
                }
                debug_assert_eq!(self.wrapping_mul(x), 1);
                x
            }
            #[inline]
            fn random<R: Rng>(rng: &mut R) -> Self {
                rng.gen()
            }
        }
    };
}

impl_limb!(u8);
impl_limb!(u16);
impl_limb!(u32);
impl_limb!(u64);

/// `ceil(log2(n))` — the number of Frobenius/Hensel doublings needed to reach
/// full 2-adic precision for a width-`n` base ring. Used by the Teichmüller
/// lifts; over-provisioning is harmless (the lift is a fixed point), but this
/// keeps the cost minimal per width.
#[inline]
pub const fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        u32::BITS - (n - 1).leading_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inv_2adic_roundtrips() {
        for &x in &[1u32, 3, 5, 15, 0xeeee_eeefu32.wrapping_mul(1), 0x7fff_ffff] {
            if x & 1 == 1 {
                assert_eq!(x.wrapping_mul(Limb::inv_2adic(x)), 1);
            }
        }
        // 15^{-1} mod 2^32 is the well-known 0xeeeeeeef.
        assert_eq!(Limb::inv_2adic(15u32), 0xeeee_eeef);
        // and mod 2^8 it is 239.
        assert_eq!(Limb::inv_2adic(15u8), 239u8);
        assert_eq!(15u8.wrapping_mul(239u8), 1);
    }

    #[test]
    fn ceil_log2_values() {
        assert_eq!(ceil_log2(8), 3);
        assert_eq!(ceil_log2(16), 4);
        assert_eq!(ceil_log2(32), 5);
        assert_eq!(ceil_log2(64), 6);
    }
}
