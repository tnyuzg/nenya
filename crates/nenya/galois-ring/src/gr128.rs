//! `R_2 = GR(2^n, 128) = Gr4<L>[Y] / f2(Y)`.
//!
//! `f2(Y) = Y^32 + Y^3 + Y + α`, where `α = Gr4([0,1,0,0])` is the natural lift
//! of the primitive element of `F_16`. `f2 mod 2` is irreducible over `F_16`
//! (verified by exhaustive search). Defined mod 2, so identical for every base
//! width — only the limb type `L` changes.
//!
//! Elements are `[a0, ..., a31] ∈ (Gr4<L>)^32`, representing
//! `a0 + a1·Y + ... + a31·Y^31`.
//!
//! Reduction rule: `Y^32 ≡ -Y^3 - Y - α` (wrapping `Gr4<L>` arithmetic).
//!
//! Mul strategy:
//!   1. Schoolbook: 1024 `Gr4<L>` muls → `c[0..63]`
//!   2. Reduce top-down (i = 30 down to 0, acting on `c[32+i]`):
//!        `c[3+i] -= c[32+i]; c[1+i] -= c[32+i]; c[0+i] -= mul_by_alpha(c[32+i])`
//!      31 specialized α-muls (each = 2 wrapping ops) + 93 `Gr4<L>` subs.

use crate::limb::Limb;
use crate::Gr4;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
// `transparent` (layout of `[Gr4<L>; 32]`): contiguous, no padding. For `u8`
// that's 128 bytes the autovectorizer can pack into SIMD; see `Gr4`.
#[repr(transparent)]
pub struct Gr128<L: Limb>(pub [Gr4<L>; 32]);

impl<L: Limb> Gr128<L> {
    pub const ZERO: Self = Self([Gr4::<L>::ZERO; 32]);
    pub const ONE: Self = {
        let mut arr = [Gr4::<L>::ZERO; 32];
        arr[0] = Gr4::<L>::ONE;
        Self(arr)
    };

    /// True iff a unit (image in `F_{2^128}` nonzero): some `Gr4` coeff is a unit.
    #[inline(always)]
    pub fn is_unit(self) -> bool {
        self.0.iter().any(|c| c.is_unit())
    }

    /// Repeated squaring: `self^n`.
    pub fn pow(self, mut n: u64) -> Self {
        let mut base = self;
        let mut acc = Self::ONE;
        while n > 0 {
            if n & 1 == 1 {
                acc = acc * base;
            }
            base = base * base;
            n >>= 1;
        }
        acc
    }

    /// Embed a `Gr4<L>` element as a constant polynomial in `Gr128<L>`.
    #[inline(always)]
    pub fn from_gr4(x: Gr4<L>) -> Self {
        let mut arr = [Gr4::<L>::ZERO; 32];
        arr[0] = x;
        Self(arr)
    }

    /// Embed an `R_0` element via the natural inclusion `R_0 ↪ R_2`.
    #[inline(always)]
    pub fn from_gr1(x: crate::Gr1<L>) -> Self {
        Self::from_gr4(Gr4::from_gr1(x))
    }

    /// Multiply by an `R_0` element lifted through `R_0 ↪ R_2`, skipping the
    /// generic 1024-inner-mul schoolbook.
    ///
    /// Under the lift, `s ∈ R_0` becomes `[Gr4([s,0,0,0]), 0, …, 0]`: only one
    /// slot is nonzero, so `s · f = Σ_i (s · f_i) Y^i` reduces to 128 wrapping
    /// muls with no reduction. Equivalent to `Gr128::from_gr1(s) * self`.
    ///
    /// Used by piop-ring's linear-check matvec.
    #[inline]
    pub fn mul_by_gr1(mut self, small: crate::Gr1<L>) -> Self {
        // "Multiply all 128 limbs by the scalar `s`." Flat in-place loop (not a
        // per-slot `Gr4` rebuild) so the autovectorizer packs it into a
        // scalar×vector multiply — `vpmullw` for `u8` (32 lanes), `vpmulld` for
        // `u32` — rather than running scalar. On the reduce hot path
        // (`mul_with_quotients` q1) and in the RAA encoder.
        let s = small.0;
        for i in 0..32 {
            for k in 0..4 {
                self.0[i].0[k] = self.0[i].0[k].wrapping_mul(s);
            }
        }
        self
    }

    /// Multiply by a `Gr4<L>` lifted through `R_1 ↪ R_2`, skipping the generic
    /// 1024-inner-mul schoolbook.
    ///
    /// `s ∈ R_1` lifts to `[s, 0, …, 0]`, so this is "32 independent `Gr4 × s`
    /// products" with no `Y^32` reduction. Implemented struct-of-arrays
    /// (transpose to 4 α-coordinate planes of 32, broadcast `s`'s coords, run
    /// the 4×4 α-schoolbook + `f1` reduction across the 32 lanes) so the
    /// per-lane `u8` muls vectorize through `vpmullw`. On the reduce hot path
    /// (`mul_with_quotients` q2 + `pack_r2`).
    #[inline]
    pub fn mul_by_gr4(self, small: Gr4<L>) -> Self {
        let [s0, s1, s2, s3] = small.0;
        // Transpose into 4 contiguous coordinate planes over the 32 Y-slots.
        let mut p0 = [L::ZERO; 32];
        let mut p1 = [L::ZERO; 32];
        let mut p2 = [L::ZERO; 32];
        let mut p3 = [L::ZERO; 32];
        for i in 0..32 {
            let v = self.0[i].0;
            p0[i] = v[0];
            p1[i] = v[1];
            p2[i] = v[2];
            p3[i] = v[3];
        }
        let mut out = Self::ZERO;
        for i in 0..32 {
            let (v0, v1, v2, v3) = (p0[i], p1[i], p2[i], p3[i]);
            // α-convolution of (v0..v3) with the broadcast scalar (s0..s3).
            let t0 = s0.wrapping_mul(v0);
            let t1 = s0.wrapping_mul(v1).wrapping_add(s1.wrapping_mul(v0));
            let t2 = s0
                .wrapping_mul(v2)
                .wrapping_add(s1.wrapping_mul(v1))
                .wrapping_add(s2.wrapping_mul(v0));
            let t3 = s0
                .wrapping_mul(v3)
                .wrapping_add(s1.wrapping_mul(v2))
                .wrapping_add(s2.wrapping_mul(v1))
                .wrapping_add(s3.wrapping_mul(v0));
            let t4 = s1
                .wrapping_mul(v3)
                .wrapping_add(s2.wrapping_mul(v2))
                .wrapping_add(s3.wrapping_mul(v1));
            let t5 = s2.wrapping_mul(v3).wrapping_add(s3.wrapping_mul(v2));
            let t6 = s3.wrapping_mul(v3);
            // Reduce mod f1 = X^4 + X + 1.
            out.0[i] = Gr4([
                t0.wrapping_sub(t4),
                t1.wrapping_sub(t4).wrapping_sub(t5),
                t2.wrapping_sub(t5).wrapping_sub(t6),
                t3.wrapping_sub(t6),
            ]);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

// The additive ops are written as flat in-place loops over the 128 contiguous
// limbs (not nested `Gr4` ops that build intermediate structs) so the
// autovectorizer packs them into SIMD: for `u8` on AVX-512BW the whole 128-byte
// `Gr128` add is a couple of `vpaddb`. Layout is `[[L;4];32]` == `[L;128]`.
impl<L: Limb> Add for Gr128<L> {
    type Output = Self;
    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..32 {
            for k in 0..4 {
                self.0[i].0[k] = self.0[i].0[k].wrapping_add(rhs.0[i].0[k]);
            }
        }
        self
    }
}

impl<L: Limb> Sub for Gr128<L> {
    type Output = Self;
    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self {
        for i in 0..32 {
            for k in 0..4 {
                self.0[i].0[k] = self.0[i].0[k].wrapping_sub(rhs.0[i].0[k]);
            }
        }
        self
    }
}

impl<L: Limb> Neg for Gr128<L> {
    type Output = Self;
    #[inline(always)]
    fn neg(mut self) -> Self {
        for i in 0..32 {
            for k in 0..4 {
                self.0[i].0[k] = self.0[i].0[k].wrapping_neg();
            }
        }
        self
    }
}

impl<L: Limb> Mul for Gr128<L> {
    type Output = Self;
    /// Struct-of-arrays schoolbook, structured for autovectorization.
    ///
    /// The naive `[Gr4;32]` schoolbook accesses each α-coordinate with stride 4,
    /// which the vectorizer can't pack. Here we transpose the multiplier into 4
    /// contiguous coordinate planes (`vp[c][j]`) and accumulate into 4 result
    /// planes (`cp[c][k]`). The hot inner loop is then a **scalar × contiguous
    /// vector** multiply-accumulate over the 32 Y-positions — exactly the shape
    /// LLVM turns into SIMD: `u8` lanes go through `vpmullw` (32 per AVX-512
    /// register, vs 16 for `u32`), so the narrow widths get more throughput.
    ///
    /// Inner α-product is the degree-3 schoolbook of `s·v` reduced mod
    /// `f1 = X^4 + X + 1` (`X^4 = -X-1`); the outer reduction is `Y^32 ≡
    /// -Y^3 - Y - α` with `α·(h0,h1,h2,h3) = (-h3, h0-h3, h1, h2)`.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Transpose rhs → contiguous coordinate planes over the 32 Y-positions.
        let mut vp = [[L::ZERO; 32]; 4];
        for j in 0..32 {
            let v = rhs.0[j].0;
            vp[0][j] = v[0];
            vp[1][j] = v[1];
            vp[2][j] = v[2];
            vp[3][j] = v[3];
        }

        // Accumulator planes over the 63 raw Y-positions.
        let mut cp = [[L::ZERO; 63]; 4];
        for i in 0..32 {
            let s = self.0[i].0; // broadcast α-scalars s0..s3
                                 // Scalar × vector multiply-accumulate over all 32 j (vectorizes).
            for j in 0..32 {
                let v0 = vp[0][j];
                let v1 = vp[1][j];
                let v2 = vp[2][j];
                let v3 = vp[3][j];
                // Raw α-convolution t0..t6.
                let t0 = s[0].wrapping_mul(v0);
                let t1 = s[0].wrapping_mul(v1).wrapping_add(s[1].wrapping_mul(v0));
                let t2 = s[0]
                    .wrapping_mul(v2)
                    .wrapping_add(s[1].wrapping_mul(v1))
                    .wrapping_add(s[2].wrapping_mul(v0));
                let t3 = s[0]
                    .wrapping_mul(v3)
                    .wrapping_add(s[1].wrapping_mul(v2))
                    .wrapping_add(s[2].wrapping_mul(v1))
                    .wrapping_add(s[3].wrapping_mul(v0));
                let t4 = s[1]
                    .wrapping_mul(v3)
                    .wrapping_add(s[2].wrapping_mul(v2))
                    .wrapping_add(s[3].wrapping_mul(v1));
                let t5 = s[2].wrapping_mul(v3).wrapping_add(s[3].wrapping_mul(v2));
                let t6 = s[3].wrapping_mul(v3);
                // Reduce mod f1 = X^4 + X + 1.
                let d0 = t0.wrapping_sub(t4);
                let d1 = t1.wrapping_sub(t4).wrapping_sub(t5);
                let d2 = t2.wrapping_sub(t5).wrapping_sub(t6);
                let d3 = t3.wrapping_sub(t6);
                let k = i + j;
                cp[0][k] = cp[0][k].wrapping_add(d0);
                cp[1][k] = cp[1][k].wrapping_add(d1);
                cp[2][k] = cp[2][k].wrapping_add(d2);
                cp[3][k] = cp[3][k].wrapping_add(d3);
            }
        }

        // Reduce Y^32 ≡ -Y^3 - Y - α, high-to-low (cascading), in SoA form.
        for i in (0..31).rev() {
            let h0 = cp[0][32 + i];
            let h1 = cp[1][32 + i];
            let h2 = cp[2][32 + i];
            let h3 = cp[3][32 + i];
            // c[3+i] -= h
            cp[0][3 + i] = cp[0][3 + i].wrapping_sub(h0);
            cp[1][3 + i] = cp[1][3 + i].wrapping_sub(h1);
            cp[2][3 + i] = cp[2][3 + i].wrapping_sub(h2);
            cp[3][3 + i] = cp[3][3 + i].wrapping_sub(h3);
            // c[1+i] -= h
            cp[0][1 + i] = cp[0][1 + i].wrapping_sub(h0);
            cp[1][1 + i] = cp[1][1 + i].wrapping_sub(h1);
            cp[2][1 + i] = cp[2][1 + i].wrapping_sub(h2);
            cp[3][1 + i] = cp[3][1 + i].wrapping_sub(h3);
            // c[i] -= α·h, where α·(h0,h1,h2,h3) = (-h3, h0-h3, h1, h2)
            cp[0][i] = cp[0][i].wrapping_sub(h3.wrapping_neg());
            cp[1][i] = cp[1][i].wrapping_sub(h0.wrapping_sub(h3));
            cp[2][i] = cp[2][i].wrapping_sub(h1);
            cp[3][i] = cp[3][i].wrapping_sub(h2);
        }

        // Pack the low 32 Y-positions back to AoS.
        let mut out = Self::ZERO;
        for k in 0..32 {
            out.0[k] = Gr4([cp[0][k], cp[1][k], cp[2][k], cp[3][k]]);
        }
        out
    }
}

#[cfg(test)]
mod tests_mul_by_gr1 {
    use crate::{GR32_1, GR32_128, GR32_4};

    fn rand_gr128(seed: u64) -> GR32_128 {
        // Tiny LCG so we don't need rand as a dev-dep here.
        let mut s: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 32) as u32
        };
        let mut arr = [GR32_4::ZERO; 32];
        for slot in arr.iter_mut() {
            *slot = GR32_4([next(), next(), next(), next()]);
        }
        GR32_128(arr)
    }

    #[test]
    fn mul_by_gr1_matches_generic_zero() {
        let f = rand_gr128(1);
        let g = f.mul_by_gr1(GR32_1(0));
        assert_eq!(g, GR32_128::ZERO);
    }

    #[test]
    fn mul_by_gr1_matches_generic_one() {
        let f = rand_gr128(2);
        let g = f.mul_by_gr1(GR32_1(1));
        assert_eq!(g, f);
    }

    #[test]
    fn mul_by_gr1_matches_generic_random() {
        for seed in 0u64..8 {
            let f = rand_gr128(seed);
            // Cycle small through a handful of u32 values, including some
            // non-units (even).
            let small = GR32_1(seed.wrapping_mul(0xDEADBEEF) as u32);
            let lhs = f.mul_by_gr1(small);
            let rhs = f * GR32_128::from_gr1(small);
            assert_eq!(lhs, rhs, "seed={seed} small={:?}", small);
        }
    }

    #[test]
    fn mul_by_gr1_is_commutative_with_lift() {
        // (lift(s) * f) should also equal f.mul_by_gr1(s).
        let f = rand_gr128(42);
        let small = GR32_1(0x1234_5678);
        let lhs = f.mul_by_gr1(small);
        let rhs = GR32_128::from_gr1(small) * f;
        assert_eq!(lhs, rhs);
    }
}

#[cfg(test)]
mod tests_mul_by_gr4 {
    use crate::{GR32_128, GR32_4};

    fn rand_gr128(seed: u64) -> GR32_128 {
        let mut s: u64 = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 32) as u32
        };
        let mut arr = [GR32_4::ZERO; 32];
        for slot in arr.iter_mut() {
            *slot = GR32_4([next(), next(), next(), next()]);
        }
        GR32_128(arr)
    }

    fn rand_gr4(seed: u64) -> GR32_4 {
        let mut s: u64 = seed.wrapping_mul(0xBF58_476D_1CE4_E5B9) | 1;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 32) as u32
        };
        GR32_4([next(), next(), next(), next()])
    }

    #[test]
    fn mul_by_gr4_matches_generic_zero() {
        let f = rand_gr128(1);
        let g = f.mul_by_gr4(GR32_4::ZERO);
        assert_eq!(g, GR32_128::ZERO);
    }

    #[test]
    fn mul_by_gr4_matches_generic_one() {
        let f = rand_gr128(2);
        let g = f.mul_by_gr4(GR32_4::ONE);
        assert_eq!(g, f);
    }

    #[test]
    fn mul_by_gr4_matches_generic_random() {
        for seed in 0u64..8 {
            let f = rand_gr128(seed);
            let small = rand_gr4(seed.wrapping_add(100));
            let lhs = f.mul_by_gr4(small);
            let rhs = f * GR32_128::from_gr4(small);
            assert_eq!(lhs, rhs, "seed={seed} small={:?}", small);
        }
    }

    #[test]
    fn mul_by_gr4_is_commutative_with_lift() {
        let f = rand_gr128(42);
        let small = GR32_4([0x1234_5678, 0x9ABC_DEF0, 0x0BAD_F00D, 0xCAFE_BABE]);
        let lhs = f.mul_by_gr4(small);
        let rhs = GR32_128::from_gr4(small) * f;
        assert_eq!(lhs, rhs);
    }
}
