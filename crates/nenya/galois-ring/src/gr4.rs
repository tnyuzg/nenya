//! `R_1 = GR(2^n, 4) = (Z/2^n)[X] / (X^4 + X + 1)`.
//!
//! `f1(X) = X^4 + X + 1`; `f1 mod 2` is irreducible over `F_2` (residue field
//! `F_16`). The polynomial is defined mod 2, so it is identical for every base
//! width — only the limb type `L` changes.
//!
//! Elements are `[a0, a1, a2, a3] ∈ (Z/2^n)^4`, representing
//! `a0 + a1·X + a2·X² + a3·X³`.
//!
//! Reduction rule: `X^4 ≡ -X - 1` (wrapping arithmetic in `L`).
//!
//! Mul strategy:
//!   1. Schoolbook: 16 wrapping `L` muls → `c[0..6]`
//!   2. Reduce via `X^4=-X-1, X^5=-X²-X, X^6=-X³-X²`: 6 wrapping subs.
//!
//! An earlier version used hand-written NEON intrinsics for add/sub/neg/mul.
//! In benchmarks they measured ~3× slower than the scalar code below, because
//! each intrinsic call forced a stack round-trip that kept LLVM from holding
//! intermediates in registers across the `Gr128` schoolbook. Scalar code
//! auto-vectorizes here (and each monomorphization preserves that), so we left
//! it scalar.

use crate::limb::Limb;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
// `transparent` (= the layout of `[L; 4]`) rather than `align(16)`: forcing
// 16-byte size padded the narrow widths catastrophically (a `Gr4<u8>` is 4
// bytes of data, so `align(16)` wasted 12 bytes and scattered `Gr128<u8>`'s
// limbs across 512 bytes, blocking SIMD). With the natural layout, `Gr128<u8>`
// is 128 contiguous bytes that vectorize. `Gr4<u32>` is 16 bytes either way.
#[repr(transparent)]
pub struct Gr4<L: Limb>(pub [L; 4]);

impl<L: Limb> Gr4<L> {
    pub const ZERO: Self = Self([L::ZERO; 4]);
    pub const ONE: Self = Self([L::ONE, L::ZERO, L::ZERO, L::ZERO]);

    /// Embed an `R_0` element as the constant polynomial.
    #[inline(always)]
    pub fn from_gr1(x: crate::Gr1<L>) -> Self {
        Self([x.0, L::ZERO, L::ZERO, L::ZERO])
    }

    /// True iff a unit (image in `F_16` is nonzero), i.e. some coefficient odd.
    #[inline(always)]
    pub fn is_unit(self) -> bool {
        self.0.iter().any(|c| c.is_odd())
    }

    /// Repeated squaring: `self^n` (for Teichmüller computation).
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
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

impl<L: Limb> Add for Gr4<L> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Gr4([
            self.0[0].wrapping_add(rhs.0[0]),
            self.0[1].wrapping_add(rhs.0[1]),
            self.0[2].wrapping_add(rhs.0[2]),
            self.0[3].wrapping_add(rhs.0[3]),
        ])
    }
}

impl<L: Limb> Sub for Gr4<L> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Gr4([
            self.0[0].wrapping_sub(rhs.0[0]),
            self.0[1].wrapping_sub(rhs.0[1]),
            self.0[2].wrapping_sub(rhs.0[2]),
            self.0[3].wrapping_sub(rhs.0[3]),
        ])
    }
}

impl<L: Limb> Neg for Gr4<L> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Gr4([
            self.0[0].wrapping_neg(),
            self.0[1].wrapping_neg(),
            self.0[2].wrapping_neg(),
            self.0[3].wrapping_neg(),
        ])
    }
}

impl<L: Limb> Mul for Gr4<L> {
    type Output = Self;
    /// Schoolbook polynomial multiplication modulo `f1 = X^4 + X + 1`.
    ///
    /// 1. 4×4 schoolbook → `c[0..7]` (16 wrapping mul-adds).
    /// 2. Reduce via `X^4=-X-1, X^5=-X^2-X, X^6=-X^3-X^2` (6 wrapping subs).
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let a = self.0;
        let b = rhs.0;
        let mut c = [L::ZERO; 8];
        for i in 0..4 {
            for j in 0..4 {
                c[i + j] = c[i + j].wrapping_add(a[i].wrapping_mul(b[j]));
            }
        }
        let c4 = c[4];
        let c5 = c[5];
        let c6 = c[6];
        Gr4([
            c[0].wrapping_sub(c4),
            c[1].wrapping_sub(c4).wrapping_sub(c5),
            c[2].wrapping_sub(c5).wrapping_sub(c6),
            c[3].wrapping_sub(c6),
        ])
    }
}
