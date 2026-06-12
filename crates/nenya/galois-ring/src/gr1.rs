//! `R_0 = GR(2^n, 1) = Z/2^n`, the base ring (one machine word).
//!
//! Generic over the limb type `L` (`u8`/`u16`/`u32`/`u64`). The historical
//! name `GR32_1` is the `u32` instantiation, kept as a type alias in `lib.rs`.

use crate::limb::Limb;
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
#[repr(transparent)]
pub struct Gr1<L: Limb>(pub L);

impl<L: Limb> Gr1<L> {
    pub const ZERO: Self = Self(L::ZERO);
    pub const ONE: Self = Self(L::ONE);

    /// Units of `Z/2^n` are exactly the odd elements.
    #[inline(always)]
    pub fn is_unit(self) -> bool {
        self.0.is_odd()
    }
}

impl<L: Limb> Add for Gr1<L> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl<L: Limb> Sub for Gr1<L> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl<L: Limb> Mul for Gr1<L> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0.wrapping_mul(rhs.0))
    }
}

impl<L: Limb> Neg for Gr1<L> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(self.0.wrapping_neg())
    }
}
