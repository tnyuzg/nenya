pub mod gr1;
pub mod gr128;
pub mod gr4;
pub mod limb;
pub mod sampling;
pub mod teichmuller;

pub use gr1::Gr1;
pub use gr128::Gr128;
pub use gr4::Gr4;
pub use limb::{ceil_log2, Limb};
pub use sampling::R2Sampler;
pub use teichmuller::{compute_t1, compute_t1_generic, teichmuller_lift_r2, OMEGA, T1};

// ---------------------------------------------------------------------------
// Backwards-compatible `u32` instantiation.
//
// The historical concrete types `GR32_1 / GR32_4 / GR32_128` are now the
// `u32` monomorphization of the generic tower. Existing code, tests, and
// benches keep compiling unchanged; new code can name `Gr1<u8>` etc.
// ---------------------------------------------------------------------------

pub type GR32_1 = Gr1<u32>;
pub type GR32_4 = Gr4<u32>;
pub type GR32_128 = Gr128<u32>;

// Constructor shims. A type alias to a generic struct cannot be used as a
// tuple-struct constructor (E0423), so we provide same-named `const fn`s in
// the *value* namespace — mirroring how a real tuple struct occupies both
// namespaces. This lets existing `GR32_4([..])` / `GR32_128(arr)` / `GR32_1(x)`
// construction sites keep compiling unchanged. Zero-cost (`#[inline(always)]`).
#[allow(non_snake_case)]
#[inline(always)]
pub const fn GR32_1(x: u32) -> Gr1<u32> {
    Gr1(x)
}
#[allow(non_snake_case)]
#[inline(always)]
pub const fn GR32_4(x: [u32; 4]) -> Gr4<u32> {
    Gr4(x)
}
#[allow(non_snake_case)]
#[inline(always)]
pub const fn GR32_128(x: [Gr4<u32>; 32]) -> Gr128<u32> {
    Gr128(x)
}

/// The `u32` Teichmüller sampler (historical name).
pub type R2TeichmullerSampler = R2Sampler<u32>;
