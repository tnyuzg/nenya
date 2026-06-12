//! Repeat–Accumulate–Accumulate code over `R = GR(2^32, 128)`.
//!
//! Variant B (diagonal-randomized) per `pcs-ring.md` §1.2, with the
//! diagonal entries restricted to `R_0^×` (odd `u32`s lifted via the natural
//! inclusion `R_0 ↪ R`). The fast path multiplies every u32 coordinate of
//! a `GR32_128` element by the same `GR32_1` scalar, so the encoder is
//! `Z/2^32`-linear **and** coord-wise on the 128 u32 components.
//!
//! Two encoders are provided:
//!
//! - [`RaaParams::encode`] — input `&[GR32_128]`, output `Vec<GR32_128>`. Used
//!   by the verifier to compute `ū = Enc(m̄)` where `m̄` is a general ring
//!   element (not in any packing image).
//! - [`RaaParams::encode_packed`] — input `&[[u32; 32]]`, output
//!   `Vec<[u32; 32]>`. Used by the committer when the message lives in the
//!   image of `reduction::pack_r2`. Runs 32 independent u32 encodes in one
//!   interleaved pass; same permutations and `diag{1,2}` values as `encode`.

use galois_ring::{Gr1, Gr128, Limb};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// `L` defaults to `u32` so this crate's own tests can write `RaaParams` /
/// `Pcs` without a width annotation; the SNARK uses `RaaParams<L>`.
#[derive(Clone, Debug)]
pub struct RaaParams<L: Limb = u32> {
    pub k: usize,
    /// Repetition factor / inverse rate.
    pub q: usize,
    /// Codeword length, `n = q · k`.
    pub n: usize,
    /// Permutation Π_1 of `[n]`. `perm1[i]` is the source position for output `i`.
    pub perm1: Vec<u32>,
    /// Permutation Π_2 of `[n]`.
    pub perm2: Vec<u32>,
    /// Diagonal D_1 (units of `R_0`, i.e. odd elements). Length `n`.
    pub diag1: Vec<Gr1<L>>,
    /// Diagonal D_2 (units of `R_0`). Length `n`.
    pub diag2: Vec<Gr1<L>>,
}

impl<L: Limb> RaaParams<L> {
    /// Sample fresh public parameters. The same seed reproduces the same code.
    pub fn setup(k: usize, q: usize, seed: u64) -> Self {
        assert!(k > 0 && q >= 2);
        let n = k * q;
        let mut rng = StdRng::seed_from_u64(seed);

        let perm1 = sample_permutation(&mut rng, n);
        let perm2 = sample_permutation(&mut rng, n);

        let diag1: Vec<Gr1<L>> = (0..n).map(|_| sample_unit(&mut rng)).collect();
        let diag2: Vec<Gr1<L>> = (0..n).map(|_| sample_unit(&mut rng)).collect();

        Self {
            k,
            q,
            n,
            perm1,
            perm2,
            diag1,
            diag2,
        }
    }

    /// Encode a length-`k` `R₂` message into a length-`n` `R₂` codeword.
    pub fn encode(&self, msg: &[Gr128<L>]) -> Vec<Gr128<L>> {
        assert_eq!(msg.len(), self.k);
        let n = self.n;
        let q = self.q;

        let mut y: Vec<Gr128<L>> = Vec::with_capacity(n);
        for &m in msg {
            for _ in 0..q {
                y.push(m);
            }
        }

        let mut t: Vec<Gr128<L>> = Vec::with_capacity(n);
        for i in 0..n {
            let src = self.perm1[i] as usize;
            t.push(y[src].mul_by_gr1(self.diag1[i]));
        }
        for i in 1..n {
            t[i] = t[i] + t[i - 1];
        }

        let mut z: Vec<Gr128<L>> = Vec::with_capacity(n);
        for i in 0..n {
            let src = self.perm2[i] as usize;
            z.push(t[src].mul_by_gr1(self.diag2[i]));
        }
        for i in 1..n {
            z[i] = z[i] + z[i - 1];
        }

        z
    }

    /// Encode a length-`k` packed message (32 base-limb slots per position)
    /// into a length-`n` packed codeword.
    ///
    /// Equivalent to `encode` lifted through `pack_r2`: for any `a` in the
    /// domain, `pack_r2_rows(encode_packed(a)) == encode(pack_r2_rows(a))`.
    /// See the `encode_commutes_with_pack` test.
    pub fn encode_packed(&self, msg: &[[L; 32]]) -> Vec<[L; 32]> {
        assert_eq!(msg.len(), self.k);
        let n = self.n;
        let q = self.q;

        let mut y: Vec<[L; 32]> = Vec::with_capacity(n);
        for &m in msg {
            for _ in 0..q {
                y.push(m);
            }
        }

        let mut t: Vec<[L; 32]> = Vec::with_capacity(n);
        for i in 0..n {
            let src = self.perm1[i] as usize;
            let d = self.diag1[i].0;
            let row = y[src];
            let mut out = [L::ZERO; 32];
            for s in 0..32 {
                out[s] = row[s].wrapping_mul(d);
            }
            t.push(out);
        }
        for i in 1..n {
            for s in 0..32 {
                t[i][s] = t[i][s].wrapping_add(t[i - 1][s]);
            }
        }

        let mut z: Vec<[L; 32]> = Vec::with_capacity(n);
        for i in 0..n {
            let src = self.perm2[i] as usize;
            let d = self.diag2[i].0;
            let row = t[src];
            let mut out = [L::ZERO; 32];
            for s in 0..32 {
                out[s] = row[s].wrapping_mul(d);
            }
            z.push(out);
        }
        for i in 1..n {
            for s in 0..32 {
                z[i][s] = z[i][s].wrapping_add(z[i - 1][s]);
            }
        }

        z
    }
}

fn sample_permutation(rng: &mut StdRng, n: usize) -> Vec<u32> {
    let mut p: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        p.swap(i, j);
    }
    p
}

/// A unit of `R₀ = Z/2^n`: any odd element. `| 1` forces the low bit; the
/// truncation in `from_u64` keeps it odd for every width.
fn sample_unit<L: Limb>(rng: &mut StdRng) -> Gr1<L> {
    Gr1(L::from_u64(rng.gen::<u64>() | 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use galois_ring::{GR32_1, GR32_128};
    use reduction::pack_r2;

    fn pack_rows(a: &[[u32; 32]]) -> Vec<GR32_128> {
        a.iter()
            .map(|row| {
                let mut e = [GR32_1::ZERO; 32];
                for s in 0..32 {
                    e[s] = GR32_1(row[s]);
                }
                pack_r2(e)
            })
            .collect()
    }

    #[test]
    fn encode_is_linear() {
        let params = RaaParams::setup(16, 4, 0xC0FFEE);
        let mut rng = StdRng::seed_from_u64(7);

        let mut a = vec![GR32_128::ZERO; params.k];
        let mut b = vec![GR32_128::ZERO; params.k];
        for i in 0..params.k {
            a[i] = GR32_128::from_gr1(GR32_1(rng.gen()));
            b[i] = GR32_128::from_gr1(GR32_1(rng.gen()));
        }

        let ea = params.encode(&a);
        let eb = params.encode(&b);

        let sum: Vec<GR32_128> = a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();
        let esum = params.encode(&sum);

        for i in 0..params.n {
            assert_eq!(esum[i], ea[i] + eb[i], "linearity failed at i={i}");
        }
    }

    #[test]
    fn encode_zero_is_zero() {
        let params = RaaParams::setup(8, 4, 1);
        let z = vec![GR32_128::ZERO; params.k];
        let c = params.encode(&z);
        for x in c {
            assert_eq!(x, GR32_128::ZERO);
        }
    }

    #[test]
    fn encode_packed_zero_is_zero() {
        let params = RaaParams::setup(8, 4, 1);
        let z = vec![[0u32; 32]; params.k];
        let c = params.encode_packed(&z);
        for row in c {
            assert_eq!(row, [0u32; 32]);
        }
    }

    /// The core soundness statement of the packed commit: encoding on the
    /// u32 slot representation is equivalent to encoding on the packed
    /// `GR32_128` representation.
    #[test]
    fn encode_commutes_with_pack() {
        for (k, q, seed) in [(16_usize, 4_usize, 1u64), (32, 4, 2), (8, 6, 3)] {
            let params = RaaParams::setup(k, q, seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0xABCD);
            let msg_packed: Vec<[u32; 32]> = (0..k)
                .map(|_| {
                    let mut a = [0u32; 32];
                    for s in 0..32 {
                        a[s] = rng.gen();
                    }
                    a
                })
                .collect();

            let packed_out = params.encode_packed(&msg_packed);
            let lhs = pack_rows(&packed_out);

            let msg_gr = pack_rows(&msg_packed);
            let rhs = params.encode(&msg_gr);

            assert_eq!(lhs.len(), rhs.len());
            for i in 0..lhs.len() {
                assert_eq!(lhs[i], rhs[i], "mismatch at i={i} (k={k}, q={q})");
            }
        }
    }

    /// Slot-rotation commute (Thm 4.12): the slot-uniform encoder commutes with
    /// a cyclic rotation of the 32 slots, `Enc(rot_ℓ·m) = rot_ℓ·Enc(m)`. This is
    /// the linchpin of the virtual rotated PCS opening for the general-circuit
    /// reduction — the verifier derives `ρ̂^(ℓ)`'s columns by slot-rotating `ŵ`'s
    /// already-opened columns, with no additional commitment.
    #[test]
    fn encode_commutes_with_slot_rotation() {
        // rot_ℓ(x)[i] = x[(i+ℓ) mod 32], matching `reduction::reduce::rot_l`.
        let rot = |e: &[u32; 32], ell: usize| {
            let mut o = [0u32; 32];
            for i in 0..32 {
                o[i] = e[(i + ell) % 32];
            }
            o
        };
        for (k, q, seed) in [(16_usize, 4_usize, 11u64), (8, 6, 22)] {
            let params = RaaParams::setup(k, q, seed);
            let mut rng = StdRng::seed_from_u64(seed ^ 0x5555);
            let msg: Vec<[u32; 32]> = (0..k)
                .map(|_| {
                    let mut a = [0u32; 32];
                    for s in 0..32 {
                        a[s] = rng.gen();
                    }
                    a
                })
                .collect();
            let enc = params.encode_packed(&msg);
            for ell in 0..32 {
                let rot_msg: Vec<[u32; 32]> = msg.iter().map(|m| rot(m, ell)).collect();
                let lhs = params.encode_packed(&rot_msg); // Enc(rot_ℓ·m)
                let rhs: Vec<[u32; 32]> = enc.iter().map(|c| rot(c, ell)).collect(); // rot_ℓ·Enc(m)
                assert_eq!(
                    lhs, rhs,
                    "encode/rotation commute failed at ell={ell} (k={k}, q={q})"
                );
            }
        }
    }
}
