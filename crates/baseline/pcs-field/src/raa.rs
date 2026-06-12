//! Repeat–Accumulate–Accumulate code over BabyBear — the field mirror of
//! [`pcs_ring::RaaParams`]. Diagonals are nonzero `BB` (units), permutations
//! are random. `encode_base` is the committer path (message in `BB`);
//! `encode_ext` is the verifier path (`m̄` lives in the extension `BB4`,
//! diagonals lifted).

use crate::lift;
use field::{AbstractField, BB, BB4};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const P: u32 = 0x7800_0001; // BabyBear modulus

pub struct FieldRaa {
    pub k: usize,
    pub q: usize,
    pub n: usize,
    perm1: Vec<u32>,
    perm2: Vec<u32>,
    diag1: Vec<BB>,
    diag2: Vec<BB>,
}

impl FieldRaa {
    pub fn setup(k: usize, q: usize, seed: u64) -> Self {
        assert!(k > 0 && q >= 2);
        let n = k * q;
        let mut rng = StdRng::seed_from_u64(seed);
        let perm1 = sample_perm(&mut rng, n);
        let perm2 = sample_perm(&mut rng, n);
        let diag1 = (0..n).map(|_| nonzero_bb(&mut rng)).collect();
        let diag2 = (0..n).map(|_| nonzero_bb(&mut rng)).collect();
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

    /// Encode a length-`k` `BB` message into a length-`n` `BB` codeword.
    pub fn encode_base(&self, msg: &[BB]) -> Vec<BB> {
        debug_assert_eq!(msg.len(), self.k);
        let n = self.n;
        let mut y = Vec::with_capacity(n);
        for &m in msg {
            for _ in 0..self.q {
                y.push(m);
            }
        }
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            t.push(y[self.perm1[i] as usize] * self.diag1[i]);
        }
        for i in 1..n {
            t[i] = t[i] + t[i - 1];
        }
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            z.push(t[self.perm2[i] as usize] * self.diag2[i]);
        }
        for i in 1..n {
            z[i] = z[i] + z[i - 1];
        }
        z
    }

    /// Encode a length-`k` `BB4` message (verifier side); same perms/diagonals,
    /// diagonal scalars lifted to `BB4`.
    pub fn encode_ext(&self, msg: &[BB4]) -> Vec<BB4> {
        debug_assert_eq!(msg.len(), self.k);
        let n = self.n;
        let mut y = Vec::with_capacity(n);
        for &m in msg {
            for _ in 0..self.q {
                y.push(m);
            }
        }
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            t.push(y[self.perm1[i] as usize] * lift(self.diag1[i]));
        }
        for i in 1..n {
            t[i] = t[i] + t[i - 1];
        }
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            z.push(t[self.perm2[i] as usize] * lift(self.diag2[i]));
        }
        for i in 1..n {
            z[i] = z[i] + z[i - 1];
        }
        z
    }
}

fn sample_perm(rng: &mut StdRng, n: usize) -> Vec<u32> {
    let mut p: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        p.swap(i, j);
    }
    p
}

fn nonzero_bb(rng: &mut StdRng) -> BB {
    let mut x = rng.gen::<u32>() % P;
    if x == 0 {
        x = 1;
    }
    BB::from_canonical_u32(x)
}
