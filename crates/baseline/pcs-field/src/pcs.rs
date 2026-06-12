//! Field tensor PCS over BabyBear — mirror of [`pcs_ring::Pcs`].
//!
//! Commit a `k1 × k2` matrix of `BB` evaluations, RAA-encode each row, blake3
//! Merkle-commit the columns. Open/verify at a `BB4` point with the merged
//! test+eval combiner `e_0 = eq(z^0, ·)` (Brakedown §4.4).

use crate::lift;
use crate::raa::FieldRaa;
use blake3::Hasher;
use field::{AbstractExtensionField, AbstractField, BB, BB4};
use p3_field::PrimeField32;
use pcs_ring::merkle::{verify_path, Hash, MerkleTree};
use rayon::prelude::*;

/// Multilinear eq-table, MSB-first in `z` (matches `piop`/`pcs-ring`
/// convention so the row/column split is an outer product).
pub fn eq_table(z: &[BB4]) -> Vec<BB4> {
    let mut t = vec![BB4::one()];
    for &zi in z {
        let mut nt = Vec::with_capacity(t.len() * 2);
        let one_minus = BB4::one() - zi;
        for &e in &t {
            nt.push(e * one_minus);
            nt.push(e * zi);
        }
        t = nt;
    }
    t
}

pub struct FieldPcs {
    pub raa: FieldRaa,
    pub k1: usize,
    pub k2: usize,
    pub ell: usize,
}

pub struct FieldCommitment {
    pub root: Hash,
    /// Encoded matrix, row-major `k1 × n` over `BB`.
    pub encoded: Vec<BB>,
    /// Original message, row-major `k1 × k2` over `BB`.
    pub message: Vec<BB>,
    pub tree: MerkleTree,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldOpening {
    pub mbar: Vec<BB4>,
    pub query_indices: Vec<u32>,
    pub query_columns: Vec<Vec<BB>>,
    pub query_paths: Vec<Vec<Hash>>,
}

impl FieldOpening {
    pub fn size_bytes(&self) -> usize {
        let mbar = self.mbar.len() * 16; // BB4 = 4 × u32
        let cols: usize = self.query_columns.iter().map(|c| c.len() * 4).sum();
        let paths: usize = self.query_paths.iter().map(|p| p.len() * 32).sum();
        let idx = self.query_indices.len() * 4;
        mbar + cols + paths + idx
    }
}

impl FieldPcs {
    pub fn new(k1: usize, k2: usize, q: usize, ell: usize, seed: u64) -> Self {
        assert!(k1.is_power_of_two() && k2.is_power_of_two());
        Self {
            raa: FieldRaa::setup(k2, q, seed),
            k1,
            k2,
            ell,
        }
    }

    /// Soundness-driven setup for a length-`2^num_vars` polynomial: derive the
    /// query budget from `(delta, lambda)` and pick the proof-size-optimal shape.
    /// Field mirror of `pcs_ring::Pcs::setup` — by the distance-preserving lift
    /// the field RAA code shares the ring's `delta`, so passing the same
    /// `(delta, lambda)` yields the same query count, keeping the comparison
    /// matched. Opening sends `m̄` as `BB4` (16 B) and column entries as `BB`
    /// (4 B), the same 4:1 ratio as `Pcs`.
    pub fn setup(num_vars: usize, q: usize, delta: f64, lambda: usize, seed: u64) -> Self {
        let t = pcs_ring::recommended_query_size(delta, lambda);
        let (k1, k2) = pcs_ring::optimal_shape_for(num_vars, q, t, 16, 4);
        Self::new(k1, k2, q, t, seed)
    }

    pub fn num_vars(&self) -> usize {
        (self.k1 * self.k2).trailing_zeros() as usize
    }

    pub fn n(&self) -> usize {
        self.raa.n
    }

    fn split_eq(&self, z: &[BB4]) -> (Vec<BB4>, Vec<BB4>) {
        assert_eq!(z.len(), self.num_vars());
        let log_k1 = self.k1.trailing_zeros() as usize;
        (eq_table(&z[..log_k1]), eq_table(&z[log_k1..]))
    }

    pub fn commit(&self, evals: &[BB]) -> FieldCommitment {
        assert_eq!(evals.len(), self.k1 * self.k2);
        let n = self.raa.n;
        // Rows are independent (each writes its own length-`n` chunk) → parallel
        // over `k1`; the per-row RAA encode stays serial (sequential prefix-sums).
        let mut encoded = vec![BB::zero(); self.k1 * n];
        encoded.par_chunks_mut(n).enumerate().for_each(|(i, dst)| {
            let row = &evals[i * self.k2..(i + 1) * self.k2];
            dst.copy_from_slice(&self.raa.encode_base(row));
        });
        // Disjoint column strides → parallel leaf hashing (order-preserving
        // collect ⇒ identical root to the serial loop).
        let leaves: Vec<Hash> = (0..n)
            .into_par_iter()
            .map(|j| hash_bb_column_strided(&encoded, self.k1, n, j))
            .collect();
        let tree = MerkleTree::build(leaves);
        let root = tree.root();
        FieldCommitment {
            root,
            encoded,
            message: evals.to_vec(),
            tree,
        }
    }

    /// Commit, returning `(commitment, encode_ms, merkle_ms)` — RAA-encode vs
    /// leaf-hash + tree build, split out for the cost breakdown (Exp 2).
    pub fn commit_with_timing(&self, evals: &[BB]) -> (FieldCommitment, f64, f64) {
        use std::time::Instant;
        assert_eq!(evals.len(), self.k1 * self.k2);
        let n = self.raa.n;
        let t = Instant::now();
        let mut encoded = vec![BB::zero(); self.k1 * n];
        encoded.par_chunks_mut(n).enumerate().for_each(|(i, dst)| {
            let row = &evals[i * self.k2..(i + 1) * self.k2];
            dst.copy_from_slice(&self.raa.encode_base(row));
        });
        let encode_ms = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let leaves: Vec<Hash> = (0..n)
            .into_par_iter()
            .map(|j| hash_bb_column_strided(&encoded, self.k1, n, j))
            .collect();
        let tree = MerkleTree::build(leaves);
        let root = tree.root();
        let merkle_ms = t.elapsed().as_secs_f64() * 1e3;
        (
            FieldCommitment {
                root,
                encoded,
                message: evals.to_vec(),
                tree,
            },
            encode_ms,
            merkle_ms,
        )
    }

    pub fn open(&self, comm: &FieldCommitment, z: &[BB4]) -> (BB4, FieldOpening) {
        let (e0, e1) = self.split_eq(z);
        let n = self.raa.n;

        // m̄ = Σ_i e0[i] · M[i, :]   ∈ BB4^{k2}. Parallelize over the disjoint
        // output column `j` (each sums over `i` in the serial order ⇒ bit-identical).
        let mut mbar = vec![BB4::zero(); self.k2];
        mbar.par_iter_mut().enumerate().for_each(|(j, out)| {
            let mut acc = BB4::zero();
            for i in 0..self.k1 {
                acc = acc + e0[i] * lift(comm.message[i * self.k2 + j]);
            }
            *out = acc;
        });
        let mut y = BB4::zero();
        for j in 0..self.k2 {
            y = y + mbar[j] * e1[j];
        }

        let query_indices = derive_query_indices(comm.root, &mbar, n, self.ell);
        let mut query_columns = Vec::with_capacity(self.ell);
        let mut query_paths = Vec::with_capacity(self.ell);
        for &j in &query_indices {
            let j = j as usize;
            let col: Vec<BB> = (0..self.k1).map(|i| comm.encoded[i * n + j]).collect();
            query_columns.push(col);
            query_paths.push(comm.tree.open(j));
        }
        (
            y,
            FieldOpening {
                mbar,
                query_indices,
                query_columns,
                query_paths,
            },
        )
    }

    pub fn verify(&self, root: Hash, z: &[BB4], y_claimed: BB4, proof: &FieldOpening) -> bool {
        let (e0, e1) = self.split_eq(z);
        let n = self.raa.n;
        if proof.mbar.len() != self.k2 {
            return false;
        }
        let mut y = BB4::zero();
        for j in 0..self.k2 {
            y = y + proof.mbar[j] * e1[j];
        }
        if y != y_claimed {
            return false;
        }
        let expected = derive_query_indices(root, &proof.mbar, n, self.ell);
        if expected != proof.query_indices {
            return false;
        }
        let ubar = self.raa.encode_ext(&proof.mbar);
        for q in 0..self.ell {
            let j = proof.query_indices[q] as usize;
            let col = &proof.query_columns[q];
            if col.len() != self.k1 {
                return false;
            }
            let mut acc = BB4::zero();
            for i in 0..self.k1 {
                acc = acc + e0[i] * lift(col[i]);
            }
            if acc != ubar[j] {
                return false;
            }
            let leaf = hash_bb_column(col);
            if !verify_path(leaf, j, &proof.query_paths[q], root) {
                return false;
            }
        }
        true
    }
}

fn hash_bb_column_strided(flat: &[BB], k1: usize, n: usize, j: usize) -> Hash {
    let mut h = Hasher::new();
    for i in 0..k1 {
        h.update(&flat[i * n + j].as_canonical_u32().to_le_bytes());
    }
    *h.finalize().as_bytes()
}

fn hash_bb_column(col: &[BB]) -> Hash {
    let mut h = Hasher::new();
    for e in col {
        h.update(&e.as_canonical_u32().to_le_bytes());
    }
    *h.finalize().as_bytes()
}

fn derive_query_indices(root: Hash, mbar: &[BB4], n: usize, ell: usize) -> Vec<u32> {
    let mut h = Hasher::new();
    h.update(b"pcs-field/fs/queries");
    h.update(&root);
    for elem in mbar {
        for c in <BB4 as AbstractExtensionField<BB>>::as_base_slice(elem) {
            h.update(&c.as_canonical_u32().to_le_bytes());
        }
    }
    let mut xof = h.finalize_xof();
    let mut out = Vec::with_capacity(ell);
    let n32 = n as u32;
    let mask = (n.next_power_of_two() - 1) as u32;
    let mut buf = [0u8; 4];
    while out.len() < ell {
        xof.fill(&mut buf);
        let v = u32::from_le_bytes(buf) & mask;
        if v < n32 {
            out.push(v);
        }
    }
    out
}
