//! Tensor PCS for *generic* `R_2 = GR(2^n, 128)`-valued multilinears, generic
//! over the base-limb width `L`.
//!
//! Mirror of [`pcs_ring::Pcs`], differing only in:
//!
//! - Commits store full `Gr128<L>` entries (4× larger leaves than the
//!   `[L; 32]` packed form), since the message is not assumed to be in
//!   `Im(φ)`. Used for `q1, q2` in the SIMD SNARK.
//! - Encoding uses `RaaParams::encode` (the generic-`R_2` path) rather
//!   than `encode_packed`.
//! - Internal Fiat–Shamir over `(root, m̄)` derives query indices, same
//!   convention as `pcs_ring`.
//!
//! Layered on top of `pcs_ring::raa::RaaParams` and
//! `pcs_ring::merkle::{MerkleTree, verify_path}`.

use blake3::Hasher;
use galois_ring::{compute_t1_generic, Gr128, Limb};
use pcs_ring::{
    merkle::{verify_path, MerkleTree},
    optimal_shape_for,
    raa::RaaParams,
    recommended_query_size, Hash,
};
use piop_ring::build_eq_table;
use rayon::prelude::*;
use reduction::in_kernel_with_t1;
use std::time::Instant;

/// Generic-R_2 tensor PCS. `L` defaults to `u32` so existing call sites stay
/// concrete; the SIMD SNARK instantiates `R2Pcs<L>`.
#[derive(Clone)]
pub struct R2Pcs<L: Limb = u32> {
    pub params: RaaParams<L>,
    pub k1: usize,
    pub k2: usize,
    /// Number of column queries per opening.
    pub ell: usize,
}

impl<L: Limb> R2Pcs<L> {
    pub fn new(k1: usize, k2: usize, q: usize, ell: usize, seed: u64) -> Self {
        assert!(k1.is_power_of_two() && k2.is_power_of_two());
        Self {
            params: RaaParams::setup(k2, q, seed),
            k1,
            k2,
            ell,
        }
    }

    /// Soundness-driven setup for a length-`2^num_vars` error polynomial: derive
    /// the query budget from `(delta, lambda)` and pick the proof-size-optimal
    /// shape. Unlike [`pcs_ring::Pcs`], this PCS sends full `Gr128` column
    /// entries (128 limbs, same as an `m̄` entry), so the optimal split is the
    /// symmetric one. Mirrors `Pcs::setup`.
    pub fn setup(num_vars: usize, q: usize, delta: f64, lambda: usize, seed: u64) -> Self {
        let elem = 128 * (L::BITS / 8) as usize; // Gr128 for both m̄ and columns
        let t = recommended_query_size(delta, lambda);
        let (k1, k2) = optimal_shape_for(num_vars, q, t, elem, elem);
        Self::new(k1, k2, q, t, seed)
    }

    pub fn num_vars(&self) -> usize {
        (self.k1 * self.k2).trailing_zeros() as usize
    }

    pub fn n(&self) -> usize {
        self.params.n
    }
}

pub struct R2Commitment<L: Limb = u32> {
    pub root: Hash,
    /// Encoded matrix `M̂`, row-major over `(i ∈ [k_1], j ∈ [n])`.
    pub encoded: Vec<Gr128<L>>,
    /// Original message, row-major `(i ∈ [k_1], j ∈ [k_2])`.
    pub message: Vec<Gr128<L>>,
    pub tree: MerkleTree,
}

/// Commitment to `s` polynomials stacked along the **height** (`k_1`) axis, so
/// they share one Merkle tree. The message matrix is `(s·k_1) × k_2`, with rows
/// `[t·k_1, (t+1)·k_1)` holding polynomial `t`. Encoding/leaf layout is exactly
/// [`R2Commitment`]'s with the height extended to `s·k_1` — `n`, the leaf count,
/// and `derive_query_indices` are unchanged, so a single opening (one root, one
/// `m̄`, `ell` paths) serves a weighted combination of all `s` polynomials. The
/// stacking variable absorbs into `k_1` (not `k_2`) because the RAA code length
/// is bound to `k_2` ([`R2Pcs::new`]).
pub struct StackedR2Commitment<L: Limb = u32> {
    pub root: Hash,
    /// Encoded matrix, row-major over `(i ∈ [s·k_1], j ∈ [n])`.
    pub encoded: Vec<Gr128<L>>,
    /// Stacked message, row-major over `(i ∈ [s·k_1], j ∈ [k_2])`.
    pub messages: Vec<Gr128<L>>,
    /// Number of stacked polynomials.
    pub s: usize,
    pub tree: MerkleTree,
}

#[derive(Clone, Debug, PartialEq)]
pub struct R2OpeningProof<L: Limb = u32> {
    pub mbar: Vec<Gr128<L>>,
    pub query_indices: Vec<u32>,
    pub query_columns: Vec<Vec<Gr128<L>>>,
    pub query_paths: Vec<Vec<Hash>>,
}

impl<L: Limb> R2OpeningProof<L> {
    /// Serialize proof to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let limb_bytes = (L::BITS / 8) as usize;

        // mbar: Vec<Gr128<L>>
        buf.extend_from_slice(&(self.mbar.len() as u64).to_le_bytes());
        for gr in &self.mbar {
            for c in gr.0.iter() {
                for w in c.0.iter() {
                    buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
                }
            }
        }

        // query_indices: Vec<u32>
        buf.extend_from_slice(&(self.query_indices.len() as u64).to_le_bytes());
        for idx in &self.query_indices {
            buf.extend_from_slice(&idx.to_le_bytes());
        }

        // query_columns: Vec<Vec<Gr128<L>>>
        buf.extend_from_slice(&(self.query_columns.len() as u64).to_le_bytes());
        for col in &self.query_columns {
            buf.extend_from_slice(&(col.len() as u64).to_le_bytes());
            for gr in col {
                for c in gr.0.iter() {
                    for w in c.0.iter() {
                        buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
                    }
                }
            }
        }

        // query_paths: Vec<Vec<Hash>>
        buf.extend_from_slice(&(self.query_paths.len() as u64).to_le_bytes());
        for path in &self.query_paths {
            buf.extend_from_slice(&(path.len() as u64).to_le_bytes());
            for hash in path {
                buf.extend_from_slice(hash);
            }
        }

        buf
    }

    /// Actual serialized size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.serialize().len()
    }
}

// ---------------------------------------------------------------------------
// Free-subspace (kernel) check
// ---------------------------------------------------------------------------

/// Free-subspace (kernel) check on the opened columns of an **error**
/// commitment: every Merkle-authenticated codeword entry in the opening must
/// lie in `ker ψ`.
///
/// Soundness (paper §4.2.2, "submodule checks are free"): the lifted code is
/// `R_0`-linear and `ker ψ` is an `R_0`-submodule, so a codeword column inherits
/// kernel membership from its message row; the proximity gap then lifts this
/// `ell`-column spot-check to "the committed error is a kernel-valued codeword",
/// hence the opened evaluation `ê(z)` lies in `ker ψ`. Forcing `ê ∈ ker ψ`
/// turns the mult-check into a real R1CS check: a dishonest `ê = â∘b̂ − ĉ` with
/// `a∘b ≠ c` satisfies `ψ(ê) = a∘b − c ≠ 0`, so it is rejected here.
///
/// This is **pure verifier work**: it reads only the columns already present in
/// the proof, adding nothing to the prover or the proof. Apply it ONLY to
/// commitments that are supposed to be kernel errors (`ê, ê_mult', ê_A, ê_B`),
/// never to generic `R_2` data (e.g. `w_phi` lives in `Im φ`, not `ker ψ`).
pub fn columns_in_kernel<L: Limb>(proof: &R2OpeningProof<L>) -> bool {
    let t1 = compute_t1_generic::<L>();
    proof
        .query_columns
        .par_iter()
        .all(|col| col.iter().all(|e| in_kernel_with_t1(e, &t1)))
}

// ---------------------------------------------------------------------------
// Hashing (width-correct: low BITS/8 bytes per limb)
// ---------------------------------------------------------------------------

fn hash_r2_bytes<L: Limb>(h: &mut Hasher, x: &Gr128<L>) {
    let nbytes = (L::BITS / 8) as usize;
    for c in x.0.iter() {
        for w in c.0.iter() {
            h.update(&w.to_u64().to_le_bytes()[..nbytes]);
        }
    }
}

/// Hash column `j` of an encoded `k1 × n` row-major matrix.
fn hash_r2_column<L: Limb>(flat: &[Gr128<L>], k1: usize, n: usize, j: usize) -> Hash {
    let mut h = Hasher::new();
    for i in 0..k1 {
        hash_r2_bytes(&mut h, &flat[i * n + j]);
    }
    *h.finalize().as_bytes()
}

fn hash_r2_column_standalone<L: Limb>(col: &[Gr128<L>]) -> Hash {
    let mut h = Hasher::new();
    for entry in col {
        hash_r2_bytes(&mut h, entry);
    }
    *h.finalize().as_bytes()
}

fn derive_query_indices<L: Limb>(root: Hash, mbar: &[Gr128<L>], n: usize, ell: usize) -> Vec<u32> {
    assert!(n > 0);
    let nbytes = (L::BITS / 8) as usize;
    let mut h = Hasher::new();
    h.update(b"snark-ring/r2-pcs/fs/queries");
    h.update(&root);
    for elem in mbar {
        for c in elem.0.iter() {
            for w in c.0.iter() {
                h.update(&w.to_u64().to_le_bytes()[..nbytes]);
            }
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

// ---------------------------------------------------------------------------
// Protocol
// ---------------------------------------------------------------------------

impl<L: Limb> R2Pcs<L> {
    pub fn commit(&self, evals: &[Gr128<L>]) -> R2Commitment<L> {
        self.commit_with_timing(evals).0
    }

    /// Commit and return `(commitment, encode_ms, merkle_ms)` — the RAA-encode
    /// time and the leaf-hash + tree-build time, split out for benchmarking.
    pub fn commit_with_timing(&self, evals: &[Gr128<L>]) -> (R2Commitment<L>, f64, f64) {
        assert_eq!(evals.len(), self.k1 * self.k2);
        let n = self.params.n;

        let t = Instant::now();
        // Rows are independent (each writes its own length-`n` chunk) → parallel
        // over `k1`; the per-row RAA encode stays serial (sequential prefix-sums).
        let mut encoded = vec![Gr128::<L>::ZERO; self.k1 * n];
        encoded.par_chunks_mut(n).enumerate().for_each(|(i, dst)| {
            let row = &evals[i * self.k2..(i + 1) * self.k2];
            dst.copy_from_slice(&self.params.encode(row));
        });
        let encode_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        // Disjoint column strides → parallel leaf hashing (order-preserving
        // collect ⇒ identical root to the serial loop).
        let leaves: Vec<Hash> = (0..n)
            .into_par_iter()
            .map(|j| hash_r2_column(&encoded, self.k1, n, j))
            .collect();
        let tree = MerkleTree::build(leaves);
        let root = tree.root();
        let merkle_ms = t.elapsed().as_secs_f64() * 1e3;

        (
            R2Commitment {
                root,
                encoded,
                message: evals.to_vec(),
                tree,
            },
            encode_ms,
            merkle_ms,
        )
    }

    /// Open at `z`. Returns `f(z)` and the opening proof.
    pub fn open(&self, comm: &R2Commitment<L>, z: &[Gr128<L>]) -> (Gr128<L>, R2OpeningProof<L>) {
        let (e0, e1) = self.split_eq(z);
        let n = self.params.n;

        // m̄ = e_0^⊤ · M, length k_2. Parallelize over the disjoint output column
        // `j` (each sums over `i` in the serial order ⇒ bit-identical).
        let mut mbar = vec![Gr128::<L>::ZERO; self.k2];
        mbar.par_iter_mut().enumerate().for_each(|(j, out)| {
            let mut acc = Gr128::<L>::ZERO;
            for i in 0..self.k1 {
                acc = acc + e0[i] * comm.message[i * self.k2 + j];
            }
            *out = acc;
        });

        // y = ⟨m̄, e_1⟩.
        let mut y = Gr128::<L>::ZERO;
        for j in 0..self.k2 {
            y = y + mbar[j] * e1[j];
        }

        let query_indices = derive_query_indices(comm.root, &mbar, n, self.ell);

        let mut query_columns: Vec<Vec<Gr128<L>>> = Vec::with_capacity(self.ell);
        let mut query_paths: Vec<Vec<Hash>> = Vec::with_capacity(self.ell);
        for &j in &query_indices {
            let j = j as usize;
            let col: Vec<Gr128<L>> = (0..self.k1).map(|i| comm.encoded[i * n + j]).collect();
            query_columns.push(col);
            query_paths.push(comm.tree.open(j));
        }

        (
            y,
            R2OpeningProof {
                mbar,
                query_indices,
                query_columns,
                query_paths,
            },
        )
    }

    pub fn verify(
        &self,
        root: Hash,
        z: &[Gr128<L>],
        y: Gr128<L>,
        proof: &R2OpeningProof<L>,
    ) -> bool {
        let (e0, e1) = self.split_eq(z);
        let n = self.params.n;

        if proof.mbar.len() != self.k2 {
            return false;
        }
        let mut y_chk = Gr128::<L>::ZERO;
        for j in 0..self.k2 {
            y_chk = y_chk + proof.mbar[j] * e1[j];
        }
        if y_chk != y {
            return false;
        }

        let expected = derive_query_indices(root, &proof.mbar, n, self.ell);
        if expected != proof.query_indices {
            return false;
        }

        let ubar = self.params.encode(&proof.mbar);

        for q in 0..proof.query_indices.len() {
            let j = proof.query_indices[q] as usize;
            if j >= n {
                return false;
            }
            let col = &proof.query_columns[q];
            if col.len() != self.k1 {
                return false;
            }

            // acc_j = Σ_i e_0[i] · col[i].
            let mut acc = Gr128::<L>::ZERO;
            for i in 0..self.k1 {
                acc = acc + e0[i] * col[i];
            }
            if acc != ubar[j] {
                return false;
            }

            let leaf = hash_r2_column_standalone(col);
            if !verify_path(leaf, j, &proof.query_paths[q], root) {
                return false;
            }
        }
        true
    }

    fn split_eq(&self, z: &[Gr128<L>]) -> (Vec<Gr128<L>>, Vec<Gr128<L>>) {
        assert_eq!(z.len(), self.num_vars());
        let log_k1 = self.k1.trailing_zeros() as usize;
        let e0 = build_eq_table(&z[..log_k1]).evals().to_vec();
        let e1 = build_eq_table(&z[log_k1..]).evals().to_vec();
        (e0, e1)
    }

    // -----------------------------------------------------------------------
    // Same-point stacking: commit `s` polynomials under one tree (height axis),
    // open a *weighted* combination `Σ_t w_t · poly_t` at a single point for the
    // cost of one opening. Used to bind ê_A and ê_B (both opened at s₁)
    // separately while paying one root / one m̄ / `ell` paths.
    // -----------------------------------------------------------------------

    /// Commit `messages.len()` polynomials (each length `k_1·k_2`) stacked along
    /// the height axis. See [`StackedR2Commitment`].
    pub fn commit_stacked(&self, messages: &[&[Gr128<L>]]) -> StackedR2Commitment<L> {
        let s = messages.len();
        assert!(s >= 1);
        for m in messages {
            assert_eq!(m.len(), self.k1 * self.k2);
        }
        let n = self.params.n;
        let rows = s * self.k1;

        // Stacked message, row-major (s·k1)×k2: rows [t·k1, (t+1)·k1) = poly t.
        let mut stacked = vec![Gr128::<L>::ZERO; rows * self.k2];
        for (t, m) in messages.iter().enumerate() {
            stacked[t * self.k1 * self.k2..(t + 1) * self.k1 * self.k2].copy_from_slice(m);
        }

        // Encode each of the `s·k1` rows independently (parallel over rows).
        let mut encoded = vec![Gr128::<L>::ZERO; rows * n];
        encoded.par_chunks_mut(n).enumerate().for_each(|(i, dst)| {
            let row = &stacked[i * self.k2..(i + 1) * self.k2];
            dst.copy_from_slice(&self.params.encode(row));
        });

        // Column-Merkle over `n` columns, each now `s·k1` entries tall.
        let leaves: Vec<Hash> = (0..n)
            .into_par_iter()
            .map(|j| hash_r2_column(&encoded, rows, n, j))
            .collect();
        let tree = MerkleTree::build(leaves);
        let root = tree.root();

        StackedR2Commitment {
            root,
            encoded,
            messages: stacked,
            s,
            tree,
        }
    }

    /// `commit_stacked` returning `(commitment, encode_ms, merkle_ms)` for Exp 2.
    pub fn commit_stacked_with_timing(
        &self,
        messages: &[&[Gr128<L>]],
    ) -> (StackedR2Commitment<L>, f64, f64) {
        use std::time::Instant;
        let s = messages.len();
        assert!(s >= 1);
        for m in messages {
            assert_eq!(m.len(), self.k1 * self.k2);
        }
        let n = self.params.n;
        let rows = s * self.k1;
        let mut stacked = vec![Gr128::<L>::ZERO; rows * self.k2];
        for (t, m) in messages.iter().enumerate() {
            stacked[t * self.k1 * self.k2..(t + 1) * self.k1 * self.k2].copy_from_slice(m);
        }
        let tm = Instant::now();
        let mut encoded = vec![Gr128::<L>::ZERO; rows * n];
        encoded.par_chunks_mut(n).enumerate().for_each(|(i, dst)| {
            let row = &stacked[i * self.k2..(i + 1) * self.k2];
            dst.copy_from_slice(&self.params.encode(row));
        });
        let encode_ms = tm.elapsed().as_secs_f64() * 1e3;
        let tm = Instant::now();
        let leaves: Vec<Hash> = (0..n)
            .into_par_iter()
            .map(|j| hash_r2_column(&encoded, rows, n, j))
            .collect();
        let tree = MerkleTree::build(leaves);
        let root = tree.root();
        let merkle_ms = tm.elapsed().as_secs_f64() * 1e3;
        (
            StackedR2Commitment {
                root,
                encoded,
                messages: stacked,
                s,
                tree,
            },
            encode_ms,
            merkle_ms,
        )
    }

    /// `e_0` weighted per stacked block: block `t` is `y_weights[t] · e_0`.
    /// Length `s·k_1`. The raw weights (not an `eq` axis, whose weights would
    /// sum to 1) let the opening reproduce an arbitrary combination, e.g.
    /// `[ONE, γ]` ⇒ `ê_A(z) + γ·ê_B(z)`.
    fn stacked_e0(&self, e0: &[Gr128<L>], y_weights: &[Gr128<L>]) -> Vec<Gr128<L>> {
        let mut out = vec![Gr128::<L>::ZERO; y_weights.len() * self.k1];
        for (t, &w) in y_weights.iter().enumerate() {
            for i in 0..self.k1 {
                out[t * self.k1 + i] = w * e0[i];
            }
        }
        out
    }

    /// Open the weighted combination `Σ_t y_weights[t]·poly_t` at point `z`.
    /// Returns `(value, proof)` with `value = Σ_t y_weights[t]·poly_t(z)`.
    pub fn open_stacked(
        &self,
        comm: &StackedR2Commitment<L>,
        z: &[Gr128<L>],
        y_weights: &[Gr128<L>],
    ) -> (Gr128<L>, R2OpeningProof<L>) {
        assert_eq!(y_weights.len(), comm.s);
        let (e0, e1) = self.split_eq(z);
        let e0s = self.stacked_e0(&e0, y_weights);
        let n = self.params.n;
        let rows = comm.s * self.k1;

        // m̄[j] = Σ_i e0s[i]·M[i·k2+j] = Σ_t y_weights[t]·(e0ᵀ M_t)[j], length k2.
        let mut mbar = vec![Gr128::<L>::ZERO; self.k2];
        mbar.par_iter_mut().enumerate().for_each(|(j, out)| {
            let mut acc = Gr128::<L>::ZERO;
            for i in 0..rows {
                acc = acc + e0s[i] * comm.messages[i * self.k2 + j];
            }
            *out = acc;
        });

        let mut y = Gr128::<L>::ZERO;
        for j in 0..self.k2 {
            y = y + mbar[j] * e1[j];
        }

        let query_indices = derive_query_indices(comm.root, &mbar, n, self.ell);

        let mut query_columns: Vec<Vec<Gr128<L>>> = Vec::with_capacity(self.ell);
        let mut query_paths: Vec<Vec<Hash>> = Vec::with_capacity(self.ell);
        for &j in &query_indices {
            let j = j as usize;
            let col: Vec<Gr128<L>> = (0..rows).map(|i| comm.encoded[i * n + j]).collect();
            query_columns.push(col);
            query_paths.push(comm.tree.open(j));
        }

        (
            y,
            R2OpeningProof {
                mbar,
                query_indices,
                query_columns,
                query_paths,
            },
        )
    }

    /// Verify a stacked opening. `y_weights.len()` is the stacked count `s`.
    pub fn verify_stacked(
        &self,
        root: Hash,
        z: &[Gr128<L>],
        y_weights: &[Gr128<L>],
        value: Gr128<L>,
        proof: &R2OpeningProof<L>,
    ) -> bool {
        let (e0, e1) = self.split_eq(z);
        let e0s = self.stacked_e0(&e0, y_weights);
        let n = self.params.n;
        let rows = y_weights.len() * self.k1;

        if proof.mbar.len() != self.k2 {
            return false;
        }
        let mut y_chk = Gr128::<L>::ZERO;
        for j in 0..self.k2 {
            y_chk = y_chk + proof.mbar[j] * e1[j];
        }
        if y_chk != value {
            return false;
        }

        let expected = derive_query_indices(root, &proof.mbar, n, self.ell);
        if expected != proof.query_indices {
            return false;
        }

        // Encode is R_2-linear ⇒ Enc(Σ_t w_t·m̄_t) = Σ_t w_t·Enc(m̄_t), so the
        // weighted column combination must match the re-encoded combined m̄.
        let ubar = self.params.encode(&proof.mbar);

        for q in 0..proof.query_indices.len() {
            let j = proof.query_indices[q] as usize;
            if j >= n {
                return false;
            }
            let col = &proof.query_columns[q];
            if col.len() != rows {
                return false;
            }
            let mut acc = Gr128::<L>::ZERO;
            for i in 0..rows {
                acc = acc + e0s[i] * col[i];
            }
            if acc != ubar[j] {
                return false;
            }

            let leaf = hash_r2_column_standalone(col);
            if !verify_path(leaf, j, &proof.query_paths[q], root) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use galois_ring::{GR32_1, GR32_128, GR32_4};

    fn rand_r2(seed: u64) -> GR32_128 {
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

    fn rand_point(num_vars: usize, seed: u64) -> Vec<GR32_128> {
        (0..num_vars)
            .map(|i| GR32_128::from_gr1(GR32_1(seed.wrapping_add(i as u64) as u32)))
            .collect()
    }

    fn eval_r2_mle(evals: &[GR32_128], z: &[GR32_128]) -> GR32_128 {
        let table = build_eq_table(z);
        let t = table.evals();
        assert_eq!(t.len(), evals.len());
        let mut acc = GR32_128::ZERO;
        for i in 0..evals.len() {
            acc = acc + t[i] * evals[i];
        }
        acc
    }

    #[test]
    fn r2_pcs_open_verify_small() {
        let pcs = R2Pcs::new(8, 8, 4, 16, 0xDEAD_BEEF);
        let evals: Vec<GR32_128> = (0..64).map(|i| rand_r2(0xABCD ^ i as u64)).collect();
        let comm = pcs.commit(&evals);
        let z = rand_point(pcs.num_vars(), 0xCAFE);
        let (y, proof) = pcs.open(&comm, &z);
        let y_ref = eval_r2_mle(&evals, &z);
        assert_eq!(y, y_ref);
        assert!(pcs.verify(comm.root, &z, y, &proof));

        let bad_y = y + GR32_128::ONE;
        assert!(!pcs.verify(comm.root, &z, bad_y, &proof));
    }

    #[test]
    fn r2_pcs_tampered_column_rejects() {
        let pcs = R2Pcs::new(8, 8, 4, 8, 0x1234);
        let evals: Vec<GR32_128> = (0..64).map(|i| rand_r2(i as u64)).collect();
        let comm = pcs.commit(&evals);
        let z = rand_point(pcs.num_vars(), 0xBEEF);
        let (y, mut proof) = pcs.open(&comm, &z);
        proof.query_columns[0][0] = proof.query_columns[0][0] + GR32_128::ONE;
        assert!(!pcs.verify(comm.root, &z, y, &proof));
    }

    #[test]
    fn stacked_open_verify_round_trip() {
        let pcs = R2Pcs::new(8, 8, 4, 16, 0xDEAD_BEEF);
        let a: Vec<GR32_128> = (0..64).map(|i| rand_r2(0xA0 ^ i as u64)).collect();
        let b: Vec<GR32_128> = (0..64).map(|i| rand_r2(0xB0 ^ i as u64)).collect();
        let comm = pcs.commit_stacked(&[&a, &b]);
        let z = rand_point(pcs.num_vars(), 0xCAFE);
        let gamma = rand_r2(0x6A33);
        let (val, proof) = pcs.open_stacked(&comm, &z, &[GR32_128::ONE, gamma]);

        // value == ê_A(z) + γ·ê_B(z).
        let want = eval_r2_mle(&a, &z) + gamma * eval_r2_mle(&b, &z);
        assert_eq!(val, want);
        assert!(pcs.verify_stacked(comm.root, &z, &[GR32_128::ONE, gamma], val, &proof));

        // Wrong claimed value rejects.
        assert!(!pcs.verify_stacked(
            comm.root,
            &z,
            &[GR32_128::ONE, gamma],
            val + GR32_128::ONE,
            &proof
        ));
    }

    #[test]
    fn stacked_tampered_column_rejects() {
        let pcs = R2Pcs::new(8, 8, 4, 8, 0x1234);
        let a: Vec<GR32_128> = (0..64).map(|i| rand_r2(0x11 ^ i as u64)).collect();
        let b: Vec<GR32_128> = (0..64).map(|i| rand_r2(0x22 ^ i as u64)).collect();
        let comm = pcs.commit_stacked(&[&a, &b]);
        let z = rand_point(pcs.num_vars(), 0xBEEF);
        let w = [GR32_128::ONE, rand_r2(0x99)];
        let (val, mut proof) = pcs.open_stacked(&comm, &z, &w);
        // Tampering any entry of the (combined) column breaks the binding.
        proof.query_columns[0][0] = proof.query_columns[0][0] + GR32_128::ONE;
        assert!(!pcs.verify_stacked(comm.root, &z, &w, val, &proof));
    }

    /// The free-subspace check distinguishes kernel-valued commitments
    /// (accepted) from image-valued ones (rejected) — a malicious prover who
    /// commits a non-kernel error passes the PCS opening but fails here.
    #[test]
    fn columns_in_kernel_accepts_kernel_rejects_image() {
        use reduction::pack_r2_matrix;

        fn rand_a16(mut s: u64) -> [[GR32_1; 2]; 16] {
            let mut next = || {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                GR32_1((s >> 32) as u32)
            };
            let mut a = [[GR32_1::ZERO; 2]; 16];
            for row in a.iter_mut() {
                row[0] = next();
                row[1] = next();
            }
            a
        }

        let pcs = R2Pcs::new(8, 8, 4, 16, 0x9E11);
        let z = rand_point(pcs.num_vars(), 0xABCD);

        let mut kernel = Vec::with_capacity(64);
        let mut image = Vec::with_capacity(64);
        for i in 0..64u64 {
            let a = rand_a16(0x100 + i);
            let b = rand_a16(0x200 + i);
            let mut c = [[GR32_1::ZERO; 2]; 16];
            for r in 0..16 {
                for j in 0..2 {
                    c[r][j] = a[r][j] * b[r][j];
                }
            }
            // φ(a)·φ(b) − φ(a∘b) ∈ ker ψ.
            kernel.push(pack_r2_matrix(a) * pack_r2_matrix(b) - pack_r2_matrix(c));
            // φ(a) ∈ Im φ, nonzero ⇒ ∉ ker ψ.
            image.push(pack_r2_matrix(a));
        }

        let (_, kproof) = pcs.open(&pcs.commit(&kernel), &z);
        assert!(columns_in_kernel(&kproof), "kernel-valued columns must pass");

        let (_, iproof) = pcs.open(&pcs.commit(&image), &z);
        assert!(
            !columns_in_kernel(&iproof),
            "image-valued columns must fail the free-subspace check"
        );
    }

    #[test]
    fn stacked_weight_swap_changes_value() {
        // Pins the block ordering: weights [1,γ] (= ê_A + γ·ê_B) must differ
        // from [γ,1] (= γ·ê_A + ê_B) for generic inputs.
        let pcs = R2Pcs::new(8, 8, 4, 8, 0x5151);
        let a: Vec<GR32_128> = (0..64).map(|i| rand_r2(0x33 ^ i as u64)).collect();
        let b: Vec<GR32_128> = (0..64).map(|i| rand_r2(0x44 ^ i as u64)).collect();
        let comm = pcs.commit_stacked(&[&a, &b]);
        let z = rand_point(pcs.num_vars(), 0x0102);
        let gamma = rand_r2(0x7777);
        let (v1, _) = pcs.open_stacked(&comm, &z, &[GR32_128::ONE, gamma]);
        let (v2, _) = pcs.open_stacked(&comm, &z, &[gamma, GR32_128::ONE]);
        assert_ne!(v1, v2);
    }
}
