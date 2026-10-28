//! Brakedown-style tensor PCS over `R₂ = GR(2^n, 128)`, packed commit, generic
//! over the base-limb width `L`.
//!
//! A multilinear polynomial in `m = log_2(N)` variables whose evaluations live
//! in the image of `reduction::pack_r2` is reshaped into a `k_1 × k_2` matrix
//! of packed tuples (32 base-limb slots per matrix entry). Each row is encoded
//! with the RAA fast-path code (`q = 1/rate`) into a length-`n` packed
//! codeword. The encoded matrix `M̂` is Merkle-committed column-wise; each
//! Merkle leaf hashes `k_1 × 32 × (BITS/8)` bytes (4× less than a full `R₂`
//! commit, and 4× less again for `u8` vs `u32`).
//!
//! Testing and evaluation are merged into a single pass using
//! `e_0 := eq(z^0, ·)` as both the proximity-test combiner and the
//! evaluation combiner (Brakedown §4.4 / Blaze §6).

use crate::merkle::{hash_column, hash_packed_column, verify_path, Hash, MerkleTree};
use crate::raa::RaaParams;
use blake3::Hasher;
use galois_ring::{Gr1, Gr128, Limb};
use piop_ring::build_eq_table;
use rayon::prelude::*;
use reduction::constants::HasPackingConstants;
use reduction::pack_r2;

// ---------------------------------------------------------------------------
// PCS
// ---------------------------------------------------------------------------

/// `L` defaults to `u32` so this crate's tests can write `Pcs`
/// unannotated; the SNARK instantiates `Pcs<L>`.
#[derive(Clone)]
pub struct Pcs<L: Limb = u32> {
    pub params: RaaParams<L>,
    pub k1: usize,
    pub k2: usize,
    /// Number of column queries per opening (`t`).
    pub ell: usize,
    /// The 32 constants `v_k := pack_r2(e_k)`, precomputed once. They let the
    /// verifier compute `Σ_i e_0[i] · pack_r2(col[i])` as
    /// `Σ_k v_k · Σ_i e_0[i].mul_by_gr1(col[i][k])` — no `pack_r2` per
    /// column entry. Stored in the struct rather than a per-type `static`.
    pub basis: Vec<Gr128<L>>,
}

impl<L: HasPackingConstants> Pcs<L> {
    /// Construct a PCS with an RAA fast-path code of shape `(k2, n = q·k2)`.
    pub fn new(k1: usize, k2: usize, q: usize, ell: usize, seed: u64) -> Self {
        assert!(k1.is_power_of_two() && k2.is_power_of_two());
        let basis = (0..32)
            .map(|k| {
                let mut e = [Gr1::<L>::ZERO; 32];
                e[k] = Gr1::<L>::ONE;
                pack_r2(e)
            })
            .collect();
        Self {
            params: RaaParams::setup(k2, q, seed),
            k1,
            k2,
            ell,
            basis,
        }
    }

    /// Protocol-driven setup (paper Theorem 5.8).
    ///
    /// Given the multilinear size `num_vars` (`N = 2^num_vars`), the RAA inverse
    /// rate `q` (`n = q·k2`, rate `ρ = 1/q`), the code's assumed relative
    /// distance `delta`, and a target soundness `lambda` (bits), this:
    ///
    /// 1. derives the query count `t` from the soundness bound — the dominant
    ///    `ε_test = (1−δ/3)^t` term of the paper's Theorem 5.8 ([`recommended_query_size`]);
    /// 2. picks the matrix shape `(k1, k2)` with `k1·k2 = N` that **minimizes
    ///    the opening-proof size** at that `t` and rate ([`optimal_shape`]).
    ///
    /// The `n/|T|` term (`|T| = 2^128`) is independent of `t`, so it does not
    /// enter the query count; it is an n-dependent floor on the achievable
    /// security ([`achievable_security`]). For non-degenerate soundness the size
    /// must be large enough that the chosen `q·k2 ≥ t`.
    pub fn setup(num_vars: usize, q: usize, delta: f64, lambda: usize, seed: u64) -> Self {
        let limb_bytes = (L::BITS / 8) as usize;
        let t = recommended_query_size(delta, lambda);
        let (k1, k2) = optimal_shape(num_vars, q, t, limb_bytes);
        Self::new(k1, k2, q, t, seed)
    }

    /// Like [`setup`], but the shape is optimized for `num_openings` openings of
    /// this commitment that **share** their queried columns (see
    /// [`rotated_open_batch`]). The `m̄` is sent once *per opening* while the
    /// columns are sent once *total*, so the cost is `num_openings·k2·mbar +
    /// t·k1·col`; weighting the `m̄` term by `num_openings` shifts the optimum to a
    /// smaller `k2` (smaller `m̄`) and larger `k1` (taller columns, paid once).
    pub fn setup_batched(
        num_vars: usize,
        q: usize,
        delta: f64,
        lambda: usize,
        seed: u64,
        num_openings: usize,
    ) -> Self {
        let limb_bytes = (L::BITS / 8) as usize;
        let t = recommended_query_size(delta, lambda);
        let mbar_bytes = num_openings * 128 * limb_bytes;
        let col_bytes = 32 * limb_bytes;
        // Minimize the batched cost (m̄ weighted ×num_openings) subject to the
        // codeword floor q·k2 ≥ t — the batched optimum pushes k2 down, and below
        // this floor there are fewer than t distinct columns to query.
        let mut best: Option<(usize, (usize, usize))> = None;
        for log_k1 in 1..num_vars {
            let k1 = 1usize << log_k1;
            let k2 = 1usize << (num_vars - log_k1);
            if q * k2 < t {
                continue;
            }
            let size = proof_size_bytes_for(k1, k2, q * k2, t, mbar_bytes, col_bytes);
            if best.map_or(true, |(b, _)| size < b) {
                best = Some((size, (k1, k2)));
            }
        }
        // Always feasible for secure sizes (k1=2 ⇒ k2=N/2 ⇒ q·k2 ≥ t); fall back
        // to the standard shape otherwise.
        let (k1, k2) = best
            .map(|(_, kk)| kk)
            .unwrap_or_else(|| optimal_shape_for(num_vars, q, t, 128 * limb_bytes, col_bytes));
        Self::new(k1, k2, q, t, seed)
    }

    /// Shape for a **single** opening of a `stack`-message slot commitment
    /// ([`commit_stacked`]/[`open_stacked`]): one `m̄` (128-limb) over columns
    /// `stack·k1` tall (32-limb each). Unlike [`setup_batched`] (which weights the
    /// `m̄` for many openings → small `k1`), this weights the **columns** by
    /// `stack` → small `k1` so the tall stacked columns stay cheap. Used for the
    /// `(â,b̂,ĉ)` commitment in the general/MIMD provers.
    pub fn setup_stacked(
        num_vars: usize,
        q: usize,
        delta: f64,
        lambda: usize,
        seed: u64,
        stack: usize,
    ) -> Self {
        let limb_bytes = (L::BITS / 8) as usize;
        let t = recommended_query_size(delta, lambda);
        let mbar_bytes = 128 * limb_bytes; // one m̄
        let col_bytes = stack * 32 * limb_bytes; // stack·k1-tall slot columns
        let mut best: Option<(usize, (usize, usize))> = None;
        for log_k1 in 1..num_vars {
            let k1 = 1usize << log_k1;
            let k2 = 1usize << (num_vars - log_k1);
            if q * k2 < t {
                continue;
            }
            let size = proof_size_bytes_for(k1, k2, q * k2, t, mbar_bytes, col_bytes);
            if best.map_or(true, |(b, _)| size < b) {
                best = Some((size, (k1, k2)));
            }
        }
        let (k1, k2) = best
            .map(|(_, kk)| kk)
            .unwrap_or_else(|| optimal_shape_for(num_vars, q, t, mbar_bytes, col_bytes));
        Self::new(k1, k2, q, t, seed)
    }

    pub fn num_vars(&self) -> usize {
        (self.k1 * self.k2).trailing_zeros() as usize
    }

    pub fn n(&self) -> usize {
        self.params.n
    }
}

// ---------------------------------------------------------------------------
// Setup: soundness-driven query count + proof-size-optimal matrix shape
// ---------------------------------------------------------------------------

/// Relative minimum distance of the lifted RAA code (proved in the paper).
/// The single source of truth for the secure default; the field baseline's RAA
/// code shares it by the distance-preserving lift.
pub const SECURE_DELTA: f64 = 0.19;

/// Default target soundness (bits) for `setup`-based secure params.
pub const SECURE_LAMBDA: usize = 100;

/// RAA inverse rate (repetition factor); rate `ρ = 1/SECURE_Q`.
pub const SECURE_Q: usize = 4;

/// Smallest query count `t` for which the tensor-PCS soundness error
/// `ε ≤ ε_test + ε_img + ε_eval` clears `2^{−lambda}` (paper Theorem 5.8). `t` is
/// governed by the dominant `ε_test = (1−δ/3)^t` term; `ε_img = ε_eval =
/// (1−2δ/3)^t` are far smaller at the chosen `t` (≈ `2^{−207}` for the paper's
/// δ=0.19, λ=100) and only ever nudge it by ±1, which the trailing check absorbs.
///
/// The `n/|T|` term of `ε_test` is independent of `t`, so it is *not* part of
/// this computation — see [`achievable_security`] for the n-dependent floor.
///
/// For the paper's parameters, `(δ=0.19, λ=100) → 1060`.
pub fn recommended_query_size(delta: f64, lambda: usize) -> usize {
    assert!(
        delta > 0.0 && delta < 1.5,
        "need 0 < δ < 1.5 so that 1−2δ/3 > 0"
    );
    let lam = lambda as f64;
    let base_test = 1.0 - delta / 3.0; //   dominant ε_test base
    let base_other = 1.0 - 2.0 * delta / 3.0; // ε_img / ε_eval base
                                              // Closed form from the dominant term: (1−δ/3)^t ≤ 2^{−λ}.
    let mut t = (lam / (-base_test.log2())).ceil() as usize;
    // Tighten until the *summed* geometric bound clears 2^{−λ} (≈ never trips).
    let target = (-lam).exp2();
    while base_test.powi(t as i32) + 2.0 * base_other.powi(t as i32) > target {
        t += 1;
    }
    t
}

/// Exact serialized opening-proof size in bytes for a `(k1, k2)` shape with
/// codeword length `n = q·k2` and query count `t`, at base-limb width
/// `limb_bytes = BITS/8`. Matches [`OpeningProof::serialize`] byte-for-byte:
/// `m̄` is `k2` ring elements (128 limbs each); each queried column is `k1`
/// packed entries (32 limbs each); one Merkle path of `⌈log2 n⌉` hashes per
/// query; plus the `u64` length headers.
pub fn proof_size_bytes(k1: usize, k2: usize, n: usize, t: usize, limb_bytes: usize) -> usize {
    // `Pcs` layout: m̄ entries are Gr128 (128 limbs); column entries are the
    // packed `[L; 32]` form (32 limbs).
    proof_size_bytes_for(k1, k2, n, t, 128 * limb_bytes, 32 * limb_bytes)
}

/// [`proof_size_bytes`] generalized over the per-element byte sizes of an `m̄`
/// entry (`mbar_bytes`) and a queried-column entry (`col_bytes`). Tensor PCSs
/// whose columns are *not* the packed 32-limb form use this directly: the
/// generic-R₂ error PCS sends full 128-limb column entries
/// (`col_bytes = 128·limb_bytes`), and the field PCS sends one field element for
/// both (`mbar_bytes = col_bytes`).
pub fn proof_size_bytes_for(
    k1: usize,
    k2: usize,
    n: usize,
    t: usize,
    mbar_bytes: usize,
    col_bytes: usize,
) -> usize {
    let path_len = n.next_power_of_two().trailing_zeros() as usize;
    let mbar = k2 * mbar_bytes; //          m̄: k2 entries
    let cols = t * (8 + k1 * col_bytes); // per query: len hdr + k1 column entries
    let paths = t * (8 + path_len * 32); // per query: len hdr + path hashes
    let idx = t * 4; //                     query_indices: t × u32
    32 + mbar + cols + paths + idx //       4 × u64 top-level length headers
}

/// Matrix shape `(k1, k2)` with `k1·k2 = 2^num_vars` (both powers of two) that
/// **minimizes** [`proof_size_bytes`] for the given inverse rate `q` and query
/// count `t`. Exhaustive over the *interior* power-of-two splits — `k1, k2 ≥ 2`,
/// so neither eq-table factor (`eq(z^0,·)` / `eq(z^1,·)`) is empty, which the
/// PCS requires (`num_vars ≥ 2`).
///
/// The proof trades `m̄` (∝ k2 = N/k1) against the column openings (∝ t·k1);
/// the continuous optimum is `k1 ≈ 2·√(N/t)` — the factor 2 is the `128:32`
/// limb ratio between an `m̄` entry and a column entry — rounded to the better
/// neighbouring power of two by the exact search (which also accounts for the
/// `⌈log2 n⌉` Merkle-path term and width-independent overheads).
pub fn optimal_shape(num_vars: usize, q: usize, t: usize, limb_bytes: usize) -> (usize, usize) {
    optimal_shape_for(num_vars, q, t, 128 * limb_bytes, 32 * limb_bytes)
}

/// [`optimal_shape`] generalized over the per-element byte sizes (`mbar_bytes`,
/// `col_bytes`) — see [`proof_size_bytes_for`]. Exhaustive over interior
/// power-of-two splits (`k1, k2 ≥ 2`). When columns and `m̄` are the same size
/// (`mbar_bytes == col_bytes`, the R₂-error and field PCSs) the continuous
/// optimum is the symmetric `k1 = √(N/t)`; the packed `Pcs` (`col_bytes =
/// mbar_bytes/4`) skews toward larger `k1`.
pub fn optimal_shape_for(
    num_vars: usize,
    q: usize,
    t: usize,
    mbar_bytes: usize,
    col_bytes: usize,
) -> (usize, usize) {
    assert!(q >= 2);
    assert!(
        num_vars >= 2,
        "need num_vars ≥ 2 for a non-degenerate k1×k2 split (k1, k2 ≥ 2)"
    );
    let mut best: Option<(usize, (usize, usize))> = None;
    for log_k1 in 1..num_vars {
        let k1 = 1usize << log_k1;
        let k2 = 1usize << (num_vars - log_k1);
        let size = proof_size_bytes_for(k1, k2, q * k2, t, mbar_bytes, col_bytes);
        if best.map_or(true, |(b, _)| size < b) {
            best = Some((size, (k1, k2)));
        }
    }
    best.unwrap().1
}

/// Achievable soundness in bits — `−log2(ε)` for codeword length `n` and query
/// count `t`, *including* the `n/|T|` floor with `|T| = 2^128` (width-independent
/// for `R₂ = GR(2^k, 128)`). Even as `t → ∞`, `ε ≥ n/|T|`, so security is capped
/// near `128 − ⌈log2 n⌉`. Use to sanity-check that a chosen `(t, n)` actually
/// delivers the target λ rather than being floored by the ring size.
pub fn achievable_security(n: usize, t: usize, delta: f64) -> f64 {
    let t_floor = (n as f64) / 2f64.powi(128); // n / |T|, |T| = 2^128
    let eps_test = (1.0 - delta / 3.0).powi(t as i32) + t_floor;
    let eps_other = 2.0 * (1.0 - 2.0 * delta / 3.0).powi(t as i32);
    -(eps_test + eps_other).log2()
}

// ---------------------------------------------------------------------------
// Commitment & proof
// ---------------------------------------------------------------------------

/// Prover state retained between `commit` and `open`.
pub struct Commitment<L: Limb = u32> {
    pub root: Hash,
    /// Encoded matrix `M̂ ∈ (Z/2^n)^{k1 × n × 32}`, row-major over (i, j).
    pub encoded_packed: Vec<[L; 32]>,
    /// Original rows `M ∈ R^{k1 × k2}`, row-major, already lifted via
    /// `pack_r2` for the open-phase matvec.
    pub message: Vec<Gr128<L>>,
    /// Original rows in slot form `[L; 32]` (the un-lifted `evals`). Retained so
    /// the virtual rotated opening (`rotated_open`) can form `ρ̂^(ℓ)`'s message
    /// `pack_r2(rot_ℓ(slots))` without re-deriving slots from `message`.
    pub message_slots: Vec<[L; 32]>,
    pub tree: MerkleTree,
}

/// Opening proof.
#[derive(PartialEq)]
pub struct OpeningProof<L: Limb = u32> {
    /// Short message `m̄ = Σ (e_0)_i · m_i ∈ R^{k_2}`.
    pub mbar: Vec<Gr128<L>>,
    /// Column indices queried (`j ∈ [n]`).
    pub query_indices: Vec<u32>,
    /// For each queried column `j`: the `k_1` packed entries `M̂[:, j]`.
    pub query_columns: Vec<Vec<[L; 32]>>,
    /// Merkle authentication path for each queried column.
    pub query_paths: Vec<Vec<Hash>>,
}

/// Stacked **slot-form** commitment: `s` messages (each a `k1·k2` matrix of
/// `[L; 32]` slot tuples) laid as `s·k1` rows, encoded independently, and
/// Merkle-committed by columns. Opened at a *shared* point with weights → one
/// `m̄` binding all `s` (Schwartz–Zippel over the weights). The queried columns
/// stay **32-limb** (image-space succinct), unlike the generic-`R₂` stack. Used
/// to commit `(â,b̂,ĉ)` — all in `Im φ`, so slot form is exact and image
/// membership is structural (no image check needed).
pub struct StackedSlotCommitment<L: Limb = u32> {
    pub root: Hash,
    /// Encoded `(s·k1)·n` slot tuples, row-major over (row, column).
    pub encoded_packed: Vec<[L; 32]>,
    /// The `s·k1·k2` rows lifted to `R₂` (for the open-phase `m̄` matvec).
    pub message: Vec<Gr128<L>>,
    /// Number of stacked messages.
    pub s: usize,
    pub tree: MerkleTree,
}

impl<L: Limb> OpeningProof<L> {
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

        // query_columns: Vec<Vec<[L; 32]>>
        buf.extend_from_slice(&(self.query_columns.len() as u64).to_le_bytes());
        for col in &self.query_columns {
            buf.extend_from_slice(&(col.len() as u64).to_le_bytes());
            for packed_entry in col {
                for limb in packed_entry.iter() {
                    buf.extend_from_slice(&limb.to_u64().to_le_bytes()[..limb_bytes]);
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

/// Batched virtual rotated opening of ONE commitment at ONE point across
/// `num_layers` rotations. The proximity test — queried columns + Merkle paths —
/// is **shared**: the query set is derived from a single Fiat–Shamir challenge
/// over *all* the rotated `m̄`s, so the (large) column/path data is sent once
/// instead of once per layer. Only `m̄^(ℓ)` is per-layer. The verifier rotates
/// the shared un-rotated columns locally. Produced by [`Pcs::rotated_open_batch`].
#[derive(Clone, Debug, PartialEq)]
pub struct RotatedOpeningBatch<L: Limb = u32> {
    /// `m̄^(ℓ)` for each layer ℓ (the only per-layer data).
    pub mbars: Vec<Vec<Gr128<L>>>,
    /// Shared queried column indices (one set, bound to all `m̄`s).
    pub query_indices: Vec<u32>,
    /// Shared un-rotated columns `ŵ[:, j]`.
    pub query_columns: Vec<Vec<[L; 32]>>,
    /// Shared Merkle paths for the queried columns.
    pub query_paths: Vec<Vec<Hash>>,
}

impl<L: Limb> RotatedOpeningBatch<L> {
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let limb_bytes = (L::BITS / 8) as usize;
        // mbars: Vec<Vec<Gr128<L>>>
        buf.extend_from_slice(&(self.mbars.len() as u64).to_le_bytes());
        for mbar in &self.mbars {
            buf.extend_from_slice(&(mbar.len() as u64).to_le_bytes());
            for gr in mbar {
                for c in gr.0.iter() {
                    for w in c.0.iter() {
                        buf.extend_from_slice(&w.to_u64().to_le_bytes()[..limb_bytes]);
                    }
                }
            }
        }
        // query_indices: Vec<u32>
        buf.extend_from_slice(&(self.query_indices.len() as u64).to_le_bytes());
        for idx in &self.query_indices {
            buf.extend_from_slice(&idx.to_le_bytes());
        }
        // query_columns: Vec<Vec<[L; 32]>>  (shared, once)
        buf.extend_from_slice(&(self.query_columns.len() as u64).to_le_bytes());
        for col in &self.query_columns {
            buf.extend_from_slice(&(col.len() as u64).to_le_bytes());
            for entry in col {
                for limb in entry.iter() {
                    buf.extend_from_slice(&limb.to_u64().to_le_bytes()[..limb_bytes]);
                }
            }
        }
        // query_paths: Vec<Vec<Hash>>  (shared, once)
        buf.extend_from_slice(&(self.query_paths.len() as u64).to_le_bytes());
        for path in &self.query_paths {
            buf.extend_from_slice(&(path.len() as u64).to_le_bytes());
            for hash in path {
                buf.extend_from_slice(hash);
            }
        }
        buf
    }

    pub fn size_bytes(&self) -> usize {
        self.serialize().len()
    }
}

// ---------------------------------------------------------------------------
// Protocol
// ---------------------------------------------------------------------------

impl<L: HasPackingConstants> Pcs<L> {
    /// Commit (§2.2): encode rows, Merkle-commit columns.
    ///
    /// `evals` is a row-major `k1 × k2` matrix of packed tuples. The polynomial
    /// `f` it represents is `f(bin(i)||bin(j)) = pack_r2(evals[i*k2 + j])`.
    pub fn commit(&self, evals: &[[L; 32]]) -> Commitment<L> {
        self.commit_with_timing(evals).0
    }

    /// Commit and return `(commitment, encode_ms, merkle_ms)`. `encode_ms` is
    /// the RAA row-encode plus the `pack_r2` message lift (the open-phase prep);
    /// `merkle_ms` is the leaf-hash + tree build. Split out for benchmarking.
    pub fn commit_with_timing(&self, evals: &[[L; 32]]) -> (Commitment<L>, f64, f64) {
        assert_eq!(evals.len(), self.k1 * self.k2);
        let n = self.params.n;

        // Encode rows in packed form. Rows are independent (each writes its own
        // length-`n` chunk), so parallelize over the `k1` rows; the RAA encode
        // itself stays serial inside each row (its prefix-sums are sequential).
        let t = std::time::Instant::now();
        let mut encoded_packed = vec![[L::ZERO; 32]; self.k1 * n];
        encoded_packed
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, dst)| {
                let row = &evals[i * self.k2..(i + 1) * self.k2];
                dst.copy_from_slice(&self.params.encode_packed(row));
            });
        let mut encode_ms = t.elapsed().as_secs_f64() * 1e3;

        // Merkle over columns.
        let t = std::time::Instant::now();
        // Each column hashes a disjoint stride of `encoded_packed`; the
        // order-preserving collect keeps the leaf vector (and hence the root)
        // identical to the serial loop.
        let leaves: Vec<Hash> = (0..n)
            .into_par_iter()
            .map(|j| hash_column(&encoded_packed, self.k1, n, j))
            .collect();
        let tree = MerkleTree::build(leaves);
        let root = tree.root();
        let merkle_ms = t.elapsed().as_secs_f64() * 1e3;

        // Lift the message into R₂ once for the open-phase matvec (counted as
        // prep). Each entry lifts independently → parallel map (order-preserving).
        let t = std::time::Instant::now();
        let message: Vec<Gr128<L>> = evals
            .par_iter()
            .map(|entry| {
                let mut e = [Gr1::<L>::ZERO; 32];
                for s in 0..32 {
                    e[s] = Gr1(entry[s]);
                }
                pack_r2(e)
            })
            .collect();
        encode_ms += t.elapsed().as_secs_f64() * 1e3;

        let comm = Commitment {
            root,
            encoded_packed,
            message,
            message_slots: evals.to_vec(),
            tree,
        };
        (comm, encode_ms, merkle_ms)
    }

    /// Open (§2.4): compute m̄ = e_0^⊤ M, open queried columns.
    /// Returns `(f(z), proof)`.
    pub fn open(&self, comm: &Commitment<L>, z: &[Gr128<L>]) -> (Gr128<L>, OpeningProof<L>) {
        let (e0, e1) = self.split_eq(z);
        let n = self.params.n;

        // m̄ = Σ_i (e_0)_i · m_i  ∈ R^{k_2}. Parallelize over the disjoint output
        // column `j` (each `mbar[j]` sums over `i` in the same fixed order as the
        // serial loop ⇒ bit-identical, no cross-thread accumulator merge).
        let mut mbar = vec![Gr128::<L>::ZERO; self.k2];
        mbar.par_iter_mut().enumerate().for_each(|(j, out)| {
            let mut acc = Gr128::<L>::ZERO;
            for i in 0..self.k1 {
                acc = acc + e0[i] * comm.message[i * self.k2 + j];
            }
            *out = acc;
        });

        // f(z) = ⟨m̄, e_1⟩
        let mut y = Gr128::<L>::ZERO;
        for j in 0..self.k2 {
            y = y + mbar[j] * e1[j];
        }

        let query_indices = derive_query_indices(comm.root, &mbar, n, self.ell);

        let mut query_columns: Vec<Vec<[L; 32]>> = Vec::with_capacity(self.ell);
        let mut query_paths = Vec::with_capacity(self.ell);
        for &j in &query_indices {
            let j = j as usize;
            let col: Vec<[L; 32]> = (0..self.k1)
                .map(|i| comm.encoded_packed[i * n + j])
                .collect();
            query_columns.push(col);
            query_paths.push(comm.tree.open(j));
        }

        (
            y,
            OpeningProof {
                mbar,
                query_indices,
                query_columns,
                query_paths,
            },
        )
    }

    /// Verify (§2.4): check ⟨m̄, e_1⟩ = y, re-encode m̄, check columns.
    pub fn verify(
        &self,
        commit_root: Hash,
        z: &[Gr128<L>],
        y_claimed: Gr128<L>,
        proof: &OpeningProof<L>,
    ) -> bool {
        let (e0, e1) = self.split_eq(z);
        let n = self.params.n;

        if proof.mbar.len() != self.k2 {
            return false;
        }
        let mut y = Gr128::<L>::ZERO;
        for j in 0..self.k2 {
            y = y + proof.mbar[j] * e1[j];
        }
        if y != y_claimed {
            return false;
        }

        let expected = derive_query_indices(commit_root, &proof.mbar, n, self.ell);
        if expected != proof.query_indices {
            return false;
        }

        // ū = Enc(m̄) over R₂ (m̄ is not in the packing image).
        let ubar = self.params.encode(&proof.mbar);

        for q in 0..self.ell {
            let j = proof.query_indices[q] as usize;
            let col = &proof.query_columns[q];
            if col.len() != self.k1 {
                return false;
            }

            // acc_j = Σ_k v_k · (Σ_i e_0[i].mul_by_gr1(col[i][k]))
            let mut acc = Gr128::<L>::ZERO;
            for k in 0..32 {
                let mut inner = Gr128::<L>::ZERO;
                for i in 0..self.k1 {
                    inner = inner + e0[i].mul_by_gr1(Gr1(col[i][k]));
                }
                acc = acc + self.basis[k] * inner;
            }
            if acc != ubar[j] {
                return false;
            }

            let leaf = hash_packed_column(col);
            if !verify_path(leaf, j, &proof.query_paths[q], commit_root) {
                return false;
            }
        }
        true
    }

    /// `e_0` replicated per stacked message and weighted: `e0s[t·k1+i] = w_t·e0[i]`.
    fn stacked_e0(&self, e0: &[Gr128<L>], y_weights: &[Gr128<L>]) -> Vec<Gr128<L>> {
        let mut out = vec![Gr128::<L>::ZERO; y_weights.len() * self.k1];
        for (t, &w) in y_weights.iter().enumerate() {
            for i in 0..self.k1 {
                out[t * self.k1 + i] = w * e0[i];
            }
        }
        out
    }

    /// Commit `s` slot-form messages stacked as `s·k1` rows under one Merkle tree
    /// (columns shared). Each message is a `k1·k2` slot matrix. See
    /// [`StackedSlotCommitment`].
    pub fn commit_stacked(&self, messages: &[&[[L; 32]]]) -> StackedSlotCommitment<L> {
        let (c, _, _) = self.commit_stacked_with_timing(messages);
        c
    }

    pub fn commit_stacked_with_timing(
        &self,
        messages: &[&[[L; 32]]],
    ) -> (StackedSlotCommitment<L>, f64, f64) {
        let s = messages.len();
        assert!(s >= 1);
        for m in messages {
            assert_eq!(m.len(), self.k1 * self.k2);
        }
        let n = self.params.n;
        let rows = s * self.k1;

        let t = std::time::Instant::now();
        let mut encoded_packed = vec![[L::ZERO; 32]; rows * n];
        encoded_packed
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, dst)| {
                let (t, r) = (i / self.k1, i % self.k1);
                let row = &messages[t][r * self.k2..(r + 1) * self.k2];
                dst.copy_from_slice(&self.params.encode_packed(row));
            });
        let mut encode_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = std::time::Instant::now();
        let leaves: Vec<Hash> = (0..n)
            .into_par_iter()
            .map(|j| hash_column(&encoded_packed, rows, n, j))
            .collect();
        let tree = MerkleTree::build(leaves);
        let root = tree.root();
        let merkle_ms = t.elapsed().as_secs_f64() * 1e3;

        // Lift each row's slots to R₂ once (open-phase prep).
        let t = std::time::Instant::now();
        let message: Vec<Gr128<L>> = (0..rows * self.k2)
            .into_par_iter()
            .map(|idx| {
                let (t, off) = (idx / (self.k1 * self.k2), idx % (self.k1 * self.k2));
                let entry = &messages[t][off];
                let mut e = [Gr1::<L>::ZERO; 32];
                for s in 0..32 {
                    e[s] = Gr1(entry[s]);
                }
                pack_r2(e)
            })
            .collect();
        encode_ms += t.elapsed().as_secs_f64() * 1e3;

        (
            StackedSlotCommitment {
                root,
                encoded_packed,
                message,
                s,
                tree,
            },
            encode_ms,
            merkle_ms,
        )
    }

    /// Open the stacked slot commitment at `z` with `y_weights` (length `s`):
    /// returns `Σ_t w_t·m_t(z)` and one proof (one m̄, columns `s·k1` tall).
    pub fn open_stacked(
        &self,
        comm: &StackedSlotCommitment<L>,
        z: &[Gr128<L>],
        y_weights: &[Gr128<L>],
    ) -> (Gr128<L>, OpeningProof<L>) {
        assert_eq!(y_weights.len(), comm.s);
        let (e0, e1) = self.split_eq(z);
        let e0s = self.stacked_e0(&e0, y_weights);
        let n = self.params.n;
        let rows = comm.s * self.k1;

        let mut mbar = vec![Gr128::<L>::ZERO; self.k2];
        mbar.par_iter_mut().enumerate().for_each(|(j, out)| {
            let mut acc = Gr128::<L>::ZERO;
            for i in 0..rows {
                acc = acc + e0s[i] * comm.message[i * self.k2 + j];
            }
            *out = acc;
        });

        let mut y = Gr128::<L>::ZERO;
        for j in 0..self.k2 {
            y = y + mbar[j] * e1[j];
        }

        let query_indices = derive_query_indices(comm.root, &mbar, n, self.ell);
        let mut query_columns: Vec<Vec<[L; 32]>> = Vec::with_capacity(self.ell);
        let mut query_paths = Vec::with_capacity(self.ell);
        for &j in &query_indices {
            let j = j as usize;
            let col: Vec<[L; 32]> = (0..rows).map(|i| comm.encoded_packed[i * n + j]).collect();
            query_columns.push(col);
            query_paths.push(comm.tree.open(j));
        }

        (
            y,
            OpeningProof {
                mbar,
                query_indices,
                query_columns,
                query_paths,
            },
        )
    }

    /// Verify a stacked slot opening. `y_weights.len()` is the stack count `s`.
    pub fn verify_stacked(
        &self,
        root: Hash,
        z: &[Gr128<L>],
        y_weights: &[Gr128<L>],
        value: Gr128<L>,
        proof: &OpeningProof<L>,
    ) -> bool {
        let (e0, e1) = self.split_eq(z);
        let e0s = self.stacked_e0(&e0, y_weights);
        let n = self.params.n;
        let rows = y_weights.len() * self.k1;

        if proof.mbar.len() != self.k2 {
            return false;
        }
        let mut y = Gr128::<L>::ZERO;
        for j in 0..self.k2 {
            y = y + proof.mbar[j] * e1[j];
        }
        if y != value {
            return false;
        }
        let expected = derive_query_indices(root, &proof.mbar, n, self.ell);
        if expected != proof.query_indices {
            return false;
        }
        let ubar = self.params.encode(&proof.mbar);
        for q in 0..self.ell {
            let j = proof.query_indices[q] as usize;
            if j >= n {
                return false;
            }
            let col = &proof.query_columns[q];
            if col.len() != rows {
                return false;
            }
            // Σ_i e0s[i]·pack_r2(col[i]) = Σ_k v_k·(Σ_i e0s[i].mul_by_gr1(col[i][k])).
            let mut acc = Gr128::<L>::ZERO;
            for k in 0..32 {
                let mut inner = Gr128::<L>::ZERO;
                for i in 0..rows {
                    inner = inner + e0s[i].mul_by_gr1(Gr1(col[i][k]));
                }
                acc = acc + self.basis[k] * inner;
            }
            if acc != ubar[j] {
                return false;
            }
            let leaf = hash_packed_column(col);
            if !verify_path(leaf, j, &proof.query_paths[q], root) {
                return false;
            }
        }
        true
    }

    /// Virtual rotated opening (general circuit, paper §4.3; virtual PCS, paper §5.3).
    ///
    /// Opens the *rotated* witness `ρ̂^(ℓ)` whose block `q` is
    /// `pack_r2(rot_ℓ(slots_q))`, **reusing the single commitment to `ŵ`** — no
    /// separate commitment to `ρ̂^(ℓ)`. Because the slot-uniform encoder commutes
    /// with the slot rotation (`Enc(rot_ℓ·m) = rot_ℓ·Enc(m)`; see
    /// `raa::tests::encode_commutes_with_slot_rotation`), the codeword columns of
    /// `ρ̂^(ℓ)` are just `rot_ℓ` of `ŵ`'s columns. The proof therefore carries
    /// `ŵ`'s **un-rotated** columns (authenticated against `ŵ`'s root) and the
    /// verifier rotates them locally; only `m̄^(ℓ) = e_0^⊤·ρ̂^(ℓ)` is
    /// rotation-specific. `ell ∈ [0, 32)`.
    pub fn rotated_open(
        &self,
        comm: &Commitment<L>,
        z: &[Gr128<L>],
        ell: usize,
    ) -> (Gr128<L>, OpeningProof<L>) {
        let (e0, e1) = self.split_eq(z);
        let n = self.params.n;

        // m̄^(ℓ)[j] = Σ_i e0[i] · pack_r2(rot_ℓ(slots_{i,j})). Column-parallel
        // over `j` (bit-identical to a serial fold). The 32 distinct ℓ are the
        // outer parallel axis at the SNARK layer.
        let mut mbar = vec![Gr128::<L>::ZERO; self.k2];
        mbar.par_iter_mut().enumerate().for_each(|(j, out)| {
            let mut acc = Gr128::<L>::ZERO;
            for i in 0..self.k1 {
                let rot = rotate_slots(&comm.message_slots[i * self.k2 + j], ell);
                let mut e = [Gr1::<L>::ZERO; 32];
                for s in 0..32 {
                    e[s] = Gr1(rot[s]);
                }
                acc = acc + e0[i] * pack_r2(e);
            }
            *out = acc;
        });

        let mut y = Gr128::<L>::ZERO;
        for j in 0..self.k2 {
            y = y + mbar[j] * e1[j];
        }

        let query_indices = derive_query_indices(comm.root, &mbar, n, self.ell);

        // Share `ŵ`'s un-rotated columns; the verifier applies rot_ℓ locally.
        let mut query_columns: Vec<Vec<[L; 32]>> = Vec::with_capacity(self.ell);
        let mut query_paths = Vec::with_capacity(self.ell);
        for &j in &query_indices {
            let j = j as usize;
            let col: Vec<[L; 32]> = (0..self.k1)
                .map(|i| comm.encoded_packed[i * n + j])
                .collect();
            query_columns.push(col);
            query_paths.push(comm.tree.open(j));
        }

        (
            y,
            OpeningProof {
                mbar,
                query_indices,
                query_columns,
                query_paths,
            },
        )
    }

    /// Verify a virtual rotated opening produced by [`rotated_open`]. Identical
    /// to [`verify`] except each queried column is slot-rotated by `ℓ` before the
    /// column-consistency check; the Merkle authentication is on `ŵ`'s
    /// un-rotated columns against `ŵ`'s root.
    pub fn rotated_verify(
        &self,
        commit_root: Hash,
        z: &[Gr128<L>],
        ell: usize,
        y_claimed: Gr128<L>,
        proof: &OpeningProof<L>,
    ) -> bool {
        let (e0, e1) = self.split_eq(z);
        let n = self.params.n;

        if proof.mbar.len() != self.k2 {
            return false;
        }
        let mut y = Gr128::<L>::ZERO;
        for j in 0..self.k2 {
            y = y + proof.mbar[j] * e1[j];
        }
        if y != y_claimed {
            return false;
        }

        let expected = derive_query_indices(commit_root, &proof.mbar, n, self.ell);
        if expected != proof.query_indices {
            return false;
        }

        let ubar = self.params.encode(&proof.mbar);

        for q in 0..self.ell {
            let j = proof.query_indices[q] as usize;
            let col = &proof.query_columns[q];
            if col.len() != self.k1 {
                return false;
            }

            // Column of `ρ̂^(ℓ)` = rot_ℓ(column of `ŵ`): slot `k` reads `ŵ`'s
            // slot `(k+ℓ) mod 32`. acc_j = Σ_k v_k·(Σ_i e0[i]·rot_ℓ(col[i])[k]).
            let mut acc = Gr128::<L>::ZERO;
            for k in 0..32 {
                let src = (k + ell) % 32;
                let mut inner = Gr128::<L>::ZERO;
                for i in 0..self.k1 {
                    inner = inner + e0[i].mul_by_gr1(Gr1(col[i][src]));
                }
                acc = acc + self.basis[k] * inner;
            }
            if acc != ubar[j] {
                return false;
            }

            let leaf = hash_packed_column(col);
            if !verify_path(leaf, j, &proof.query_paths[q], commit_root) {
                return false;
            }
        }
        true
    }

    /// Batched [`rotated_open`] over layers `0..num_layers` sharing one proximity
    /// test. The query set is derived from a single FS challenge over *all* the
    /// rotated `m̄`s, so the columns + paths are opened once. Returns the per-layer
    /// evaluations `y_ℓ` and the batch.
    pub fn rotated_open_batch(
        &self,
        comm: &Commitment<L>,
        z: &[Gr128<L>],
        num_layers: usize,
    ) -> (Vec<Gr128<L>>, RotatedOpeningBatch<L>) {
        let (e0, e1) = self.split_eq(z);
        let n = self.params.n;

        // m̄^(ℓ) per layer — serial over layers, parallel over `j` (as in
        // `rotated_open`; bit-identical fold).
        let mbars: Vec<Vec<Gr128<L>>> = (0..num_layers)
            .map(|ell| {
                let mut mbar = vec![Gr128::<L>::ZERO; self.k2];
                mbar.par_iter_mut().enumerate().for_each(|(j, out)| {
                    let mut acc = Gr128::<L>::ZERO;
                    for i in 0..self.k1 {
                        let rot = rotate_slots(&comm.message_slots[i * self.k2 + j], ell);
                        let mut e = [Gr1::<L>::ZERO; 32];
                        for s in 0..32 {
                            e[s] = Gr1(rot[s]);
                        }
                        acc = acc + e0[i] * pack_r2(e);
                    }
                    *out = acc;
                });
                mbar
            })
            .collect();

        // y_ℓ = m̄^(ℓ) · e1.
        let ys: Vec<Gr128<L>> = mbars
            .iter()
            .map(|mbar| {
                let mut y = Gr128::<L>::ZERO;
                for j in 0..self.k2 {
                    y = y + mbar[j] * e1[j];
                }
                y
            })
            .collect();

        // ONE shared query set, bound to ALL the m̄s.
        let flat: Vec<Gr128<L>> = mbars.iter().flatten().copied().collect();
        let query_indices = derive_query_indices(comm.root, &flat, n, self.ell);

        // Open the shared un-rotated columns once.
        let mut query_columns: Vec<Vec<[L; 32]>> = Vec::with_capacity(self.ell);
        let mut query_paths = Vec::with_capacity(self.ell);
        for &j in &query_indices {
            let j = j as usize;
            let col: Vec<[L; 32]> = (0..self.k1)
                .map(|i| comm.encoded_packed[i * n + j])
                .collect();
            query_columns.push(col);
            query_paths.push(comm.tree.open(j));
        }

        (
            ys,
            RotatedOpeningBatch {
                mbars,
                query_indices,
                query_columns,
                query_paths,
            },
        )
    }

    /// Verify a [`rotated_open_batch`]: one shared proximity test + per-layer
    /// `m̄`/evaluation consistency reusing the shared columns.
    pub fn rotated_verify_batch(
        &self,
        commit_root: Hash,
        z: &[Gr128<L>],
        y_claimed: &[Gr128<L>],
        batch: &RotatedOpeningBatch<L>,
    ) -> bool {
        let (e0, e1) = self.split_eq(z);
        let n = self.params.n;
        let num_layers = batch.mbars.len();
        if y_claimed.len() != num_layers
            || batch.query_columns.len() != self.ell
            || batch.query_paths.len() != self.ell
        {
            return false;
        }

        // Per-layer evaluation y_ℓ = m̄^(ℓ) · e1.
        for ell in 0..num_layers {
            if batch.mbars[ell].len() != self.k2 {
                return false;
            }
            let mut y = Gr128::<L>::ZERO;
            for j in 0..self.k2 {
                y = y + batch.mbars[ell][j] * e1[j];
            }
            if y != y_claimed[ell] {
                return false;
            }
        }

        // Shared query set, bound to all m̄s.
        let flat: Vec<Gr128<L>> = batch.mbars.iter().flatten().copied().collect();
        if derive_query_indices(commit_root, &flat, n, self.ell) != batch.query_indices {
            return false;
        }

        // Merkle-authenticate each shared column ONCE.
        for q in 0..self.ell {
            let j = batch.query_indices[q] as usize;
            let col = &batch.query_columns[q];
            if col.len() != self.k1 {
                return false;
            }
            let leaf = hash_packed_column(col);
            if !verify_path(leaf, j, &batch.query_paths[q], commit_root) {
                return false;
            }
        }

        // Per-layer column consistency reuses the shared columns (parallel over ℓ).
        (0..num_layers).into_par_iter().all(|ell| {
            let ubar = self.params.encode(&batch.mbars[ell]);
            for q in 0..self.ell {
                let j = batch.query_indices[q] as usize;
                let col = &batch.query_columns[q];
                let mut acc = Gr128::<L>::ZERO;
                for k in 0..32 {
                    let src = (k + ell) % 32;
                    let mut inner = Gr128::<L>::ZERO;
                    for i in 0..self.k1 {
                        inner = inner + e0[i].mul_by_gr1(Gr1(col[i][src]));
                    }
                    acc = acc + self.basis[k] * inner;
                }
                if acc != ubar[j] {
                    return false;
                }
            }
            true
        })
    }

    fn split_eq(&self, z: &[Gr128<L>]) -> (Vec<Gr128<L>>, Vec<Gr128<L>>) {
        assert_eq!(z.len(), self.num_vars());
        let log_k1 = self.k1.trailing_zeros() as usize;
        let e0 = build_eq_table(&z[..log_k1]).evals().to_vec();
        let e1 = build_eq_table(&z[log_k1..]).evals().to_vec();
        (e0, e1)
    }
}

// ---------------------------------------------------------------------------
// Fiat–Shamir
// ---------------------------------------------------------------------------

fn derive_query_indices<L: Limb>(root: Hash, mbar: &[Gr128<L>], n: usize, ell: usize) -> Vec<u32> {
    assert!(n > 0);
    let lbytes = (L::BITS / 8) as usize;
    let mut h = Hasher::new();
    h.update(b"pcs-ring/fs/queries");
    h.update(&root);
    for elem in mbar {
        for c in elem.0.iter() {
            for w in c.0.iter() {
                h.update(&w.to_u64().to_le_bytes()[..lbytes]);
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

/// Cyclic slot rotation `out[i] = entry[(i+ℓ) mod 32]`, matching
/// `reduction::reduce::rot_l`. The local operation a verifier applies to `ŵ`'s
/// opened columns to derive `ρ̂^(ℓ)`'s columns in the virtual rotated opening.
#[inline]
fn rotate_slots<L: Limb>(entry: &[L; 32], ell: usize) -> [L; 32] {
    let mut out = [L::ZERO; 32];
    for i in 0..32 {
        out[i] = entry[(i + ell) % 32];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use galois_ring::{GR32_128, GR32_4};

    fn rand_gr128(seed: &mut u64) -> GR32_128 {
        let mut next = || {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*seed >> 32) as u32
        };
        let mut arr = [GR32_4::ZERO; 32];
        for slot in arr.iter_mut() {
            *slot = GR32_4([next(), next(), next(), next()]);
        }
        GR32_128(arr)
    }

    /// The virtual rotated opening of `ŵ` for every `ℓ` must (a) verify against
    /// `ŵ`'s commitment, and (b) reproduce exactly what a *real* commitment to
    /// the explicitly slot-rotated witness `rot_ℓ(ŵ)` would open — no separate
    /// commitment needed. This is the end-to-end correctness of the virtual PCS.
    #[test]
    fn rotated_open_matches_real_rotated_commit() {
        let (k1, k2, q, ellq) = (4usize, 4usize, 4usize, 8usize);
        let pcs = Pcs::<u32>::new(k1, k2, q, ellq, 0xABCD_EF01);

        let mut s = 0x1234_5678u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 32) as u32
        };
        let evals: Vec<[u32; 32]> = (0..k1 * k2)
            .map(|_| std::array::from_fn(|_| next()))
            .collect();
        let comm = pcs.commit(&evals);

        let mut zs = 0xC0FFEEu64;
        let z: Vec<GR32_128> = (0..pcs.num_vars()).map(|_| rand_gr128(&mut zs)).collect();

        for ell in 0..32 {
            // (a) virtual rotated opening verifies.
            let (y, proof) = pcs.rotated_open(&comm, &z, ell);
            assert!(
                pcs.rotated_verify(comm.root, &z, ell, y, &proof),
                "rotated_verify rejected its own proof at ell={ell}"
            );

            // (b) equals the real opening of an explicit commitment to rot_ℓ(ŵ).
            let rot_evals: Vec<[u32; 32]> = evals.iter().map(|e| rotate_slots(e, ell)).collect();
            let comm_rot = pcs.commit(&rot_evals);
            let (y_real, proof_real) = pcs.open(&comm_rot, &z);
            assert_eq!(y, y_real, "evaluation mismatch at ell={ell}");
            assert!(pcs.verify(comm_rot.root, &z, y_real, &proof_real));
        }
    }

    /// Stacking **3** slot messages (the `â,b̂,ĉ` case): the weighted stacked
    /// opening `Σ_t ρ^t·m_t(z)` must (a) verify against the shared commitment and
    /// (b) equal the same combination of the three individual openings — i.e. the
    /// single opening soundly binds all three evals (Schwartz–Zippel layout).
    #[test]
    fn stacked_slot_three_matches_individual() {
        let (k1, k2, q, ellq) = (4usize, 4usize, 4usize, 8usize);
        let pcs = Pcs::<u32>::new(k1, k2, q, ellq, 0x5151_2727);
        let mut s = 0xBEEF_1234u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 32) as u32
        };
        let a: Vec<[u32; 32]> = (0..k1 * k2)
            .map(|_| std::array::from_fn(|_| next()))
            .collect();
        let b: Vec<[u32; 32]> = (0..k1 * k2)
            .map(|_| std::array::from_fn(|_| next()))
            .collect();
        let c: Vec<[u32; 32]> = (0..k1 * k2)
            .map(|_| std::array::from_fn(|_| next()))
            .collect();

        let comm = pcs.commit_stacked(&[&a, &b, &c]);
        let mut zs = 0xC0FFEE99u64;
        let z: Vec<GR32_128> = (0..pcs.num_vars()).map(|_| rand_gr128(&mut zs)).collect();
        let rho = rand_gr128(&mut zs);
        let weights = [GR32_128::ONE, rho, rho * rho];

        let (y, proof) = pcs.open_stacked(&comm, &z, &weights);
        assert!(
            pcs.verify_stacked(comm.root, &z, &weights, y, &proof),
            "verify_stacked rejected its own proof"
        );

        // (b) equals weighted sum of the three individual openings.
        let mut expected = GR32_128::ZERO;
        for (m, w) in [(&a, weights[0]), (&b, weights[1]), (&c, weights[2])] {
            let cm = pcs.commit(m);
            let (yi, pi) = pcs.open(&cm, &z);
            assert!(pcs.verify(cm.root, &z, yi, &pi));
            expected = expected + w * yi;
        }
        assert_eq!(y, expected, "stacked weighted value != Σ ρ^t·m_t(z)");
        assert!(!pcs.verify_stacked(comm.root, &z, &weights, y + GR32_128::ONE, &proof));
    }
}

#[cfg(test)]
mod setup_tests {
    use super::*;
    use galois_ring::{Gr1, GR32_128};

    /// Cheap LCG → `u32` stream for filling test data.
    fn lcg(seed: u64) -> impl FnMut() -> u32 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u32
        }
    }

    #[test]
    fn query_size_matches_paper_params() {
        assert_eq!(recommended_query_size(0.19, 100), 1060);
        // The returned t actually clears the full geometric bound.
        for (delta, lambda) in [(0.19, 100usize), (0.5, 128), (0.3, 128), (0.4, 80)] {
            let t = recommended_query_size(delta, lambda);
            let b1 = (1.0 - delta / 3.0).powi(t as i32);
            let b2 = 2.0 * (1.0 - 2.0 * delta / 3.0).powi(t as i32);
            assert!(b1 + b2 <= (-(lambda as f64)).exp2(), "t={t} too small");
            // ...and t−1 does not (tightness).
            let t = t - 1;
            let b1 = (1.0 - delta / 3.0).powi(t as i32);
            assert!(b1 > (-(lambda as f64)).exp2(), "t not minimal");
        }
        // Monotone in λ and in tightness (smaller δ ⇒ more queries).
        assert!(recommended_query_size(0.5, 100) < recommended_query_size(0.5, 128));
        assert!(recommended_query_size(0.5, 128) < recommended_query_size(0.3, 128));
    }

    #[test]
    fn optimal_shape_beats_every_alternative() {
        for (m, q, t, lb) in [
            (20usize, 4usize, 487usize, 4usize),
            (18, 6, 842, 4),
            (16, 4, 487, 1),
        ] {
            let n_total = 1usize << m;
            let (k1, k2) = optimal_shape(m, q, t, lb);
            assert_eq!(k1 * k2, n_total);
            let best = proof_size_bytes(k1, k2, q * k2, t, lb);
            for log_k1 in 0..=m {
                let a = 1usize << log_k1;
                let b = n_total >> log_k1;
                assert!(best <= proof_size_bytes(a, b, q * b, t, lb));
            }
            // Within a factor 2 of the continuous optimum k1 ≈ 2·√(N/t).
            let cont = 2.0 * ((n_total as f64) / (t as f64)).sqrt();
            assert!(
                (k1 as f64) >= cont / 2.0 && (k1 as f64) <= cont * 2.0,
                "k1={k1} far from 2·sqrt(N/t)={cont} (m={m}, t={t})"
            );
        }
    }

    /// The static predictor must equal the real serialized proof, byte-for-byte.
    #[test]
    fn proof_size_predictor_matches_serialize() {
        let (m, q, t) = (8usize, 4usize, 8usize);
        let (k1, k2) = optimal_shape(m, q, t, 4);
        let pcs = Pcs::<u32>::new(k1, k2, q, t, 0xBEEF);
        let mut next = lcg(1);
        let evals: Vec<[u32; 32]> = (0..k1 * k2)
            .map(|_| std::array::from_fn(|_| next()))
            .collect();
        let comm = pcs.commit(&evals);
        let z: Vec<GR32_128> = (0..pcs.num_vars())
            .map(|_| GR32_128::from_gr1(Gr1(next())))
            .collect();
        let (_y, proof) = pcs.open(&comm, &z);
        assert_eq!(proof.size_bytes(), proof_size_bytes(k1, k2, pcs.n(), t, 4));
    }

    /// `setup` picks sound parameters and the resulting PCS round-trips.
    #[test]
    fn setup_roundtrip() {
        let pcs = Pcs::<u32>::setup(12, 4, 0.5, 128, 0xC0DE);
        assert_eq!(pcs.k1 * pcs.k2, 1 << 12);
        assert_eq!(pcs.ell, recommended_query_size(0.5, 128));
        assert_eq!((pcs.k1, pcs.k2), optimal_shape(12, 4, pcs.ell, 4));

        let mut next = lcg(7);
        let evals: Vec<[u32; 32]> = (0..pcs.k1 * pcs.k2)
            .map(|_| std::array::from_fn(|_| next()))
            .collect();
        let comm = pcs.commit(&evals);
        let z: Vec<GR32_128> = (0..pcs.num_vars())
            .map(|_| GR32_128::from_gr1(Gr1(next())))
            .collect();
        let (y, proof) = pcs.open(&comm, &z);
        assert!(pcs.verify(comm.root, &z, y, &proof));
        assert!(!pcs.verify(comm.root, &z, y + GR32_128::ONE, &proof));
    }
}
