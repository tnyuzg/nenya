//! End-to-end zkSNARK for the **general-circuit** reduction (paper §4.3,
//! Halevi–Shoup cyclic-diagonal packing), generic over the base-limb width `L`.
//!
//! ## Structure (mirrors the SIMD prover, with two differences)
//!
//! 1. The linear map `A·w` is the **32-layer diagonal sum** `â = Σ_ℓ D̂_A^(ℓ)·ρ̂^(ℓ)`
//!    where `ρ̂^(ℓ) = rot_ℓ(ŵ)` are the cyclic rotations of the block-packed
//!    witness. The `ρ̂^(ℓ)` are **not committed** — they are virtual openings of
//!    the single `ŵ` commitment ([`pcs_ring::Pcs::rotated_open`]).
//! 2. The packed matvec carries a kernel error, so unlike SIMD the linear check
//!    has an error term. We use **two** committed errors (low-degree, in `ker ψ`):
//!    - `ê_mult' = â∘b̂ − C_sum` (opened at the mult target `r`), folding the
//!      C-relation kernel error into the mult error (C-row uses the raw matvec).
//!    - `ê_lin = ê_A + γ·ê_B` (opened at `s₁`), committed after γ is sampled.
//!
//! ## Setup vs. prover cost
//!
//! The packed diagonals `D̂_{A,B,C}^(ℓ)` are **public and deterministic** — they
//! are precomputed once into [`GeneralSnarkParams`] ([`reduction::pack_general_matrices`])
//! and are **not** part of the per-proof cost (paper §4.3). The prover's
//! reduction is only the witness-dependent part ([`reduction::general_reduce_witness`]):
//! the 32 block-rotations `ρ̂` and the A,B kernel quotients.
//!
//! ## v0 caveats
//! - Per-round Fiat–Shamir (same as the SIMD prover).
//! - The kernel checks `ψ(ê)=0` (free-subspace check) are enforced by the
//!   verifier via [`crate::r2_pcs::columns_in_kernel`] on the opened error
//!   columns (Remark 4.15) — pure verifier work, no prover/proof cost.
//! - The 32 virtual rotated openings derive independent query indices (no
//!   column sharing yet) — a proof-size optimization.

use galois_ring::{compute_t1_generic, Gr1, Gr128, Gr4, Limb};
use pcs_ring::{
    Hash, OpeningProof, Pcs, RotatedOpeningBatch, SECURE_DELTA, SECURE_LAMBDA, SECURE_Q,
};
use rayon::prelude::*;
use reduction::constants::{constants, HasPackingConstants};
use reduction::matrix::CsrMatrix;
use reduction::{general_reduce_witness, pack_r2, psi_r2_with_t1, GeneralDiagonals};
use std::time::Instant;

use piop_ring::{
    build_eq_table, prove_linear_check_general, prove_mult_check_sub, verify_linear_check_general,
    verify_mult_check_sub, LinearCheckGeneralProof, Mle, MultCheckSubProof, LAYERS,
};

use crate::error::VerifyError;
use crate::r2_pcs::{columns_in_kernel, R2OpeningProof, R2Pcs};
use crate::transcript::Transcript;

/// Public parameters. Holds the **precomputed** diagonals (setup) plus the two
/// PCSs (sized for the block count `n_blocks = n/32 = 2^log_blocks`).
pub struct GeneralSnarkParams<L: Limb = u32> {
    pub log_blocks: usize,
    pub pcs_w: Pcs<L>,
    /// PCS for the slot-stacked (â,b̂,ĉ) — shaped for a single 3-stacked opening.
    pub pcs_abc: Pcs<L>,
    pub pcs_e: R2Pcs<L>,
    /// Precomputed packed cyclic-diagonal layers of `A, B, C` (setup, not prover cost).
    pub diag: GeneralDiagonals<L>,
}

impl<L: HasPackingConstants> GeneralSnarkParams<L> {
    /// **Secure default** params at the 100-bit operating point (δ = [`SECURE_DELTA`],
    /// λ = [`SECURE_LAMBDA`], q = [`SECURE_Q`]); query budget + optimal shape via
    /// `*::setup`. `diag` is the packed cyclic-diagonal preprocessing
    /// (`pack_general_matrices`), witness-independent setup. Real proofs use this;
    /// raw construction with a small explicit `ell` is for tests/benchmarks only.
    pub fn secure(log_blocks: usize, diag: GeneralDiagonals<L>, seed_w: u64, seed_e: u64) -> Self {
        Self {
            log_blocks,
            // Witness PCS is opened LAYERS times sharing columns (rotated_open_batch),
            // so optimize the shape for "m̄ ×LAYERS, columns ×1".
            pcs_w: Pcs::setup_batched(
                log_blocks,
                SECURE_Q,
                SECURE_DELTA,
                SECURE_LAMBDA,
                seed_w,
                LAYERS,
            ),
            // (â,b̂,ĉ) PCS: ONE 3-stacked opening, so shape for "columns ×3, m̄ ×1"
            // (small k1) — NOT the rotated witness shape (which would bloat it).
            pcs_abc: Pcs::setup_stacked(
                log_blocks,
                SECURE_Q,
                SECURE_DELTA,
                SECURE_LAMBDA,
                seed_w ^ 0xABC,
                3,
            ),
            pcs_e: R2Pcs::setup(log_blocks, SECURE_Q, SECURE_DELTA, SECURE_LAMBDA, seed_e),
            diag,
        }
    }
}

/// General-circuit SNARK proof.
pub struct GeneralSnarkProof<L: Limb = u32> {
    pub root_w: Hash,
    /// Root of the slot-stacked (â,b̂,ĉ) commitment (image-form; all open at s₁).
    pub root_abc: Hash,
    pub root_emult: Hash,
    /// Root of the stacked (ê_A, ê_B, ê_C) commitment (all open at s₁).
    pub root_estack: Hash,
    pub mult_proof: MultCheckSubProof<L>,
    pub lin_proof: LinearCheckGeneralProof<L>,
    /// ê_mult opened at s₁ (binds final_e).
    pub open_emult: R2OpeningProof<L>,
    /// (â,b̂,ĉ) opened at s₁ with weights (1,γ,γ²); value `â+γb̂+γ²ĉ`(s₁), checked
    /// by the verifier against `final_a+γ·final_b+γ²·final_c`.
    pub open_abc: OpeningProof<L>,
    /// (ê_A,ê_B,ê_C) opened at s₁ with weights (1,γ,γ²); value used in `claim_lin`.
    pub open_estack: R2OpeningProof<L>,
    /// Batched rotated openings of ŵ for the 32 layers — one shared proximity
    /// test (columns/paths once), 32 per-layer `m̄`s.
    pub open_rho: RotatedOpeningBatch<L>,
}

impl<L: Limb> GeneralSnarkProof<L> {
    /// Serialize proof to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let limb_bytes = (L::BITS / 8) as usize;

        let _ = limb_bytes;
        // Commitments
        buf.extend_from_slice(&self.root_w);
        buf.extend_from_slice(&self.root_abc);
        buf.extend_from_slice(&self.root_emult);
        buf.extend_from_slice(&self.root_estack);

        // Proofs
        buf.extend_from_slice(&self.mult_proof.serialize());
        buf.extend_from_slice(&self.lin_proof.serialize());

        // PCS openings
        buf.extend_from_slice(&self.open_emult.serialize());
        buf.extend_from_slice(&self.open_abc.serialize());
        buf.extend_from_slice(&self.open_estack.serialize());
        buf.extend_from_slice(&self.open_rho.serialize());

        buf
    }

    /// Actual serialized size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.serialize().len()
    }
}

/// Per-stage prover timing (ms). The diagonal packing (setup) is excluded.
#[derive(Clone, Debug, Default)]
pub struct GeneralProveTiming {
    /// Witness reduction `ρ̂` + A,B kernel quotients (per-proof, *not* diagonals).
    pub reduce_ms: f64,
    /// The 32-layer matvecs `Σ_ℓ D̂^(ℓ)·ρ̂^(ℓ)` for A, B, C + `ê_mult'`.
    pub matvec_ms: f64,
    /// Commit `ŵ` + `ê_mult'` + `ê_lin`.
    pub commit_ms: f64,
    /// commit split by phase (across all commitments): RAA-encode vs Merkle.
    pub commit_encode_ms: f64,
    pub commit_merkle_ms: f64,
    pub mult_sumcheck_ms: f64,
    /// Build the 32 layer rows `M^(ℓ)` + the linear sumcheck.
    pub linear_ms: f64,
    /// `linear_ms` split: building the 32 layer rows vs the linear sumcheck.
    pub linear_rowbuild_ms: f64,
    pub linear_sumcheck_ms: f64,
    /// PCS opens: 32 rotated `ρ̂` + the two errors.
    pub opens_ms: f64,
    pub total_ms: f64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn inner_product<L: Limb>(a: &[Gr128<L>], b: &[Gr128<L>]) -> Gr128<L> {
    debug_assert_eq!(a.len(), b.len());
    a.par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| x * y)
        .reduce(|| Gr128::<L>::ZERO, |u, v| u + v)
}

/// `ψ(x)` (Teichmüller-eval — the SAME map the kernel check uses) flattened to
/// the 32 natural slot order (`[i][0],[i][1]` interleaved). It is the left
/// inverse of `pack_r2`, so for `x ∉ Im φ` the residue `x − pack_r2(unpack_flat(x))`
/// lies in `ker ψ` exactly. (`unpack_r2`, the linear inverse, agrees only on the
/// image and would NOT give a ψ-kernel residue.)
fn unpack_flat<L: Limb>(x: Gr128<L>, t1: &[Gr4<L>; 16]) -> [Gr1<L>; 32] {
    let m = psi_r2_with_t1(&x, t1);
    let mut out = [Gr1::<L>::ZERO; 32];
    for i in 0..16 {
        out[2 * i] = m[i][0];
        out[2 * i + 1] = m[i][1];
    }
    out
}

/// The 32 base-limb slots of an image element (for slot-form commitment).
fn slots_of<L: Limb>(x: Gr128<L>, t1: &[Gr4<L>; 16]) -> [L; 32] {
    let f = unpack_flat(x, t1);
    std::array::from_fn(|i| f[i].0)
}

/// `Σ_ℓ D̂^(ℓ)·ρ̂^(ℓ)`. Parallel over the **output rows** (the big axis, ≫ #cores)
/// via the precomputed CSR layers: each output `p` independently sums its
/// nonzeros across all 32 layers — no scatter, no per-layer scratch, all cores.
fn layered_matvec<L: Limb>(
    layers: &[CsrMatrix<Gr128<L>>; LAYERS],
    rho: &[Vec<Gr128<L>>; LAYERS],
    nb: usize,
) -> Vec<Gr128<L>> {
    (0..nb)
        .into_par_iter()
        .map(|p| {
            let mut acc = Gr128::<L>::ZERO;
            for ell in 0..LAYERS {
                let m = &layers[ell];
                let r = &rho[ell];
                for k in m.row_ptr[p]..m.row_ptr[p + 1] {
                    acc = acc + m.vals[k] * r[m.col_idx[k]];
                }
            }
            acc
        })
        .collect()
}

/// 32 layer-row MLEs `M^(ℓ)(q) = Σ_p eq_s1[p]·(D̂_A+γD̂_B)^(ℓ)[p,q] + γ²·Σ_p eq_r[p]·D̂_C^(ℓ)[p,q]`.
fn build_layer_rows<L: Limb>(
    diag: &GeneralDiagonals<L>,
    eq_s1: &[Gr128<L>],
    eq_r: &[Gr128<L>],
    gamma: Gr128<L>,
    gamma2: Gr128<L>,
    nb: usize,
) -> [Mle<L>; LAYERS] {
    // Serial over the 32 layers; within each, parallelize over the output column
    // `q` (the big axis) via the CSC layers — each `q` sums its column's entries
    // independently (no scatter). In the CSC, `col_idx` holds the original row p.
    let rows: Vec<Mle<L>> = (0..LAYERS)
        .map(|ell| {
            let ca = &diag.csc_a[ell];
            let cb = &diag.csc_b[ell];
            let cc = &diag.csc_c[ell];
            let row: Vec<Gr128<L>> = (0..nb)
                .into_par_iter()
                .map(|q| {
                    let mut a_acc = Gr128::<L>::ZERO;
                    for k in ca.row_ptr[q]..ca.row_ptr[q + 1] {
                        a_acc = a_acc + eq_s1[ca.col_idx[k]] * ca.vals[k];
                    }
                    let mut b_acc = Gr128::<L>::ZERO;
                    for k in cb.row_ptr[q]..cb.row_ptr[q + 1] {
                        b_acc = b_acc + eq_s1[cb.col_idx[k]] * cb.vals[k];
                    }
                    let mut c_acc = Gr128::<L>::ZERO;
                    for k in cc.row_ptr[q]..cc.row_ptr[q + 1] {
                        c_acc = c_acc + eq_r[cc.col_idx[k]] * cc.vals[k];
                    }
                    a_acc + gamma * b_acc + gamma2 * c_acc
                })
                .collect();
            Mle::new(row)
        })
        .collect();
    rows.try_into().expect("LAYERS rows")
}

/// Finer linear-check decomposition (ms) for the cost breakdown. Splits the
/// linear check into its two value-independent matrix operations, leaving the
/// value-dependent sumcheck to the real prover:
///
/// - **matrix random evaluation** — build the 32 combined rows
///   `M^(ℓ) = A + γ·B + γ²·C` at the random point (the per-layer A,B,C matvec and
///   the γ-fold, as one term — [`build_layer_rows`]). Shared by prover & verifier.
/// - **eval** — verifier-only: recompute the 32 row evaluations `⟨eq_s2, M^(ℓ)⟩`.
///
/// Both are **value-independent** (`build_layer_rows` and `inner_product` iterate
/// every nonzero / entry regardless of the values), so mocking the challenges with
/// random `Gr128` faithfully reflects `prove`/`verify`'s cost. The **sumcheck** is
/// deliberately *not* here: its cost is value-dependent (the `x²=y` rows/rhos are
/// sparse), so it must be read from the real prover
/// (`GeneralProveTiming::linear_sumcheck_ms`) rather than mocked.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearDecomp {
    /// Matrix random evaluation: build `M^(ℓ) = A + γB + γ²C` (matvec + γ-fold).
    pub matrix_eval_ms: f64,
    /// Verifier row eval: recompute `⟨eq_s2, M^(ℓ)⟩` over the 32 rows.
    pub eval_ms: f64,
}

pub fn measure_linear_decomp<L: Limb>(
    diag: &GeneralDiagonals<L>,
    log_blocks: usize,
) -> LinearDecomp {
    let nb = 1usize << log_blocks;
    // Mock the verifier challenges — the per-phase cost is value-independent.
    let mut s = 0x5EED_1357_u64;
    let mut rnd = move || {
        let mut arr = [Gr4::<L>::ZERO; 32];
        for slot in arr.iter_mut() {
            let mut g = [L::ZERO; 4];
            for x in g.iter_mut() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *x = L::from_u64(s >> 32);
            }
            *slot = Gr4(g);
        }
        Gr128(arr)
    };
    let eq_s1: Vec<Gr128<L>> = (0..nb).map(|_| rnd()).collect();
    let eq_r: Vec<Gr128<L>> = (0..nb).map(|_| rnd()).collect();
    let eq_s2: Vec<Gr128<L>> = (0..nb).map(|_| rnd()).collect();
    let gamma = rnd();
    let gamma2 = gamma * gamma;

    // Matrix random evaluation: build `M^(ℓ) = A + γB + γ²C` (best-of-2; the
    // matvec is memory-bound, so min() rejects scheduler/first-touch noise).
    let mut matrix_eval_ms = f64::INFINITY;
    let mut rows = build_layer_rows(diag, &eq_s1, &eq_r, gamma, gamma2, nb);
    for _ in 0..2 {
        let t = Instant::now();
        rows = build_layer_rows(diag, &eq_s1, &eq_r, gamma, gamma2, nb);
        matrix_eval_ms = matrix_eval_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }

    // Verifier: recompute the 32 row evaluations `⟨eq_s2, M^(ℓ)⟩` (best-of-2).
    let mut eval_ms = f64::INFINITY;
    for _ in 0..2 {
        let t = Instant::now();
        let mut acc = Gr128::<L>::ZERO;
        for ell in 0..LAYERS {
            acc = acc + inner_product(&eq_s2, rows[ell].evals());
        }
        std::hint::black_box(acc);
        eval_ms = eval_ms.min(t.elapsed().as_secs_f64() * 1e3);
    }

    LinearDecomp {
        matrix_eval_ms,
        eval_ms,
    }
}

// Bind the mult-check finals. Round messages are bound per-round via
// `Challenger::observe` inside the sumcheck; the claim is bound by the
// `claim_mult` absorb before the sumcheck.
fn absorb_mult_finals<L: Limb>(t: &mut Transcript, proof: &MultCheckSubProof<L>) {
    t.absorb_gr128_slice(
        b"mult/finals",
        &[proof.final_a, proof.final_b, proof.final_c, proof.final_e],
    );
}

// ---------------------------------------------------------------------------
// Prover
// ---------------------------------------------------------------------------

/// Prove that the witness `w` satisfies the general circuit whose diagonals are
/// precomputed in `params`.
pub fn prove_general<L: HasPackingConstants>(
    w: &[Gr1<L>],
    params: &GeneralSnarkParams<L>,
) -> GeneralSnarkProof<L> {
    prove_general_with_timing(w, params).0
}

pub fn prove_general_with_timing<L: HasPackingConstants>(
    w: &[Gr1<L>],
    params: &GeneralSnarkParams<L>,
) -> (GeneralSnarkProof<L>, GeneralProveTiming) {
    let mut timing = GeneralProveTiming::default();
    let total_t = Instant::now();
    let diag = &params.diag;
    let log_b = params.log_blocks;
    let nb = 1usize << log_b;

    // ----- 1. Per-proof witness reduction (ρ̂ + A,B quotients). Diagonals are setup. -----
    let t = Instant::now();
    let parts = general_reduce_witness(w, diag);
    timing.reduce_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 2. 32-layer matvecs → â, b̂, ê_mult'. -----
    let t = Instant::now();
    let cst = constants::<L>();
    let v_alpha: Gr128<L> = Gr128::<L>::from_gr4(cst.v_alpha);
    let v_beta: Gr128<L> = cst.v_beta;
    // Per-row independent → parallel maps (bit-identical).
    let err = |q1: &[Gr128<L>], q2: &[Gr128<L>]| -> Vec<Gr128<L>> {
        (0..nb)
            .into_par_iter()
            .map(|i| q1[i] * v_alpha + q2[i] * v_beta)
            .collect()
    };
    let e_a = err(&parts.q1_a, &parts.q2_a);
    let e_b = err(&parts.q1_b, &parts.q2_b);
    let a_sum = layered_matvec(&diag.csr_a, &parts.rho, nb);
    let b_sum = layered_matvec(&diag.csr_b, &parts.rho, nb);
    let c_sum = layered_matvec(&diag.csr_c, &parts.rho, nb);
    let a_hat: Vec<Gr128<L>> = (0..nb).into_par_iter().map(|i| a_sum[i] - e_a[i]).collect();
    let b_hat: Vec<Gr128<L>> = (0..nb).into_par_iter().map(|i| b_sum[i] - e_b[i]).collect();
    // C is now treated like A,B: ĉ = image(Ĉŵ), ê_C = Ĉŵ − ĉ ∈ ker ψ.
    // ĉ = pack_r2(ψ(c_sum)); since c_sum = ĉ + ê_C, ψ(c_sum) = c (image part).
    let t1 = compute_t1_generic::<L>();
    let c_hat: Vec<Gr128<L>> = c_sum
        .par_iter()
        .map(|&x| pack_r2(unpack_flat(x, &t1)))
        .collect();
    let e_c: Vec<Gr128<L>> = (0..nb)
        .into_par_iter()
        .map(|i| c_sum[i] - c_hat[i])
        .collect();
    // ê_mult' = â∘b̂ − ĉ (the committed image ĉ, NOT the raw Ĉŵ — no ê_C fold).
    let e_mult: Vec<Gr128<L>> = (0..nb)
        .into_par_iter()
        .map(|i| a_hat[i] * b_hat[i] - c_hat[i])
        .collect();
    // Slot form of â,b̂,ĉ (all ∈ Im φ) for the slot-stacked commitment.
    let a_slots: Vec<[L; 32]> = a_hat.par_iter().map(|&x| slots_of(x, &t1)).collect();
    let b_slots: Vec<[L; 32]> = b_hat.par_iter().map(|&x| slots_of(x, &t1)).collect();
    let c_slots: Vec<[L; 32]> = c_hat.par_iter().map(|&x| slots_of(x, &t1)).collect();
    timing.matvec_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 3. Commit ŵ + (â,b̂,ĉ) slot-stacked + (ê_A,ê_B,ê_C) stacked + ê_mult'.
    // â,b̂,ĉ are committed in slot form ⇒ structurally in Im φ (no image check
    // needed); they bind the mult-check values so a kernel-polluted â can't forge.
    // The three errors are stacked (all open at s₁) and ψ-checked. -----
    let w_slots: Vec<[L; 32]> = (0..nb)
        .map(|qb| std::array::from_fn(|i| w[32 * qb + i].0))
        .collect();
    let (comm_w, we, wm) = params.pcs_w.commit_with_timing(&w_slots);
    let (comm_abc, ae, am) = params
        .pcs_abc
        .commit_stacked_with_timing(&[&a_slots, &b_slots, &c_slots]);
    let (comm_emult, ee, em) = params.pcs_e.commit_with_timing(&e_mult);
    let (comm_estack, se, sm) = params.pcs_e.commit_stacked_with_timing(&[&e_a, &e_b, &e_c]);
    timing.commit_encode_ms = we + ae + ee + se;
    timing.commit_merkle_ms = wm + am + em + sm;
    timing.commit_ms = timing.commit_encode_ms + timing.commit_merkle_ms;

    let mut tr = Transcript::new(b"general");
    tr.absorb_root(&comm_w.root);
    tr.absorb_root(&comm_abc.root);
    tr.absorb_root(&comm_emult.root);
    tr.absorb_root(&comm_estack.root);
    let r_target: Vec<Gr128<L>> = tr.challenge_target::<L>(b"mult_check_target", log_b);

    // ----- 4. Mult-check (sub form): Σ_h eq(r,h)·(â·b̂ − ĉ − ê_mult)(h) = 0.
    // Finals â(s₁),b̂(s₁),ĉ(s₁),ê_mult(s₁) all at the single point s₁. -----
    let t = Instant::now();
    tr.absorb_label(b"mult_check_sumcheck");
    let a_mle = Mle::new(a_hat);
    let b_mle = Mle::new(b_hat);
    let c_mle = Mle::new(c_hat);
    let e_mle = Mle::new(e_mult);
    let (mult_proof, s_1) =
        prove_mult_check_sub(&a_mle, &b_mle, &c_mle, &e_mle, &r_target, &mut tr);
    timing.mult_sumcheck_ms = t.elapsed().as_secs_f64() * 1e3;
    absorb_mult_finals(&mut tr, &mult_proof);

    let gamma: Gr128<L> = tr.challenge_gr128::<L>(b"linear_check_gamma");
    let gamma2 = gamma * gamma;

    // ----- 5. Open the stacked errors (committed pre-γ) at s₁ with weights
    // (1,γ,γ²) → ê_A(s₁)+γ·ê_B(s₁)+γ²·ê_C(s₁); and the slot-stacked (â,b̂,ĉ) at s₁
    // with the same weights → bound to the mult finals (Schwartz–Zippel over γ).
    // ê_mult is opened at s₁ too (bind final_e). -----
    let t = Instant::now();
    let weights = [Gr128::<L>::ONE, gamma, gamma2];
    let (estack_val, open_estack) = params.pcs_e.open_stacked(&comm_estack, &s_1, &weights);
    let (_abc_val, open_abc) = params.pcs_abc.open_stacked(&comm_abc, &s_1, &weights);
    let (_em_s1, open_emult) = params.pcs_e.open(&comm_emult, &s_1);
    debug_assert_eq!(
        _abc_val,
        mult_proof.final_a + gamma * mult_proof.final_b + gamma2 * mult_proof.final_c
    );
    debug_assert_eq!(_em_s1, mult_proof.final_e);
    timing.opens_ms += t.elapsed().as_secs_f64() * 1e3;

    // ----- 6. Build the 32 layer rows + linear-check. All of â,b̂,ĉ now bind at
    // s₁, so the C-row uses eq_s1 too (no eq_r asymmetry). -----
    let t = Instant::now();
    let tr_rb = Instant::now();
    let eq_s1 = build_eq_table(&s_1);
    debug_assert_eq!(
        estack_val,
        inner_product(eq_s1.evals(), &e_a)
            + gamma * inner_product(eq_s1.evals(), &e_b)
            + gamma2 * inner_product(eq_s1.evals(), &e_c)
    );
    let rows = build_layer_rows(diag, eq_s1.evals(), eq_s1.evals(), gamma, gamma2, nb);
    timing.linear_rowbuild_ms = tr_rb.elapsed().as_secs_f64() * 1e3;
    // Move ρ̂ into the MLEs (no clone): `parts.rho` is not used after the matvec
    // above, so cloning 32×nb Gr128 (16 GB @2^25) was pure waste.
    let rhos: [Mle<L>; LAYERS] = parts.rho.map(Mle::new);
    let claim_lin =
        mult_proof.final_a + gamma * mult_proof.final_b + gamma2 * mult_proof.final_c + estack_val;
    tr.absorb_label(b"linear_check_sumcheck");
    let tr_sc = Instant::now();
    let (lin_proof, s_2) = prove_linear_check_general(rows, rhos, claim_lin, &mut tr);
    timing.linear_sumcheck_ms = tr_sc.elapsed().as_secs_f64() * 1e3;
    timing.linear_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 7. Opens: 32 virtual rotated ρ̂ (one shared proximity test). -----
    let t = Instant::now();
    let (ys, open_rho) = params.pcs_w.rotated_open_batch(&comm_w, &s_2, LAYERS);
    debug_assert_eq!(
        ys.as_slice(),
        &lin_proof.final_rho[..],
        "rotated opens ≠ final_rho"
    );
    timing.opens_ms += t.elapsed().as_secs_f64() * 1e3;

    timing.total_ms = total_t.elapsed().as_secs_f64() * 1e3;

    (
        GeneralSnarkProof {
            root_w: comm_w.root,
            root_abc: comm_abc.root,
            root_emult: comm_emult.root,
            root_estack: comm_estack.root,
            mult_proof,
            lin_proof,
            open_emult,
            open_abc,
            open_estack,
            open_rho,
        },
        timing,
    )
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

/// Verify against the precomputed diagonals in `params` (no witness, no
/// re-reduction).
pub fn verify_general<L: HasPackingConstants>(
    params: &GeneralSnarkParams<L>,
    proof: &GeneralSnarkProof<L>,
) -> Result<(), VerifyError> {
    verify_general_with_timing(params, proof).0
}

/// Per-stage verifier timing (ms): `piop` = transcript + sumcheck verifies +
/// per-layer row recompute; `pcs` = PCS opening verifies (incl. 32 rotated) +
/// the ψ-kernel checks.
#[derive(Clone, Debug, Default)]
pub struct GeneralVerifyTiming {
    pub piop_ms: f64,
    /// Of `piop_ms`: the 32-layer row recompute (`build_layer_rows` + the
    /// per-layer `⟨eq_s2,M^ℓ⟩` check). The rest of `piop_ms` is sumcheck-verify.
    pub row_recompute_ms: f64,
    pub pcs_ms: f64,
    pub total_ms: f64,
}

pub fn verify_general_with_timing<L: HasPackingConstants>(
    params: &GeneralSnarkParams<L>,
    proof: &GeneralSnarkProof<L>,
) -> (Result<(), VerifyError>, GeneralVerifyTiming) {
    let diag = &params.diag;
    let log_b = params.log_blocks;
    let mut tm = GeneralVerifyTiming::default();
    let total_t = Instant::now();

    // ---- PIOP phase ----
    let t = Instant::now();
    let mut row_recompute_ms = 0.0f64;
    type PiopOut<L> = (
        Vec<Gr128<L>>,
        Vec<Gr128<L>>,
        Gr128<L>,
        Gr128<L>,
        Gr128<L>,
        Gr128<L>,
    );
    let piop = (|| -> Result<PiopOut<L>, VerifyError> {
        let mut tr = Transcript::new(b"general");
        tr.absorb_root(&proof.root_w);
        tr.absorb_root(&proof.root_abc);
        tr.absorb_root(&proof.root_emult);
        tr.absorb_root(&proof.root_estack);
        let r_target: Vec<Gr128<L>> = tr.challenge_target::<L>(b"mult_check_target", log_b);

        tr.absorb_label(b"mult_check_sumcheck");
        let s_1 = verify_mult_check_sub(&proof.mult_proof, &r_target, &mut tr)
            .map_err(|_| VerifyError::MultSumcheck)?;
        absorb_mult_finals(&mut tr, &proof.mult_proof);

        let gamma: Gr128<L> = tr.challenge_gr128::<L>(b"linear_check_gamma");
        let gamma2 = gamma * gamma;

        tr.absorb_label(b"linear_check_sumcheck");
        let s_2 = verify_linear_check_general(&proof.lin_proof, &mut tr)
            .map_err(|_| VerifyError::LinSumcheck)?;

        // Recompute the three opened values from their m̄s (⟨m̄, e₁(s₁)⟩).
        let dot = |mbar: &[Gr128<L>], e1: &[Gr128<L>]| -> Gr128<L> {
            let mut v = Gr128::<L>::ZERO;
            for j in 0..e1.len() {
                v = v + mbar[j] * e1[j];
            }
            v
        };
        let log_k1_e = params.pcs_e.k1.trailing_zeros() as usize;
        let e1_e = build_eq_table(&s_1[log_k1_e..]);
        let estack_val = dot(&proof.open_estack.mbar, e1_e.evals());
        let emult_s1 = dot(&proof.open_emult.mbar, e1_e.evals());
        let log_k1_abc = params.pcs_abc.k1.trailing_zeros() as usize;
        let e1_abc = build_eq_table(&s_1[log_k1_abc..]);
        let abc_val = dot(&proof.open_abc.mbar, e1_abc.evals());

        // (â,b̂,ĉ) and ê_mult opened at s₁ bind the mult-check finals.
        let abc_expected = proof.mult_proof.final_a
            + gamma * proof.mult_proof.final_b
            + gamma2 * proof.mult_proof.final_c;
        if abc_val != abc_expected || emult_s1 != proof.mult_proof.final_e {
            return Err(VerifyError::LinClaimMismatch);
        }

        // Linear check ties â,b̂,ĉ to the matvec via ê_A,ê_B,ê_C (all at s₁).
        let expected_claim_lin = abc_expected + estack_val;
        if proof.lin_proof.claimed_sum != expected_claim_lin {
            return Err(VerifyError::LinClaimMismatch);
        }

        let eq_s1 = build_eq_table(&s_1);
        let eq_s2 = build_eq_table(&s_2);
        let nb = 1usize << log_b;
        let tr_rr = Instant::now();
        let rows = build_layer_rows(diag, eq_s1.evals(), eq_s1.evals(), gamma, gamma2, nb);
        for ell in 0..LAYERS {
            if inner_product(eq_s2.evals(), rows[ell].evals()) != proof.lin_proof.final_row[ell] {
                return Err(VerifyError::LinARowMismatch);
            }
        }
        row_recompute_ms = tr_rr.elapsed().as_secs_f64() * 1e3;
        Ok((s_1, s_2, gamma, abc_val, emult_s1, estack_val))
    })();
    tm.piop_ms = t.elapsed().as_secs_f64() * 1e3;
    tm.row_recompute_ms = row_recompute_ms;
    let (s_1, s_2, gamma, abc_val, emult_s1, estack_val) = match piop {
        Ok(x) => x,
        Err(e) => return (Err(e), tm),
    };
    let gamma2 = gamma * gamma;

    // ---- PCS phase ----
    let t = Instant::now();
    let pcs = (|| -> Result<(), VerifyError> {
        let w = [Gr128::<L>::ONE, gamma, gamma2];
        if !params
            .pcs_e
            .verify(proof.root_emult, &s_1, emult_s1, &proof.open_emult)
        {
            return Err(VerifyError::PcsE);
        }
        // (â,b̂,ĉ) — slot form ⇒ structurally in Im φ, so NO image check needed.
        if !params
            .pcs_abc
            .verify_stacked(proof.root_abc, &s_1, &w, abc_val, &proof.open_abc)
        {
            return Err(VerifyError::PcsW);
        }
        if !params
            .pcs_e
            .verify_stacked(proof.root_estack, &s_1, &w, estack_val, &proof.open_estack)
        {
            return Err(VerifyError::PcsE);
        }
        if proof.open_rho.mbars.len() != LAYERS {
            return Err(VerifyError::PcsW);
        }
        if !params.pcs_w.rotated_verify_batch(
            proof.root_w,
            &s_2,
            &proof.lin_proof.final_rho,
            &proof.open_rho,
        ) {
            return Err(VerifyError::PcsW);
        }
        if !columns_in_kernel(&proof.open_emult) || !columns_in_kernel(&proof.open_estack) {
            return Err(VerifyError::KernelCheck);
        }
        Ok(())
    })();
    tm.pcs_ms = t.elapsed().as_secs_f64() * 1e3;
    tm.total_ms = total_t.elapsed().as_secs_f64() * 1e3;
    (pcs, tm)
}
