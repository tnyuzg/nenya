//! End-to-end zkSNARK for the **MIMD** reduction (paper §4.2): 32 *independent*
//! R1CS instances packed into one over `R_2`. Generic over the base-limb width `L`.
//!
//! Structure = SIMD prover + **dense `GR128` matrices** `Â,B̂,Ĉ` (matvec /
//! linear-check row via full `R_2` mult, no `mul_by_gr1` fast path) + **one
//! extra linear error**. Two committed errors (low-degree, in `ker ψ`):
//! - `ê_mult' = â∘b̂ − C_sum` (opened at the mult target `r`), folding `ê_C`.
//! - `ê_lin = ê_A + γ·ê_B` (opened at `s₁`), committed after γ.
//! Verifier: `claim_lin = final_a + γ·final_b + ê_lin(s₁) + γ²·(claim_mult − ê_mult'(r))`.
//!
//! The packed matrices `Â,B̂,Ĉ` are public/deterministic → precomputed once into
//! [`MimdSnarkParams`] (setup), **not** per-proof cost. The prover's reduction is
//! the witness-dependent part only ([`reduction::mimd_reduce_witness`]).
//!
//! Per-round Fiat–Shamir (both sumchecks drive the [`Transcript`] challenger);
//! `ψ(ê)=0` kernel checks enforced on the opened error columns
//! ([`crate::r2_pcs::columns_in_kernel`], verifier-only).

use galois_ring::{compute_t1_generic, Gr1, Gr128, Gr4, Limb};
use pcs_ring::{Hash, OpeningProof, Pcs, SECURE_DELTA, SECURE_LAMBDA, SECURE_Q};
use rayon::prelude::*;
use reduction::constants::{constants, HasPackingConstants};
use reduction::matrix::{matvec_gr128_dense_csr, SparseMatrix};
use reduction::{mimd_reduce_witness, pack_r2, psi_r2_with_t1, MimdMatrices};
use std::time::Instant;

use piop_ring::linear_check_mimd::{
    prove_linear_check_mimd_sumcheck_only, verify_linear_check_mimd, LinearCheckMimdProof,
};
use piop_ring::{
    build_eq_table, prove_mult_check_sub, verify_mult_check_sub, Mle, MultCheckSubProof,
};

use crate::error::VerifyError;
use crate::r2_pcs::{columns_in_kernel, R2OpeningProof, R2Pcs};
use crate::transcript::Transcript;

/// `ψ(x)` flattened to the 32 natural slot order — the left inverse of `pack_r2`
/// (Teichmüller-eval, matching the kernel check), so `x − pack_r2(unpack_flat(x))`
/// is exactly in `ker ψ`.
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

/// MIMD parameters: dimensions, the two PCSs, and the **precomputed** packed
/// matrices (setup, not prover cost).
pub struct MimdSnarkParams<L: Limb = u32> {
    pub log_n: usize,
    pub log_m: usize,
    pub pcs_w: Pcs<L>,
    /// PCS for the slot-stacked (â,b̂,ĉ) — shaped for a single 3-stacked opening.
    pub pcs_abc: Pcs<L>,
    pub pcs_e: R2Pcs<L>,
    pub mats: MimdMatrices<L>,
}

impl<L: Limb> MimdSnarkParams<L> {
    pub fn n(&self) -> usize {
        1 << self.log_n
    }
    pub fn m(&self) -> usize {
        1 << self.log_m
    }
}

impl<L: HasPackingConstants> MimdSnarkParams<L> {
    /// **Secure default** params at the 100-bit operating point (δ = [`SECURE_DELTA`],
    /// λ = [`SECURE_LAMBDA`], q = [`SECURE_Q`]); query budget + optimal shape via
    /// `*::setup`. `mats` is the packed-matrix preprocessing (`pack_mimd_matrices`),
    /// witness-independent setup. Real proofs use this; raw construction with a
    /// small explicit `ell` is for tests/benchmarks only.
    pub fn secure(
        log_n: usize,
        log_m: usize,
        mats: MimdMatrices<L>,
        seed_w: u64,
        seed_e: u64,
    ) -> Self {
        Self {
            log_n,
            log_m,
            pcs_w: Pcs::setup(log_m, SECURE_Q, SECURE_DELTA, SECURE_LAMBDA, seed_w),
            // (â,b̂,ĉ) are length-n, image-form: one 3-stacked opening ⇒ setup_stacked.
            pcs_abc: Pcs::setup_stacked(
                log_n,
                SECURE_Q,
                SECURE_DELTA,
                SECURE_LAMBDA,
                seed_w ^ 0xABC,
                3,
            ),
            pcs_e: R2Pcs::setup(log_n, SECURE_Q, SECURE_DELTA, SECURE_LAMBDA, seed_e),
            mats,
        }
    }
}

pub struct MimdSnarkProof<L: Limb = u32> {
    pub root_w: Hash,
    /// Root of the slot-stacked (â,b̂,ĉ) commitment (image-form; open at s₁).
    pub root_abc: Hash,
    pub root_emult: Hash,
    /// Root of the stacked (ê_A, ê_B, ê_C) commitment (all open at s₁).
    pub root_estack: Hash,
    pub mult_proof: MultCheckSubProof<L>,
    pub lin_proof: LinearCheckMimdProof<L>,
    pub open_w: OpeningProof<L>,
    /// ê_mult opened at s₁ (binds final_e).
    pub open_emult: R2OpeningProof<L>,
    /// (â,b̂,ĉ) at s₁ with weights (1,γ,γ²) — checked against the mult finals.
    pub open_abc: OpeningProof<L>,
    /// (ê_A,ê_B,ê_C) at s₁ with weights (1,γ,γ²); value used in claim_lin.
    pub open_estack: R2OpeningProof<L>,
}

impl<L: Limb> MimdSnarkProof<L> {
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
        buf.extend_from_slice(&self.open_w.serialize());
        buf.extend_from_slice(&self.open_emult.serialize());
        buf.extend_from_slice(&self.open_abc.serialize());
        buf.extend_from_slice(&self.open_estack.serialize());

        buf
    }

    /// Actual serialized size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.serialize().len()
    }
}

/// Per-stage prover timing (ms). Matrix packing (setup) is excluded.
#[derive(Clone, Debug, Default)]
pub struct MimdProveTiming {
    pub reduce_ms: f64,
    pub matvec_ms: f64,
    pub commit_ms: f64,
    /// commit sub-split (diagnostic): ŵ, ê_mult', stacked (ê_A, ê_B).
    pub commit_w_ms: f64,
    pub commit_emult_ms: f64,
    pub commit_estack_ms: f64,
    /// commit split by phase (across all three commitments): RAA-encode vs Merkle.
    pub commit_encode_ms: f64,
    pub commit_merkle_ms: f64,
    pub mult_sumcheck_ms: f64,
    pub linear_ms: f64,
    /// `linear_ms` split: the matrix random evaluation (build `M_γ` row) vs the
    /// rest (the linear sumcheck + glue).
    pub linear_rowbuild_ms: f64,
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

/// `M_γ_row[col] = Σ_y eq_s1[y]·(Â+γB̂)[y,col] + γ²·Σ_y eq_r[y]·Ĉ[y,col]` (dense `R_2`).
/// The three accumulations are independent → run concurrently.
fn build_combined_row<L: Limb>(
    mats: &MimdMatrices<L>,
    eq_s1: &[Gr128<L>],
    eq_r: &[Gr128<L>],
    gamma: Gr128<L>,
    gamma2: Gr128<L>,
    m: usize,
) -> Vec<Gr128<L>> {
    let acc = |hat: &SparseMatrix<Gr128<L>>, eq: &[Gr128<L>]| -> Vec<Gr128<L>> {
        let mut v = vec![Gr128::<L>::ZERO; m];
        for &(y, col, val) in &hat.triples {
            v[col] = v[col] + eq[y] * val;
        }
        v
    };
    let (mut out, (bb, cc)) = rayon::join(
        || acc(&mats.a_hat, eq_s1),
        || rayon::join(|| acc(&mats.b_hat, eq_s1), || acc(&mats.c_hat, eq_r)),
    );
    out.par_iter_mut().enumerate().for_each(|(h, o)| {
        *o = *o + gamma * bb[h] + gamma2 * cc[h];
    });
    out
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

pub fn prove_mimd<L: HasPackingConstants>(
    witnesses: &[Vec<Gr1<L>>; 32],
    params: &MimdSnarkParams<L>,
) -> MimdSnarkProof<L> {
    prove_mimd_with_timing(witnesses, params).0
}

pub fn prove_mimd_with_timing<L: HasPackingConstants>(
    witnesses: &[Vec<Gr1<L>>; 32],
    params: &MimdSnarkParams<L>,
) -> (MimdSnarkProof<L>, MimdProveTiming) {
    let mut timing = MimdProveTiming::default();
    let total_t = Instant::now();
    let mats = &params.mats;
    let n = params.n();
    let m = params.m();
    let log_n = params.log_n;

    // ----- 1. Per-proof witness reduction (ŵ + A,B quotients). -----
    let t = Instant::now();
    let parts = mimd_reduce_witness(witnesses, mats);
    timing.reduce_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 2. Dense matvecs → â, b̂, ê_mult'. -----
    let t = Instant::now();
    let cst = constants::<L>();
    let v_alpha: Gr128<L> = Gr128::<L>::from_gr4(cst.v_alpha);
    let v_beta: Gr128<L> = cst.v_beta;
    // All of these are per-row independent → parallel maps (bit-identical).
    let err = |q1: &[Gr128<L>], q2: &[Gr128<L>]| -> Vec<Gr128<L>> {
        (0..n)
            .into_par_iter()
            .map(|i| q1[i] * v_alpha + q2[i] * v_beta)
            .collect()
    };
    let e_a = err(&parts.q1_a, &parts.q2_a);
    let e_b = err(&parts.q1_b, &parts.q2_b);
    // Three sparse matvecs over R_2, each parallel over rows (precomputed CSR).
    let a_sum = matvec_gr128_dense_csr(&mats.a_csr, &parts.w_phi);
    let b_sum = matvec_gr128_dense_csr(&mats.b_csr, &parts.w_phi);
    let c_sum = matvec_gr128_dense_csr(&mats.c_csr, &parts.w_phi);
    let a_hat: Vec<Gr128<L>> = (0..n).into_par_iter().map(|i| a_sum[i] - e_a[i]).collect();
    let b_hat: Vec<Gr128<L>> = (0..n).into_par_iter().map(|i| b_sum[i] - e_b[i]).collect();
    // C treated like A,B: ĉ = image(Ĉŵ), ê_C = Ĉŵ − ĉ ∈ ker ψ; ê_mult = â∘b̂ − ĉ.
    let t1 = compute_t1_generic::<L>();
    let c_hat: Vec<Gr128<L>> = c_sum
        .par_iter()
        .map(|&x| pack_r2(unpack_flat(x, &t1)))
        .collect();
    let e_c: Vec<Gr128<L>> = (0..n)
        .into_par_iter()
        .map(|i| c_sum[i] - c_hat[i])
        .collect();
    let e_mult: Vec<Gr128<L>> = (0..n)
        .into_par_iter()
        .map(|i| a_hat[i] * b_hat[i] - c_hat[i])
        .collect();
    let a_slots: Vec<[L; 32]> = a_hat.par_iter().map(|&x| slots_of(x, &t1)).collect();
    let b_slots: Vec<[L; 32]> = b_hat.par_iter().map(|&x| slots_of(x, &t1)).collect();
    let c_slots: Vec<[L; 32]> = c_hat.par_iter().map(|&x| slots_of(x, &t1)).collect();
    timing.matvec_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 3. Commit ŵ (slot form), ê_mult', and the stacked (ê_A, ê_B). -----
    // ê_A and ê_B are committed SEPARATELY and BEFORE γ — stacked under one tree
    // (both open at s₁) so each is bound pre-challenge and individually
    // ψ-checkable. A single γ-combined error committed *after* γ would be unsound
    // (the prover could pick a kernel lump hiding per-relation discrepancies).
    let w_slots: Vec<[L; 32]> = (0..m)
        .map(|j| std::array::from_fn(|k| witnesses[k][j].0))
        .collect();
    let (comm_w, we, wm) = params.pcs_w.commit_with_timing(&w_slots);
    let (comm_abc, ae, am) = params
        .pcs_abc
        .commit_stacked_with_timing(&[&a_slots, &b_slots, &c_slots]);
    let (comm_emult, ee, em) = params.pcs_e.commit_with_timing(&e_mult);
    let (comm_estack, se, sm) = params.pcs_e.commit_stacked_with_timing(&[&e_a, &e_b, &e_c]);
    timing.commit_w_ms = we + wm;
    timing.commit_emult_ms = ee + em;
    timing.commit_estack_ms = se + sm;
    timing.commit_encode_ms = we + ae + ee + se;
    timing.commit_merkle_ms = wm + am + em + sm;
    timing.commit_ms = timing.commit_encode_ms + timing.commit_merkle_ms;

    let mut tr = Transcript::new(b"mimd");
    tr.absorb_root(&comm_w.root);
    tr.absorb_root(&comm_abc.root);
    tr.absorb_root(&comm_emult.root);
    tr.absorb_root(&comm_estack.root);
    let r_target: Vec<Gr128<L>> = tr.challenge_target::<L>(b"mult_check_target", log_n);

    // ----- 4. Mult-check (sub form): Σ eq(r,h)·(â·b̂ − ĉ − ê_mult)(h) = 0. -----
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

    // ----- 5. Open (ê_A,ê_B,ê_C) and (â,b̂,ĉ) at s₁ with weights (1,γ,γ²);
    // ê_mult at s₁. The abc opening binds the mult finals (S-Z over γ). -----
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

    // ----- 6. Build M_γ_row + linear-check (C at s₁ now — eq_s1 for all). -----
    let t = Instant::now();
    let tr_rb = Instant::now();
    let eq_s1 = build_eq_table(&s_1);
    debug_assert_eq!(
        estack_val,
        inner_product(eq_s1.evals(), &e_a)
            + gamma * inner_product(eq_s1.evals(), &e_b)
            + gamma2 * inner_product(eq_s1.evals(), &e_c)
    );
    let m_gamma_row = build_combined_row(mats, eq_s1.evals(), eq_s1.evals(), gamma, gamma2, m);
    timing.linear_rowbuild_ms = tr_rb.elapsed().as_secs_f64() * 1e3;
    let claim_lin =
        mult_proof.final_a + gamma * mult_proof.final_b + gamma2 * mult_proof.final_c + estack_val;
    tr.absorb_label(b"linear_check_sumcheck");
    let g1 = Mle::new(m_gamma_row);
    let g2 = Mle::new(parts.w_phi);
    let (lin_proof, s_2) = prove_linear_check_mimd_sumcheck_only(g1, g2, claim_lin, &mut tr);
    timing.linear_ms = t.elapsed().as_secs_f64() * 1e3;

    // ----- 7. Open the witness at s₂. (abc/emult/estack opened in step 5.) -----
    let t = Instant::now();
    let (_yw, open_w) = params.pcs_w.open(&comm_w, &s_2);
    debug_assert_eq!(_yw, lin_proof.final_w);
    timing.opens_ms += t.elapsed().as_secs_f64() * 1e3;

    timing.total_ms = total_t.elapsed().as_secs_f64() * 1e3;

    (
        MimdSnarkProof {
            root_w: comm_w.root,
            root_abc: comm_abc.root,
            root_emult: comm_emult.root,
            root_estack: comm_estack.root,
            mult_proof,
            lin_proof,
            open_w,
            open_emult,
            open_abc,
            open_estack,
        },
        timing,
    )
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

pub fn verify_mimd<L: HasPackingConstants>(
    params: &MimdSnarkParams<L>,
    proof: &MimdSnarkProof<L>,
) -> Result<(), VerifyError> {
    verify_mimd_with_timing(params, proof).0
}

/// Per-stage verifier timing (ms): `piop` = transcript + sumcheck verifies +
/// row/claim recompute; `pcs` = PCS opening verifies + the ψ-kernel checks.
#[derive(Clone, Debug, Default)]
pub struct MimdVerifyTiming {
    pub piop_ms: f64,
    /// Of `piop_ms`: the `M_γ`-row recompute (`build_combined_row` + the
    /// `⟨eq_s2,M_γ⟩` check). The rest of `piop_ms` is sumcheck-verify.
    pub row_recompute_ms: f64,
    pub pcs_ms: f64,
    pub total_ms: f64,
}

pub fn verify_mimd_with_timing<L: HasPackingConstants>(
    params: &MimdSnarkParams<L>,
    proof: &MimdSnarkProof<L>,
) -> (Result<(), VerifyError>, MimdVerifyTiming) {
    let mats = &params.mats;
    let log_n = params.log_n;
    let mut tm = MimdVerifyTiming::default();
    let total_t = Instant::now();

    // ---- PIOP phase: transcript replay + sumcheck verifies + row recompute ----
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
        let mut tr = Transcript::new(b"mimd");
        tr.absorb_root(&proof.root_w);
        tr.absorb_root(&proof.root_abc);
        tr.absorb_root(&proof.root_emult);
        tr.absorb_root(&proof.root_estack);
        let r_target: Vec<Gr128<L>> = tr.challenge_target::<L>(b"mult_check_target", log_n);

        tr.absorb_label(b"mult_check_sumcheck");
        let s_1 = verify_mult_check_sub(&proof.mult_proof, &r_target, &mut tr)
            .map_err(|_| VerifyError::MultSumcheck)?;
        absorb_mult_finals(&mut tr, &proof.mult_proof);

        let gamma: Gr128<L> = tr.challenge_gr128::<L>(b"linear_check_gamma");
        let gamma2 = gamma * gamma;

        tr.absorb_label(b"linear_check_sumcheck");
        let s_2 = verify_linear_check_mimd(&proof.lin_proof, &mut tr)
            .map_err(|_| VerifyError::LinSumcheck)?;

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

        let abc_expected = proof.mult_proof.final_a
            + gamma * proof.mult_proof.final_b
            + gamma2 * proof.mult_proof.final_c;
        if abc_val != abc_expected || emult_s1 != proof.mult_proof.final_e {
            return Err(VerifyError::LinClaimMismatch);
        }

        let expected_claim_lin = abc_expected + estack_val;
        if proof.lin_proof.claimed_sum != expected_claim_lin {
            return Err(VerifyError::LinClaimMismatch);
        }

        let eq_s1 = build_eq_table(&s_1);
        let eq_s2 = build_eq_table(&s_2);
        let tr_rr = Instant::now();
        let m_row = build_combined_row(
            mats,
            eq_s1.evals(),
            eq_s1.evals(),
            gamma,
            gamma2,
            params.m(),
        );
        if inner_product(eq_s2.evals(), &m_row) != proof.lin_proof.final_m_row {
            return Err(VerifyError::LinARowMismatch);
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

    // ---- PCS phase: opening verifies + free-subspace (ψ-kernel) checks ----
    let t = Instant::now();
    let pcs = (|| -> Result<(), VerifyError> {
        let w = [Gr128::<L>::ONE, gamma, gamma2];
        if !params
            .pcs_w
            .verify(proof.root_w, &s_2, proof.lin_proof.final_w, &proof.open_w)
        {
            return Err(VerifyError::PcsW);
        }
        if !params
            .pcs_e
            .verify(proof.root_emult, &s_1, emult_s1, &proof.open_emult)
        {
            return Err(VerifyError::PcsE);
        }
        // (â,b̂,ĉ) slot form ⇒ structurally in Im φ; no image check needed.
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
        if !columns_in_kernel(&proof.open_emult) || !columns_in_kernel(&proof.open_estack) {
            return Err(VerifyError::KernelCheck);
        }
        Ok(())
    })();
    tm.pcs_ms = t.elapsed().as_secs_f64() * 1e3;
    tm.total_ms = total_t.elapsed().as_secs_f64() * 1e3;
    (pcs, tm)
}
