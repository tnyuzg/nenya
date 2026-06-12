//! End-to-end SIMD SNARK test: 32 batched `x·x = y` instances.
//!
//! Builds a satisfied `SimdR1csInstance` over R_0 = Z/2^L, runs
//! `prove_simd`, runs `verify_simd`, and asserts acceptance. Negative tests
//! corrupt one witness entry and assert rejection. Instantiated per base-limb
//! width below (u8/u16/u32/u64).

use galois_ring::{Gr1, Gr128, Limb};
use pcs_ring::Pcs;
use reduction::constants::HasPackingConstants;
use reduction::{matrix::SparseMatrix, SimdR1csInstance};
use snark_ring::{prove_simd, verify_simd, R2Pcs, SimdSnarkParams, SimdSnarkProof};

/// Build a satisfied SIMD instance with `n × m = 2^log_size × 2^log_size`
/// matrices: `n` rows, `m = n` cols. We pack as many `x*x = y` constraints
/// as fit in the witness layout `[1, x_0, y_0, x_1, y_1, …]`.
fn build_simd<L: Limb>(log_size: usize) -> SimdR1csInstance<L> {
    let n = 1usize << log_size;
    let m = n;
    let k = (m - 1) / 2;
    assert!(k <= n);

    let mut a = SparseMatrix::<Gr1<L>>::new(n, m);
    let mut b = SparseMatrix::<Gr1<L>>::new(n, m);
    let mut c = SparseMatrix::<Gr1<L>>::new(n, m);
    for i in 0..k {
        let x_col = 1 + 2 * i;
        let y_col = 1 + 2 * i + 1;
        a.push(i, x_col, Gr1::<L>::ONE);
        b.push(i, x_col, Gr1::<L>::ONE);
        c.push(i, y_col, Gr1::<L>::ONE);
    }

    let mut witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|_| vec![]);
    for kk in 0..32 {
        let mut w = vec![Gr1::<L>::ZERO; m];
        w[0] = Gr1::<L>::ONE;
        for i in 0..k {
            let x = Gr1(L::from_u64((((kk * 0x9E37) ^ (i * 0x85EB)) as u64).wrapping_add(1)));
            w[1 + 2 * i] = x;
            w[1 + 2 * i + 1] = x * x;
        }
        witnesses[kk] = w;
    }
    SimdR1csInstance { a, b, c, witnesses }
}

/// **INSECURE — unit-test params only.** Square shape, `ell = 16` (~4-bit): tiny
/// so the tests run fast. Real proofs use [`SimdSnarkParams::secure`] (δ=0.19,
/// λ=100 ⇒ `ell = 1060`, optimal shape) — exercised by
/// [`secure_default_end_to_end_u8`].
fn make_params<L: HasPackingConstants>(log_size: usize) -> SimdSnarkParams<L> {
    let half = log_size / 2;
    let k1 = 1 << half;
    let k2 = 1 << (log_size - half);
    let pcs_w = Pcs::new(k1, k2, 4, 16, 0xDEAD_BEEF);
    let pcs_q = R2Pcs::new(k1, k2, 4, 16, 0xCAFE_BABE);
    SimdSnarkParams {
        log_n: log_size,
        log_m: log_size,
        pcs_w,
        pcs_q,
    }
}

/// The wired **secure default** must produce a verifying proof end-to-end. Run at
/// a moderate size so `ell = 1060` fits the codeword and the test stays fast;
/// u8 only.
#[test]
fn secure_default_end_to_end_u8() {
    let log_size = 12; // 32 × 2^12 = 2^17 scalar constraints
    let simd = build_simd::<u8>(log_size);
    assert!(simd.is_satisfied());

    let params = SimdSnarkParams::<u8>::secure(log_size, log_size, 0xDEAD_BEEF, 0xCAFE_BABE);
    assert!(
        params.pcs_q.ell >= 1000 && params.pcs_w.ell >= 1000,
        "secure params must use the real (≥1000) query budget, got w={} q={}",
        params.pcs_w.ell,
        params.pcs_q.ell
    );

    let proof = prove_simd(&simd, &params);
    assert!(
        verify_simd(&simd.a, &simd.b, &simd.c, &params, &proof).is_ok(),
        "secure-default proof must verify"
    );
}

fn check_satisfied_small<L: HasPackingConstants>() {
    let log_size = 4; // 16 × 16
    let simd = build_simd::<L>(log_size);
    assert!(simd.is_satisfied());

    let params = make_params::<L>(log_size);
    let proof = prove_simd(&simd, &params);
    let r = verify_simd(&simd.a, &simd.b, &simd.c, &params, &proof);
    assert!(r.is_ok(), "verify failed: {r:?}");
}

fn check_satisfied_medium<L: HasPackingConstants>() {
    let log_size = 6; // 64 × 64
    let simd = build_simd::<L>(log_size);
    assert!(simd.is_satisfied());

    let params = make_params::<L>(log_size);
    let proof = prove_simd(&simd, &params);
    let r = verify_simd(&simd.a, &simd.b, &simd.c, &params, &proof);
    assert!(r.is_ok(), "verify failed: {r:?}");

    println!("proof size: {} bytes", proof.size_bytes());
}

fn check_corrupted_witness<L: HasPackingConstants>() {
    let log_size = 4;
    let mut simd = build_simd::<L>(log_size);
    // Flip one entry of one SIMD witness.
    simd.witnesses[5][2] = Gr1(L::from_u64(0xBAD_C0DE));
    assert!(!simd.is_satisfied());

    let params = make_params::<L>(log_size);
    let proof = prove_simd(&simd, &params);
    let r = verify_simd(&simd.a, &simd.b, &simd.c, &params, &proof);
    assert!(r.is_err(), "verify should reject corrupted witness");
}

fn check_tampered_proof<L: HasPackingConstants>() {
    let log_size = 4;
    let simd = build_simd::<L>(log_size);
    let params = make_params::<L>(log_size);
    let mut proof = prove_simd(&simd, &params);

    // Flip mult_proof.final_a (a_phi(s_1)).
    proof.mult_proof.final_a = proof.mult_proof.final_a + Gr128::<L>::ONE;
    let r = verify_simd(&simd.a, &simd.b, &simd.c, &params, &proof);
    assert!(r.is_err(), "verify should reject tampered final_a");
}

/// Headline parallelism gate: the proof must be **byte-for-byte identical**
/// whether the prover runs on 1 thread or many. All parallel reductions fold
/// `Z/2^n` ring elements (associative + commutative), so reordering by the
/// rayon work-splitter is exact — any difference would mean a nondeterministic
/// reduction slipped in. `prove_simd` is deterministic (Fiat–Shamir only), so
/// the sole variable here is the thread count.
fn check_parallel_bit_identical<L: HasPackingConstants>() {
    let log_size = 10; // 1024 × 1024 — enough to exercise real parallel splitting
    let simd = build_simd::<L>(log_size);
    assert!(simd.is_satisfied());
    let params = make_params::<L>(log_size);

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pooln = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();

    let proof_serial = pool1.install(|| prove_simd(&simd, &params));
    let proof_parallel = pooln.install(|| prove_simd(&simd, &params));

    assert!(
        proof_serial == proof_parallel,
        "serial (1-thread) and parallel (8-thread) proofs differ — a parallel reduction is nondeterministic"
    );
    assert!(verify_simd(&simd.a, &simd.b, &simd.c, &params, &proof_serial).is_ok());
    assert!(verify_simd(&simd.a, &simd.b, &simd.c, &params, &proof_parallel).is_ok());
}

fn check_proof_size_reported<L: HasPackingConstants>() {
    let log_size = 4;
    let simd = build_simd::<L>(log_size);
    let params = make_params::<L>(log_size);
    let proof = prove_simd(&simd, &params);
    let _: usize = proof.size_bytes();
    let _: SimdSnarkProof<L> = proof; // keeps the value alive
}

// ---------------------------------------------------------------------------
// Per-width instantiations.
// ---------------------------------------------------------------------------

macro_rules! width_tests {
    ($modname:ident, $t:ty) => {
        mod $modname {
            use super::*;

            #[test]
            fn end_to_end_satisfied_small() {
                check_satisfied_small::<$t>();
            }
            #[test]
            fn end_to_end_satisfied_medium() {
                check_satisfied_medium::<$t>();
            }
            #[test]
            fn negative_corrupted_witness() {
                check_corrupted_witness::<$t>();
            }
            #[test]
            fn negative_tampered_proof() {
                check_tampered_proof::<$t>();
            }
            #[test]
            fn parallel_proof_is_bit_identical_to_serial() {
                check_parallel_bit_identical::<$t>();
            }
            #[test]
            fn proof_size_reported() {
                check_proof_size_reported::<$t>();
            }
        }
    };
}

width_tests!(w8, u8);
width_tests!(w16, u16);
width_tests!(w32, u32);
