//! End-to-end field SNARK test: a satisfied `x·x = y` R1CS over BabyBear.

use field::{AbstractField, BB, BB4};
use pcs_field::FieldPcs;
use snark_field::instance::{FieldR1csInstance, SparseMatrixBB};
use snark_field::{prove_field, verify_field, FieldSnarkParams};

fn build(log_size: usize) -> FieldR1csInstance {
    let n = 1usize << log_size;
    let m = n;
    let k = (m - 1) / 2;
    let mut a = SparseMatrixBB::new(n, m);
    let mut b = SparseMatrixBB::new(n, m);
    let mut c = SparseMatrixBB::new(n, m);
    for i in 0..k {
        a.push(i, 1 + 2 * i, BB::one());
        b.push(i, 1 + 2 * i, BB::one());
        c.push(i, 1 + 2 * i + 1, BB::one());
    }
    let mut w = vec![BB::zero(); m];
    w[0] = BB::one();
    for i in 0..k {
        let x = BB::from_canonical_u32(
            (i as u32).wrapping_mul(2654435761).wrapping_add(7) % 0x7800_0001,
        );
        w[1 + 2 * i] = x;
        w[1 + 2 * i + 1] = x * x;
    }
    FieldR1csInstance { a, b, c, w }
}

/// **INSECURE — unit-test params only.** Square shape, `ell = 16` (~4-bit): tiny
/// so the tests run fast. Real proofs use [`FieldSnarkParams::secure`] (the same
/// δ=0.19, λ=100 ⇒ `ell = 1060` as the ring) — exercised by
/// [`secure_default_end_to_end`].
fn params(log_size: usize) -> FieldSnarkParams {
    let half = log_size / 2;
    let k1 = 1 << half;
    let k2 = 1 << (log_size - half);
    FieldSnarkParams {
        log_n: log_size,
        log_m: log_size,
        pcs_w: FieldPcs::new(k1, k2, 4, 16, 0xDEAD_BEEF),
    }
}

/// The wired **secure default** (`FieldSnarkParams::secure`) must produce a
/// verifying proof end-to-end, at a moderate size where `ell = 1060` fits.
#[test]
fn secure_default_end_to_end() {
    let log_size = 12;
    let inst = build(log_size);
    assert!(inst.is_satisfied());
    let p = FieldSnarkParams::secure(log_size, log_size, 0xDEAD_BEEF);
    assert!(
        p.pcs_w.ell >= 1000,
        "secure params must use the real (≥1000) query budget, got {}",
        p.pcs_w.ell
    );
    let proof = prove_field(&inst, &p);
    assert!(
        verify_field(&inst.a, &inst.b, &inst.c, &p, &proof).is_ok(),
        "secure-default proof must verify"
    );
}

#[test]
fn prove_verify_roundtrip() {
    for log_size in [4usize, 6, 8] {
        let inst = build(log_size);
        assert!(
            inst.is_satisfied(),
            "instance not satisfied (log_size={log_size})"
        );
        let p = params(log_size);
        let proof = prove_field(&inst, &p);
        assert!(
            verify_field(&inst.a, &inst.b, &inst.c, &p, &proof).is_ok(),
            "verify rejected a valid proof (log_size={log_size})"
        );
    }
}

/// Parallelism gate (field side): the proof must be byte-for-byte identical on
/// 1 thread vs many. BB4 addition is associative + commutative, so the parallel
/// reductions are exact and the proof cannot depend on thread count.
#[test]
fn parallel_proof_is_bit_identical_to_serial() {
    let log_size = 10;
    let inst = build(log_size);
    assert!(inst.is_satisfied());
    let p = params(log_size);

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pooln = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();

    let proof_serial = pool1.install(|| prove_field(&inst, &p));
    let proof_parallel = pooln.install(|| prove_field(&inst, &p));

    assert!(
        proof_serial == proof_parallel,
        "serial (1-thread) and parallel (8-thread) field proofs differ — a parallel reduction is nondeterministic"
    );
    assert!(verify_field(&inst.a, &inst.b, &inst.c, &p, &proof_serial).is_ok());
    assert!(verify_field(&inst.a, &inst.b, &inst.c, &p, &proof_parallel).is_ok());
}

#[test]
fn tampered_final_a_rejects() {
    let log_size = 6;
    let inst = build(log_size);
    let p = params(log_size);
    let mut proof = prove_field(&inst, &p);
    proof.mult_proof.final_a += BB4::one();
    assert!(verify_field(&inst.a, &inst.b, &inst.c, &p, &proof).is_err());
}

#[test]
fn tampered_witness_opening_rejects() {
    let log_size = 6;
    let inst = build(log_size);
    let p = params(log_size);
    let mut proof = prove_field(&inst, &p);
    // Corrupt one opened column entry.
    proof.open_w.query_columns[0][0] += BB::one();
    assert!(verify_field(&inst.a, &inst.b, &inst.c, &p, &proof).is_err());
}
