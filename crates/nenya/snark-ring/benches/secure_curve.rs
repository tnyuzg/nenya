//! Matched ring-vs-field proof-size + verify curve at the real 100-bit operating
//! point. PCS query budget and matrix shape come from
//! `*::setup(num_vars, q, δ=0.19, λ=100)` (`recommended_query_size` +
//! `optimal_shape`), compared against the old square-shape `ell=16` benchmarking
//! params. Both stacks use the SAME (δ, λ) — by the distance-preserving lift the
//! field code shares the ring's δ — so the comparison is apples-to-apples.
//!
//! For each `log_scalar`, the ring proves 32 SIMD instances of `2^(log_scalar−5)`
//! and the field one `2^log_scalar` R1CS (same scalar-constraint count).
//!
//! Usage: `secure_curve [log_scalar ...]`   (default: 17 20 23)

use field::{AbstractField, BB};
use galois_ring::{Gr1, Limb};
use pcs_field::FieldPcs;
use pcs_ring::Pcs;
use reduction::constants::HasPackingConstants;
use reduction::{matrix::SparseMatrix, SimdR1csInstance};
use snark_field::instance::{FieldR1csInstance, SparseMatrixBB};
use snark_field::{prove_field_with_timing, verify_field_with_timing, FieldSnarkParams};
use snark_ring::{prove_simd_with_timing, verify_simd_with_timing, R2Pcs, SimdSnarkParams};

const DELTA: f64 = 0.19;
const LAMBDA: usize = 100;
const Q: usize = 4;

fn square(log: usize) -> (usize, usize) {
    let h = log / 2;
    (1 << h, 1 << (log - h))
}

// ----- ring SIMD x·x=y builders (u8) -----
fn xy_witness<L: Limb>(m: usize, seed: usize) -> Vec<Gr1<L>> {
    let k = (m - 1) / 2;
    let mut w = vec![Gr1::<L>::ZERO; m];
    w[0] = Gr1::<L>::ONE;
    for i in 0..k {
        let x = Gr1(L::from_u64((((seed * 0x9E37) ^ (i * 0x85EB)) as u64).wrapping_add(1)));
        w[1 + 2 * i] = x;
        w[2 + 2 * i] = x * x;
    }
    w
}
fn xy_matrices<L: Limb>(n: usize, m: usize) -> [SparseMatrix<Gr1<L>>; 3] {
    let k = (m - 1) / 2;
    let (mut a, mut b, mut c) = (
        SparseMatrix::new(n, m),
        SparseMatrix::new(n, m),
        SparseMatrix::new(n, m),
    );
    for i in 0..k {
        a.push(i, 1 + 2 * i, Gr1::<L>::ONE);
        b.push(i, 1 + 2 * i, Gr1::<L>::ONE);
        c.push(i, 2 + 2 * i, Gr1::<L>::ONE);
    }
    [a, b, c]
}

/// Returns (proof_bytes, verify_ms, kernel_ms).
fn ring_simd<L: HasPackingConstants>(log_size: usize, secure: bool) -> (usize, f64, f64) {
    let n = 1usize << log_size;
    let [a, b, c] = xy_matrices::<L>(n, n);
    let witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|k| xy_witness::<L>(n, k));
    let simd = SimdR1csInstance { a, b, c, witnesses };
    let (pcs_w, pcs_q) = if secure {
        (
            Pcs::<L>::setup(log_size, Q, DELTA, LAMBDA, 0xDEAD),
            R2Pcs::<L>::setup(log_size, Q, DELTA, LAMBDA, 0xCAFE),
        )
    } else {
        let (k1, k2) = square(log_size);
        (
            Pcs::<L>::new(k1, k2, Q, 16, 0xDEAD),
            R2Pcs::<L>::new(k1, k2, Q, 16, 0xCAFE),
        )
    };
    let params = SimdSnarkParams {
        log_n: log_size,
        log_m: log_size,
        pcs_w,
        pcs_q,
    };
    let (proof, _) = prove_simd_with_timing(&simd, &params);
    let vt = verify_simd_with_timing(&simd.a, &simd.b, &simd.c, &params, &proof).expect("verify");
    (proof.size_bytes(), vt.total_ms, vt.kernel_ms)
}

/// Field baseline. Returns (proof_bytes, verify_ms).
fn field_snark(log: usize, secure: bool) -> (usize, f64) {
    let n = 1usize << log;
    let k = (n - 1) / 2;
    let (mut a, mut b, mut c) = (
        SparseMatrixBB::new(n, n),
        SparseMatrixBB::new(n, n),
        SparseMatrixBB::new(n, n),
    );
    for i in 0..k {
        a.push(i, 1 + 2 * i, BB::one());
        b.push(i, 1 + 2 * i, BB::one());
        c.push(i, 2 + 2 * i, BB::one());
    }
    let mut w = vec![BB::zero(); n];
    w[0] = BB::one();
    for i in 0..k {
        let x = BB::from_canonical_u32(((i as u32) ^ 0x85EB).wrapping_add(1) % 0x7800_0001);
        w[1 + 2 * i] = x;
        w[2 + 2 * i] = x * x;
    }
    let inst = FieldR1csInstance { a, b, c, w };
    let pcs_w = if secure {
        FieldPcs::setup(log, Q, DELTA, LAMBDA, 0xDEAD)
    } else {
        let (k1, k2) = square(log);
        FieldPcs::new(k1, k2, Q, 16, 0xDEAD)
    };
    let params = FieldSnarkParams {
        log_n: log,
        log_m: log,
        pcs_w,
    };
    let (proof, _) = prove_field_with_timing(&inst, &params);
    let vt = verify_field_with_timing(&inst.a, &inst.b, &inst.c, &params, &proof).expect("verify");
    (proof.size_bytes(), vt.total_ms)
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1e6
}

fn main() {
    let scalars: Vec<usize> = {
        let v: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
        if v.is_empty() {
            vec![17, 20, 23]
        } else {
            v
        }
    };
    let ell = pcs_ring::recommended_query_size(DELTA, LAMBDA);
    println!(
        "Matched 100-bit operating point: δ={DELTA}, λ={LAMBDA} ⇒ ell={ell} (vs insecure ell=16).\n\
         ring = SIMD u8 (32×2^(L−5));  field = BabyBear (2^L). proof in MB, verify in ms.\n"
    );
    println!(
        "{:>4} | {:^28} | {:^28} | {:^20}",
        "", "ring SIMD-u8 proof (MB)", "field proof (MB)", "verify (ms)"
    );
    println!(
        "{:>4} | {:>12} {:>12} | {:>12} {:>12} | {:>9} {:>9}",
        "2^L", "sq/ell16", "secure", "sq/ell16", "secure", "ring sec", "field sec"
    );
    for &ls in &scalars {
        let rs = ls - 5; // ring per-instance size
        let (rp_sq, _rv_sq, _) = ring_simd::<u8>(rs, false);
        let (rp_se, rv_se, rk) = ring_simd::<u8>(rs, true);
        let (fp_sq, _fv_sq) = field_snark(ls, false);
        let (fp_se, fv_se) = field_snark(ls, true);
        println!(
            "{:>4} | {:>12.3} {:>12.3} | {:>12.3} {:>12.3} | {:>9.1} {:>9.1}   (ring kernel {:.1}ms)",
            ls,
            mb(rp_sq),
            mb(rp_se),
            mb(fp_sq),
            mb(fp_se),
            rv_se,
            fv_se,
            rk,
        );
    }
    println!(
        "\nsecure proof = optimal_shape @ ell={ell}; sq/ell16 = old square shape, ell=16 (~4-bit, insecure)."
    );
}
