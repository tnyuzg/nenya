//! One-shot timing for the field SNARK with per-stage breakdown.
//!
//! Usage: `snark_field_oneshot [log_size] [ell]`  (defaults 20, 487).
//!
//! Size note: the ring SIMD SNARK at `log_size = L` proves 32 instances ×
//! 2^L constraints = 2^(L+5) base constraints. The matched field run is
//! therefore `log_size = L + 5` (e.g. field 2^21 ↔ ring 2^16).

use field::{AbstractField, BB};
use pcs_field::FieldPcs;
use snark_field::instance::{FieldR1csInstance, SparseMatrixBB};
use snark_field::{prove_field_with_timing, verify_field_with_timing, FieldSnarkParams};
use std::time::Instant;

fn build_field(log_size: usize) -> FieldR1csInstance {
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
        let x = BB::from_canonical_u32(((i as u32) ^ 0x85EB).wrapping_add(1) % 0x7800_0001);
        w[1 + 2 * i] = x;
        w[1 + 2 * i + 1] = x * x;
    }
    FieldR1csInstance { a, b, c, w }
}

/// Print one stage row: `label`, serial ms, parallel ms, and the speedup.
fn row(label: &str, s: f64, p: f64) {
    let sp = if p > 0.0 { s / p } else { 0.0 };
    println!("  {label:<20} {s:>11.2} {p:>11.2}   {sp:>6.1}x");
}

fn main() {
    let log_size: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let ell: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(487);

    let half = log_size / 2;
    let k1 = 1 << half;
    let k2 = 1 << (log_size - half);
    let pcs_w = FieldPcs::new(k1, k2, 4, ell, 0xDEAD_BEEF);
    let params = FieldSnarkParams {
        log_n: log_size,
        log_m: log_size,
        pcs_w,
    };

    let t = Instant::now();
    let inst = build_field(log_size);
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("\n##### FIELD (BabyBear) SNARK  2^{log_size}  (ell={ell}) #####");
    println!("build:               {:>10.2?}", t.elapsed());
    debug_assert!(inst.is_satisfied());

    // Same code under a 1-thread pool (serial column) vs an all-cores pool
    // (parallel column); `pool.install` confines all nested `par_iter`.
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pooln = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .unwrap();

    let (proof_s, ts) = pool1.install(|| prove_field_with_timing(&inst, &params));
    let (proof_p, tp) = pooln.install(|| prove_field_with_timing(&inst, &params));
    assert!(
        proof_s == proof_p,
        "serial and parallel proofs differ — nondeterministic reduction!"
    );

    println!("--- prove: serial(1 thread) vs parallel({nthreads} threads) ---");
    println!(
        "  {:<20} {:>11} {:>11}   {:>7}",
        "stage", "serial ms", "par ms", "speedup"
    );
    row("commit w", ts.commit_w_ms, tp.commit_w_ms);
    row("matvec A,B,C·w", ts.matvec_abc_ms, tp.matvec_abc_ms);
    row("eq_r + claim ⟨·⟩", ts.eq_r_ms, tp.eq_r_ms);
    row("mult sumcheck", ts.mult_sumcheck_ms, tp.mult_sumcheck_ms);
    row("build M_γ_row", ts.linear_arow_ms, tp.linear_arow_ms);
    row(
        "linear sumcheck",
        ts.linear_sumcheck_ms,
        tp.linear_sumcheck_ms,
    );
    row("open w", ts.open_w_ms, tp.open_w_ms);
    row("TOTAL prove", ts.total_ms, tp.total_ms);

    let vt = pooln
        .install(|| verify_field_with_timing(&inst.a, &inst.b, &inst.c, &params, &proof_p))
        .expect("verify failed");
    println!("  verify:             {:>9.2} ms", vt.total_ms);
    println!("  proof size:         {} bytes", proof_p.size_bytes());
}
