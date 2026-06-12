//! One-shot timing for the SIMD SNARK with per-stage prover/verifier
//! breakdowns, generic over the base-limb width.
//!
//! Usage: `snark_simd_oneshot [log_size] [ell] [width]`
//!   log_size default 15, ell default 16, width ∈ {8,16,32,64} default 32.

use galois_ring::{Gr1, Limb};
use pcs_ring::Pcs;
use reduction::constants::HasPackingConstants;
use reduction::{matrix::SparseMatrix, SimdR1csInstance};
use snark_ring::{prove_simd_with_timing, verify_simd_with_timing, R2Pcs, SimdSnarkParams};
use std::time::Instant;

fn build_simd<L: Limb>(log_size: usize) -> SimdR1csInstance<L> {
    let n = 1usize << log_size;
    let m = n;
    let k = (m - 1) / 2;
    let mut a = SparseMatrix::<Gr1<L>>::new(n, m);
    let mut b = SparseMatrix::<Gr1<L>>::new(n, m);
    let mut c = SparseMatrix::<Gr1<L>>::new(n, m);
    for i in 0..k {
        a.push(i, 1 + 2 * i, Gr1::<L>::ONE);
        b.push(i, 1 + 2 * i, Gr1::<L>::ONE);
        c.push(i, 1 + 2 * i + 1, Gr1::<L>::ONE);
    }
    let mut witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|_| vec![]);
    for kk in 0..32 {
        let mut w = vec![Gr1::<L>::ZERO; m];
        w[0] = Gr1::<L>::ONE;
        for i in 0..k {
            let x = Gr1(L::from_u64(
                (((kk * 0x9E37) ^ (i * 0x85EB)) as u64).wrapping_add(1),
            ));
            w[1 + 2 * i] = x;
            w[1 + 2 * i + 1] = x * x;
        }
        witnesses[kk] = w;
    }
    SimdR1csInstance { a, b, c, witnesses }
}

/// Print one stage row: `label`, serial ms, parallel ms, and the speedup.
fn row(label: &str, s: f64, p: f64) {
    let sp = if p > 0.0 { s / p } else { 0.0 };
    println!("  {label:<20} {s:>11.2} {p:>11.2}   {sp:>6.1}x");
}

fn run<L: HasPackingConstants>(width: u32, log_size: usize, ell: usize) {
    let half = log_size / 2;
    let k1 = 1 << half;
    let k2 = 1 << (log_size - half);
    let pcs_w = Pcs::<L>::new(k1, k2, 4, ell, 0xDEAD_BEEF);
    let pcs_q = R2Pcs::<L>::new(k1, k2, 4, ell, 0xCAFE_BABE);
    let params = SimdSnarkParams {
        log_n: log_size,
        log_m: log_size,
        pcs_w,
        pcs_q,
    };

    let t = Instant::now();
    let simd = build_simd::<L>(log_size);
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!("\n##### u{width}  SIMD SNARK  2^{log_size}  (ell={ell}) #####");
    println!("build_simd:          {:>10.2?}", t.elapsed());

    // Run the *same* parallelized prover under a 1-thread pool (serial column)
    // and an all-cores pool (parallel column). `pool.install` confines every
    // nested `par_iter` to that pool, so the 1-thread number is the honest
    // serial baseline for the identical code. (rayon@1 carries a small
    // work-splitter overhead vs a raw `for`, so the reported speedup slightly
    // under-states the true parallel gain.)
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pooln = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .unwrap();

    let (proof_s, ts) = pool1.install(|| prove_simd_with_timing(&simd, &params));
    let (proof_p, tp) = pooln.install(|| prove_simd_with_timing(&simd, &params));
    assert!(
        proof_s == proof_p,
        "serial and parallel proofs differ — nondeterministic reduction!"
    );

    println!("--- prove: serial(1 thread) vs parallel({nthreads} threads) ---");
    println!(
        "  {:<20} {:>11} {:>11}   {:>7}",
        "stage", "serial ms", "par ms", "speedup"
    );
    row("reduce (φ_2)", ts.reduce_ms, tp.reduce_ms);
    row(
        "commit w encode",
        ts.commit_w_encode_ms,
        tp.commit_w_encode_ms,
    );
    row(
        "commit w merkle",
        ts.commit_w_merkle_ms,
        tp.commit_w_merkle_ms,
    );
    row(
        "commit e encode",
        ts.commit_e_encode_ms,
        tp.commit_e_encode_ms,
    );
    row(
        "commit e merkle",
        ts.commit_e_merkle_ms,
        tp.commit_e_merkle_ms,
    );
    row("matvec A,B,C·w", ts.matvec_abc_ms, tp.matvec_abc_ms);
    row("eq_r + claim ⟨·⟩", ts.eq_r_ms, tp.eq_r_ms);
    row("mult sumcheck", ts.mult_sumcheck_ms, tp.mult_sumcheck_ms);
    row("build M_γ_row", ts.linear_arow_ms, tp.linear_arow_ms);
    row(
        "linear sumcheck",
        ts.linear_sumcheck_ms,
        tp.linear_sumcheck_ms,
    );
    row(
        "opens w+e",
        ts.open_w_ms + ts.open_e_ms,
        tp.open_w_ms + tp.open_e_ms,
    );
    row("TOTAL prove", ts.total_ms, tp.total_ms);

    let vt = pooln
        .install(|| verify_simd_with_timing(&simd.a, &simd.b, &simd.c, &params, &proof_p))
        .expect("verify failed");
    println!("  verify:             {:>9.2} ms", vt.total_ms);
    println!(
        "    kernel check:     {:>9.2} ms  ({:.1}% of verify)",
        vt.kernel_ms,
        if vt.total_ms > 0.0 {
            vt.kernel_ms / vt.total_ms * 100.0
        } else {
            0.0
        }
    );
    println!("  proof size:         {} bytes", proof_p.size_bytes());
}

fn main() {
    let log_size: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let ell: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let width: u32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    match width {
        8 => run::<u8>(8, log_size, ell),
        16 => run::<u16>(16, log_size, ell),
        32 => run::<u32>(32, log_size, ell),
        64 => run::<u64>(64, log_size, ell),
        _ => panic!("width must be 8, 16, 32, or 64"),
    }
}
