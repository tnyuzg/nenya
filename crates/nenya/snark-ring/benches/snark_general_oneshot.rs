//! One-shot timing for the general-circuit SNARK (paper §3.1.3, Halevi–Shoup),
//! serial vs parallel, generic over the base-limb width.
//!
//! Usage: `snark_general_oneshot [log_blocks] [width]`
//!   log_blocks default 15 (⇒ circuit size n = 32·2^log_blocks), width ∈ {8,16,32,64} default 32.

use galois_ring::{Gr1, Limb};
use pcs_ring::Pcs;
use reduction::constants::HasPackingConstants;
use reduction::{matrix::SparseMatrix, pack_general_matrices, GeneralR1csInstance};
use snark_ring::{prove_general_with_timing, verify_general, GeneralSnarkParams, R2Pcs};
use std::time::Instant;

/// Satisfied general instance of size `n×n` with `x·x=y` constraints.
fn build_general<L: Limb>(n: usize) -> GeneralR1csInstance<L> {
    let k = (n - 1) / 2;
    let mut a = SparseMatrix::<Gr1<L>>::new(n, n);
    let mut b = SparseMatrix::<Gr1<L>>::new(n, n);
    let mut c = SparseMatrix::<Gr1<L>>::new(n, n);
    for i in 0..k {
        a.push(i, 1 + 2 * i, Gr1::<L>::ONE);
        b.push(i, 1 + 2 * i, Gr1::<L>::ONE);
        c.push(i, 2 + 2 * i, Gr1::<L>::ONE);
    }
    let mut w = vec![Gr1::<L>::ZERO; n];
    w[0] = Gr1::<L>::ONE;
    for i in 0..k {
        let x = Gr1(L::from_u64(
            (((i * 0x9E37) ^ 0x85EB) as u64).wrapping_add(1),
        ));
        w[1 + 2 * i] = x;
        w[2 + 2 * i] = x * x;
    }
    GeneralR1csInstance { a, b, c, w }
}

fn row(label: &str, s: f64, p: f64) {
    let sp = if p > 0.0 { s / p } else { 0.0 };
    println!("  {label:<18} {s:>11.2} {p:>11.2}   {sp:>6.1}x");
}

fn run<L: HasPackingConstants>(width: u32, log_blocks: usize) {
    let nb = 1usize << log_blocks;
    let n = 32 * nb; // circuit size (witness length)
    let half = log_blocks / 2;
    let k1 = 1 << half;
    let k2 = 1 << (log_blocks - half);

    let g = build_general::<L>(n);
    let nthreads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    println!(
        "\n##### u{width}  GENERAL SNARK  n=2^{} (nb=2^{log_blocks})  #####",
        n.trailing_zeros()
    );

    // Setup: the packed diagonals are precomputed (public, deterministic) — NOT
    // part of the prover cost (paper §3.1.3). Timed separately.
    let t = Instant::now();
    let params = GeneralSnarkParams::<L> {
        log_blocks,
        pcs_w: Pcs::new(k1, k2, 4, 16, 0xDEAD_BEEF),
        pcs_abc: Pcs::new(k1, k2, 4, 16, 0xABC_BEEF),
        pcs_e: R2Pcs::new(k1, k2, 4, 16, 0xCAFE_BABE),
        diag: pack_general_matrices(&g.a, &g.b, &g.c),
    };
    println!(
        "setup (diagonal packing, excluded from prove): {:.2} ms",
        t.elapsed().as_secs_f64() * 1e3
    );

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let pooln = rayon::ThreadPoolBuilder::new()
        .num_threads(nthreads)
        .build()
        .unwrap();

    let (proof_s, ts) = pool1.install(|| prove_general_with_timing(&g.w, &params));
    let (proof_p, tp) = pooln.install(|| prove_general_with_timing(&g.w, &params));

    println!("--- prove (excl. setup): serial(1) vs parallel({nthreads}) ---");
    println!(
        "  {:<18} {:>11} {:>11}   {:>7}",
        "stage", "serial ms", "par ms", "speedup"
    );
    row("reduce (ρ̂ + q_AB)", ts.reduce_ms, tp.reduce_ms);
    row("matvec Σ_ℓ D̂·ρ̂", ts.matvec_ms, tp.matvec_ms);
    row("commit w+e", ts.commit_ms, tp.commit_ms);
    row("mult sumcheck", ts.mult_sumcheck_ms, tp.mult_sumcheck_ms);
    row("linear (rows+sc)", ts.linear_ms, tp.linear_ms);
    row("opens (32 ρ̂ + e)", ts.opens_ms, tp.opens_ms);
    row("TOTAL prove", ts.total_ms, tp.total_ms);

    let t = Instant::now();
    let ok = pooln.install(|| verify_general(&params, &proof_p)).is_ok();
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;
    assert!(ok, "verify failed");
    assert!(verify_general(&params, &proof_s).is_ok());
    println!("  verify:            {:>11.2} ms", verify_ms);
    println!("  proof size:        {} bytes", proof_p.size_bytes());
}

fn main() {
    let log_blocks: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let width: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    match width {
        8 => run::<u8>(8, log_blocks),
        16 => run::<u16>(16, log_blocks),
        32 => run::<u32>(32, log_blocks),
        64 => run::<u64>(64, log_blocks),
        _ => panic!("width must be 8, 16, 32, or 64"),
    }
}
