//! One-shot timing for the MIMD SNARK (paper §3.2.2), serial vs parallel,
//! generic over the base-limb width.
//!
//! Usage: `snark_mimd_oneshot [log_size] [width]`
//!   log_size default 15 (⇒ 32·2^log_size scalar constraints), width ∈ {8,16,32,64} default 32.
//!
//! Note: this benchmark uses 32 instances with *identical* matrix support
//! (distinct witnesses), so `Â,B̂,Ĉ` are as sparse as a single matrix — a
//! best case for the matvec. A real MIMD with distinct supports would have a
//! denser packed matrix and heavier matvec; the reduce/sumcheck stages dominate
//! either way.

use galois_ring::{Gr1, Limb};
use pcs_ring::Pcs;
use reduction::constants::HasPackingConstants;
use reduction::{matrix::SparseMatrix, pack_mimd_matrices, MimdR1csInstance};
use snark_ring::{prove_mimd_with_timing, verify_mimd, MimdSnarkParams, R2Pcs};
use std::time::Instant;

fn build_mimd<L: Limb>(log_size: usize) -> MimdR1csInstance<L> {
    let n = 1usize << log_size;
    let m = n;
    let k = (m - 1) / 2;
    let mut a: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| SparseMatrix::new(n, m));
    let mut b: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| SparseMatrix::new(n, m));
    let mut c: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| SparseMatrix::new(n, m));
    for inst in 0..32 {
        for i in 0..k {
            a[inst].push(i, 1 + 2 * i, Gr1::<L>::ONE);
            b[inst].push(i, 1 + 2 * i, Gr1::<L>::ONE);
            c[inst].push(i, 2 + 2 * i, Gr1::<L>::ONE);
        }
    }
    let mut witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|_| vec![Gr1::<L>::ZERO; m]);
    for inst in 0..32 {
        witnesses[inst][0] = Gr1::<L>::ONE;
        for i in 0..k {
            let x = Gr1(L::from_u64(
                (((inst * 0x9E37) ^ (i * 0x85EB)) as u64).wrapping_add(1),
            ));
            witnesses[inst][1 + 2 * i] = x;
            witnesses[inst][2 + 2 * i] = x * x;
        }
    }
    MimdR1csInstance {
        a_per_instance: a,
        b_per_instance: b,
        c_per_instance: c,
        witnesses,
    }
}

fn row(label: &str, s: f64, p: f64) {
    let sp = if p > 0.0 { s / p } else { 0.0 };
    println!("  {label:<18} {s:>11.2} {p:>11.2}   {sp:>6.1}x");
}

fn run<L: HasPackingConstants>(width: u32, log_size: usize) {
    let half = log_size / 2;
    let k1 = 1 << half;
    let k2 = 1 << (log_size - half);
    let nthreads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);

    let mimd = build_mimd::<L>(log_size);
    println!(
        "\n##### u{width}  MIMD SNARK  32·2^{log_size} = 2^{} scalar constraints  #####",
        log_size + 5
    );

    let t = Instant::now();
    let params = MimdSnarkParams::<L> {
        log_n: log_size,
        log_m: log_size,
        pcs_w: Pcs::new(k1, k2, 4, 16, 0xDEAD_BEEF),
        pcs_abc: Pcs::new(k1, k2, 4, 16, 0xABC_BEEF),
        pcs_e: R2Pcs::new(k1, k2, 4, 16, 0xCAFE_BABE),
        mats: pack_mimd_matrices(
            &mimd.a_per_instance,
            &mimd.b_per_instance,
            &mimd.c_per_instance,
        ),
    };
    println!(
        "setup (matrix packing, excluded from prove): {:.2} ms",
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

    let (proof_s, ts) = pool1.install(|| prove_mimd_with_timing(&mimd.witnesses, &params));
    let (proof_p, tp) = pooln.install(|| prove_mimd_with_timing(&mimd.witnesses, &params));

    println!("--- prove (excl. setup): serial(1) vs parallel({nthreads}) ---");
    println!(
        "  {:<18} {:>11} {:>11}   {:>7}",
        "stage", "serial ms", "par ms", "speedup"
    );
    row("reduce (ŵ + q_AB)", ts.reduce_ms, tp.reduce_ms);
    row("matvec Â,B̂,Ĉ·ŵ", ts.matvec_ms, tp.matvec_ms);
    row("commit w+e", ts.commit_ms, tp.commit_ms);
    row("  · commit w", ts.commit_w_ms, tp.commit_w_ms);
    row("  · commit e_mult", ts.commit_emult_ms, tp.commit_emult_ms);
    row("  · commit e_stack", ts.commit_estack_ms, tp.commit_estack_ms);
    row("mult sumcheck", ts.mult_sumcheck_ms, tp.mult_sumcheck_ms);
    row("linear (row+sc)", ts.linear_ms, tp.linear_ms);
    row("opens w+e", ts.opens_ms, tp.opens_ms);
    row("TOTAL prove", ts.total_ms, tp.total_ms);

    let t = Instant::now();
    let ok = pooln.install(|| verify_mimd(&params, &proof_p)).is_ok();
    let verify_ms = t.elapsed().as_secs_f64() * 1e3;
    assert!(ok, "verify failed");
    assert!(verify_mimd(&params, &proof_s).is_ok());
    println!("  verify:            {:>11.2} ms", verify_ms);
    println!("  proof size:        {} bytes", proof_p.size_bytes());
}

fn main() {
    let log_size: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(15);
    let width: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    match width {
        8 => run::<u8>(8, log_size),
        16 => run::<u16>(16, log_size),
        32 => run::<u32>(32, log_size),
        64 => run::<u64>(64, log_size),
        _ => panic!("width must be 8, 16, 32, or 64"),
    }
}
