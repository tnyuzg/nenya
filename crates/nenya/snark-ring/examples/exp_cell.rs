//! Experiment runner for the paper's evaluation (Section 6). Each invocation
//! makes one measurement at the paper's secure parameters (δ = 0.19, λ = 100,
//! t = 1060, proof-size-optimal shape) on the circuit `x·x = y`.
//!
//! Usage:
//!   exp_cell cell <field|simd|mimd|general> <u8|u16|u32|u64> <logN> <threads>
//!       -> "prove_s proof_bytes verify_ms"        (Figures 5 and 8, Table 2)
//!   exp_cell breakdown <field|simd|mimd|general> <threads> [logN]
//!       -> "metric component value" lines, u32    (Table 1; logN defaults to 25)
//!   exp_cell ringops
//!       -> "k add_ring_ns add_field_ns mult_ring_ns mult_field_ns" rows (Table 3)
//!
//! `logN` is log2 of the total number of constraints; the word size is ignored
//! for `field`.

use field::{AbstractField, BB, BB4};
use galois_ring::{Gr1, Gr128, Gr4, Limb};
use reduction::constants::HasPackingConstants;
use reduction::{
    matrix::SparseMatrix, pack_general_matrices, pack_mimd_matrices, GeneralR1csInstance,
    SimdR1csInstance,
};
use snark_field::instance::{FieldR1csInstance, SparseMatrixBB};
use snark_field::{
    prove_field_with_timing, verify_field, verify_field_with_timing, FieldSnarkParams,
};
use snark_ring::{
    prove_general_with_timing, prove_mimd_with_timing, prove_simd_with_timing, verify_general,
    verify_general_with_timing, verify_mimd, verify_mimd_with_timing, verify_simd,
    verify_simd_with_timing, GeneralSnarkParams, MimdSnarkParams, SimdSnarkParams,
};
use std::time::Instant;

// ---------------------------------------------------------------------------
// x·x = y instance builders (generic over base width L for the ring side)
// ---------------------------------------------------------------------------

fn xy_witness<L: Limb>(m: usize, seed: usize) -> Vec<Gr1<L>> {
    let k = (m - 1) / 2;
    let mut w = vec![Gr1::<L>::ZERO; m];
    w[0] = Gr1::<L>::ONE;
    for i in 0..k {
        let x = Gr1(L::from_u64(
            (((seed * 0x9E37) ^ (i * 0x85EB)) as u64).wrapping_add(1),
        ));
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

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

// Returns (prove_s, proof_bytes, verify_ms).
fn ring_simd<L: HasPackingConstants>(log: usize) -> (f64, usize, f64) {
    let n = 1usize << log;
    let [a, b, c] = xy_matrices::<L>(n, n);
    let witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|s| xy_witness::<L>(n, s));
    let simd = SimdR1csInstance { a, b, c, witnesses };
    let params = SimdSnarkParams::<L>::secure(log, log, 0xDEAD, 0xCAFE);
    let (proof, t) = prove_simd_with_timing(&simd, &params);
    let tv = Instant::now();
    verify_simd(&simd.a, &simd.b, &simd.c, &params, &proof).expect("verify simd");
    (t.total_ms / 1e3, proof.size_bytes(), ms(tv))
}

fn ring_mimd<L: HasPackingConstants>(log: usize) -> (f64, usize, f64) {
    let n = 1usize << log;
    let a: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| xy_matrices::<L>(n, n)[0].clone());
    let b: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| xy_matrices::<L>(n, n)[1].clone());
    let c: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| xy_matrices::<L>(n, n)[2].clone());
    let witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|s| xy_witness::<L>(n, s));
    let mats = pack_mimd_matrices(&a, &b, &c);
    let params = MimdSnarkParams::<L>::secure(log, log, mats, 0xDEAD, 0xCAFE);
    let (proof, t) = prove_mimd_with_timing(&witnesses, &params);
    let tv = Instant::now();
    verify_mimd(&params, &proof).expect("verify mimd");
    (t.total_ms / 1e3, proof.size_bytes(), ms(tv))
}

fn ring_general<L: HasPackingConstants>(log_blocks: usize) -> (f64, usize, f64) {
    let n = 32usize << log_blocks; // = 2^(log_blocks+5)
    let [a, b, c] = xy_matrices::<L>(n, n);
    let w = xy_witness::<L>(n, 1);
    let diag = pack_general_matrices(&a, &b, &c);
    let params = GeneralSnarkParams::<L>::secure(log_blocks, diag, 0xDEAD, 0xCAFE);
    let g = GeneralR1csInstance { a, b, c, w };
    let (proof, t) = prove_general_with_timing(&g.w, &params);
    let tv = Instant::now();
    verify_general(&params, &proof).expect("verify general");
    (t.total_ms / 1e3, proof.size_bytes(), ms(tv))
}

fn field_snark(log: usize) -> (f64, usize, f64) {
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
    let params = FieldSnarkParams::secure(log, log, 0xDEAD);
    let (proof, t) = prove_field_with_timing(&inst, &params);
    let tv = Instant::now();
    verify_field(&inst.a, &inst.b, &inst.c, &params, &proof).expect("verify field");
    (t.total_ms / 1e3, proof.size_bytes(), ms(tv))
}

/// Dispatch a ring system over the chosen base width. `log` is the per-instance
/// (SIMD/MIMD) or log_blocks (general) size.
fn ring_cell(system: &str, word: &str, log: usize) -> (f64, usize, f64) {
    macro_rules! by_word {
        ($f:ident) => {
            match word {
                "u8" => $f::<u8>(log),
                "u16" => $f::<u16>(log),
                "u32" => $f::<u32>(log),
                "u64" => $f::<u64>(log),
                _ => panic!("word must be u8|u16|u32|u64"),
            }
        };
    }
    match system {
        "simd" => by_word!(ring_simd),
        "mimd" => by_word!(ring_mimd),
        "general" => by_word!(ring_general),
        _ => panic!("ring system must be simd|mimd|general"),
    }
}

// ---------------------------------------------------------------------------
// Ring-arithmetic microbenchmark (Exp 4)
// ---------------------------------------------------------------------------

fn rand_gr128<L: Limb>(s: &mut u64) -> Gr128<L> {
    let mut next = || {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        L::from_u64(*s >> 32)
    };
    let mut arr = [Gr4::<L>::ZERO; 32];
    for slot in arr.iter_mut() {
        *slot = Gr4([next(), next(), next(), next()]);
    }
    Gr128(arr)
}

/// Median per-op latency (ns) of `op` over `iters`, fed a small rotating pool so
/// the optimizer can't fold it. Warm-up first.
fn time_op_ns<T: Copy, F: Fn(T, T) -> T>(pool: &[T], iters: usize, op: F) -> f64 {
    let m = pool.len();
    // warm-up
    let mut acc = pool[0];
    for i in 0..(iters / 10).max(1000) {
        acc = op(acc, pool[i % m]);
    }
    std::hint::black_box(acc);
    let t = Instant::now();
    let mut acc = pool[1 % m];
    for i in 0..iters {
        acc = op(acc, pool[i % m]);
    }
    let el = t.elapsed().as_secs_f64();
    std::hint::black_box(acc);
    el / iters as f64 * 1e9
}

fn ringops_row<L: Limb>(k: u32, iters: usize) -> (f64, f64) {
    let mut s = 0xC0FFEE ^ (k as u64);
    let pool: Vec<Gr128<L>> = (0..256).map(|_| rand_gr128::<L>(&mut s)).collect();
    let add = time_op_ns(&pool, iters, |a, b| a + b);
    let mul = time_op_ns(&pool, iters, |a, b| a * b);
    (add, mul)
}

fn ringops() {
    let iters = 20_000_000usize;
    // Field reference: BB4 add / mul (degree-4 BabyBear extension).
    let mut s = 0xBEEF_u64;
    let mut nb = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        BB::from_canonical_u32((s >> 33) as u32 % 0x7800_0001)
    };
    let fpool: Vec<BB4> = (0..256)
        .map(|_| {
            let b = [nb(), nb(), nb(), nb()];
            <BB4 as field::AbstractExtensionField<BB>>::from_base_slice(&b)
        })
        .collect();
    let add_f = time_op_ns(&fpool, iters, |a, b| a + b);
    let mul_f = time_op_ns(&fpool, iters, |a, b| a * b);

    println!("k add_ring_ns add_field_ns mult_ring_ns mult_field_ns");
    for k in [8u32, 16, 32, 64] {
        let (add_r, mul_r) = match k {
            8 => ringops_row::<u8>(k, iters),
            16 => ringops_row::<u16>(k, iters),
            32 => ringops_row::<u32>(k, iters),
            64 => ringops_row::<u64>(k, iters),
            _ => unreachable!(),
        };
        println!("{k} {add_r:.4} {add_f:.4} {mul_r:.4} {mul_f:.4}");
    }
}

// ---------------------------------------------------------------------------
// Cost breakdown (Table 1): per-system phase split at a fixed size.
// Emits "metric component value" lines.
// ---------------------------------------------------------------------------

fn fmt(v: f64) -> String {
    if v.is_nan() {
        "nan".into()
    } else {
        format!("{v:.6}")
    }
}

/// Prover phase split (ms in, seconds out). `mxeval` is the **matrix random
/// evaluation** (building the γ-combined `M` row at the random point), pulled out
/// of `piop_linear` into its own component; `piop_linear` then carries the rest
/// (the linear sumcheck + glue). Residual glue (transcript, eq builds) folded
/// into `piop_mult` so the components sum to `total`.
fn emit_prover(
    red: f64,
    lin: f64,
    mult: f64,
    enc: f64,
    merk: f64,
    open: f64,
    total: f64,
    mxeval: f64,
) {
    let redp = if red.is_nan() { 0.0 } else { red };
    let mult = mult + (total - (redp + lin + mult + enc + merk + open));
    println!("prover_s reduction {}", fmt(red / 1e3));
    println!("prover_s piop_mxeval {}", fmt(mxeval / 1e3));
    println!("prover_s piop_linear {}", fmt((lin - mxeval) / 1e3));
    println!("prover_s piop_mult {}", fmt(mult / 1e3));
    println!("prover_s pcs_encode {}", fmt(enc / 1e3));
    println!("prover_s pcs_merkle {}", fmt(merk / 1e3));
    println!("prover_s pcs_open {}", fmt(open / 1e3));
    println!("prover_s total {}", fmt(total / 1e3));
}
fn emit_proof(piop_b: usize, total_b: usize) {
    let pcs_b = total_b.saturating_sub(piop_b);
    println!("proof_kb piop {:.3}", piop_b as f64 / 1000.0);
    println!("proof_kb pcs {:.3}", pcs_b as f64 / 1000.0);
    println!("proof_kb total {:.3}", total_b as f64 / 1000.0);
}
/// Verify split: `piop_mxeval` = the verifier's
/// matrix random evaluation (recomputing the `M = A+γB+γ²C` row at the random
/// point), pulled OUT of `piop`; `piop` = the rest of the PIOP verify (the
/// sumcheck checks + glue). First three components sum to `total`.
fn emit_verify(v_piop: f64, v_mxeval: f64, v_pcs: f64) {
    println!("verify_ms piop_mxeval {v_mxeval:.4}");
    println!("verify_ms piop {v_piop:.4}");
    println!("verify_ms pcs {v_pcs:.4}");
    println!("verify_ms total {:.4}", v_piop + v_mxeval + v_pcs);
}
/// One-time, witness-independent circuit packing (pack_mimd/general_matrices).
/// Reported separately — NOT part of `prover_s total` — since it amortizes over
/// many proofs of the same circuit. 0 for field/SIMD (no packing; their matvec
/// runs inside the prover).
fn emit_preprocess(secs: f64) {
    println!("prover_s preprocessing {}", fmt(secs));
}

fn bd_simd<L: HasPackingConstants>(log: usize) {
    let n = 1usize << log;
    let [a, b, c] = xy_matrices::<L>(n, n);
    let witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|s| xy_witness::<L>(n, s));
    let simd = SimdR1csInstance { a, b, c, witnesses };
    let params = SimdSnarkParams::<L>::secure(log, log, 0xDEAD, 0xCAFE);
    let (proof, t) = prove_simd_with_timing(&simd, &params);
    let vt = verify_simd_with_timing(&simd.a, &simd.b, &simd.c, &params, &proof).unwrap();
    emit_prover(
        t.reduce_ms,
        t.linear_arow_ms + t.linear_sumcheck_ms,
        t.matvec_abc_ms + t.eq_r_ms + t.mult_sumcheck_ms,
        t.commit_w_encode_ms + t.commit_e_encode_ms,
        t.commit_w_merkle_ms + t.commit_e_merkle_ms,
        t.open_w_ms + t.open_e_ms,
        t.total_ms,
        t.linear_arow_ms,
    );
    emit_preprocess(0.0); // SIMD shares the sparse matrices — no packing
    emit_proof(
        proof.mult_proof.serialize().len() + proof.lin_proof.serialize().len(),
        proof.size_bytes(),
    );
    emit_verify(
        vt.mult_sumcheck_ms + vt.linear_sumcheck_ms,
        vt.linear_arow_recompute_ms,
        vt.pcs_e_ms + vt.pcs_w_ms + vt.kernel_ms,
    );
}

fn bd_mimd<L: HasPackingConstants>(log: usize) {
    let n = 1usize << log;
    let a: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| xy_matrices::<L>(n, n)[0].clone());
    let b: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| xy_matrices::<L>(n, n)[1].clone());
    let c: [SparseMatrix<Gr1<L>>; 32] = std::array::from_fn(|_| xy_matrices::<L>(n, n)[2].clone());
    let witnesses: [Vec<Gr1<L>>; 32] = std::array::from_fn(|s| xy_witness::<L>(n, s));
    let tp = Instant::now();
    let mats = pack_mimd_matrices(&a, &b, &c);
    let preprocess_s = tp.elapsed().as_secs_f64();
    let params = MimdSnarkParams::<L>::secure(log, log, mats, 0xDEAD, 0xCAFE);
    let (proof, t) = prove_mimd_with_timing(&witnesses, &params);
    let (res, vt) = verify_mimd_with_timing(&params, &proof);
    res.unwrap();
    emit_prover(
        t.reduce_ms,
        t.linear_ms,
        t.matvec_ms + t.mult_sumcheck_ms,
        t.commit_encode_ms,
        t.commit_merkle_ms,
        t.opens_ms,
        t.total_ms,
        t.linear_rowbuild_ms,
    );
    emit_preprocess(preprocess_s);
    emit_proof(
        proof.mult_proof.serialize().len() + proof.lin_proof.serialize().len(),
        proof.size_bytes(),
    );
    emit_verify(
        vt.piop_ms - vt.row_recompute_ms,
        vt.row_recompute_ms,
        vt.pcs_ms,
    );
}

fn bd_general<L: HasPackingConstants>(log_blocks: usize) {
    let n = 32usize << log_blocks;
    let [a, b, c] = xy_matrices::<L>(n, n);
    let w = xy_witness::<L>(n, 1);
    let tp = Instant::now();
    let diag = pack_general_matrices(&a, &b, &c);
    let preprocess_s = tp.elapsed().as_secs_f64();
    let params = GeneralSnarkParams::<L>::secure(log_blocks, diag, 0xDEAD, 0xCAFE);
    let g = GeneralR1csInstance { a, b, c, w };
    let (proof, t) = prove_general_with_timing(&g.w, &params);
    let (res, vt) = verify_general_with_timing(&params, &proof);
    res.unwrap();
    emit_prover(
        t.reduce_ms,
        t.linear_ms,
        t.matvec_ms + t.mult_sumcheck_ms,
        t.commit_encode_ms,
        t.commit_merkle_ms,
        t.opens_ms,
        t.total_ms,
        t.linear_rowbuild_ms,
    );
    emit_preprocess(preprocess_s);
    emit_proof(
        proof.mult_proof.serialize().len() + proof.lin_proof.serialize().len(),
        proof.size_bytes(),
    );
    emit_verify(
        vt.piop_ms - vt.row_recompute_ms,
        vt.row_recompute_ms,
        vt.pcs_ms,
    );
}

fn bd_field(log: usize) {
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
    let params = FieldSnarkParams::secure(log, log, 0xDEAD);
    let (proof, t) = prove_field_with_timing(&inst, &params);
    let vt = verify_field_with_timing(&inst.a, &inst.b, &inst.c, &params, &proof).unwrap();
    emit_prover(
        f64::NAN, // field has no packing reduction
        t.linear_arow_ms + t.linear_sumcheck_ms,
        t.matvec_abc_ms + t.eq_r_ms + t.mult_sumcheck_ms,
        t.commit_w_encode_ms,
        t.commit_w_merkle_ms,
        t.open_w_ms,
        t.total_ms,
        t.linear_arow_ms,
    );
    emit_preprocess(0.0); // field uses sparse A,B,C directly — no packing
                          // piop bytes per FieldSnarkProof::size_bytes accounting (BB4 = 16 B).
    let e = 16usize;
    let piop_b = (3 * e + proof.mult_proof.round_polys.len() * 4 * e)
        + (3 * e + proof.lin_proof.round_polys.len() * 3 * e);
    emit_proof(piop_b, proof.size_bytes());
    emit_verify(
        vt.mult_sumcheck_ms + vt.linear_sumcheck_ms,
        vt.linear_arow_recompute_ms,
        vt.pcs_w_ms,
    );
}

// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("ringops") => ringops(),
        Some("breakdown") => {
            // @ 2^25 (field log=25; batched per-instance/block log=20), u32.
            let system = &args[2];
            let threads: usize = args[3].parse().unwrap();
            // Optional logN (default 25). Batched modes use logN-5 per instance/block.
            let logn: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(25);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| match system.as_str() {
                "field" => bd_field(logn),
                "simd" => bd_simd::<u32>(logn - 5),
                "mimd" => bd_mimd::<u32>(logn - 5),
                "general" => bd_general::<u32>(logn - 5),
                _ => panic!("breakdown system must be field|simd|mimd|general"),
            });
        }
        Some("cell") => {
            let system = &args[2];
            let word = &args[3];
            let logn: usize = args[4].parse().unwrap();
            let threads: usize = args[5].parse().unwrap();
            // batched modes: per-instance / block log = logN - 5; field: full logN.
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let (prove_s, bytes, verify_ms) = pool.install(|| {
                if system == "field" {
                    field_snark(logn)
                } else {
                    ring_cell(system, word, logn - 5)
                }
            });
            println!("{prove_s:.6} {bytes} {verify_ms:.4}");
        }
        _ => {
            eprintln!(
                "usage: exp_cell cell <system> <word> <logN> <threads> \
                 | exp_cell breakdown <system> <threads> [logN] | exp_cell ringops"
            );
            std::process::exit(2);
        }
    }
}
