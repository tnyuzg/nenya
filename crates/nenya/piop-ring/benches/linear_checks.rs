//! Side-by-side bench for the three linear-check provers, started from a
//! **R1CS instance of the same size** (witness length `N = 2^LOG_ORIG`
//! u32 elements). After packing 32 u32 → 1 `R_2` element, the sumcheck
//! over `R_2` sees `LOG_SUMCHECK = LOG_ORIG - 5` variables in every
//! variant.
//!
//! For `LOG_ORIG = 20` (the default → `LOG_SUMCHECK = 15`):
//! - **SIMD**: pre-built `a_row` of length `2^15`, witness MLE of length
//!   `2^15`. Sumcheck: 15 rounds, one `R_2 × R_2` product per fold-pair.
//! - **MIMD**: identical sumcheck shape to SIMD (the only structural
//!   difference between the two is in the matvec preprocessing, which
//!   this bench excludes — see below).
//! - **General**: 32 pre-built row MLEs of length `2^15` plus 32 rho MLEs.
//!   Sumcheck: 15 rounds, **32 `(row, rho)` factor pairs at each fold
//!   index** — ~32× more `R_2` muls per round than SIMD/MIMD.
//!
//! ## What this bench measures
//!
//! **PIOP prover time only.** All three benches *exclude* the row-eq
//! preprocessing (`build_eq_table` + sparse matvec on `A`, `B`, `C`).
//! That stage is verifier-recomputable from public inputs (`A`, `B`, `C`,
//! `r_target`) — it is symmetric preprocessing, not novel prover work,
//! and the verifier already does it during the final check. Excluding it
//! puts SIMD/MIMD on the same footing as the general bench, which has
//! always taken pre-built row MLEs.
//!
//! Run: `cargo bench -p piop-ring --bench linear_checks`.

/// Original R1CS witness size in u32 elements (`N = 2^LOG_ORIG`).
const LOG_ORIG: usize = 20;

/// Sumcheck-over-R_2 size after packing 32 u32 → 1 R_2 slot. Same for
/// all three variants. For general it is also the per-layer MLE length;
/// the 32 layers are stacked on top.
const LOG_SUMCHECK: usize = LOG_ORIG - 5;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use galois_ring::{R2TeichmullerSampler, GR32_1, GR32_128, GR32_4};
use piop_ring::linear_check_general::LAYERS;
use piop_ring::linear_check_mimd::build_m_row;
use piop_ring::linear_check_simd::build_a_row;
use piop_ring::{
    build_eq_table, prove_linear_check_general, prove_linear_check_mimd_sumcheck_only,
    prove_linear_check_simd_sumcheck_only, Mle, SparseMatrix,
};
use reduction::SparseMatrix as SparseR2;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 32) as u32
}

fn rand_gr128(state: &mut u64) -> GR32_128 {
    let mut arr = [GR32_4::ZERO; 32];
    for slot in arr.iter_mut() {
        *slot = GR32_4([lcg(state), lcg(state), lcg(state), lcg(state)]);
    }
    GR32_128(arr)
}

fn rand_mle(num_vars: usize, state: &mut u64) -> Mle {
    let len = 1usize << num_vars;
    let mut evals = Vec::with_capacity(len);
    for _ in 0..len {
        evals.push(rand_gr128(state));
    }
    Mle::new(evals)
}

fn rand_target(num_vars: usize, state: &mut u64) -> Vec<GR32_128> {
    (0..num_vars).map(|_| rand_gr128(state)).collect()
}

fn rand_sparse_r0(log_n: usize, nnz_per_row: usize, state: &mut u64) -> SparseMatrix {
    let nrows = 1usize << log_n;
    let ncols = 1usize << log_n;
    let mut m = SparseMatrix::new(log_n, log_n);
    for r in 0..nrows {
        for _ in 0..nnz_per_row {
            let c = (lcg(state) as usize) % ncols;
            let v = GR32_1(lcg(state));
            m.push(r, c, v);
        }
    }
    m
}

fn rand_sparse_r2(log_n: usize, nnz_per_row: usize, state: &mut u64) -> SparseR2<GR32_128> {
    let n = 1usize << log_n;
    let mut m = SparseR2::<GR32_128>::new(n, n);
    for r in 0..n {
        for _ in 0..nnz_per_row {
            let c = (lcg(state) as usize) % n;
            m.push(r, c, rand_gr128(state));
        }
    }
    m
}

fn simd_claim(matrix: &SparseMatrix, w: &Mle, r_target: &[GR32_128]) -> GR32_128 {
    let eq = build_eq_table(r_target);
    let mut a_row = vec![GR32_128::ZERO; matrix.num_cols()];
    for &(row, col, val) in &matrix.entries {
        a_row[col] = a_row[col] + eq.evals()[row].mul_by_gr1(val);
    }
    let mut acc = GR32_128::ZERO;
    for i in 0..w.len() {
        acc = acc + a_row[i] * w.evals()[i];
    }
    acc
}

fn mimd_claim(matrix: &SparseR2<GR32_128>, w: &Mle, r_target: &[GR32_128]) -> GR32_128 {
    let eq = build_eq_table(r_target);
    let mut m_row = vec![GR32_128::ZERO; matrix.cols];
    for &(row, col, val) in &matrix.triples {
        m_row[col] = m_row[col] + eq.evals()[row] * val;
    }
    let mut acc = GR32_128::ZERO;
    for i in 0..w.len() {
        acc = acc + m_row[i] * w.evals()[i];
    }
    acc
}

fn general_claim(rows: &[Mle; LAYERS], rhos: &[Mle; LAYERS]) -> GR32_128 {
    let len = rows[0].len();
    let mut acc = GR32_128::ZERO;
    for ell in 0..LAYERS {
        for i in 0..len {
            acc = acc + rows[ell].evals()[i] * rhos[ell].evals()[i];
        }
    }
    acc
}

fn bench_linear_check_simd(c: &mut Criterion) {
    let mut g = c.benchmark_group("linear_check_simd");
    g.sample_size(10);
    let mut state = 0xC0FFEE_u64;
    let matrix = rand_sparse_r0(LOG_SUMCHECK, 3, &mut state);
    let w = rand_mle(LOG_SUMCHECK, &mut state);
    let r_target = rand_target(LOG_SUMCHECK, &mut state);
    let claim = simd_claim(&matrix, &w, &r_target);

    // Preprocessing (verifier-recomputable): build eq_row and the row MLE.
    // Excluded from the timed region.
    let eq_row = build_eq_table(&r_target);
    let a_row = build_a_row(&matrix, eq_row.evals());

    g.bench_with_input(
        BenchmarkId::from_parameter(format!("orig=2^{LOG_ORIG} (sumcheck=2^{LOG_SUMCHECK})")),
        &(a_row, w, claim),
        |bencher, (a_row, w, claim)| {
            bencher.iter(|| {
                let mut sampler = R2TeichmullerSampler::new(0xBEEF);
                let _ = black_box(prove_linear_check_simd_sumcheck_only(
                    a_row.clone(),
                    w.clone(),
                    *claim,
                    &mut sampler,
                ));
            });
        },
    );
    g.finish();
}

fn bench_linear_check_mimd(c: &mut Criterion) {
    let mut g = c.benchmark_group("linear_check_mimd");
    g.sample_size(10);
    let mut state = 0xCAFEBABE_u64;
    let matrix = rand_sparse_r2(LOG_SUMCHECK, 3, &mut state);
    let w = rand_mle(LOG_SUMCHECK, &mut state);
    let r_target = rand_target(LOG_SUMCHECK, &mut state);
    let claim = mimd_claim(&matrix, &w, &r_target);

    // Preprocessing (verifier-recomputable): build eq_row and the row MLE.
    // Excluded from the timed region.
    let eq_row = build_eq_table(&r_target);
    let m_row = build_m_row(&matrix, eq_row.evals());

    g.bench_with_input(
        BenchmarkId::from_parameter(format!("orig=2^{LOG_ORIG} (sumcheck=2^{LOG_SUMCHECK})")),
        &(m_row, w, claim),
        |bencher, (m_row, w, claim)| {
            bencher.iter(|| {
                let mut sampler = R2TeichmullerSampler::new(0xBEEF);
                let _ = black_box(prove_linear_check_mimd_sumcheck_only(
                    m_row.clone(),
                    w.clone(),
                    *claim,
                    &mut sampler,
                ));
            });
        },
    );
    g.finish();
}

fn bench_linear_check_general(c: &mut Criterion) {
    let mut g = c.benchmark_group("linear_check_general");
    g.sample_size(10);
    // Per-layer MLE length = 2^LOG_SUMCHECK; sumcheck has LOG_SUMCHECK
    // rounds with 32 inner factors at each fold index → total inner work
    // = 32 · 2^LOG_SUMCHECK, i.e. 32× more R_2 muls than SIMD/MIMD at the
    // same sumcheck instance size.
    let mut state = 0xFEEDFACE_u64;
    let rows: [Mle; LAYERS] = std::array::from_fn(|_| rand_mle(LOG_SUMCHECK, &mut state));
    let rhos: [Mle; LAYERS] = std::array::from_fn(|_| rand_mle(LOG_SUMCHECK, &mut state));
    let claim = general_claim(&rows, &rhos);

    g.bench_with_input(
        BenchmarkId::from_parameter(format!(
            "orig=2^{LOG_ORIG} (sumcheck=2^{LOG_SUMCHECK}, layers=32)"
        )),
        &(rows, rhos, claim),
        |bencher, (rows, rhos, claim)| {
            bencher.iter(|| {
                let mut sampler = R2TeichmullerSampler::new(0xBEEF);
                let _ = black_box(prove_linear_check_general(rows.clone(), rhos.clone(), *claim, &mut sampler));
            });
        },
    );
    g.finish();
}

criterion_group!(
    benches,
    bench_linear_check_simd,
    bench_linear_check_mimd,
    bench_linear_check_general,
);
criterion_main!(benches);
