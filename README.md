# Nenya

Rust implementation of **Nenya**, a zkSNARK for rank-1 constraint systems over
the integer rings `Z/2^kZ` (k ∈ {8, 16, 32, 64}), from the paper

> **Nenya: Efficient zkSNARK for Integer Rings Z/2^kZ via Reverse
> Multiplication-Friendly Embeddings** (to appear at IEEE S&P).

Nenya packs 32 ring elements into one element of the Galois ring
`GR(2^k, 128)` and runs a Spartan-style PIOP and a Brakedown-style polynomial
commitment over that ring. The repository contains Nenya's three computation
models (SIMD, MIMD, General) and the field SNARK used as the baseline in the
paper (the same design over the degree-4 extension of BabyBear).

## Build and test

Requires [rustup](https://rustup.rs) on x86-64 Linux; the compiler version
used for the paper is pinned in `rust-toolchain.toml`.

```sh
cargo test --workspace --release
RUSTFLAGS="-C target-cpu=native" cargo build --release -p snark-ring --examples
B=target/release/examples
```

## Reproducing the paper's results

All runs use the paper's parameters (λ = 100, RAA code at rate 1/4,
δ = 0.19, t = 1060 queried columns). `T` is the number of threads; the paper
uses `T=192` on an AWS c7i.48xlarge (2 × Xeon Platinum 8488C, 371 GB RAM) and
reports the median of repeated runs.

```sh
T=192

# Figure 5 (and Figure 8 with T=1): prover time and proof size, 2^10..2^25 constraints.
# Prints: prover seconds, proof bytes, verifier milliseconds.
for L in $(seq 10 25); do for S in field simd mimd general; do
  echo "$L $S $($B/exp_cell cell $S u32 $L $T)"; done; done

# Table 1: cost breakdown at 2^25.
for S in field simd mimd general; do $B/exp_cell breakdown $S $T 25; done

# Table 2: base word size at 2^20 (the Field column is the field run of Figure 5 at 2^20).
for S in simd mimd general; do for W in u8 u16 u32 u64; do
  echo "$S $W $($B/exp_cell cell $S $W 20 $T)"; done; done

# Table 3: ring vs field operation latency (ns). Speedup = 32 × field / ring.
$B/exp_cell ringops

# Figure 7: PNG Sub/Up filtering over Z/2^8, Nenya vs field, 2^10..2^20-byte images.
for L in $(seq 10 20); do $B/png_filter $L $T | grep APP_RESULT; done
```

Figure 6's numbers for the prior work [30] are taken from that paper.

## Limitations

This is a research prototype. It has not been audited and must not be used to
protect real assets. The verifier is linear in the circuit size (the
sublinear, holographic variant is not implemented), and knowledge extraction is
proved in the paper but not implemented.

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option.
