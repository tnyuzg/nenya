# sumcheck-ring

Reference deg-3 single-product sumcheck over $R_2 = \mathrm{GR}(2^{32}, 128)$:

$$
c \;\stackrel{?}{=}\; \sum_{h \in \{0,1\}^n} f_1(h)\cdot f_2(h)\cdot f_3(h).
$$

**Protocol specification, benchmark numbers, and design rationale live
in [sumcheck-ring.md](sumcheck-ring.md).** This README is the quickstart.

## Role in the workspace

Executable reference for the deg-3 ring sumcheck. `piop-ring` does **not**
call this crate's prover — its `mult_check` has its own specialized
implementation — but the two share the same coefficient-form round
message and are cross-checked against each other in
[`piop-ring/tests/cross_check_sumcheck_ring.rs`](../piop-ring/tests/cross_check_sumcheck_ring.rs).

Companion: [`sumcheck-field`](../sumcheck-field/) runs the same protocol
over $\mathrm{BB}_4 = \mathrm{BabyBear}^4$ for direct comparison.

## Commands

```sh
cargo test  -p sumcheck-ring            # completeness + soundness
cargo test  -p piop-ring --test cross_check_sumcheck_ring
cargo bench -p sumcheck-ring            # n ∈ {10, 13, 15}
cargo bench -p sumcheck-field           # field-side mirror
```

## Pointers

- Protocol, benchmarks, and design: [sumcheck-ring.md](sumcheck-ring.md).
- PIOP layer built conceptually on top of this spec:
  [`piop-ring`](../piop-ring/piop-ring.md).
