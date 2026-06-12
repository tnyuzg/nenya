# sumcheck-ring

Two reference provers for the single-product sumcheck over
$R_2 = \mathrm{GR}(2^{32}, 128)$:

$$
c \;\stackrel{?}{=}\; \sum_{h \in \{0,1\}^n} \prod_{j=1}^{d} f_j(h),
\qquad f_j \in \mathrm{Mle}_n(R_2),\quad d \in \{2, 3\}.
$$

- [`prove_deg2`](src/prover.rs) — deg-2, `∑ f₁·f₂`. Cross-checked against
  `piop-ring::prove_linear_check`.
- [`prove`](src/prover.rs) — deg-3, `∑ f₁·f₂·f₃`. Cross-checked against
  `piop-ring::prove_mult_check`.

This crate is the **executable spec** for the ring sumcheck. `piop-ring`
does not call it in production — each PIOP rolls its own specialized
prover — but both share coefficient-form round messages and are
cross-checked against each other in
[`piop-ring/tests/cross_check_sumcheck_ring.rs`](../piop-ring/tests/cross_check_sumcheck_ring.rs).

---

## 1. Protocol

### 1.1 Claim

Public inputs: $d$ multilinears $f_1, \ldots, f_d \in \mathrm{Mle}_n(R_2)$
($d \in \{2, 3\}$, evaluation form on $\{0,1\}^n$) and a claimed sum
$c \in R_2$. Sections 1.2–1.3 are $d$-independent. Sections 1.4–1.6
present the $d = 3$ case; the $d = 2$ case ([`prove_deg2`](src/prover.rs),
[`verify_deg2`](src/verifier.rs)) drops one factor and one round-polynomial
coefficient — per-pair work is **4 muls + 4 adds + 2 subs** (vs 10 + 6 + 3
for $d = 3$), round message is $[a_0, a_1, a_2]$, and the final check is
$c_n = \text{final}_{f_1}\cdot \text{final}_{f_2}$.

### 1.2 Challenge set

Challenges are drawn from the **Teichmüller set** $T_2 \subset R_2$:

$$
T_2 \;=\; \{\,x \in R_2 : x^{2^{128}} = x\,\}, \qquad |T_2| = 2^{128}.
$$

$T_2$ is the unique multiplicative lift of $\mathbb{F}_{2^{128}}$ into $R_2$.
It is **unit-separated**: for any $t \neq t'$ in $T_2$, the difference
$t - t'$ is a unit in $R_2$. Without this property, Schwartz–Zippel fails
over $R_2$ (a nonzero degree-$d$ univariate can have far more than $d$
roots when coefficients involve zero divisors).

Sampler lives in galois-ring (next to the primitive it wraps):
[`galois_ring::R2TeichmullerSampler`](../galois-ring/src/sampling.rs) fills
all 128 u32 coefficients of an $R_2$ element with random bits (the mod-2
image is a uniform $\mathbb{F}_{2^{128}}$ representative; extra precision
is collapsed by the lift), then Hensel-lifts via
[`galois_ring::teichmuller_lift_r2`](../galois-ring/src/teichmuller.rs).
Cost: ~640 $R_2$ muls / ~580 µs per challenge.

### 1.3 Round-message encoding

Each round message is a degree-3 univariate
$s_i(Z) = a_0 + a_1 Z + a_2 Z^2 + a_3 Z^3$ sent in **monomial-coefficient
form** $[a_0, a_1, a_2, a_3]$. The textbook Lagrange-on-$\{0,1,2,3\}$ form
would force the verifier to divide by the Vandermonde determinant of
$\{0,1,2,3\}$ — a multiple of $2$ — and $2$ is not a unit in
$\mathbb{Z}/2^{32}$. Coefficient form is division-free.

The $\{0, 1, \infty\}$ evaluation form used for deg-2 messages in
`piop-ring`'s linear check does not generalize cleanly past deg-2 (the
extra points beyond $\{0, 1, \infty\}$ run into the same divide-by-2
problem). `piop-ring`'s Gruen-optimized mult check works around this by
factoring an $\mathrm{eq}_1$ slot out, reducing the inner-factor degree
to 2 — see [`piop-ring/piop-ring.md`](../piop-ring/piop-ring.md) §2.3.

### 1.4 Prover

Start with $f_{j,0} = f_j$ for $j \in \{1,2,3\}$, $c_0 = c$. For
$i = 0, \ldots, n-1$, with $m = 2^{n-i-1}$:

1. Build $s_i(Z) = \sum_{x \in \{0,1\}^{n-i-1}} f_{1,i}(Z, x)\,f_{2,i}(Z,
   x)\,f_{3,i}(Z, x)$ as $[a_0, a_1, a_2, a_3]$ using shared sub-products
   ([src/prover.rs:compute_round_poly](src/prover.rs#L82-L127)). Per
   fold-pair, with $\Delta_j = f_{j,\text{hi}} - f_{j,\text{lo}}$:

   $$
   p_\text{lolo} = f_{1,\text{lo}}\,f_{2,\text{lo}},\quad
   p_\text{lohi} = f_{1,\text{lo}}\,\Delta_2,\quad
   p_\text{hilo} = \Delta_1\,f_{2,\text{lo}},\quad
   p_\text{hihi} = \Delta_1\,\Delta_2;
   $$
   $$
   \begin{aligned}
   a_0 &\mathrel{+}= p_\text{lolo}\,f_{3,\text{lo}},\\
   a_1 &\mathrel{+}= (p_\text{lohi} + p_\text{hilo})\,f_{3,\text{lo}} + p_\text{lolo}\,\Delta_3,\\
   a_2 &\mathrel{+}= p_\text{hihi}\,f_{3,\text{lo}} + (p_\text{lohi} + p_\text{hilo})\,\Delta_3,\\
   a_3 &\mathrel{+}= p_\text{hihi}\,\Delta_3.
   \end{aligned}
   $$

   Cost: **10 muls + ~6 adds + 3 subs** per pair.

2. Send $s_i$.
3. Draw $r_i \leftarrow T_2$.
4. Fold $f_{j,i+1}(x) = f_{j,i}(x) + r_i\,(f_{j,i}(x + m) - f_{j,i}(x))$
   for $x \in [0, m)$ and each $j \in \{1,2,3\}$. Cost: 1 mul + 1 add + 1
   sub per pair per table = 3 fold muls per pair.

Finally send $\text{final}_{f_j} = f_{j,n}$ for $j \in \{1,2,3\}$ (three
$R_2$ scalars).

Total inner-loop muls per pair: **13** (10 compute + 3 fold).

### 1.5 Verifier

Start with $c_0 = c$. For $i = 0, \ldots, n-1$:

1. Receive $s_i = [a_0, a_1, a_2, a_3]$.
2. **Round check.** Assert $s_i(0) + s_i(1) = 2 a_0 + a_1 + a_2 + a_3 = c_i$.
   Multiplication by $2$ is $x + x$; no inverse required.
3. Draw $r_i \leftarrow T_2$ (same seed as prover).
4. Update $c_{i+1} = s_i(r_i) = a_0 + r_i(a_1 + r_i(a_2 + r_i\,a_3))$ by
   Horner.

**Final check.** Assert
$c_n = \text{final}_{f_1}\cdot \text{final}_{f_2}\cdot \text{final}_{f_3}$.

Implementation: [src/verifier.rs](src/verifier.rs).

### 1.6 Soundness

A cheating prover's round-$i$ message differs from the honest $s_i$ by a
nonzero polynomial of degree $\leq 3$. By Schwartz–Zippel over the
unit-separated set $T_2$, the probability it vanishes at uniform
$r_i \in T_2$ is $\leq 3 / 2^{128}$. Union bound over $n$ rounds:

$$
\varepsilon_{\text{sumcheck}} \;\leq\; 3n / 2^{128}.
$$

The final check trusts the three final evaluations; in a real protocol
they are reduced to PCS openings at $r$.

---

## 2. Architecture

| File | Purpose |
|------|---------|
| [src/mle.rs](src/mle.rs) | `Mle`: `Vec<GR32_128>` evals + in-place fold |
| [src/proof.rs](src/proof.rs) | `SumcheckProof` (deg-3) + `SumcheckProofDeg2` |
| [src/prover.rs](src/prover.rs) | `prove(f1, f2, f3, …)` and `prove_deg2(f1, f2, …)` |
| [src/verifier.rs](src/verifier.rs) | `verify` / `verify_deg2` |

The challenge sampler `R2TeichmullerSampler` lives in
[`galois-ring/src/sampling.rs`](../galois-ring/src/sampling.rs), next to
the `teichmuller_lift_r2` primitive it wraps.

Companion crate [`sumcheck-field`](../sumcheck-field/) runs the identical
protocol over $\mathrm{BB}_4 = \mathrm{BabyBear}^4$ for direct comparison.

---

## 3. Benchmarks

Apple Silicon (aarch64), `--release`. Identical Criterion harnesses on
both sides ([benches/sumcheck.rs](benches/sumcheck.rs),
[sumcheck-field/benches/sumcheck.rs](../sumcheck-field/benches/sumcheck.rs));
same protocol, same fold loop, same proof structure; only the element type
and sampler differ.

### 3.1 Prover wall time

Benched at `num_vars ∈ {10, 13, 15}` for both variants; 15 matches the
`piop-ring` bench point.

| `num_vars` | `prove_deg2` | `prove` (deg-3) |
|-----------:|-------------:|----------------:|
|         10 |      ~11 ms  |         ~27 ms  |
|         13 |      ~46 ms  |        ~110 ms  |
|         15 |     ~185 ms  |        ~420 ms  |

Deg-3 matches `piop-ring::mult_check` (the naive 3-table prover: same
inner loop, same cost). Deg-2 is about 2.3× faster than deg-3 at matched
n — fewer muls per pair plus one fewer fold. Cross-check in
[piop-ring/piop-ring.md](../piop-ring/piop-ring.md) §5.

### 3.2 Per-witness ratio (the relevant figure)

A single $R_2$ element packs **32 $\mathbb{Z}/2^{32}$ witnesses** under
the $\varphi_2$ embedding (the paper §2.3). At $n=15$, effective per-witness
ratio vs BB4 is **~2.25×** (see
[piop-ring/piop-ring.md](../piop-ring/piop-ring.md) §5.2) — the number
that drives prover throughput on packed instances.

### 3.3 Where the ratio comes from

Per fold-pair: 10 muls in `compute_round_poly` + 3 fold muls = **13 muls
per pair**. Summing $2^{n-i-1}$ pairs over $n$ rounds: $13\,(2^n-1)$ muls.
At $n=15$: ~426 K muls. At ~900 ns/R_2-mul that predicts ~384 ms of
pure compute, closing the gap to measured with eq-build + sampler +
overhead.

---

## 4. Commands

```sh
cargo test  -p sumcheck-ring            # 4 round-trips + 2 soundness
cargo test  -p piop-ring --test cross_check_sumcheck_ring
cargo bench -p sumcheck-ring            # n ∈ {10, 13, 15}
cargo bench -p sumcheck-field           # field-side mirror
```

---

## 5. Pointers

- Specialized deg-3 prover on the same claim (Gruen's eq-MLE
  specialization, 6 muls/pair):
  [`piop-ring/piop-ring.md`](../piop-ring/piop-ring.md) §2,
  the piop-ring `mult_check` module docstring.
- Specialized deg-2 prover ($\{0, 1, \infty\}$ form, 2 muls/pair) in the
  linear check: [`piop-ring/piop-ring.md`](../piop-ring/piop-ring.md) §3.
- Cross-check against `piop-ring` mult check:
  [`piop-ring/tests/cross_check_sumcheck_ring.rs`](../piop-ring/tests/cross_check_sumcheck_ring.rs).
