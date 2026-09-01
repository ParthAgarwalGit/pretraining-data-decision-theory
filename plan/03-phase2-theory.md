# Phase 2 — Theory

**Goal:** a precise formal setup, three theorems stated exactly, proof drafts that
follow known templates step by step, and — the part you *can* fully own — a numerical
certificate suite that checks every claimed inequality on simulated data.

**Compute:** laptop.
**Ends at:** GATE-T, where proof drafts go to a human co-author.

---

## Read this before starting

You are not going to invent a proof. What you *are* going to do, and what is
genuinely valuable:

1. Write the formal setup and assumptions with no ambiguity, so a mathematician can
   pick it up cold.
2. State each theorem so precisely that it is falsifiable.
3. Write proof *skeletons* that follow the published templates (Kaufmann, Cappé &
   Garivier 2016; Garivier & Kaufmann 2016) line by line, marking every step that is
   either (a) a direct citation, (b) a mechanical adaptation, or (c) **a genuine new
   step requiring a human**. Be honest about (c) — mislabelling a hard step as
   mechanical is the single worst failure mode in this phase.
4. Build `tests/theory/` that numerically checks each claimed inequality on thousands
   of simulated instances. A theorem that fails its numerical check is wrong, and
   finding that out now is worth more than a month of proof-writing.

**Never write "it can be shown that" or "the proof follows similarly".** Either write
the step or mark it `\needshuman{...}`.

---

## P2-01 — Formal setup and assumptions

**Branch:** `phase2/formal-setup`

Write `paper/sections/setup.tex` defining, with no hand-waving:

- Arms `k in {1..K}`, scales `s in S` (a finite or continuous set of `(N, D)` pairs),
  target `s*`.
- The scaling family `mu_k(s) = g(theta_k, s)` with `g` a known parametric family and
  `theta_k in Theta ⊂ R^p` unknown. State smoothness assumptions on `g` (continuous
  differentiability in `theta`, bounded Jacobian on `Theta`).
- The observation model `y_{k,i} = mu_k(s_i) + eps_{k,i}` with `eps` sub-Gaussian of
  known proxy variance `sigma^2(s_i)` — and note explicitly that empirically
  `sigma^2` *decreases* in `s` (P1-05 measured this), which is unusual relative to
  standard BAI and matters for the allocation.
- The cost `c(s) ∝ N D` (with `C ≈ 6ND`), total budget `C = sum_i c(s_i)`.
- **The misspecification model.** This is the novel modelling choice and needs the most
  care. Adopt: `mu_k(s) = g(theta_k, s) + h_k(s)` where `h_k` lies in a bounded
  perturbation class `H` (for example `sup_s |h_k(s)| <= eta`, or `h_k` in a Hölder
  ball). Define `sigma^2_extrap,k` as the induced squared error at `s*` of the best
  in-family approximation to `mu_k` on the design `S_fit`:
  `sigma^2_extrap,k(S_fit) = ( g(theta_k^dagger(S_fit), s*) - mu_k(s*) )^2`
  where `theta_k^dagger(S_fit)` is the population-level (noise-free) projection of
  `mu_k` onto the family under the design measure. **Note that this quantity depends on
  the design `S_fit`, not only on the arm.** That dependence is the mathematical heart
  of the paper: it is why more small-scale data does not shrink the bias, and why the
  *choice of scales* changes the bias while the *number of repeats* does not.
- The policy class: adaptive `(k, s)` selection, a stopping time `tau`, a recommendation
  `k_hat`. `delta`-correctness: `P[k_hat != k*] <= delta` for every instance in the
  class.

**Definition of done:** a setup section a stranger can read without the source PDF,
plus `docs/notation.md` mapping every symbol to the code identifier that computes it.

---

## P2-02 — Theorem 1: the extrapolation-aware error bound

**Branch:** `phase2/theorem-1`

State and prove:

> For a design `S_fit` with total compute `C`, an estimator `mu_hat_k(s*)` obtained by
> least squares within the family, and gaps `Delta_k`,
> `P[k_hat != k*] <= sum_{k != k*} exp( - Delta_k^2 / (2 (sigma^2_extrap,k + v_k(C))) )`
> where `v_k(C) -> 0` as `C -> infinity` and `sigma^2_extrap,k` does not depend on `C`
> for a fixed design shape.

Proof route (this one is genuinely tractable):

1. Sub-Gaussian concentration of the least-squares parameter estimate.
2. Delta-method propagation to `s*` through the Jacobian `J = ∂g/∂theta |_{s*}`, giving
   `v_k(C) = J^T Sigma_theta(C) J`.
3. A bias-plus-deviation decomposition of the event `{mu_hat_k(s*) >= mu_hat_{k*}(s*)}`.
4. Union bound over `k != k*`.

**Corollary 1 (consistency iff correct specification).** Winner selection is consistent
as `C -> infinity` **iff** `sigma^2_extrap,k = 0` for every `k` with `Delta_k` below the
level at which the bias can flip the ordering. State this precisely — the honest
condition is about the *ordering*, not each arm's level, and involves the pairwise
difference. **P1-06 step 4 found empirically whether the pairwise version is materially
tighter; incorporate that finding here.**

**Corollary 2 (single-scale as a degenerate case).** With the constant extrapolator,
`v = sigma^2(s_p)` and `sigma^2_extrap` is the squared *ordering* distortion of the
proxy scale. This is the formal statement of "single-scale accepts a fixed proxy gap
instead of paying an extrapolation bias" and is what makes DataDecide's result
intelligible.

**Numerical certificate** (`tests/theory/test_theorem1.py`): simulate 5000 random
instances with known `theta`, known perturbation `h`, known noise; check the bound
holds in every one, and record the tightness distribution. A single violation fails the
test suite.

---

## P2-03 — Theorem 2: lower bound and the impossibility phase transition

**Branch:** `phase2/theorem-2`

Two parts.

**Part A — the change-of-measure lower bound.** Following Kaufmann, Cappé & Garivier
(2016): for any `delta`-correct policy,
`E[C] >= kl(delta, 1-delta) * T*(instance)`
where `T*` solves a **compute-weighted** optimal-design program over scales:

```
T*^{-1} = sup over allocations w on (arm, scale) pairs, with sum_i w_i c(s_i) = 1,
          of  min over k != k*  of  <information that w carries about the sign of
                                     mu_{k*}(s*) - mu_k(s*)>
```

The novelty relative to classical BAI is that information about the *target-scale*
difference arrives only through the parametric link, so the per-pull information is
`(J_k^T I_k(w)^{-1} J_k)^{-1}`-shaped rather than a per-arm KL. Write the program
explicitly; it is a concave max-min problem and therefore both provable and solvable
numerically (which P3-02 does).

**Part B — the impossibility result.** This is the part with no classical analogue and
must be foregrounded in the paper (the source document flags "just BAI applied" as the
main reviewer objection).

> If the perturbation class `H` is rich enough that two instances exist which (i) agree
> on all observable scales `s < s*` up to any achievable precision, and (ii) have
> opposite orderings at `s*`, then no policy using only scales `s < s*` is
> `delta`-correct for `delta < 1/2`, at any finite compute.

State the structural condition on `H` and `S` that separates the solvable regime from
the impossible one — this is the **phase transition**. Give both directions: a
sufficient condition for solvability, and a construction witnessing impossibility.

**Numerical certificate**: construct explicit instance pairs in the impossible regime
and verify that no estimator in a large class separates them; verify `T*` computed
numerically matches the achieved sample complexity of the P3 algorithm in the solvable
regime.

**Honest note for the draft:** Part B's construction is where the real mathematical
content is. Mark it `\needshuman` and give the human a fully worked *candidate*
construction plus the numerical evidence that it works.

---

## P2-04 — Theorem 3: identifiability and minimax rate

**Branch:** `phase2/theorem-3`

> `theta_k` is identifiable for extrapolation to `s*` iff the design spans enough
> curvature: informally, at least `p` (and in the Chinchilla case at least 3)
> well-separated scales with `J` full rank. Give the minimax rate for `mu_hat_k(s*)`
> over the assumed class.

Deliverables:

1. The precise rank/spacing condition, with the degenerate cases spelled out (what
   exactly fails with 2 scales, or with 3 clustered scales).
2. A minimax lower bound over the Hölder or parametric class via Le Cam's two-point
   method or Fano — pick one, justify it, and follow the standard template.
3. A matching (up to constants) upper bound achieved by least squares.
4. **The design-dependence corollary:** since `sigma^2_extrap` depends on `S_fit`,
   there is an optimal *spacing* of scales that trades identifiability against cost.
   Derive it, at least for the power-law family. This is directly actionable advice for
   practitioners and is worth foregrounding.

**Numerical certificate:** vary the number and spacing of scales in simulation; confirm
the estimator error follows the predicted rate and blows up exactly where the rank
condition fails. This also directly feeds ablation P5-02.

---

## P2-05 — Theorem 4: the algorithm's correctness

**Branch:** `phase2/theorem-4`

State the guarantee for the Extrapolation-Track-and-Stop algorithm that P3 implements:

- **`delta`-correctness** of the stopping rule (a Chernoff / GLR-style statistic with an
  anytime-valid threshold, adapted so the bias floor enters the threshold).
- **Asymptotic optimality** in the solvable regime: `E[C]/log(1/delta) -> T*` as
  `delta -> 0`.
- **Graceful abstention** in the impossible regime: the algorithm must *detect* it and
  output "cannot certify; fall back to single-scale at scale `s_rec`" rather than
  running forever. Define the abstention rule and prove it triggers with high
  probability in the impossible regime. **This is the practically valuable part** — it
  is the "diagnostic" contribution the source document promises.

The stopping threshold with a bias floor is subtle: with `sigma^2_extrap > 0` the GLR
statistic does not diverge, so a naive Track-and-Stop never stops. Handle this by
either (a) certifying only up to a bias-inflated confidence, or (b) assuming a known or
estimated upper bound `eta` on the perturbation and certifying `delta`-correctness
relative to `eta`. Route (b) is more honest and more useful; state clearly that the
guarantee is conditional on `eta`, and make estimating `eta` an explicit empirical
step (P1-06 provides the estimator).

---

## GATE-T — Hand proofs to a human

**Stop. Post to the PI:**

1. `paper/sections/` with setup and all four theorem statements.
2. A table of every proof step marked `\needshuman`, with a one-line description of the
   difficulty and your assessment of how hard it looks.
3. The numerical certificate report: which claimed inequalities passed on how many
   simulated instances, and any that failed.
4. Your honest assessment of which theorem is most at risk.

**Ask:** "Who is verifying these proofs, and by when? I can continue with P3 (algorithm
implementation) in parallel while that happens." Then continue to P3 — do not idle.

---

## P2-06 — Revise theory against Phase 1 evidence

**Branch:** `phase2/revise-from-p1`

After the P1 memo exists, revisit every assumption against measured reality:

- Is the sub-Gaussian noise model consistent with the P1-05 noise estimates?
- Does the perturbation class `H` you assumed actually contain the deviations observed
  in DataDecide? Measure the empirical `h_k(s)` residuals and check.
- Is `sigma^2_extrap` really flat in compute, as Theorem 1 assumes? P1-06 measured the
  curve — if it drifts, the theorem needs an amended statement.
- Do the observed noise levels *decrease* with scale, and does the `T*` program account
  for that?

Amend the theorems rather than the data. Log every amendment in `docs/decisions.md`.

---

## P2-07 — Related-theory positioning

**Branch:** `phase2/positioning`

Write `paper/sections/related.tex` that pre-empts the three objections the source
document anticipates:

1. *"Scaling laws are misspecified, so the parametric assumption is unrealistic."*
   Answer: misspecification is the **subject**, not an assumption — Theorem 1's bias
   term and Theorem 2's impossibility regime are about exactly that.
2. *"Real labs use continuous mixtures, not `K` arms."* Answer: state the extension to
   a continuous simplex via linear or GP-bandit BAI as an explicit generalisation, with
   the one-paragraph statement of what changes.
3. *"DataDecide already answered this empirically."* Answer: DataDecide *measured*
   decision accuracy; it gave no estimator, no confidence guarantee, no lower bound,
   and no allocation rule.

Also position precisely against the near-miss papers found in P0-07.

---

## P2-08 — Integrate the verified proofs

**Branch:** `phase2/integrate-proofs`

After the human returns verified or corrected proofs: integrate them, re-run the
numerical certificates against the *final* theorem statements (the statements may have
changed during verification — the certificates must track them), and make sure every
constant in the paper matches a constant in the code.

---

## P2-09 — Theory appendix

**Branch:** `phase2/appendix`

Full proofs in `paper/appendix/`, notation table, a worked example instance carried
through all four theorems, and the numerical certificate report as an appendix table so
reviewers can see the theorems were empirically stress-tested.
