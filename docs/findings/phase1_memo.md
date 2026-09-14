# Phase 1 memo: what the DataDecide re-analysis found, and which framing it supports

**To:** PI **From:** Agent **Re:** Phase 1 results (P1-01 through P1-11), and the framing
question P1-12 asks to be settled before Phase 2 theory drafting hardens.

This is not a formal GATE, but please read it before Phase 2's theorems get written
around a specific framing — reversing that later is expensive.

## What we found

- **P1-01/P1-02 — the analysis frame, and how often there's a real winner to find.**
  Full (recipe, params, seed, task) coverage: 69,300/69,300 cells, no holes. At the 1B
  target scale, **9 of 11 `macro_avg` tasks are statistically ambiguous under
  `primary_metric`** (6/11 under `acc_per_char`) — even DataDecide's own
  `olmes_10_macro_avg` composite has only a ~0.05pp margin between its top two recipes,
  well within seed noise. Most of the time, there is no single correct answer to recover.
- **P1-03/P1-04 — reproduction.** Single-scale baseline reproduces at 76.3% decision
  accuracy at 150M (79.6% excluding ties) against DataDecide's published ~80%. The
  headline negative result reproduces cleanly: **0 of 18 (fitter, design) combinations
  beat the single-scale frontier at matched compute**, across 6 scaling-law fitters and 3
  designs.
- **P1-05 — noise floor.** Seed variance does **not** shrink monotonically with model
  scale (median ~2.9e-5 to ~5.9e-5, 4M through 1B, no clear trend) — contradicts the
  naive "bigger models are less noisy" intuition the plan started from.
- **P1-06 — the core decomposition.** `sigma2_extrap` (extrapolation bias-squared-plus-
  excess-variance) is real, non-trivial, and clearly **task-dependent**: a ~14x spread
  across tasks for the same fitter and design, comparable to or larger than the plain
  estimation variance `v` almost everywhere. It is not a rounding error. But the ratio
  `sigma2_extrap / v` moves **opposite** to the plan's own predicted signature: it *falls*
  as designs grow toward the target (e.g. PowerLawN: 9.6 -> 7.2 -> 5.2), because
  `sigma2_extrap` shrinks faster than `v` does, not slower.
- **P1-07/P1-08 — does the theory explain the ceiling?** The plug-in selection-error
  bound is **never violated** (396/396 cells) and the pairwise form is reliably tighter
  than the marginal form, exactly as predicted (median tightness ratio ~20.6 vs. ~24.1) —
  clean confirmation of the theory's qualitative structure. But taken literally as a
  number, the bound is **vacuous**: it is a union bound over ~24 mostly-near-tied
  per-task comparisons (a direct consequence of P1-02's ambiguity finding), so it exceeds
  1 in all 396 cells and the resulting predicted accuracy clips to 0.0% everywhere. A
  `sigma2_extrap = 0` counterfactual is more informative but narrow: even with bias
  removed entirely, only **1 of 15** (fitter, design) pairs beat single-scale's own
  bias-free counterfactual.
- **P1-09 — rank reversals.** Naive reversal rate across the 14-size ladder was 61.7%,
  but that used 14 uncorrected simultaneous per-pair tests; Bonferroni-corrected, it's
  **15.2%** (500/3,300 pairs) — still real and non-trivial, just not the inflated number.
- **P1-10 — secondary ladder (Pythia).** Scoped to Pythia only, per the plan's own escape
  clause. Only 2 of 6 fitters can even fit (3 usable proxy sizes below target), and the
  P1-06 ratio-vs-compute finding **does not clearly replicate in either direction**
  (131.0 vs. 137.6 median ratio across the 2 usable designs, ~5% apart) — reported
  inconclusive rather than forced into a verdict.
- **P1-11 — figures.** F1-F5 generated and visually verified (not just checked for
  exceptions); several legibility bugs caught and fixed this way, documented in
  `docs/decisions.md`.

## Which framing does the evidence support?

P1-12 poses this as a choice between two framings: **(A)** `sigma2_extrap` is large,
so the extrapolation-error mechanism explains the "extrapolation doesn't win" puzzle; or
**(B)** `sigma2_extrap` is small, so the paper should pivot to an allocation-speedup
story instead (extrapolation as a cheaper route to *similar* accuracy, not *better*
accuracy).

**The evidence does not support (B).** `sigma2_extrap` is not small: P1-06 shows it is
routinely comparable to or larger than `v` at the smallest design, with a 14x spread
across tasks. There is no task or fitter where it is negligible.

**The evidence supports (A), but only qualitatively, with an important caveat the paper
needs to be honest about.** `sigma2_extrap` is real and plausibly *a* cause of the
observed gap — that part of framing (A) holds up. But its precise theoretical
consequence, the plug-in bound, is too loose to be the paper's quantitative payoff: it's
never wrong (P1-07), but it's also uninformative at these gap sizes (P1-08). More
tellingly, the `sigma2_extrap = 0` counterfactual shows that *even completely removing
bias* only flips the outcome in 1 of 15 cases — meaning `sigma2_extrap` alone cannot be
the dominant explanation for *why* extrapolation loses 18/18 (now 17/18) matched-compute
comparisons. Something else is doing most of that work, and P1-02 already identified the
likely candidate: with 9 of 11 tasks statistically ambiguous at the target scale, "beat
single-scale" is a demanding bar for *any* method to clear, extrapolation included,
simply because there is usually no stable winner to correctly recover in the first
place. P1-09's Bonferroni-corrected 15.2% reversal rate is a second, independent line of
evidence for the same underlying picture: the ranking that "matters" at the target scale
is often not stable even across seeds of the *same* method.

**Recommendation:** neither framing (A) nor (B) as originally posed is the right
headline by itself. The best-supported story is closer to an **impossibility/ambiguity
regime**: decision-relevant differences between recipes are frequently smaller than the
noise floor at practical scales, for reasons that are only partly about extrapolation
quality (`sigma2_extrap`, real but not sufficient) and mostly about the underlying
problem being genuinely close to a tie (P1-02, P1-09). Phase 2's theorems should treat
the bias/variance decomposition (Claim 1) and the near-tie/ambiguity structure (Claim 2)
as **co-equal**, not lead with the plug-in bound's numeric tightness as if it were the
main quantitative result — it isn't one, and presenting it that way would overstate what
Phase 1 actually shows. This matches the conclusion already reached in
`docs/findings/p1_06.md` after P1-07/P1-08 landed; P1-09/P1-10 since then are consistent
with it, not in tension with it.

## Threats to validity

- **The union bound is a proof-technique artifact, not necessarily a fact about the
  world.** P1-08's "vacuous" finding is partly a consequence of using an (easy-to-derive,
  conservative) union bound over ~24 pairwise comparisons per task. A tighter
  simultaneous-inference technique might tell a quantitatively different story without
  changing the qualitative one — worth flagging in Phase 2 rather than treating P1-08's
  0.0% as a deep fact about extrapolation.
- **P1-10 is underpowered.** Only 2 usable design points and 1 fitter able to run at
  both means the "does not replicate" verdict is weak evidence of absence, not strong
  evidence of a real difference from DataDecide. A second secondary ladder (OLMo 2, not
  attempted, per the plan's own scope cap) would strengthen or weaken this considerably.
- **The 3 fixed designs (`<=150M`/`<=300M`/`<=530M`) are a single, fairly coarse choice.**
  All of P1-04/06/07/08's headline numbers are computed at exactly these 3 points; P5-02
  (number and spacing of scales) is explicitly planned to stress-test this later, but
  until then, the reported trends (e.g. the ratio-vs-compute direction) rest on 3 points,
  not a dense curve.
- **`primary_metric` vs. `acc_per_char` disagree on how ambiguous the target scale is**
  (9/11 vs. 6/11 tasks). The paper's headline ambiguity number should say which metric it
  means, and the choice should be made deliberately, not implicitly by whichever was
  computed first.
- **Bootstrap scheme.** All P1-06/07 headline numbers use the `seed_bootstrap` scheme by
  convention; the parametric scheme is computed alongside it in the same results files
  but not the one quoted here. A disagreement between the two would be worth checking
  before Phase 2 leans on any specific number.

## Bottom line

Ship the bias/variance decomposition and the near-tie/ambiguity finding together as the
paper's two headline empirical claims. Do not lead with the plug-in bound's quantitative
tightness — it's real and never wrong, but not informative at the gap sizes this data
actually has. Phase 2's theorems should be built to support *both* claims, not just the
bound.
