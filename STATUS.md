# STATUS

Update this file at the end of **every** session (protocol step 11).

Last updated: 2026-09-11 | Session: 13

States: `TODO` | `IN PROGRESS` | `IN REVIEW` | `DONE` | `BLOCKED` | `DROPPED`

## Open GATEs
- **GATE-T** (opened 2026-09-11, after P2-05): proof drafts (setup + all 4 theorems, `paper/sections/`) ready for a human co-author's review. 4 `\needshuman` steps flagged, none faked as mechanical; numerical certificates (18 tests across `tests/theory/`) all pass, 0 violations, after catching and fixing 2 real bugs along the way (Theorem 1's originally-stated bound formula, Theorem 4's certificate round-cap). Full summary posted to the PI in-session; per `plan/03-phase2-theory.md`'s own instruction, not idling on this -- continuing directly to Phase 3 (algorithm implementation) in parallel, per protocol Rule 4 / `plan/09-review-gates.md` §7 (Phase 2 and Phase 3 are independent). No further Phase 2 theory PRs (P2-06 onward) until this resolves.
  - **Update 2026-09-11 (P3-04):** the reviewer should also weigh a new empirical finding directly relevant to Theorem 4 -- `extrapolation_track_and_stop`'s bare-minimum forced-exploration warm-up can cause false Certified decisions in practice (concrete instance found and reproduced), traced to Theorem 4's proof sketch's own `\needshuman`-flagged "nonlinear-g M-estimator concentration" gap between the asymptotic guarantee and finite-sample behavior. Substantially mitigated (not proven eliminated) with more warm-up replication. Full account: `docs/decisions.md` 2026-09-11 P3-04 entry, PR #30.

## Blocked
- (none)

---

## Phase 0 — Setup

| Task | Title | State | PR | Notes |
|---|---|---|---|---|
| P0-01 | Verify and record the toolchain | DONE | [#1](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/1) | folded into docs/environment.md |
| P0-02 | Propose repo names; create GitHub repo and skeleton | DONE | [#1](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/1) | branch protection unavailable on free plan — see docs/decisions.md |
| P0-03 | Reproducible Python environment | DONE | [#2](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/2) | `make` was missing on Windows, installed GNU Make via winget; ruff scoped away from plan/ markdown; verified on a genuine fresh clone |
| P0-04 | Provenance helper and results contract | DONE | [#3](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/3) | 21 tests; caught a real bug in the numpy encoder (array vs scalar) via the numpy-array test case; manually verified `make check` fails loudly on a hand-written result |
| P0-05 | CI incl. fabrication guard | DONE | [#4](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/4) | fabrication guard is advisory until P6-01 (paper doesn't exist yet); found and NUMBER-OK-annotated 4 real toolchain-version numbers in docs/environment.md; verified end-to-end with a deliberately fabricated README number; also fixed a CI cache-key race and an actions/checkout deprecation |
| P0-06 | Acquire and cache DataDecide | DONE | [#5](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/5) | data-recipes repo is 19.2TB (not a metadata table) -- deliberately not bulk-downloaded, only its README's recipe table is parsed; found 2 bonus tables (macro_avg, scaling_law_fit incl. DataDecide's own decision_acc); 25 recipes/14 sizes confirmed 3 independent ways; seeds are 3-per-size everywhere but relabeled at 1B (large aux vs small aux) -- resolves a risk P1-01 flagged; see docs/decisions.md for full findings |
| P0-07 | Novelty sweep #1 | DONE | [#6](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/6) | verdict: no direct prior work found; 3 named near-misses + 1 found independently (arXiv:2604.22753), all characterized; 2 real citation errors fixed in the source doc's own reference list (Muennighoff year, "Hu et al." -> Lourie et al.); also fixed 2 real bugs in the fabrication guard tool itself (arXiv-ID false positives, a Windows console encoding crash); verified target-venue deadlines -- see docs/related_work.md |
| P0-08 | Create the HF dataset repo shell | DONE | [#7](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/7) | PI confirmed via AskUserQuestion; repo live at [huggingface.co/datasets/Parth4105/pdt-datadecide-analysis](https://huggingface.co/datasets/Parth4105/pdt-datadecide-analysis) (private), card pushed via the real `push_results()` function -- doubles as a live end-to-end test of that code path |
| **GATE-0** | **End of Phase 0 — PI approval** | **DONE** | | PI approved 2026-09-02: merge #6+#7, target **COLM 2027**, proceed to Phase 1. See docs/decisions.md. |

## Phase 1 — DataDecide re-analysis

| Task | Title | State | PR | Notes |
|---|---|---|---|---|
| P1-01 | Canonical analysis frame | DONE | [#9](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/9) | full (recipe, params, seed, task) coverage matrix is 100% complete -- 69,300/69,300 cells, 0 holes; gained a `source`/`metrics` parameterization in P1-02 after finding eval_results was the wrong granularity -- see docs/decisions.md |
| P1-02 | Ground truth at target scale, and gaps | DONE | [#10](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/10) | **headline finding: 9/11 macro_avg tasks still ambiguous under primary_metric at 1B (6/11 under acc_per_char)** -- even DataDecide's own olmes_10_macro_avg has a ~0.05pp winning margin, within seed noise; found eval_results was the wrong task granularity (fixed frame.py), a cache-key bug that silently poisoned P1-01's own results (fixed), and non-deterministic tie-breaking in compute_ground_truth (fixed) -- all 3 caught by diffing clean-tree reruns against each other, see docs/decisions.md |
| P1-03 | Reproduce DataDecide single-scale baseline | DONE | [#11](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/11) | **REPRODUCED: 76.3% at 150M (79.6% excluding ties) vs published ~80%** -- all 4 sensitivity variants cluster 73.3%-77.5%; accuracy-vs-size curve smooth and monotonic-ish (53% at 4M -> 85% at 530M); pipeline confirmed correct -- see PR for full table |
| P1-04 | Scaling-law fitters; reproduce "extrapolation does not win" | DONE | [#12](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/12) | **REPRODUCED: 0/12 evaluable (fitter, design) combinations beat the single-scale frontier at matched compute (6 more unassessed, out of the single-scale range -- not losses)** -- 6 fitters (Constant/PowerLawN/PowerLawC/ChinchillaND/TwoStepLadder/LogLinear) x 3 designs (<=150M/<=300M/<=530M); ConstantExtrapolator's per-design accuracy exactly reproduces the matching P1-03 single-scale point (hard consistency check, enforced in code); the <=530M design's own compute exceeds the largest single-scale comparison point (750M) so that design's matched-compute comparison is out of range, not a failure (reported as unassessed) -- see docs/decisions.md and PR for full table |
| **GATE-1** | **Reproduction checkpoint — PI approval** | **DONE** | | PI approved 2026-09-03: "continue to P1-05". PRs #10-#12 still open/unmerged at time of approval. |
| P1-05 | Noise-floor estimation | DONE | [#13](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/13) | 3 variance components (seed/checkpoint/eval) across all 3850 (recipe,size,task) cells; **finding: sigma2_seed does NOT shrink monotonically with scale** (median ranges ~2.9e-5 to ~5.9e-5, 4M through 1B, no clear trend) -- contradicts the plan's "as expected" framing; sigma2_target(k,t) at 1B defined as sigma2_seed/n_seeds (matches P1-02's actual seed-averaging estimator exactly); found+fixed a real float-order reproducibility bug in group_by().agg(.var()) (~1e-14 relative, not a logic error) -- see docs/decisions.md |
| _(follow-up)_ | `group_by().agg()` determinism audit across `src/pdt/` (task_2d6c6192, flagged by P1-05 Decision 5) | IN REVIEW | [#14](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/14) | **confirmed the same float-order bug affected 3 functions, not just 1**: `decision_accuracy.recipe_means()` directly (24/24 fresh runs on eval_results disagreed); `ground_truth.compute_ground_truth()` and `recipe_trajectories()` didn't reproduce under a fresh warm-cache audit but **did** demonstrably corrupt the already-committed P1-02/03/04 numbers (diffed old vs. clean-fixed regeneration: 6,674/11/61 diffs respectively, incl. one real `is_ambiguous` flip in p1_02) -- the original runs hit the higher-risk first-ever cold-cache read path a same-day audit doesn't exercise; all headline figures (P1-02 9/11 ambiguous, P1-03 76.3%, P1-04 0/18) unchanged; fixed all 3, regenerated + verified deterministic (2x clean runs, byte-identical) -- see docs/decisions.md |
| P1-06 | Bias/variance decomposition (core result) | DONE | [#16](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/16) | full B=200 x 2-scheme x 6-fitter x 3-design x 11-task bootstrap, ~2M fits, 0 failures; **finding: sigma2_extrap/v ratio DECREASES with compute for every single fitter** (e.g. PowerLawN: 9.6 -> 7.2 -> 5.2 across the 3 designs) -- contradicts the plan's own stated "signature prediction" that this ratio should grow as v shrinks and sigma2_extrap stays flat; empirically both shrink as design compute grows, but sigma2_extrap shrinks faster, not slower, than v -- a real, surprising result to carry into P1-07/P1-08 and docs/findings/p1_06.md, not smoothed over; see docs/decisions.md for full numbers and the compute/BLAS/file-size engineering decisions |
| P1-07 | Plug-in bound and empirical coverage | DONE | [#17](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/17) | **no bound violations found (ratio >=1 in all 396 cells)**; pairwise form confirmed tighter than marginal as the plan predicted (min 1.71/median 20.6 vs min 4.44/median 24.1) -- reassuring result alongside P1-06's more surprising ratio-vs-compute finding; see docs/decisions.md |
| P1-08 | Does the bound predict the 80% ceiling? | DONE | [#18](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/18) | **taken literally the bound is vacuous: bound_pairwise > 1 in all 396 cells (union bound over ~24 mostly-tied comparisons/task, traces to P1-02's 9/11-ambiguous finding), so predicted accuracy clips to 0.0% everywhere** -- sigma2_extrap=0 counterfactual is more informative but narrow: only 1/15 (fitter,design) pairs beat single-scale's OWN bias-free counterfactual (LogLinear @ <=530M); docs/findings/p1_06.md written (deferred from PR #16); see docs/decisions.md |
| P1-09 | Rank-reversal census | DONE | [#15](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/15) | 3300 pairs x 11 tasks classified stable/reversing/within_noise across the 14-size ladder; **naive threshold gave an inflated 61.7% reversing rate -- caught a multiple-comparisons issue (14 per-pair tests, uncorrected) and fixed with a Bonferroni correction: primary figure is 15.2% reversing** (500/3300); still a real, meaningful fraction supporting the impossibility-regime framing, just not the dramatically larger uncorrected number -- see docs/decisions.md |
| P1-10 | Secondary ladders: Pythia and OLMo 2 | DONE | [#19](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/19) | scoped to Pythia only (OLMo 2 not attempted, per the plan's own "cap it at one ladder" clause); scope further forced by Pythia's own data: only ConstantExtrapolator+LogLinear can fit (3 sizes below target), parametric-bootstrap only (1 seed/checkpoint published), 5 of 11 headline tasks have a real counterpart; **P1-06's ratio-vs-compute finding does not clearly replicate either direction** (131.0 vs 137.6 median ratio across the 2 usable designs, ~5% apart) -- reported inconclusive; also found a real naming irregularity (no plain pythia-1b directory published) requiring retarget to 1.4b; see docs/decisions.md |
| P1-11 | Phase 1 figures (F1-F5) | DONE | [#20](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/20) | all 5 generated by `src/pdt/viz/` via `make figures`; every figure rendered to a 300-DPI PNG and visually inspected (not just checked for exceptions), which caught 5 real bugs: a clipped title (global rcParams), a legend cropped at the page edge despite `bbox_inches="tight"`, a 3-column legend overflowing the figure width, an illegible crowded log-axis, ambiguous 4-char-truncated fitter labels (PowerLawC/PowerLawN both read "Powe"), and two in-plot legends sitting on real data -- see docs/decisions.md |
| P1-12 | Publish derived tables; Phase 1 memo | DONE | [#21](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/21) | memo written (`docs/findings/phase1_memo.md`): evidence supports neither literal framing cleanly -- recommends the bias/variance decomposition and P1-02's ambiguity finding as co-equal headline claims; **PI confirmed via AskUserQuestion, HF push complete**: all 11 `results/*.json` files + an updated dataset card live at [huggingface.co/datasets/Parth4105/pdt-datadecide-analysis](https://huggingface.co/datasets/Parth4105/pdt-datadecide-analysis) (private) -- pushed a filtered copy excluding `results/README.md`, which would have silently clobbered the dataset card (see docs/decisions.md) |

## Phase 2 — Theory

| Task | Title | State | PR | Notes |
|---|---|---|---|---|
| P2-01 | Formal setup and assumptions | DONE | [#22](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/22) | `paper/sections/setup.tex` + `docs/notation.md`; every symbol mapped to a real code identifier, not described from memory; corrected the plan's own draft assumption that noise decreases with scale (P1-05 found no such trend) rather than silently carrying a falsified premise into later theorems |
| P2-02 | Theorem 1: extrapolation-aware error bound | DONE | [#23](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/23) | **numerical certificate caught a real bug in the plan's own draft formula**: the literally-stated additive bound is not a valid worst-case bound (497/5000 simulated instances violated it, several severely) -- corrected to a gap-reduction form including both compared arms' bias/variance; corrected certificate: 0 violations across all 5000 instances (1801 MC-checkable at a 20000-trial budget, tightness ratio min 1.0/median 7.1); P1-07/08's already-reported numbers are unaffected (their real-data bound values were always vacuous >=1 either way) -- see docs/decisions.md |
| P2-03 | Theorem 2: lower bound and impossibility | DONE | [#24](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/24) | Part A (change-of-measure lower bound, Fisher-information-shaped per-pull rate) proved mechanically; Part B (impossibility) `\needshuman`-flagged per the plan with a fully worked, numerically-verified candidate construction -- sup-norm class: gap-independent sufficient condition; Holder-alpha class: genuine phase transition in the gap via a bump function that provably saturates the Holder budget exactly; caught and fixed a random-sampling gotcha (naive verification under-reported the true modulus by 4-15%, a measure-zero tight point random sampling misses) -- see docs/decisions.md |
| P2-04 | Theorem 3: identifiability and minimax rate | DONE | [#25](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/25) | rank/spacing condition with rank-deficiency vs. ill-conditioning distinguished explicitly; minimax rate proved by direct reuse of Theorem 2's Fisher-information machinery; **worked power-law spacing example found a genuine, verified, non-monotone (U-shaped) result** -- optimal second-scale placement is neither maximally clustered nor maximally close to the target, contradicting the naive "spread scales out" intuition -- see docs/decisions.md |
| P2-05 | Theorem 4: algorithm correctness | DONE | [#26](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/26) | algorithm defined precisely (Certified/Abstain stopping rule, structural fix for the "never stops with a bias floor" subtlety); delta-correctness proved mechanically from Theorem 1; **graceful abstention proved as a direct corollary of Theorem 2 Part B's own construction**, not a separate argument -- ties the whole theory section together; asymptotic optimality needshuman-flagged (mechanical in structure, not written out line-by-line); certificate caught and fixed a real round-cap bug before trusting "0 violations" -- see docs/decisions.md |
| **GATE-T** | **Hand proofs to a human co-author** | **OPEN** | | opened 2026-09-11; see "Open GATEs" above and the in-session summary posted to the PI |
| P2-06 | Revise theory against Phase 1 evidence | TODO | | |
| P2-07 | Related-theory positioning | TODO | | |
| P2-08 | Integrate the verified proofs | TODO | | |
| P2-09 | Theory appendix | TODO | | |

## Phase 3 — Algorithm

| Task | Title | State | PR | Notes |
|---|---|---|---|---|
| P3-01 | Simulator and oracle interface | DONE | [#27](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/27) | `PullOracle` protocol + `SyntheticOracle` (calibrated to real P1-05/P1-06 numbers, not guessed) + `DataDecideOracle` (deterministic table lookup, seed labels discovered per (recipe,scale) not hardcoded) + `LiveTrainingOracle` stub; **verified `cost()` (6ND) against DataDecide's own compute column: ratio min 0.85/median 0.999/max 1.10** -- kept 6ND for consistency with the rest of the project rather than switching; 21 tests, 100% coverage -- see docs/decisions.md |
| P3-02 | Solve the optimal-allocation program | DONE | [#28](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/28) | Projected subgradient ascent with an exact closed-form gradient, after two scipy.optimize routes (SLSQP, trust-constr) proved unreliable near rank-deficient Fisher information; **known, documented limitation: reliably reaches a locally max-min-consistent point, not verified globally optimal** (brute_force_allocation, vectorized for practicality, found a better point on a real instance) -- shipped honestly rather than hidden, `n_restarts` is the mitigation; 19 tests, 100% coverage -- see docs/decisions.md |
| P3-03 | Extrapolation-Track-and-Stop + baselines | DONE | [#29](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/29) | Tracking/stopping/abstention rule per theorem4_algorithm.tex, plus 4 baselines sharing one `SelectionResult` type; **two real theorem-vs-implementation gaps found and resolved, not hidden**: T*(nu) has no term for k*'s own allocation (reserved-share heuristic), and a post-fit Jacobian-rank identifiability check was too numerically fragile (switched to a design-level distinct-scale-count check); **a `"certified"` outcome is a delta-level claim only under stated assumptions** (known sub-Gaussian noise `sigma2`, bias <= `eta`, prediction linear in the data -- exact for LogLinear, first-order otherwise -- and a design independent of the noise, exact only at the non-adaptive warm-up check; simultaneous over rounds and ordered arm pairs), replacing the residual-variance (HC0) + Student-t rule that second-round review showed certified the wrong arm 49% of the time on a high-leverage design; `variance_mode="hc0_heuristic"` is kept as an explicitly heuristic `"recommended"` outcome -- see docs/decisions.md |
| P3-04 | Simulation study | DONE (reduced pilot) | [#30](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/30) | Literal 108-cell/200-run grid not feasible on a laptop in-session; ran a real 8-cell/15-run pilot instead, stated honestly -- **claim 1 holds** (0/15 wrong in both well-specified cells), **claim 2 not meaningfully testable at this scale** (compute-to-stop was warm-up-dominated, identical across delta), **claim 3 mostly holds** (80-93% abstention on rank-reversal instances, baselines usually wrong) **with an open residual-uncertainty caveat**: found a real finite-sample gap where minimal forced-exploration warm-up can cause false certification (the exact `\needshuman`-flagged nonlinear-M-estimator caveat in theorem4_algorithm.tex, now with a concrete instance), substantially mitigated via a larger `min_pulls_per_pair` but not proven eliminated -- see docs/decisions.md |
| P3-05 | Offline replay on DataDecide (free real-data result) | DONE (open question) | [#31](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/31) | Guarded-oracle leakage guard delivered and tested (100% coverage); **headline finding: ETS hit its round cap on all 4 real tasks tested (0% genuine abstention, 100% round-cap exhaustion)** -- never reached a delta-correct certification OR a genuine Theorem-4 abstention on real data within this pilot's affordable compute; reported honestly per the plan's own "if it abstains everywhere, say so" contingency (adapted: this is a different, arguably more concerning failure mode). **P3-05's core question (does ETS deliver real compute savings on real DataDecide data) remains open** -- needs substantially more compute for the adaptive algorithm than this session's laptop budget allowed. Baselines ran to real, bootstrapped completion. Also found and fixed 2 real bugs (uniform_allocation recipe starvation; ETS exhausting DataDecide's finite replicate pool) -- see docs/decisions.md |
| P3-06 | Sensitivity to the bias-floor estimate eta | DONE (partial) | [#32](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/32) | **Clean, unambiguous headline result: under-estimating eta to 0 causes confident, 100%-wrong certifications (20/20 trials)** -- exactly the danger the plan asks to demonstrate, and this specific result is not compute-limited (certify is unambiguous). **The safe side (eta >= 0.25x true bias, including 3x) hit the same round-cap-exhaustion wall as P3-04/P3-05** -- 0% genuine abstention at every non-zero multiplier tested, a third independent confirmation of the same systemic compute-budget finding. Practical plug-in (residual RMSE + small-sample correction) delivered and tested; found to itself under-estimate the true bias here (a real, structural limitation of residual-based estimators against an invisible-on-fit-scales bump), though it did not falsely certify in this instance. See docs/decisions.md |
| P3-07 | Phase 3 figures (F6-F8) | DONE | [#33](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory/pull/33) | F6/F7/F8 delivered, `build_all.py` regenerates all 8 figures. **F7's "as delta varies" corrected**: solve_allocation's optimal shape is provably delta-independent (Track-and-Stop, Garivier & Kaufmann 2016) -- swept gap regimes instead, delta-dependence shown separately as `T*log(1/delta)`. F6/F8 show P3-05/P3-06's honestly-reported limitations directly in the figure (ETS's round-cap-exhausted bars/points drawn distinctly, never presented as clean results) rather than smoothing them over. New cheap experiment for F8's baseline half (no adaptive loop, seconds not minutes) shows a clean crossing point where baselines flip from 100% to 0% accuracy. See docs/decisions.md |
| P3-08 | Reference implementation polish | TODO | | |

## Phase 4 — Training runs

| Task | Title | State | PR | Notes |
|---|---|---|---|---|
| P4-01 | Training stack selection and smoke test | TODO | | laptop only |
| P4-02 | Recipe design and data preparation plan | TODO | | pre-register the reversal |
| **GATE-C** | **COMPUTE REQUEST — PI picks a tier** | TODO | | blocks all below |
| P4-03 | Cluster setup and reproducibility harness | TODO | | |
| P4-04 | Calibration runs against DataDecide | TODO | | |
| **GATE-D** | **First real run review — go/no-go** | TODO | | |
| P4-05 | Ladder runs for the new recipes | TODO | | smallest sizes first |
| P4-06 | Target-scale ground-truth runs | TODO | | seal from decision code |
| P4-07 | Live algorithm run | TODO | | |
| P4-08 | Release checkpoints and configs | TODO | | needs PI approval to push |
| P4-09 | Phase 4 analysis and figures (F9-F10) | TODO | | |

## Phase 5 — Ablations

| Task | Title | State | PR | Notes |
|---|---|---|---|---|
| P5-01 | Correct vs misspecified scaling forms | TODO | | |
| P5-02 | Number and spacing of scales | TODO | | |
| P5-03 | Continuous proxies vs discrete accuracy | TODO | | |
| P5-04 | Rank-reversal stress test | TODO | | |
| P5-05 | Robustness of the analysis pipeline | TODO | | |
| P5-06 | Target-scale generalisation | TODO | | |
| P5-07 | Continuous mixture extension (optional) | TODO | | cut if time-short |
| P5-08 | Live confirmation of ablations | TODO | | needs Phase 4 |
| P5-09 | Cross-suite agreement | TODO | | needs Phase 4 |
| P5-10 | Cost-model accuracy | TODO | | needs Phase 4 |

## Phase 6 — Paper and release

| Task | Title | State | PR | Notes |
|---|---|---|---|---|
| P6-01 | Paper skeleton; fabrication guard goes hard | TODO | | |
| P6-02 | Narrative outline (PI signs off on outline) | TODO | | framing decision |
| P6-03 | Write the theory sections | TODO | | |
| P6-04 | Write the experiment sections | TODO | | |
| P6-05 | Write intro, abstract, conclusion | TODO | | write last |
| P6-06 | Limitations, ethics, reproducibility | TODO | | |
| P6-07 | Internal review round | TODO | | verify every citation |
| P6-08 | Artifact release | TODO | | |
| P6-09 | Novelty sweep #2 | TODO | | within 2 weeks of deadline |
| **GATE-S** | **Submission approval — PI approves submit + arXiv** | TODO | | |
| P6-10 | Submit and archive | TODO | | |

---

## Decisions log
See `docs/decisions.md`.
