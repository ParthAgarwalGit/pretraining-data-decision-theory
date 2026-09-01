# Related Work — Novelty Sweep #1

Task P0-07. Swept 2026-09-02. See `plan/01-phase0-setup.md` for the required
query list and `plan/07-phase6-paper.md` task P6-09 for the mandatory
second sweep before submission.

**Verdict: as of 2026-09-02, no work found that (a) derives sample-complexity
lower bounds for data-mixture selection via scaling-law extrapolation,
(b) applies pure-exploration bandit theory to neural scaling laws, (c) gives
a formal probability-the-extrapolated-winner-is-correct guarantee, or
(d) casts "how many models at which sizes" as active experimental design for
scaling laws.** Four close papers exist (below); each is missing the defining
feature of our setting — the reward at the target scale is never observed,
only extrapolated through a fitted scaling law — either because they instead
address curve-*fitting* accuracy (not a selection decision with a guarantee),
or because their "proxy" is a same-scale correlated signal rather than a
smaller-scale extrapolation target. This sweep is a search, not a proof of
absence — see Caveats at the end.

---

## 1. Near-miss papers, characterized precisely

### arXiv:2605.17234 — "Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning"
Schram, Hiller, Beck, Cohn. Submitted 2026-05-17 (v1), revised 2026-05-25 (v2).

**What it does:** Formulates fitting a scaling law itself as a budget-allocation
problem — given a pool of runnable pilot experiments with heterogeneous cost,
choose which to run to get an accurate curve fit cheaply. Compares Successive
Halving alone, SH + parametric surrogate, and SH + non-parametric surrogate;
reports up to 98.7% compute savings over exhaustive fitting.

**What it does not do:** No decision/selection framing at all — there is no
"choose the best of K arms" objective, no `delta`-correctness guarantee, no
sample-complexity lower bound. The output is a fitted curve with better or
worse *estimation* accuracy, not a certified winner among competing recipes.

**Why we're distinct:** Our reward is a *decision* ("recipe k or k'?") with a
formal error-probability guarantee; theirs is a *point estimate* of one curve
with no stated confidence semantics. This is exactly the gap the source
document names.

### arXiv:2604.22753 — "Spend Less, Fit Better: Budget-Efficient Scaling Law Fitting via Active Experiment Selection"
Li, Li, Lin, Sun, Talwalkar, Yang. Submitted 2026-04-24 (v1), revised
2026-08-12 (v2). **Not named in the source document — found independently
during this sweep.**

**What it does:** Same problem shape as 2605.17234 — budget-aware sequential
experimental design to maximize *extrapolation accuracy* of a fitted scaling
law in a high-cost target region. Proposes SL² (Scaling Laws, Spend Less), an
uncertainty-aware sequential-allocation method; reports matching full-dataset
fit quality at ~10% of the training budget, and at ~1% on some tasks.

**What it does not do:** Same gap as 2605.17234 — this is regression/estimation
accuracy for one curve, not a selection decision between multiple candidates
with a confidence guarantee. No BAI framing, no lower bound.

**Why we're distinct:** Identical reasoning to 2605.17234. Flagging that two
independent, contemporaneous groups converged on "active design for
scaling-law *fitting*" this cycle is itself informative — it confirms the
adjacent estimation problem is active and current, which makes the
*undone* decision-theoretic version (ours) a sharper, more clearly
differentiated contribution to state up front in the paper.

### arXiv:2607.06879 — "Best-Arm Identification with Generative Proxy"
Ma, Qin, Zhu, Zuo. Submitted 2026-07-08.

**What it does:** Fixed-confidence BAI where each arm has an expensive true
reward and a cheap, *correlated* proxy score (from an ML model or LLM) at the
**same operational scale** (e.g. auto-loan pricing: proxy and reward are both
observed per-applicant, not at different model sizes). Proposes PROBE, a
phase-elimination algorithm using a control-variate/OLS residual-variance
certificate; proves `delta`-PAC-ness with sample complexity approaching the
known-correlation oracle.

**What it does not do:** The proxy is never required to *extrapolate* across
a structural axis (like model scale) — it's a same-scale correlated
observation, estimated via a learned correlation coefficient, not a fitted
functional form `g(theta, s)` propagated to an unobserved `s*`. There is no
notion of extrapolation bias, no scaling-law misspecification term, and
correspondingly no impossibility/phase-transition result.

**Why we're distinct:** This is the general "cheap correlated side-information"
BAI setting; ours is the specific "reward observed only through a fitted
extrapolation, never directly, at the decision scale" setting. Their sample
complexity has no bias floor because their proxy has no extrapolation gap by
construction — exactly the classical-BAI-vs-our-setting distinction the
source document's Theorem 1 formalizes.

### arXiv:2601.21471 — "Best Arm Identification with LLM Judges and Limited Human Audits"
Ao et al. (5 authors). Submitted 2026-01 (exact day not confirmed from search
snippets).

**What it does:** Fixed-confidence BAI where every arm has a cheap but
*biased* (arm- and context-dependent) LLM-judge score, and a costly true label
obtainable only through selective human auditing. Builds an
inverse-propensity-weighted estimator with anytime-valid confidence
sequences; the algorithm adaptively concentrates audits on unreliable
contexts and close arms, with a plug-in Neyman rule proven near-oracle-audit-
efficient.

**What it does not do:** Same-scale proxy again — the LLM judge scores the
*same* item a human would audit, at the *same* point in the design space.
No cross-scale extrapolation, no scaling-law functional form, no bias term
that grows with extrapolation distance.

**Why we're distinct:** Structurally the same distinction as 2607.06879: bias
here comes from a *judge's* systematic error at a fixed point, correctable
by selective auditing of that same point — not from extrapolating a fitted
function to a point that is never directly observable at all (which is our
setting, and which is why "just audit more" isn't an available move for us).

---

## 2. Citation-graph checks

**Papers citing DataDecide (arXiv:2504.11393)**, checked via the Semantic
Scholar API citation list (up to 100 entries, 2026-09-02): no citing paper
formalizes best-arm identification, pure-exploration bandits, or
sample-complexity lower bounds for pretraining-data-recipe selection via
scaling-law extrapolation. Citing work clusters around further data-mixture
optimization (point-estimate methods) and further scaling-law studies, none
with a selection-decision-with-guarantee framing.

**Papers citing Garivier & Kaufmann 2016 (arXiv:1602.04589, the Track-and-Stop
paper)** from 2025–2026 that mention LLMs, checked the same way: exactly one,
and it is arXiv:2601.21471 above — already characterized and distinguished.
Nobody has applied Track-and-Stop (or an adaptation of it) to neural scaling
laws or pretraining data selection.

**OpenReview search** (`"best arm identification" scaling law data mixture`,
and separately `"pretraining data" selection confidence bandit`, both scoped
to 2026–2027 cycles): no submission matching our framing found. Real, unrelated
BAI papers turned up (best-arm ID with correlated sampling, fixed-budget large
deviation, constrained BAI, covariance-adaptive BAI, rising bandits) — none
apply to scaling-law extrapolation for data selection.

## 3. Adjacent papers checked and ruled out

- **arXiv:2606.08167 — "Explaining Data Mixing Scaling Laws"** (Dai, Zheng,
  2026). Gives a *mechanistic* explanation (capacity competition, noise
  reduction) for why data-mixing scaling laws behave as they do. Not a
  statistical/decision-theoretic account — no bias/variance decomposition, no
  sample-complexity bound. Worth citing as an adjacent explanatory-mechanism
  paper in the related-work section, but not a novelty threat.
- **arXiv:2512.23407 — "Theoretical Foundations of Scaling Law in Familial
  Models"** (2025–2026). Different problem entirely: characterizing a
  granularity axis for deployable sub-models sharing one backbone. No
  selection/BAI content. Ruled out.
- **arXiv:2601.12945 — "A Component-Based Survey of Interactions between
  Large Language Models and Multi-Armed Bandits"** (2026). Abstract confirms
  MAB-for-pretraining is a covered survey topic in general, but nothing in the
  abstract indicates it catalogs a best-arm-identification-via-scaling-law-
  extrapolation line of work — consistent with (not proof of) that line not
  yet existing. Useful as a citable survey anchor for "the surrounding
  literature is mature"; not independently re-verified against its full
  bibliography given time budget — worth a deeper pass in the P6-09 sweep.

---

## 4. Source-document reference list — verified

Every reference from the source Beginner's Guide document's Part 11
("Full Reference List"), checked against arXiv, publisher proceedings pages,
or ACL/OpenReview/PMLR listings on 2026-09-02.

| Citation | Venue + year (verified) | arXiv ID | One-line claim | Corrections vs. source doc |
|---|---|---|---|---|
| Kaplan et al. | arXiv preprint, 2020 | 2001.08361 | First large-scale power-law loss scaling demonstration. | None. |
| Hoffmann et al. ("Chinchilla") | arXiv preprint, 2022 | 2203.15556 | Compute-optimal N/D balancing. | None. |
| Muennighoff et al. | **NeurIPS 2023** (Outstanding Paper Runner-up) | 2305.16264 | Repeated-data scaling; value of repetition decays to ~0. | **Resolves the source doc's own flagged 2023-vs-2024 ambiguity: 2023 is correct.** |
| Xie et al. (DoReMi) | **NeurIPS 2023 (Spotlight)** | 2305.10429 | Group-DRO proxy sets domain weights via minimax. | Source doc said "NeurIPS"; confirmed Spotlight tier. |
| Ye et al. (Data Mixing Laws) | **ICLR 2025** | 2403.16952 | Functional law predicting loss from mixture proportions. | None (arXiv 2024, venue 2025 — normal submission lag, not an error). |
| Magnusson et al. (DataDecide) | **ICML 2025** (poster) | 2504.11393 | 25 recipes × 14 sizes × 3 seeds; single-scale ranking ≈ scaling-law extrapolation. | None — our anchor paper, citation confirmed solid. |
| Ruan, Maddison & Hashimoto | **NeurIPS 2024 (Spotlight)** | 2405.10938 | Observational scaling laws from ~100 public models. | None. |
| Choshen et al. (Hitchhiker's Guide) | **ICML 2025, PMLR 267:10683–10699** | 2410.11840 | Meta-analysis of 1000+ scaling-law fits; best practices. | None — PMLR volume/page confirmed exactly. |
| **Lourie, Hu & Cho** ("Scaling Laws Are Unreliable for Downstream Tasks: A Reality Check") | **Findings of ACL: EMNLP 2025**, pp. 16167–16180 | 2507.00885 | Only 39% of tasks show clean scaling; documents inverse/nonmonotonic/breakthrough shapes. | **Source doc cites this as "Hu, S., et al." — wrong first author. First author is Lourie; Hu is second author. Corrected here; use "Lourie et al." in our own citations.** |
| Bhagia et al. (model ladder) | **COLM 2025** | 2412.04403 | Two-step compute→loss→task-metric fitting methodology. | Source doc gave no venue; confirmed COLM 2025. |
| Kaufmann, Cappé & Garivier | JMLR 17(1):1–42, 2016 | — (no arXiv; JMLR-native) | General sample-complexity lower-bound theory for fixed-confidence BAI. | None. |
| Garivier & Kaufmann | COLT 2016 | 1602.04589 | Track-and-Stop: asymptotically optimal fixed-confidence BAI algorithm. | None. |
| Koh & Liang | **ICML 2017, PMLR 70:1885–1894** | 1703.04730 | Influence functions for tracing predictions to training points. | Source doc gave no venue; confirmed PMLR v70. |
| Grosse et al. (Anthropic) | arXiv preprint, 2023 (no confirmed peer-reviewed venue found) | 2308.03296 | EK-FAC-scaled influence functions up to 52B-param LLMs. | None — arXiv-only status confirmed, not an omission. |
| Gerstgrasser et al. | **COLM 2024** (also an ICML 2024 workshop) | 2404.01413 | Accumulating (not replacing) real+synthetic data avoids model collapse. | Source doc gave no venue; confirmed COLM 2024 primary venue. |
| Miller (Anthropic) | arXiv preprint, 2024 (no confirmed peer-reviewed venue found) | 2411.00640 | Treats eval benchmarks as experiments; clustered SEs, paired analysis. | None — arXiv-only status confirmed. |

**Two corrections found and worth carrying into the paper's own bibliography:**
the Muennighoff et al. year ambiguity is resolved (2023, not 2024), and the
"Hu et al." citation should read **Lourie et al.** (Nicholas Lourie is first
author; Michael Y. Hu is second author — an easy mistake to propagate if
copied from a shorthand reference, so double-check this one specifically at
P6-07's citation-verification pass too).

---

## 5. Target venue deadlines — verified 2026-09-02

`PLAN.md` section 4 requires verifying current deadlines rather than relying
on the rough windows stated there, and ties that check to this task.

| Venue | Status | Deadline | Confidence |
|---|---|---|---|
| ICML 2026 | **Already passed** | Abstract Jan 23, 2026 AoE; full paper Jan 28, 2026 AoE | High — confirmed directly from `icml.cc/Conferences/2026/CallForPapers`. |
| NeurIPS 2026 | **Already passed** | Abstract/full paper May 4–6, 2026 AoE | High — confirmed directly from NeurIPS's own site. |
| **ICML 2027** | Upcoming | Abstract ~Jan 16, 2027; full paper ~Jan 22, 2027 AoE | **Moderate** — reported consistently by conference-deadline aggregators; `icml.cc/Conferences/2027` is not live yet (404 as of this check), so not yet confirmable from the canonical source. Re-verify once it is. |
| **COLM 2027** | Upcoming | Paper deadline March 31, 2027; conference Oct 6–9, 2027 | **Moderate** — reported consistently across two independent aggregator hits; `colmweb.org`'s own 2027 CFP page does not exist yet (only 2025/2026 are posted), so also not yet canonically confirmed. |
| NeurIPS 2027 | Upcoming | **Officially TBA.** Historical pattern (2024: May 22, 2025: May 16, 2026: May 4-6) suggests ~May 2027, but this is an estimate, not a finding. | Low — no CFP posted yet. |

**Recommendation:** target **COLM 2027** as the primary venue. It has the
better-verified date of the two realistic options, and March 31, 2027 gives
roughly 7 months from today — comfortable room after the rough 13-week
schedule in `PLAN.md` §3, including slack for GATE-C/compute delays and
GATE-T proof review, both of which are hard to schedule tightly. ICML 2027
(~Jan 22, 2027, ~4.5 months out) is tight given Phases 1–6 have not started;
treat it as a stretch target only if Phase 1 (the load-bearing empirical
result) and Phase 2 theory land faster than planned, not as the primary
plan. This is a recommendation for the PI to confirm, not a decision made
here — matching the "recommended target and why" requirement in the
GATE-0 checklist (`plan/01-phase0-setup.md`).

**Before committing to either date**, re-verify against the canonical site
directly (`icml.cc`, `colmweb.org`) once each venue's own CFP page exists —
neither is confirmable from its own primary source yet as of this check.

---

## Caveats

- This sweep covered arXiv, OpenReview, and general web search as of
  2026-09-02. It is **not provably exhaustive** — the source document's own
  caveat about "very recent or unindexed preprints" applies equally here.
  arXiv:2604.22753, found independently in this sweep despite not being named
  in the source document, is itself a demonstration that the space moves
  fast enough for a second sweep close to the submission deadline (P6-09) to
  matter.
- arXiv:2606.07616 ("Item Response Scaling Laws") is adjacent-area context per
  the source document, not a novelty threat to this project's core question —
  confirmed still just an arXiv preprint (submitted 2026-05-29) with no
  confirmed peer-reviewed venue as of this sweep.
- The survey at arXiv:2601.12945 was checked only at the abstract level, not
  its full bibliography — worth a deeper pass at P6-09.
