# 10 — Phase 7: target-noise decomposition of small-scale data decisions

Opened 2026-11-20 by PI decision. It replaces the Phase 3 (ETS) line as the paper's
main contribution. The ETS and Theorem 1–4 PRs (#17–#35) continue their own review
cycle, but nothing in this phase depends on or cites their artifacts. This phase
builds only on merged code (PRs #1–#16).

## Thesis under test

Published decision accuracy for small-scale proxies (e.g. DataDecide's single-scale
baseline against a 3-seed 1B target) mixes three error sources:

1. proxy-measurement noise;
2. **target-measurement noise** — the target's own seed-to-seed unreliability;
3. true cross-scale reordering of recipe means.

Phase 7 estimates the three separately, with valid recipe-clustered uncertainty. It
then tests what follows for decision practice:

- accuracy on target-resolvable decisions;
- the seeds-vs-scale allocation of a fixed compute budget;
- re-scoring of prior conclusions;
- a tie-aware reporting standard.

## Standing rules specific to this phase

- **Exploratory vs confirmatory.** An exploratory probe (2026-11-20, outside the
  repo) already looked at DataDecide `primary_metric` final checkpoints.
  - Every RQ1–RQ3 result on that data is **exploratory** and must be labelled so.
  - Results are **confirmatory** only on data the probe never touched: other
    metrics; instance-level data; many-seed checks; new benchmarks (N7-11);
    external suites.
  - Confirmatory analyses use only hypotheses, code and benchmark lists frozen
    in N7-05.
- **Inference.** Recipe pairs share recipes, so use cluster (recipe-level) bootstrap
  only and never per-pair CIs.
  - Enumerate seed splits.
  - Tasks are fixed effects; never pool `olmes_10_macro_avg` with its own components.
  - Method comparisons are paired on identical resamples.
- **Circularity guard.** Any stratification by target resolvability uses target
  seeds held out from the agreement being scored, or a model-based posterior.
  It never uses the same noisy target for both.
- **750M is off-ladder.** DataDecide's final 750M checkpoints saw fewer tokens than
  530M. Scale curves either exclude it or are plotted against compute, with the
  confound stated.
- **Compute.** Local GPUs (RTX 4060 laptop, Colab/Kaggle T4) are used for inference
  only. No training happens without GATE-C7 approval.

## Tasks

| ID | Title | Depends on | Output |
|---|---|---|---|
| N7-00 | This phase plan | — | `plan/10-…md`, STATUS rows |
| N7-01 | Novelty gate G0: verbatim read of Signal & Noise appendices; screen all citers of DataDecide, Signal & Noise, Bhagia et al., Choshen et al.; OpenReview search; check what `allenai/signal-and-noise` contains (do the 10 extra 1B-5xC runs appear?). **Kill/reframe if prior work already decomposes decision accuracy by target noise.** | — | `docs/related_work.md` addendum + decisions entry |
| N7-02 | Reliability core (E1): `src/pdt/analysis/reliability.py`. Covers test–retest, held-out-seed resolvability stratification, the seeds-vs-scale frontier, recipe-cluster bootstrap, the homoscedasticity test (Brown–Forsythe) and the shared-seed ANOVA. Plus `experiments/n7_02_reliability.py` → `results/n7_02_reliability.json`. | — | code, tests, results |
| N7-03 | Simulation validation (E2) of the model-based decomposition estimator: bias, CI coverage ≥ nominal−2pp, and a posterior predictive check against N7-02's observed test–retest and stratified curve. **If the PPC fails, stop and open a GATE.** | N7-02 | estimator, tests, results |
| N7-04 | Inference harness (E11 prep): load released DataDecide checkpoints (and PolyPythia), score multiple-choice benchmarks with per-question outputs, and time it on 3 checkpoints on the local GPU. | — | `src/pdt/eval/`, timing results |
| N7-05 | **Pre-registration commit.** Freeze RQ/H text, thresholds, confirmatory benchmark list, analysis entry points and the code SHA. | N7-01..04 | `docs/preregistration.md` |
| N7-06 | DataDecide decomposition (E3): all tasks × sizes, `primary_metric` + continuous proxy metrics. | N7-03, N7-05 | results |
| N7-07 | Re-score prior conclusions (E6): extrapolation vs single-scale (P1-04); continuous vs accuracy metrics; checkpoint averaging if data allow. | N7-06 | results |
| N7-08 | Alternative targets (E7): 530M as target; Pythia weak check. | N7-06 | results |
| N7-09 | Instance-level noise (E5): targeted download of `DataDecide-eval-instances` final checkpoints; item bootstrap; paired comparisons. | N7-05 | results |
| N7-10 | Many-seed checks (E4/E4b): Signal & Noise data if usable; PolyPythia inference via N7-04. | N7-04, N7-05 | results |
| N7-11 | Confirmatory re-evaluation (E11): DataDecide final checkpoints on frozen non-OLMES benchmarks via N7-04; test frozen predictions. | N7-04, N7-05 | results |
| N7-12 | Tie-aware abstention rule (E10, stretch). | N7-06 | results |
| N7-13 | Workshop draft. | N7-06..08 | `paper/` |
| **GATE-C7** | Compute for prospective training (E9) — PI approval and provisioning. | — | — |
