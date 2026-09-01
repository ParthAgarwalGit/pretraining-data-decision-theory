---
license: odc-by
task_categories:
- text-generation
tags:
- scaling-laws
- best-arm-identification
- pretraining-data-selection
---

# pdt-datadecide-analysis

Derived analysis artifacts for **"A Statistical Decision Theory for
Pretraining-Data Selection"** — code and full plan at
[github.com/ParthAgarwalGit/pretraining-data-decision-theory](https://github.com/ParthAgarwalGit/pretraining-data-decision-theory).

**Status: shell only.** This repo currently holds nothing but this card.
Populated starting Phase 1 — see task P1-12
(`plan/02-phase1-datadecide.md`) for what lands here and when.

## What will land here

- Derived analysis tables built from AI2's DataDecide suite: per-task
  decision-accuracy frontiers, the bias/variance decomposition
  (`sigma^2_extrap` per recipe/task/design), and bound-coverage results.
- Fitted scaling-law parameters per (recipe, task, extrapolation method,
  design), for every fitter compared in Phase 1.
- Per-task `sigma^2_extrap` estimates — the paper's central empirical
  quantity, and the reason this project exists (see the repo's `PLAN.md`).

Every table pushed here is written by `src/pdt/hub.py`'s `push_results()`,
which refuses to run from an uncommitted working tree or on data that
fails `src/pdt/provenance.py`'s validation — so anything you see in this
repo traces back to a specific, clean commit in the GitHub repo above.

## Attribution

Built on **AI2's DataDecide** (Magnusson et al., ICML 2025,
[arXiv:2504.11393](https://arxiv.org/abs/2504.11393)), released under
**ODC-BY**. This derived-analysis repo is likewise released under
**ODC-BY**, consistent with that upstream license.

## License

This repository: **ODC-BY** (Open Data Commons Attribution License).
The project's code is licensed separately under Apache-2.0 — see the
GitHub repo linked above.
