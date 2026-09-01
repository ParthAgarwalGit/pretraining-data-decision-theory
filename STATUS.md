# STATUS

Update this file at the end of **every** session (protocol step 11).

Last updated: 2026-09-02 | Session: 6

States: `TODO` | `IN PROGRESS` | `IN REVIEW` | `DONE` | `BLOCKED` | `DROPPED`

## Open GATEs
- GATE-0 not yet reached — P0-03 through P0-08 remain before it can be posted.

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
| P0-07 | Novelty sweep #1 | IN REVIEW | (opening) | verdict: no direct prior work found; 3 named near-misses + 1 found independently (arXiv:2604.22753), all characterized; 2 real citation errors fixed in the source doc's own reference list (Muennighoff year, "Hu et al." -> Lourie et al.); also fixed 2 real bugs in the fabrication guard tool itself (arXiv-ID false positives, a Windows console encoding crash) -- see docs/related_work.md |
| P0-08 | Create the HF dataset repo shell | TODO | | |
| **GATE-0** | **End of Phase 0 — PI approval** | TODO | | |

## Phase 1 — DataDecide re-analysis

| Task | Title | State | PR | Notes |
|---|---|---|---|---|
| P1-01 | Canonical analysis frame | TODO | | |
| P1-02 | Ground truth at target scale, and gaps | TODO | | |
| P1-03 | Reproduce DataDecide single-scale baseline | TODO | | target ~80% at 150M |
| P1-04 | Scaling-law fitters; reproduce "extrapolation does not win" | TODO | | |
| **GATE-1** | **Reproduction checkpoint — PI approval** | TODO | | |
| P1-05 | Noise-floor estimation | TODO | | |
| P1-06 | Bias/variance decomposition (core result) | TODO | | |
| P1-07 | Plug-in bound and empirical coverage | TODO | | |
| P1-08 | Does the bound predict the 80% ceiling? | TODO | | |
| P1-09 | Rank-reversal census | TODO | | |
| P1-10 | Secondary ladders: Pythia and OLMo 2 | TODO | | |
| P1-11 | Phase 1 figures (F1-F5) | TODO | | |
| P1-12 | Publish derived tables; Phase 1 memo | TODO | | |

## Phase 2 — Theory

| Task | Title | State | PR | Notes |
|---|---|---|---|---|
| P2-01 | Formal setup and assumptions | TODO | | |
| P2-02 | Theorem 1: extrapolation-aware error bound | TODO | | |
| P2-03 | Theorem 2: lower bound and impossibility | TODO | | |
| P2-04 | Theorem 3: identifiability and minimax rate | TODO | | |
| P2-05 | Theorem 4: algorithm correctness | TODO | | |
| **GATE-T** | **Hand proofs to a human co-author** | TODO | | |
| P2-06 | Revise theory against Phase 1 evidence | TODO | | |
| P2-07 | Related-theory positioning | TODO | | |
| P2-08 | Integrate the verified proofs | TODO | | |
| P2-09 | Theory appendix | TODO | | |

## Phase 3 — Algorithm

| Task | Title | State | PR | Notes |
|---|---|---|---|---|
| P3-01 | Simulator and oracle interface | TODO | | |
| P3-02 | Solve the optimal-allocation program | TODO | | |
| P3-03 | Extrapolation-Track-and-Stop + baselines | TODO | | |
| P3-04 | Simulation study | TODO | | |
| P3-05 | Offline replay on DataDecide (free real-data result) | TODO | | high value |
| P3-06 | Sensitivity to the bias-floor estimate eta | TODO | | |
| P3-07 | Phase 3 figures (F6-F8) | TODO | | |
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
