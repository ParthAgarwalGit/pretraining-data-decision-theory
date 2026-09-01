# Implementation Plan — A Statistical Decision Theory for Pretraining-Data Selection

**Project codename:** `pdt` (Pretraining-Data decision Theory)
**Principal Investigator (PI):** Parth (GitHub `ParthAgarwalGit`, HF `Parth4105`, HF org `Algoverse-AYJP`)
**Executing agent:** a coding agent operating under `plan/00-agent-protocol.md`
**Source documents:** `A Statistical Decision Theory for Pretraining-Data Selection.pdf` and `Beginner_Guide_LLM_Data_Selection_Research.pdf`, both in this folder
**Plan written:** 2026-09-01

---

## 0. Read this first (agent)

You are implementing a research paper end to end. You are **not** expected to be
clever. You are expected to be **exact, honest, and stepwise**.

Three rules override everything else in this plan:

1. **One task per work session.** Find the lowest-numbered task in `STATUS.md`
   that is not `DONE`, do only that task, open a PR for it, update `STATUS.md`,
   then stop and report to the PI.
2. **Never invent a number.** Every number that ends up in a table, a figure, or
   the paper must be produced by a committed script that writes a JSON or CSV into
   `results/`. If you cannot produce it, write `TODO` and say so out loud.
3. **Stop at every GATE.** A GATE means: post the requested summary to the PI and
   wait for an explicit reply. Do not proceed past a GATE on your own judgement.
   Every compute request is a GATE.

Read `plan/00-agent-protocol.md` in full before your first action, and re-read its
"Session checklist" section at the start of every session.

---

## 1. What we are building, in one paragraph

Choosing which pretraining-data recipe to use for a huge model, based on small cheap
training runs plus scaling-law extrapolation, is currently a heuristic with no
guarantees. AI2's **DataDecide** (Magnusson et al., ICML 2025, arXiv:2504.11393)
showed empirically that this heuristic is shaky: ranking recipes at a single small
size (about 150M params) predicts the 1B-scale winner roughly 80% of the time, and
**none of 8 scaling-law extrapolation methods beat that**. Nobody has explained why.
We formalise the problem as **fixed-confidence best-arm identification (BAI) in which
the reward at the target scale is never observed, only extrapolated**, and deliver
four things: (T1) an error bound decomposing decision error into a shrinking
*variance* term and an irreducible *extrapolation-bias* term; (T2) a change-of-measure
lower bound with a *phase transition / impossibility* regime when recipe rankings
reverse across scale; (T3) an identifiability condition and minimax rate; (T4) a
Track-and-Stop-style **active allocation over (recipe, model-size) pairs** with an
anytime-valid stopping certificate. We validate on DataDecide's released 30k+
checkpoints (free), then confirm the allocation algorithm with a small set of real
pretraining runs on the PI's A100s.

**The single most important empirical object in this project** is
`sigma^2_extrap,k`, the irreducible extrapolation bias. DataDecide is the ideal
testbed precisely because AI2 *actually trained the 1B target models*, so the
quantity our theory says is unobservable in practice is, for once, directly
measurable.

---

## 2. Phase map

| Phase | Name | Compute | Blocking? | Plan file |
|---|---|---|---|---|
| **P0** | Setup: repos, env, data acquisition, novelty sweep | Laptop only | No | `plan/01-phase0-setup.md` |
| **P1** | DataDecide re-analysis (headline empirical result) | Laptop / 1 small GPU | No | `plan/02-phase1-datadecide.md` |
| **P2** | Theory: formal setup, 3 theorems, proof drafts + numerical certificates | Laptop | Needs human co-author at GATE-T | `plan/03-phase2-theory.md` |
| **P3** | Algorithm: Extrapolation-Track-and-Stop + free offline replay on DataDecide | Laptop | No | `plan/04-phase3-algorithm.md` |
| **P4** | Confirmatory pretraining runs (algorithm running live) | **A100 cluster, GATE-C** | Yes, at GATE-C | `plan/05-phase4-training.md` |
| **P5** | Ablations and stress tests | Mixed | Partly | `plan/06-phase5-ablations.md` |
| **P6** | Paper, artifact release, submission | Laptop | Yes, at GATE-S | `plan/07-phase6-paper.md` |

Supporting files:

- `plan/08-compute-requests.md` — how to ask the PI for compute, with costed budget tiers
- `plan/09-review-gates.md` — Git/GitHub/HF workflow, PR template, review checklists
- `STATUS.md` — the running task ledger (you create it in P0-02 and update every session)

**Critical path:** P0 → P1 → (P2 in parallel with P3) → free part of P5 → P6.
**P4 is confirmatory, not load-bearing.** If the PI declines the compute request at
GATE-C, the paper is still complete and submittable on P1 + P2 + P3 + the free half
of P5. State this explicitly when you make the compute ask, so the PI can decide with
full information.

---

## 3. Ordering and rough effort

Effort is in *agent work sessions*, not wall-clock time. One session is about one
task and one PR.

```
P0-01 .. P0-08   Setup                        ~8 sessions    (week 1)
P1-01 .. P1-12   DataDecide re-analysis      ~14 sessions    (weeks 2-4)    GATE-1 after P1-04
P3-01 .. P3-08   Algorithm + offline replay  ~10 sessions    (weeks 4-6)    may interleave with P2
P2-01 .. P2-09   Theory                      ~10 sessions    (weeks 3-7)    GATE-T after P2-05
P5-01 .. P5-07   Ablations (free subset)      ~7 sessions    (weeks 6-8)
P4-01 .. P4-09   Training runs                ~9 sessions    (weeks 7-10)   GATE-C before P4-03
P5-08 .. P5-10   Ablations needing P4         ~3 sessions    (weeks 10-11)
P6-01 .. P6-10   Paper and release           ~12 sessions    (weeks 9-13)   GATE-S before submitting
```

Do **not** start P4 before P3 is merged. The entire point of P4 is to run the
algorithm from P3 live; running it before it is validated offline burns GPU hours
that cost the PI real money.

---

## 4. Target venue and calendar

Primary: **NeurIPS or ICML main track** (a theory-plus-method paper, explicitly *not*
Datasets and Benchmarks). Secondary: **COLM**.

Approximate deadline windows. **Verify current dates on the official sites before
relying on them, and report the verified dates to the PI at P0-07.**

| Venue | Typical abstract deadline | Typical full-paper deadline |
|---|---|---|
| ICML | late January | about 1 week after abstract |
| COLM | around March | around March |
| NeurIPS | around mid-May | about 1 week after abstract |

---

## 5. Deliverables checklist (definition of "project done")

- [ ] GitHub repo, public, Apache-2.0, CI green, README reproduces every figure
- [ ] HF dataset repo with derived DataDecide analysis tables, fitted scaling-law parameters, and per-task `sigma^2_extrap` estimates
- [ ] HF model repo(s) with Phase 4 checkpoints and training configs (only if GATE-C is funded)
- [ ] `paper/main.tex` compiling to a complete submission, every number traceable to a `results/*.json`
- [ ] `docs/reproduce.md` — a stranger can rerun Phase 1 with one command
- [ ] Novelty sweep re-run within 2 weeks of submission (P6-09)
- [ ] arXiv preprint posted, only after the PI approves at GATE-S

---

## 6. Honest scoping notes for the PI

Three things in this plan cannot be fully delegated to an agent. Staff or budget for
them:

1. **The proofs (P2).** An agent can write the formal setup, state the theorems
   precisely, produce proof *skeletons* that follow Kaufmann-Cappe-Garivier (2016) and
   Garivier-Kaufmann (2016) step by step, and build numerical certificates that check
   every claimed inequality on simulated and real data. It cannot be trusted to
   *certify* a novel proof. GATE-T exists to hand the proof drafts to you or a
   co-author with bandit or mathematical-statistics depth. Budget for that person.
2. **The compute decision (P4, GATE-C).** Three funding tiers are costed in
   `plan/08-compute-requests.md`. You pick one, or decline, and the paper still stands.
3. **Novelty risk.** The source document flags that novelty was checked but is "not
   provably exhaustive". P0-07 and P6-09 are two independent sweeps for concurrent
   work. If a concurrent paper appears, fallback framings are listed in
   `plan/07-phase6-paper.md` section 7.
