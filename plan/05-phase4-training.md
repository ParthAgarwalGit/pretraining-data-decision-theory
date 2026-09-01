# Phase 4 — Confirmatory Pretraining Runs

**Goal:** run the Extrapolation-Track-and-Stop algorithm **live**, deciding which
recipe to train next and at what size, on real pretraining runs we execute ourselves.

**Compute:** the PI's A100 cluster. **Nothing in this phase starts before GATE-C.**

**Status in the paper:** confirmatory. Phase 1 + 2 + 3 is already a complete
submission. Say this to the PI when asking for compute so the decision is informed.

---

## Design rationale (read before proposing the experiment)

Two design pressures pull against each other:

- If we use DataDecide's own recipes at a size DataDecide already covers, we get a free
  cross-check that our training stack is sound — but the experiment is a re-run, not a
  live decision.
- If we invent new recipes, it is a genuine new BAI instance — but we must train the
  target-scale models ourselves to know the right answer, which is most of the cost.

**Recommended resolution:** `K = 6` recipes, of which **2 are DataDecide recipes**
(calibration: our small-scale numbers must land near AI2's, which validates the stack
for a few GPU-hours) and **4 are new mixtures** we construct, chosen so that at least
one pair is *likely to reverse* across scale based on the P1-09 census. A reversing pair
in the live experiment is the most informative single thing this phase can contain,
because it exercises the abstention rule.

---

## P4-01 — Training stack selection and single-GPU smoke test

**Branch:** `phase4/training-stack`

**Runs on the PI's laptop GPU (RTX 4060, 8 GB) — no cluster needed, no GATE.**

1. Evaluate candidate stacks and pick one. Priority order, but **verify current
   maintenance status before committing** — this plan was written 2026-09-01 and
   framework health changes:
   - **OLMo-core / the OLMo model ladder** — first choice, because DataDecide used OLMo
     ladder configs, which maximises comparability with our Phase 1 results.
   - **SmolLM training configs** — second choice, well-documented small-model recipes.
   - **TorchTitan** or **litgpt** — fallbacks if the above are stale.
2. Get a 4M-parameter model training end to end on the laptop GPU for a few hundred
   steps. Confirm: loss decreases, checkpointing works, resume-from-checkpoint works,
   the eval harness runs, and logs are structured.
3. Write `src/pdt/train/` wrapping the chosen stack behind a single function
   `train_and_eval(recipe, scale, seed, config) -> metrics`, which is what
   `LiveTrainingOracle` will call.
4. Record measured throughput (tokens/sec) so the GATE-C budget uses **measured**
   numbers rather than the estimates in `plan/08-compute-requests.md`.

**Definition of done:** a 4M model trains and evaluates locally; measured
tokens/sec/GPU recorded in `results/p4_01_smoke.json`.

---

## P4-02 — Recipe design and data preparation plan

**Branch:** `phase4/recipe-design`

**No cluster compute; this is planning and small-scale data work.**

1. Choose the 6 recipes. Document each as a mixture over named public corpora with
   proportions, and justify each choice in one sentence. Sources to draw from: DataDecide's
   own released corpora (`allenai/DataDecide-data-recipes`), DCLM-baseline, Dolma,
   FineWeb-Edu, StarCoder/code, and a maths or academic slice.
2. Use P1-09's stress-test pairs to deliberately include a pair expected to reverse.
   State the prediction *before* running — a pre-registered prediction that comes true
   is far more persuasive than a post-hoc observation, and pre-registering it costs
   nothing. Write it into `docs/preregistration.md` with a timestamp and commit it.
3. Fix the scale ladder and the token budget rule (recommend 5xC, i.e. `D = 100 N`,
   matching DataDecide).
4. Compute exact storage requirements: tokenised bytes per recipe, checkpoint bytes,
   log bytes. Include these in the GATE-C ask — storage is the most commonly forgotten
   part of a compute request and the most annoying to fix mid-run.
5. Write the data pipeline: download, dedup if the recipe calls for it, tokenise, shard,
   and verify with a checksum manifest. Test it on 1% of one recipe locally.

**Definition of done:** `configs/recipes/*.yaml` for all 6, `docs/preregistration.md`
committed, storage estimate computed, pipeline tested on a 1% slice.

---

## GATE-C — The compute request

**This is the main compute GATE. Stop and ask.**

Use the template and the costed tiers in `plan/08-compute-requests.md`. Your ask must
contain, with no exceptions:

1. **What the compute buys scientifically** — one paragraph, and one sentence on what
   the paper loses if the answer is no (answer: a confirmatory section; the paper still
   stands).
2. **Three tiers** (A/B/C) with, for each: GPU-hours, wall-clock on a stated cluster
   shape, storage in TB, and exactly which experiments become possible.
3. **Measured, not estimated, throughput** from P4-01, with the arithmetic shown.
4. **The failure budget** — what fraction you have reserved for restarts and mistakes
   (reserve at least 25%), and what you will do if a run diverges.
5. **A decision request**, phrased as a choice: "Tier A, B, C, or decline?"

Also ask for the operational details you will need and cannot guess:

- Cluster access method (SSH, Slurm, Kubernetes, a cloud account), and who provisions it
- Node shape: how many A100s per node, 40 GB or 80 GB, interconnect
- Whether jobs are pre-emptible, and the maximum job wall-time
- Shared filesystem path and quota
- Whether there is a Weights & Biases or equivalent account for logging, or whether to
  log to files only
- Any billing ceiling you must not exceed without re-asking

**Do not provision, launch, or spend anything until the PI answers.** If the PI says
"go ahead", record the approved tier and ceiling in `docs/decisions.md`, and treat that
ceiling as hard: if you are on track to exceed it, stop and re-ask.

---

## P4-03 — Cluster setup and reproducibility harness

**Branch:** `phase4/cluster-setup`

1. Environment on the cluster mirrors the local one — same lockfile, plus the `train`
   optional dependency group with a pinned torch/CUDA build matching the driver.
2. Job submission wrapper that, for every run, records: git SHA, config hash, dataset
   manifest checksum, node/GPU inventory, and the full environment.
3. Structured logging: loss, learning rate, throughput, gradient norm, wall-clock, at a
   fixed step cadence, to files that survive job pre-emption.
4. Checkpointing at a fixed token cadence (not step cadence — token cadence keeps
   different sizes comparable), with resume tested by deliberately killing a job and
   restarting it.
5. A hard spend counter: the wrapper tracks cumulative GPU-hours against the approved
   ceiling and **refuses to launch** when a new job would exceed it.

**Definition of done:** a 4M run completes on the cluster through the wrapper; a killed
job resumes correctly; the spend counter works.

---

## P4-04 — Calibration runs against DataDecide

**Branch:** `phase4/calibration`

Train the 2 DataDecide recipes at the 2 or 3 smallest sizes and compare our metrics
against AI2's published values for the same recipe and size.

**Acceptance criterion, stated before you look:** our final-checkpoint metric should sit
within the seed-to-seed spread AI2 reports (from P1-05's `sigma^2_seed`). Write the
criterion into the config before running.

If we land outside it, **stop and diagnose** — tokeniser mismatch, token-budget
mismatch, LR schedule, sequence length, and data ordering are the usual causes. Do not
proceed to the expensive runs with an uncalibrated stack; that is how a compute budget
is wasted. This task is cheap insurance.

---

## GATE-D — First real run review

After the first non-trivial run (the largest calibration run), post to the PI: the loss
curve, throughput against the estimate used at GATE-C, the calibration comparison, and
the projected total spend at the observed throughput. **Ask for a go/no-go on launching
the remaining runs.** If throughput came in materially below the estimate, re-scope to a
smaller tier *before* spending, not after.

---

## P4-05 — Ladder runs for the new recipes

**Branch:** `phase4/ladder-runs`

Execute the proxy-scale ladder for all 6 recipes at the approved tier. Run smallest
sizes first, across all recipes, before any larger size — so that a budget cut at any
point still leaves a complete, analysable design rather than a ragged one. This
ordering is worth more than it looks: it converts a hard budget failure into a smaller
but valid experiment.

After each size completes across all recipes, update `results/p4_05_ladder.json` and
report progress and spend to the PI in one line. Do not wait until the end.

---

## P4-06 — Target-scale ground-truth runs

**Branch:** `phase4/target-runs`

Train all 6 recipes at the target scale `s*`. These are the most expensive runs and
exist solely to score the decision.

**Critical discipline:** the algorithm in P4-07 must produce its decision from the
proxy-scale data *before* any target-scale result is examined. Enforce this by (a)
running P4-07's decision first and committing its output with a timestamp, then (b)
running the target-scale training. If scheduling forces target runs to start earlier,
keep their results in a directory the analysis code cannot read (a separate
`results/sealed/` path, checked by a test) until the decision is committed.

---

## P4-07 — Live algorithm run

**Branch:** `phase4/live-algorithm`

1. Wire `LiveTrainingOracle` into the algorithm from P3-03.
2. Run Extrapolation-Track-and-Stop against the live oracle, letting it choose which
   `(recipe, scale)` to train next — within the pre-approved compute ceiling.
3. Run the baselines on the same data.
4. Record: the decision (or abstention), the certificate, total compute spent, and the
   full pull log with timestamps.
5. Score against the P4-06 ground truth.

**The headline claim to test:** the active allocation reaches the correct winner at the
target confidence using materially less compute than uniform allocation and than the
single-scale baseline.

**If it does not:** report that. A negative confirmatory result alongside a strong
Phase 1 + Phase 3 result is publishable and honest; a massaged positive is not.

---

## P4-08 — Release checkpoints and configs

**Branch:** `phase4/release-artifacts`

Push to Hugging Face under the approved namespace, **after PI approval** (uploads are
public-facing publication):

- One model repo per `(recipe, size)`, or one repo with per-run subfolders — choose
  based on count and document the choice. Naming: `pdt-<recipe>-<size>`.
- Model cards: recipe composition, token budget, hyperparameters, eval results,
  licence, and the compute used.
- A dataset repo entry with the tokenised-data manifests and checksums (manifests, not
  the corpora themselves, unless licences permit redistribution — **check each corpus
  licence before uploading any data**).

---

## P4-09 — Phase 4 analysis and figures

**Branch:** `phase4/analysis`

- **F9** — live compute-to-decision, our algorithm against baselines (the Phase 4 analogue
  of F6).
- **F10** — the observed ladder curves for all 6 recipes with the fitted extrapolations
  and the target-scale truth marked, including the pre-registered reversal pair.
- Re-estimate `sigma^2_extrap` on our own runs and compare to the DataDecide estimates.
  Agreement across two independent suites is a strong result; disagreement is
  informative and must be reported either way.
