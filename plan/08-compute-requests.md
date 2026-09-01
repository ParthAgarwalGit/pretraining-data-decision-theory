# 08 — Compute Requests

**Rule: every request for the PI's GPUs is a GATE. You stop, you ask, you wait.**
This applies to the first big ask and to every subsequent "can I also run…".

---

## 1. When you must ask

Ask before:

- Any job on the PI's cluster, including a 10-minute smoke test.
- Any local job expected to run longer than 2 hours or download more than 50 GB.
- Any re-run after a failure that would spend more than 5% of the approved ceiling.
- Any request to exceed a previously approved ceiling. **Never quietly exceed it.**

You do **not** need to ask for: analysis on the laptop, downloading the DataDecide
result tables (about 700 MB), simulations, or anything in Phases 1-3 that fits on the
laptop.

---

## 2. The request template

Send exactly this shape. Fill every field. No field may say "TBD".

```
COMPUTE REQUEST — <task ID> — <one-line title>

WHY
<2-4 sentences: what scientific question this compute answers, and what the paper
loses if the answer is no.>

WHAT I WILL RUN
<Concrete list: N runs, which recipes, which sizes, how many seeds, what token budget.>

COST
Measured throughput: <X> tokens/sec/GPU at <size>, measured in <task ID>.
Arithmetic: C = 6ND per run; GPU-hours = C / (throughput_FLOPs * 3600).
| Item | GPU-hours | Storage |
|---|---|---|
| ... | ... | ... |
| Subtotal | ... | ... |
| Failure reserve (25%) | ... | |
| TOTAL | <H> GPU-hours | <T> TB |

Wall clock: <H>/<gpus> = <hours> on a <n>xA100 node, assuming <utilisation>.

TIERS
Tier A (minimal): <what it buys> — <H_A> GPU-h
Tier B (recommended): <what it buys> — <H_B> GPU-h
Tier C (ideal): <what it buys> — <H_C> GPU-h

RISKS
<What could waste this compute, and the guard against each.>

WHAT I NEED FROM YOU
1. Which tier (A / B / C / decline)?
2. Cluster access details: <the specific list from plan/05, section GATE-C>
3. A hard GPU-hour ceiling I must not exceed without asking again.
```

---

## 3. Costed tiers for GATE-C

**These are planning estimates written 2026-09-01. Replace every number with measured
throughput from P4-01 before sending the request.** They exist so the PI can form an
expectation now, not so you can skip measuring.

### Assumptions

- Token budget 5xC, i.e. `D = 100 N` tokens, matching DataDecide.
- `C = 6 N D = 600 N^2` FLOPs per run.
- A100 40 GB at **35% MFU** on bf16 → about `1.1e14` effective FLOP/s → about
  `3.96e17` FLOPs per GPU-hour.
- **Small models will do worse than 35% MFU** (they are memory-bandwidth-bound). The
  sub-60M rows are negligible in total, so this does not move the totals much, but do
  not be surprised when a 4M run is inefficient.

### Cost per single run

| Size `N` | Tokens `D` | FLOPs `C` | GPU-hours |
|---|---|---|---|
| 4M | 0.4B | 9.6e15 | 0.02 |
| 20M | 2B | 2.4e17 | 0.6 |
| 60M | 6B | 2.2e18 | 5.5 |
| 90M | 9B | 4.9e18 | 12 |
| 150M | 15B | 1.35e19 | 34 |
| 300M | 30B | 5.4e19 | 136 |
| 530M | 53B | 1.7e20 | 425 |
| 1B | 100B | 6.0e20 | 1515 |

### Tier A — minimal, target `s* = 150M`

`K = 6` recipes, proxy ladder {4M, 20M, 60M, 90M}, target 150M, 1 seed at large sizes
plus 2 extra seeds at 4M/20M for noise estimation.

| Item | GPU-hours |
|---|---|
| Proxy ladders, 6 recipes × 18.1 | 109 |
| Target-scale runs, 6 × 34 | 204 |
| Extra seeds | 4 |
| Failure reserve 25% | 79 |
| **Total** | **~400 GPU-hours** |

Storage: roughly 0.25 TB. Wall clock on 8×A100: about **2 days**.
Buys: a complete live demonstration at a small target scale. Enough to say the
algorithm was validated on real runs we executed.

### Tier B — recommended, target `s* = 300M`

`K = 6`, ladder {4M, 20M, 60M, 90M, 150M}, target 300M.

| Item | GPU-hours |
|---|---|
| Proxy ladders, 6 × 52 | 313 |
| Target-scale runs, 6 × 136 | 816 |
| Extra seeds | 10 |
| Failure reserve 25% | 285 |
| **Total** | **~1,420 GPU-hours** |

Storage: roughly 0.6 TB. Wall clock on 8×A100: about **7-8 days**.
Buys: a 5-point ladder extrapolating a full 2× beyond the largest proxy, which is
enough for the identifiability ablation (P5-02) to be meaningful on our own runs. **This
is the recommended tier** — Tier A's 4-point ladder is thin for testing scale-design
claims.

### Tier C — ideal, target `s* = 530M`

`K = 6`, ladder {4M, 20M, 60M, 90M, 150M, 300M}, target 530M.

| Item | GPU-hours |
|---|---|
| Proxy ladders, 6 × 188 | 1,130 |
| Target-scale runs, 6 × 425 | 2,550 |
| Failure reserve 25% | 920 |
| **Total** | **~4,600 GPU-hours** |

Storage: roughly 1.5 TB. Wall clock: about **24 days on 8×A100**, or **6 days on
32×A100**. Only sensible with a larger cluster or a long window.
Buys: a 6-point ladder and an extrapolation ratio comparable to DataDecide's own.

### Not recommended: target `s* = 1B`

6 recipes at 1B is roughly 9,100 GPU-hours for the target runs alone, about 47 days on
8×A100. **DataDecide already provides 1B ground truth for free.** If a 1B target is
wanted, use DataDecide's runs (Phase 1 and the Phase 3 replay already do exactly this)
rather than paying for it. Say this to the PI if they offer more compute than Tier C —
recommending they *not* spend it is the right answer here.

---

## 4. Storage and data-transfer asks

Easy to forget, annoying to fix mid-run. Always include:

- **Tokenised corpora:** roughly 2 bytes/token. Tier B needs about 30B tokens per recipe
  for the target run; proxy runs read prefixes of the same shards, so budget per recipe,
  not per run. Six recipes at Tier B: about 360 GB.
- **Checkpoints:** weights-only bf16 at about `2 * N` bytes each. Ten checkpoints per run
  for the token-cadence schedule. Tier B: roughly 100-200 GB. Do **not** save full
  optimizer state at every checkpoint (that is 3-4× larger); save it only for the
  latest, for resume.
- **Logs and eval outputs:** small, under 10 GB, but keep them.
- **Headroom:** ask for 1.5× the computed total.

---

## 5. Spending discipline

1. Record the approved tier and hard ceiling in `docs/decisions.md` the moment the PI
   approves.
2. The job wrapper (P4-03) tracks cumulative GPU-hours and **refuses to launch** a job
   that would exceed the ceiling.
3. Report spend against ceiling in every session summary once Phase 4 starts, as one
   line: `Spend: 412 / 1420 GPU-h (29%)`.
4. At 50% and 80% of the ceiling, proactively report to the PI with a projection to
   completion.
5. If a run fails, do not immediately relaunch. Diagnose first, then relaunch once. A
   second failure of the same job is a stop-and-ask.
6. Run smallest sizes first across all recipes (see P4-05), so an early stop still
   leaves a complete design.
