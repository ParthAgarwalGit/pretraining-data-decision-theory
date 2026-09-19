# A Statistical Decision Theory for Pretraining-Data Selection

**Status: Phase 3 (algorithm) in progress — theory (Phase 2) drafted and
awaiting human co-author review (GATE-T); see [`STATUS.md`](STATUS.md)
for the live task ledger.**

Choosing which pretraining-data recipe to use for a large target-scale LLM,
based on small cheap training runs plus scaling-law extrapolation, is
currently a heuristic with no guarantees. AI2's **DataDecide**
(Magnusson et al., ICML 2025, [arXiv:2504.11393](https://arxiv.org/abs/2504.11393))
showed empirically that this heuristic is shaky: ranking recipes at a single
small size predicts the target-scale winner about 80% of the time, and none
of eight scaling-law extrapolation methods beat that. Nobody has explained
why.

This project formalizes "choosing the best pretraining-data recipe from
small-scale runs and scaling-law extrapolation" as a **fixed-confidence
best-arm identification (BAI) problem in which the reward at the target
scale is never observed, only extrapolated** — then derives:

1. an error bound decomposing decision error into a shrinking *variance*
   term and an irreducible *extrapolation-bias* term,
2. a compute/sample-complexity lower bound with a phase-transition /
   impossibility result when recipe rankings reverse across scale,
3. an identifiability condition and minimax rate, and
4. an active compute-allocation algorithm (Extrapolation-Track-and-Stop)
   for deciding which model sizes to train and how many times.

## Quickstart: selecting a recipe

The reference implementation (Extrapolation-Track-and-Stop, `pdt.bai.ets`)
is a deliverable in its own right. Given any object with `pull(recipe,
scale, seed)`, `cost(scale)`, and `available_scales()` methods (a
`PullOracle`), it returns a recipe recommendation plus a certificate:

```python
import numpy as np
from pdt.bai.ets import extrapolation_track_and_stop
from pdt.bai.oracle import SyntheticOracle  # or your own PullOracle
from pdt.scaling.base import Scale

scales = [Scale(n=1e6, d=2e7), Scale(n=3e6, d=6e7), Scale(n=1e7, d=2e8), Scale(n=3e7, d=6e8)]
target = Scale(n=1e9, d=2e10)
oracle = SyntheticOracle(
    recipes=["a", "b"], scales=scales, target_scale=target, rng=np.random.default_rng(0)
)

result = extrapolation_track_and_stop(
    oracle,
    ["a", "b"],
    scales,
    target,
    delta=0.1,
    eta={"a": 0.03, "b": 0.03},
    sigma2=lambda s: 1e-4,
)
print(result.outcome, result.recipe, result.certificate)
```

**Read [`docs/when_to_trust_extrapolation.md`](docs/when_to_trust_extrapolation.md)
before picking `eta` and `sigma2`** — a `"certified"` outcome is a
delta-level claim (a bound on `P[certifies AND wrong] <= delta`, not
"certification is never wrong," and not the conditional error rate given
certification) that holds **only under stated assumptions**, printed in
`result.certificate["assumptions"]`: `sigma2` is the *known* noise variance,
`eta` is a valid upper bound on each recipe's extrapolation bias, the
prediction is linear in the data (exact for `LogLinear`, first-order for the
nonlinear fits), and the design is independent of the noise being certified
(exact for the non-adaptive warm-up check, an unproved heuristic once tracking
adapts). Violating the first two can produce confident, wrong answers far more
often than `delta`. With `variance_mode="hc0_heuristic"` the algorithm returns
`"recommended"` instead: no error-probability claim at all.

Or from the command line, against a YAML config
([`configs/my_selection.yaml`](configs/my_selection.yaml) is a runnable
example):

```bash
pdt select --config configs/my_selection.yaml
```

## Where to start

- [`PLAN.md`](PLAN.md) — the master plan: phase map, critical path, deliverables.
- [`plan/00-agent-protocol.md`](plan/00-agent-protocol.md) — the operating
  rules this project is executed under (one task per session, no fabricated
  numbers, every compute request is a gate).
- [`plan/`](plan/) — one file per phase, with every task fully specified.
- [`STATUS.md`](STATUS.md) — the live task ledger.
- [`docs/decisions.md`](docs/decisions.md) — append-only decision log.

## Repository layout

```
src/pdt/        library code (importable, tested)
experiments/    thin runnable scripts, one per task
configs/        YAML configs — experiments never hardcode parameters
results/        machine-written JSON/CSV only, never hand-edited
figures/        script-generated figures only, never hand-edited
paper/          LaTeX submission
tests/          pytest
docs/           environment notes, decisions log, findings memos
plan/           the phase-by-phase implementation plan
```

## License

Apache-2.0 (see [`LICENSE`](LICENSE)). Derived analysis built on AI2's
DataDecide artifacts, which are released under ODC-BY / CC BY 4.0 — see
attribution notices in `docs/` once the Phase 1 analysis lands.
