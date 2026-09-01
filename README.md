# A Statistical Decision Theory for Pretraining-Data Selection

**Status: Phase 0 (setup) — no research results yet.**

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
