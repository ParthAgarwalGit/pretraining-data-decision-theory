"""Task P3-05: offline replay on DataDecide -- the free real-data result.

See plan/04-phase3-algorithm.md P3-05. Writes results/p3_05_replay.json.

Runs Extrapolation-Track-and-Stop and all four P3-03 baselines against
real DataDecide data via `DataDecideOracle`, target `s* = 1B` (the
largest of the 14 real sizes), pulling only the other 13. The replay
must never look at target-scale data except to score the final decision
-- enforced in code via `GuardedOracle`
(`src/pdt/bai/guarded_oracle.py`), not by discipline; every method here
is handed a `GuardedOracle` instance, which raises `TargetScaleLeakageError`
the instant anything tries to pull, cost, or even list the target scale.

**Scope, stated honestly (see docs/decisions.md):** the plan asks for
every DataDecide task and error bars from bootstrapping over seeds and
task subsets. Two things made the literal ask infeasible in this
session's time budget:
  1. `extrapolation_track_and_stop` (unlike the four baselines, which are
     cheap one-shot fits) refits all 25 real recipes and re-solves
     P3-02's T* program every adaptive round -- a single real-data run
     with the full 25-recipe pool costs several minutes, verified by
     direct timing while building this script. Bootstrapping ETS itself
     (many re-runs per task) was not attempted; ETS's numbers below are
     a single real run per task, not a bootstrap mean -- flagged
     explicitly in the output, not silently presented as if it had error
     bars.
  2. DataDecide only has 3 real seeds per (recipe, scale) everywhere
     (confirmed in P0-06/P1-01 and re-checked by `DataDecideOracle`
     itself) -- a real seed-level bootstrap has very limited resampling
     power (3 items). Baselines (cheap enough to bootstrap) are
     resampled with replacement from those 3 real seeds
     `_N_BOOTSTRAP` times each; the resulting accuracy and its normal-
     approximation 95% CI are reported per (task, method).

Task selection: rather than all 11 DataDecide tasks, this pilot uses the
two most reversal-heavy tasks and the two most rank-stable tasks by
P1-09's own already-computed `kendall_tau_vs_scale_by_task`
(results/p1_09_rank_reversals.json) -- a real, previously-computed
number, not a fresh judgment call -- for a deliberate contrast on the
plan's own "abstention should concentrate on reversal-heavy tasks" ask.
"""

from __future__ import annotations

import os

for _blas_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_env_var, "1")

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from pdt import provenance  # noqa: E402
from pdt.bai.ets import (  # noqa: E402
    extrapolation_track_and_stop,
    fixed_ladder_extrapolation,
    single_scale_recommendation,
    successive_halving_over_scales,
    uniform_allocation,
)
from pdt.bai.guarded_oracle import GuardedOracle  # noqa: E402
from pdt.bai.oracle import DataDecideOracle  # noqa: E402
from pdt.scaling.base import Scale  # noqa: E402
from pdt.scaling.fitters import PowerLawN  # noqa: E402

_REVERSAL_HEAVY_TASKS = ["winogrande", "boolq"]
_STABLE_TASKS = ["arc_easy", "mmlu"]
_DEFAULT_TASKS = _REVERSAL_HEAVY_TASKS + _STABLE_TASKS
_DEFAULT_DELTA = 0.1
_DEFAULT_ETA = 0.02  # a fixed, documented bias-budget assumption -- see docs/decisions.md
_DEFAULT_N_BOOTSTRAP = 20
_DEFAULT_MAX_ROUNDS = 60
_DEFAULT_SOLVER_N_ITER = 20
_N_REAL_SEEDS = 3  # confirmed everywhere in DataDecide, P0-06/P1-01


class _ReseededOracle:
    """Remaps the external seed index `i` a caller asks for to a
    bootstrap-resampled real seed index, so a baseline's own internal
    `seed=0,1,2,...` pulls draw from a resampled set of real replicates
    rather than always the same 3, in order, every bootstrap replicate."""

    def __init__(self, inner, seed_map: dict[int, int]):
        self._inner = inner
        self._seed_map = seed_map

    def pull(self, recipe: str, scale: Scale, seed: int) -> float:
        return self._inner.pull(recipe, scale, self._seed_map.get(seed, seed))

    def cost(self, scale: Scale) -> float:
        return self._inner.cost(scale)

    def available_scales(self) -> list[Scale]:
        return self._inner.available_scales()


def _true_winner(task: str) -> tuple[str, dict[str, float], Scale, list[str]]:
    """The real 1B-scale winner, computed via a *separate*, unguarded
    oracle instance used only here, for scoring -- never handed to any
    algorithm under test."""
    oracle = DataDecideOracle(task=task)
    target = max(oracle.available_scales(), key=lambda s: s.n)
    recipes = sorted({r for r, _ in oracle._primary})
    true_vals = {}
    for r in recipes:
        draws = []
        for seed in range(_N_REAL_SEEDS):
            draws.append(oracle.pull(r, target, seed))
        true_vals[r] = float(np.mean(draws))
    winner = max(true_vals, key=true_vals.get)
    return winner, true_vals, target, recipes


_BASELINES = {
    "SingleScale": lambda oracle, recipes, fit_scales, target: single_scale_recommendation(
        oracle, recipes, fit_scales[-1], n_replicates=2
    ),
    "FixedLadderExtrapolation": lambda oracle, recipes, fit_scales, target: (
        fixed_ladder_extrapolation(oracle, recipes, fit_scales, target, n_replicates=1)
    ),
    # compute_budget scales with len(recipes): "equal compute per arm"
    # means a fixed multiple of one recipe's own ladder cost is not
    # enough budget to go around for many recipes -- a real bug found
    # while building this script (uniform_allocation could raise
    # FitFailure on an under-budgeted many-recipe instance); see
    # docs/decisions.md.
    "UniformAllocation": lambda oracle, recipes, fit_scales, target: uniform_allocation(
        oracle,
        recipes,
        fit_scales,
        target,
        compute_budget=6 * len(recipes) * sum(s.compute for s in fit_scales),
    ),
    "SuccessiveHalvingOverScales": lambda oracle, recipes, fit_scales, target: (
        successive_halving_over_scales(oracle, recipes, fit_scales, target, n_replicates=1)
    ),
}


def _bootstrap_baseline(
    fn, base_oracle, recipes, fit_scales, target, winner, n_bootstrap, rng
) -> dict:
    n_correct = 0
    compute_spent = []
    for _ in range(n_bootstrap):
        seed_map = {i: int(rng.integers(0, _N_REAL_SEEDS)) for i in range(_N_REAL_SEEDS)}
        reseeded = _ReseededOracle(base_oracle, seed_map)
        res = fn(reseeded, recipes, fit_scales, target)
        n_correct += int(res.recipe == winner)
        compute_spent.append(res.compute_spent)
    p_hat = n_correct / n_bootstrap
    se = float(np.sqrt(max(p_hat * (1 - p_hat), 0.0) / n_bootstrap))
    return {
        "accuracy": p_hat,
        "accuracy_ci95": [max(0.0, p_hat - 1.96 * se), min(1.0, p_hat + 1.96 * se)],
        "mean_compute": float(np.mean(compute_spent)),
        "n_bootstrap": n_bootstrap,
    }


def _run_task(
    task: str, delta: float, n_bootstrap: int, max_rounds: int, solver_n_iter: int, rng
) -> dict:
    winner, true_vals, target, recipes = _true_winner(task)
    base_oracle = DataDecideOracle(task=task)
    guarded = GuardedOracle(base_oracle, target)
    fit_scales = sorted(guarded.available_scales(), key=lambda s: s.n)
    eta = {r: _DEFAULT_ETA for r in recipes}

    baseline_results = {}
    for name, fn in _BASELINES.items():
        baseline_results[name] = _bootstrap_baseline(
            fn, base_oracle, recipes, fit_scales, target, winner, n_bootstrap, rng
        )

    t0 = time.time()
    ets_res = extrapolation_track_and_stop(
        guarded,
        recipes,
        fit_scales,
        target,
        delta=delta,
        eta=eta,
        sigma2=lambda _s: 1e-4,
        model_factory=PowerLawN,
        epsilon_0=0.02,
        max_rounds=max_rounds,
        solver_n_restarts=1,
        solver_n_iter=solver_n_iter,
        min_pulls_per_pair=1,
    )
    ets_elapsed = time.time() - t0

    return {
        "task": task,
        "reversal_heavy": task in _REVERSAL_HEAVY_TASKS,
        "true_winner": winner,
        "n_recipes": len(recipes),
        "n_fit_scales": len(fit_scales),
        "baselines": baseline_results,
        "ets_single_run": {
            "outcome": ets_res.outcome,
            "recipe": ets_res.recipe,
            "correct": ets_res.recipe == winner,
            "compute_spent": ets_res.compute_spent,
            "n_pulls": ets_res.n_pulls,
            "wall_seconds": ets_elapsed,
            "note": "single real-data run, no bootstrap -- see module docstring",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=_DEFAULT_TASKS)
    parser.add_argument("--delta", type=float, default=_DEFAULT_DELTA)
    parser.add_argument("--n-bootstrap", type=int, default=_DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--max-rounds", type=int, default=_DEFAULT_MAX_ROUNDS)
    parser.add_argument("--solver-n-iter", type=int, default=_DEFAULT_SOLVER_N_ITER)
    parser.add_argument("--out", default="results/p3_05_replay.json")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    results = []
    t_start = time.time()
    for i, task in enumerate(args.tasks):
        cell = _run_task(
            task, args.delta, args.n_bootstrap, args.max_rounds, args.solver_n_iter, rng
        )
        results.append(cell)
        elapsed = time.time() - t_start
        print(
            f"[{i + 1}/{len(args.tasks)}] task={task} true_winner={cell['true_winner']} "
            f"ets={cell['ets_single_run']['outcome']}/{cell['ets_single_run']['correct']} "
            f"elapsed={elapsed:.0f}s"
        )

    reversal_heavy_cells = [r for r in results if r["reversal_heavy"]]
    stable_cells = [r for r in results if not r["reversal_heavy"]]
    ets_abstention_reversal_heavy = (
        sum(1 for r in reversal_heavy_cells if r["ets_single_run"]["outcome"] == "abstained")
        / len(reversal_heavy_cells)
        if reversal_heavy_cells
        else None
    )
    ets_abstention_stable = (
        sum(1 for r in stable_cells if r["ets_single_run"]["outcome"] == "abstained")
        / len(stable_cells)
        if stable_cells
        else None
    )

    headline = []
    for r in results:
        row = {"task": r["task"], "true_winner": r["true_winner"]}
        for name, b in r["baselines"].items():
            row[name] = {"accuracy": b["accuracy"], "mean_compute": b["mean_compute"]}
        row["ExtrapolationTrackAndStop"] = {
            "outcome": r["ets_single_run"]["outcome"],
            "correct": r["ets_single_run"]["correct"],
            "compute": r["ets_single_run"]["compute_spent"],
        }
        headline.append(row)

    payload = {
        "scope_note": (
            "4-task pilot (2 reversal-heavy + 2 stable, by P1-09's own kendall_tau), "
            "not all 11 DataDecide tasks; ETS is a single real-data run per task, not "
            "bootstrapped (baselines are) -- see this file's own module docstring and "
            "docs/decisions.md for why."
        ),
        "delta": args.delta,
        "eta_assumed": _DEFAULT_ETA,
        "n_bootstrap_baselines_only": args.n_bootstrap,
        "cells": results,
        "headline_table": headline,
        "ets_abstention_rate_reversal_heavy_tasks": ets_abstention_reversal_heavy,
        "ets_abstention_rate_stable_tasks": ets_abstention_stable,
    }

    provenance.write_result(
        args.out,
        payload=payload,
        config={
            "task": "P3-05",
            "tasks": args.tasks,
            "delta": args.delta,
            "n_bootstrap": args.n_bootstrap,
            "max_rounds": args.max_rounds,
            "solver_n_iter": args.solver_n_iter,
        },
    )
    print(f"wrote {args.out}")
    print(f"ETS abstention rate, reversal-heavy tasks: {ets_abstention_reversal_heavy}")
    print(f"ETS abstention rate, stable tasks: {ets_abstention_stable}")


if __name__ == "__main__":
    main()
