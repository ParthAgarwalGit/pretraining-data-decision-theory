"""Task P3-04: simulation study for Extrapolation-Track-and-Stop.

See plan/04-phase3-algorithm.md P3-04. Writes results/p3_04_simulation.json.

**Scope, stated honestly up front (see docs/decisions.md for the full
account):** the plan specifies a sweep of `K in {5,10,25} x delta in
{0.05,0.1,0.2} x eta_level in {0,small,medium,large} x gap_structure in
{well-separated, close top-two, reversing}` with >= 200 runs per cell
(108 cells, 21,600+ runs). `extrapolation_track_and_stop` refits every
recipe's extrapolator every round and re-solves P3-02's T* program every
round -- each run can take seconds to tens of seconds even at K=3-5, and
scales up sharply with K. Running the literal grid to completion was not
feasible within this session's time budget. This script runs a smaller
but real pilot by default (the module-level `_DEFAULT_*` constants below)
and reports genuine results for it, with every default overridable via
CLI flags so the full grid can be run later given more compute time. This
is a stated, deliberate scope reduction, not a silently-cut corner.

**Instance construction:** each cell's K recipes are power-law curves
(`e - a*N^-alpha`, the same functional family SyntheticOracle and
PowerLawN both use) with a controlled gap structure:
  - `well_separated`: spread-out ceilings/decay rates, stable ranking,
    comfortable gaps throughout.
  - `close_top_two`: same, but the top two ceilings are pulled to within
    0.005 of each other.
  - `reversing`: the true runner-up trails at every *fit* scale but is
    constructed (higher ceiling, steeper decay, solved to cross the
    apparent leader's curve at a scale chosen strictly between the
    fit ladder and the target) to genuinely overtake it by the target
    scale -- a real, resolvable-in-principle rank reversal (the same
    phenomenon P1-01/P1-03 found empirically in DataDecide), not
    Theorem 2 Part B's stronger "literally unresolvable" construction.

`eta_level` layers an *additional*, target-only bump (Theorem 2 Part B's
`phi(x)=clip(x,0,1)` shape) on top of every non-winning recipe,
proportional to its own true gap to the winner (`ETA_FRACTIONS[eta_level]
* gap`). The algorithm is always given the *exactly correct* `eta` for
this bump (a perfectly-calibrated-eta scenario) -- so Theorem 4's
delta-correctness hypothesis holds by construction at every eta_level,
letting claim 1 (error rate respects delta) be checked cleanly across the
whole eta sweep while claims 2/3 (compute, abstention) are expected to
degrade as eta_level grows.
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
import itertools  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

from pdt import provenance  # noqa: E402
from pdt.analysis.intervals import clopper_pearson  # noqa: E402
from pdt.bai.ets import (  # noqa: E402
    extrapolation_track_and_stop,
    fixed_ladder_extrapolation,
    single_scale_recommendation,
    successive_halving_over_scales,
    uniform_allocation,
)
from pdt.bai.oracle import _stable_seed  # noqa: E402
from pdt.scaling.base import Scale  # noqa: E402
from pdt.scaling.fitters import PowerLawN  # noqa: E402

_FIT_SCALES = [Scale(n=n, d=20 * n) for n in [1e6, 3e6, 1e7, 3e7]]
_TARGET = Scale(n=1e8, d=20e8)
_SIGMA = 0.05

ETA_FRACTIONS = {"none": 0.0, "small": 0.3, "medium": 1.0, "large": 2.0}

_DEFAULT_KS = [3]
_DEFAULT_DELTAS = [0.05, 0.2]
_DEFAULT_ETA_LEVELS = ["none", "large"]
_DEFAULT_GAP_STRUCTURES = ["well_separated", "reversing"]
_DEFAULT_N_RUNS = 15
_DEFAULT_MAX_ROUNDS = 100
_DEFAULT_SOLVER_N_ITER = 25
# See docs/decisions.md for why this default is > ETS's own default of 1:
# with only 1 pull per (recipe, scale) pair, the sandwich-covariance
# variance estimate (analytic_v_k) has essentially zero residual degrees
# of freedom for a 3-parameter PowerLawN fit against exactly 4 scales,
# and was found -- concretely, not hypothetically -- to sometimes report
# an implausibly small v_hat purely by chance, causing a false Certified
# decision as early as round 1. A modest amount of front-loaded warm-up
# replication was found to eliminate this in direct testing.
_DEFAULT_MIN_PULLS_PER_PAIR = 8


class _Instance:
    """A PullOracle over K recipes with a controlled gap structure --
    same generative shape (and the same Theorem 2 Part B bump for
    misspecification) as tests/test_ets.py's `_BumpOracle`, generalized
    to K arms and to the power-law family directly."""

    def __init__(self, params: dict, fit_scales: list[Scale], target_scale: Scale, sigma: float):
        self.params = params
        self.fit_scales = fit_scales
        self.target_scale = target_scale
        self.sigma = sigma

    def _h(self, recipe: str, scale: Scale) -> float:
        p = self.params[recipe]
        s_max = max(s.n for s in self.fit_scales)
        s_star = self.target_scale.n
        if scale.n <= s_max:
            return 0.0
        x = (scale.n - s_max) / (s_star - s_max)
        return p["bias_at_target"] * float(np.clip(x, 0.0, 1.0))

    def _true_mean(self, recipe: str, scale: Scale) -> float:
        p = self.params[recipe]
        return p["e"] - p["a"] * scale.n ** (-p["alpha"]) + self._h(recipe, scale)

    def pull(self, recipe: str, scale: Scale, seed: int) -> float:
        mean = self._true_mean(recipe, scale)
        rng = np.random.default_rng(_stable_seed(recipe, scale.n, scale.d, seed))
        return float(mean + rng.normal(0.0, self.sigma))

    def cost(self, scale: Scale) -> float:
        return scale.compute

    def available_scales(self) -> list[Scale]:
        return list(self.fit_scales)

    def true_value_at_target(self, recipe: str) -> float:
        return self._true_mean(recipe, self.target_scale)


def _base_params(rng: np.random.Generator, k: int) -> tuple[dict, list[str]]:
    # Evenly-spaced ceilings, not k raw uniform draws: k independent draws
    # from Uniform(0.5, 0.85) can land arbitrarily close together by
    # chance for small k (found the hard way -- an early version of this
    # script used raw uniform draws and "well_separated" cells sometimes
    # had a near-zero true top-two gap purely from unlucky sampling,
    # producing a misleadingly high apparent error rate that had nothing
    # to do with the algorithm -- see docs/decisions.md). Spacing
    # guarantees a real minimum gap of `spacing - 2*jitter` between every
    # adjacent pair; jitter is capped well below that so ordering never
    # inverts.
    spacing = 0.35 / max(k - 1, 1)
    jitter = min(0.01, spacing / 4)
    es = np.linspace(0.85, 0.5, k) + rng.uniform(-jitter, jitter, size=k)
    alphas = rng.uniform(0.2, 0.4, size=k)
    a_s = rng.uniform(1.0, 3.0, size=k)
    recipes = [f"r{i}" for i in range(k)]
    params = {
        r: {"e": float(es[i]), "a": float(a_s[i]), "alpha": float(alphas[i]), "bias_at_target": 0.0}
        for i, r in enumerate(recipes)
    }
    return params, recipes


def make_instance(
    rng: np.random.Generator,
    k: int,
    gap_structure: str,
    eta_level: str,
    fit_scales: list[Scale] = _FIT_SCALES,
    target_scale: Scale = _TARGET,
    sigma: float = _SIGMA,
) -> tuple[_Instance, list[str], str, dict[str, float]]:
    """Returns `(instance, recipes, k_star, eta_assumed)`."""
    params, recipes = _base_params(rng, k)

    if gap_structure == "close_top_two":
        params[recipes[1]]["e"] = params[recipes[0]]["e"] - 0.005

    if gap_structure == "reversing":
        if k < 2:
            raise ValueError("reversing gap structure needs at least 2 recipes")
        # A random (alpha, a) perturbation almost never lands the crossover
        # exactly inside the (max fit scale, target scale) window: too
        # steep an alpha1 and recipe1 has already overtaken before the
        # fit ladder even ends; too shallow and it never catches up by
        # the target -- found by direct numerical search while building
        # this script (see docs/decisions.md). So the crossover scale
        # `n_c` is chosen directly (log-uniform, close to but strictly
        # past the max fit scale), and `a1` is *solved for* so the two
        # curves meet exactly there -- `alpha1 > alpha0` (steeper decay)
        # then guarantees a single crossover: recipe1 trails everywhere
        # below n_c and leads everywhere above it.
        #
        # A second, more consequential problem was found the same way: a
        # crossover placed too close to the *target* (rather than to the
        # fit boundary) needs `a1` to be huge (tens to ~100) to make up
        # the gap over a comparatively short remaining distance --
        # `PowerLawN`'s own fit bounds cap `a` at +-10
        # (src/pdt/scaling/fitters.py), so a demanded `a1` outside that
        # range is a curve PowerLawN cannot even represent, not a
        # genuinely hard-but-fittable extrapolation. An early version of
        # this script did not check for this and produced "reversing"
        # instances the fitter was structurally unable to identify no
        # matter how much data it got -- see docs/decisions.md for the
        # full account of how this was traced (comparing the fitted
        # theta against the true one on a large-sample probe run showed
        # the fit pinned at the `a=-10` boundary). Retrying with the
        # crossover placed close to the fit boundary (where less
        # catching-up is needed) keeps `|a1|` in a representable range in
        # roughly half of draws.
        #
        # For SOME base (e0, a0, alpha0) draws, no crossover in the
        # searched window satisfies the bound at all within 200 tries
        # (found the same way, on a rare seed) -- the outer loop below
        # redraws the base parameters too, not just the crossover, rather
        # than raising on an unlucky base draw.
        s_max_fit = max(s.n for s in fit_scales)
        for _outer_attempt in range(20):
            found = False
            for _attempt in range(200):
                log_cross = rng.uniform(np.log10(s_max_fit) + 0.02, np.log10(s_max_fit) + 0.25)
                n_c = 10**log_cross
                e0, a0, alpha0 = (
                    params[recipes[0]]["e"],
                    params[recipes[0]]["a"],
                    params[recipes[0]]["alpha"],
                )
                alpha1 = alpha0 * rng.uniform(1.05, 1.25)
                e1 = e0 + rng.uniform(0.003, 0.012)
                a1 = (e1 - e0 + a0 * n_c ** (-alpha0)) / (n_c ** (-alpha1))
                if abs(a1) > 8.0:
                    continue
                params[recipes[1]]["e"] = e1
                params[recipes[1]]["alpha"] = alpha1
                params[recipes[1]]["a"] = a1

                probe = _Instance(params, fit_scales, target_scale, sigma)
                v0_fit = [probe._true_mean(recipes[0], s) for s in fit_scales]
                v1_fit = [probe._true_mean(recipes[1], s) for s in fit_scales]
                v0_t = probe.true_value_at_target(recipes[0])
                v1_t = probe.true_value_at_target(recipes[1])
                if all(a > b for a, b in zip(v0_fit, v1_fit, strict=True)) and v1_t > v0_t:
                    found = True
                    break
            if found:
                break
            params, recipes = _base_params(rng, k)  # redraw and retry
        else:
            raise RuntimeError(
                "could not construct a fittable reversing instance after 20 x 200 attempts"
            )

    base_instance = _Instance(params, fit_scales, target_scale, sigma)
    base_true_vals = {r: base_instance.true_value_at_target(r) for r in recipes}
    k_star = max(base_true_vals, key=base_true_vals.get)

    frac = ETA_FRACTIONS[eta_level]
    for r in recipes:
        if r != k_star:
            params[r]["bias_at_target"] = frac * (base_true_vals[k_star] - base_true_vals[r])

    instance = _Instance(params, fit_scales, target_scale, sigma)
    true_vals = {r: instance.true_value_at_target(r) for r in recipes}
    k_star = max(true_vals, key=true_vals.get)  # re-resolved: a large enough
    # eta_level can flip the true ranking -- a real, intended possibility
    # (Theorem 2 Part B's construction can do exactly this), not a bug.

    # eta_assumed[r] must be each recipe's OWN bias magnitude, regardless
    # of whether r happens to be k_star -- a recipe can BE the (re-resolved)
    # winner precisely BECAUSE its own target-only bump was large enough to
    # overtake the original leader, so "is the winner" and "has zero bias"
    # are not the same thing. An earlier version special-cased eta=0 for
    # k_star unconditionally; PR #30's review reproduced a concrete case
    # (make_instance(default_rng(0), 3, "well_separated", "large")) where
    # the re-resolved winner's own bias_at_target is ~0.623, not 0 --
    # falsely telling the algorithm the winner is bias-free breaks the
    # "eta is given, exactly correct" calibration this whole pilot's
    # claim-1 check depends on.
    eta_assumed = {r: abs(params[r]["bias_at_target"]) for r in recipes}
    return instance, recipes, k_star, eta_assumed


def _run_one_cell(
    rng: np.random.Generator,
    k: int,
    delta: float,
    eta_level: str,
    gap_structure: str,
    n_runs: int,
    max_rounds: int,
    solver_n_iter: int,
    min_pulls_per_pair: int,
) -> dict:
    ets_outcomes: Counter[str] = Counter()
    n_abstained_bias_floor = 0
    n_abstained_timeout = 0
    ets_correct_given_certified = 0
    ets_compute_certified: list[float] = []
    ets_compute_all: list[float] = []
    baseline_correct: dict[str, int] = {
        "SingleScale": 0,
        "FixedLadderExtrapolation": 0,
        "UniformAllocation": 0,
        "SuccessiveHalvingOverScales": 0,
    }

    for run_idx in range(n_runs):
        instance, recipes, k_star, eta_assumed = make_instance(rng, k, gap_structure, eta_level)

        res = extrapolation_track_and_stop(
            instance,
            recipes,
            _FIT_SCALES,
            _TARGET,
            delta=delta,
            eta=eta_assumed,
            sigma2=lambda _s: _SIGMA**2,
            model_factory=PowerLawN,
            epsilon_0=0.03,
            max_rounds=max_rounds,
            solver_n_restarts=1,
            solver_n_iter=solver_n_iter,
            min_pulls_per_pair=min_pulls_per_pair,
            rng=np.random.default_rng(_stable_seed("ets-internal", k, delta, eta_level, run_idx)),
        )
        ets_outcomes[res.outcome] += 1
        ets_compute_all.append(res.compute_spent)
        if res.outcome == "certified":
            ets_compute_certified.append(res.compute_spent)
            if res.recipe == k_star:
                ets_correct_given_certified += 1
        elif res.outcome == "abstained":
            # ets.py's SelectionResult.certificate["reason"] distinguishes
            # a genuine bias-floor abstention ("bias floor" -- Theorem 4's
            # abstention condition, c_t <= epsilon_0 and the margin can't
            # clear it) from simply running out of rounds
            # ("max_rounds exhausted without certifying or abstaining").
            # Lumping both under one "abstention rate" cannot support a
            # claim about the algorithm RECOGNIZING an impossible instance
            # (PR #30's review, P2) -- a timeout says nothing about
            # whether Theorem 4's own abstention condition was ever met,
            # only that this pilot's max_rounds budget was too small.
            if res.certificate.get("reason") == "bias floor":
                n_abstained_bias_floor += 1
            else:
                n_abstained_timeout += 1

        budget = 20 * sum(s.compute for s in _FIT_SCALES)
        single = single_scale_recommendation(instance, recipes, _FIT_SCALES[-1], n_replicates=3)
        ladder = fixed_ladder_extrapolation(instance, recipes, _FIT_SCALES, _TARGET, n_replicates=2)
        uniform = uniform_allocation(instance, recipes, _FIT_SCALES, _TARGET, compute_budget=budget)
        halving = successive_halving_over_scales(
            instance, recipes, _FIT_SCALES, _TARGET, n_replicates=2
        )
        baseline_correct["SingleScale"] += int(single.recipe == k_star)
        baseline_correct["FixedLadderExtrapolation"] += int(ladder.recipe == k_star)
        baseline_correct["UniformAllocation"] += int(uniform.recipe == k_star)
        baseline_correct["SuccessiveHalvingOverScales"] += int(halving.recipe == k_star)

    n_certified = ets_outcomes["certified"]
    return {
        "k": k,
        "delta": delta,
        "eta_level": eta_level,
        "gap_structure": gap_structure,
        "n_runs": n_runs,
        "ets_outcomes": dict(ets_outcomes),
        # NOT a valid proxy for "the algorithm recognized this instance
        # was impossible" -- includes both genuine bias-floor abstention
        # and simple round-cap timeout. Kept for backward-compatible
        # context; use the two split rates below for anything claim-3
        # actually needs to say.
        "ets_abstention_rate": ets_outcomes["abstained"] / n_runs,
        "ets_abstention_rate_bias_floor": n_abstained_bias_floor / n_runs,
        "ets_abstention_rate_timeout": n_abstained_timeout / n_runs,
        "ets_error_rate_given_certified": (
            1.0 - ets_correct_given_certified / n_certified if n_certified > 0 else None
        ),
        "ets_error_rate_overall": (n_certified - ets_correct_given_certified) / n_runs,
        # The delta-correctness guarantee bounds the JOINT rate P[certified AND wrong],
        # i.e. `ets_error_rate_overall`; an exact 95% interval on it is what a
        # violation must be judged against (the conditional rate above can rest on
        # one certified run).
        "ets_n_wrong_certified": n_certified - ets_correct_given_certified,
        "ets_joint_error_ci95": list(
            clopper_pearson(n_certified - ets_correct_given_certified, n_runs)
        ),
        "ets_mean_compute_given_certified": (
            float(np.mean(ets_compute_certified)) if ets_compute_certified else None
        ),
        "ets_mean_compute_overall": float(np.mean(ets_compute_all)),
        "baseline_accuracy": {name: count / n_runs for name, count in baseline_correct.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks", type=int, nargs="+", default=_DEFAULT_KS)
    parser.add_argument("--deltas", type=float, nargs="+", default=_DEFAULT_DELTAS)
    parser.add_argument("--eta-levels", nargs="+", default=_DEFAULT_ETA_LEVELS)
    parser.add_argument("--gap-structures", nargs="+", default=_DEFAULT_GAP_STRUCTURES)
    parser.add_argument("--n-runs", type=int, default=_DEFAULT_N_RUNS)
    parser.add_argument("--max-rounds", type=int, default=_DEFAULT_MAX_ROUNDS)
    parser.add_argument("--solver-n-iter", type=int, default=_DEFAULT_SOLVER_N_ITER)
    parser.add_argument("--min-pulls-per-pair", type=int, default=_DEFAULT_MIN_PULLS_PER_PAIR)
    parser.add_argument("--out", default="results/p3_04_simulation.json")
    args = parser.parse_args()

    cells = list(itertools.product(args.ks, args.deltas, args.eta_levels, args.gap_structures))
    n_total = len(cells) * args.n_runs
    print(f"p3_04_simulation: {len(cells)} cells x {args.n_runs} runs = {n_total} runs")

    results = []
    t_start = time.time()
    for i, (k, delta, eta_level, gap_structure) in enumerate(cells):
        rng = np.random.default_rng(_stable_seed("p3-04-cell", k, delta, eta_level, gap_structure))
        cell = _run_one_cell(
            rng,
            k,
            delta,
            eta_level,
            gap_structure,
            args.n_runs,
            args.max_rounds,
            args.solver_n_iter,
            args.min_pulls_per_pair,
        )
        results.append(cell)
        elapsed = time.time() - t_start
        print(
            f"[{i + 1}/{len(cells)}] K={k} delta={delta} eta={eta_level} gap={gap_structure} "
            f"-> outcomes={cell['ets_outcomes']} error_given_certified="
            f"{cell['ets_error_rate_given_certified']} elapsed={elapsed:.0f}s"
        )

    # --- Claim checks -------------------------------------------------
    well_specified_cells = [r for r in results if r["gap_structure"] != "reversing"]
    # A violation of the JOINT guarantee P[certified AND wrong] <= delta is detected
    # only when the exact interval's lower end exceeds delta.
    claim1_violations = [
        r for r in well_specified_cells if r["ets_joint_error_ci95"][0] > r["delta"]
    ]
    claim1_holds = len(claim1_violations) == 0

    by_delta: dict[float, list[float]] = {}
    for r in results:
        if r["ets_mean_compute_given_certified"] is not None:
            by_delta.setdefault(r["delta"], []).append(r["ets_mean_compute_given_certified"])
    deltas_sorted = sorted(by_delta)
    claim2_note = (
        "compute-to-stop by delta (smaller delta should cost more, if the "
        "T*log(1/delta) scaling holds): "
        + ", ".join(f"delta={d}: mean={np.mean(by_delta[d]):.3g}" for d in deltas_sorted)
    )

    reversing_cells = [r for r in results if r["gap_structure"] == "reversing"]
    claim3_summary = [
        {
            "eta_level": r["eta_level"],
            "delta": r["delta"],
            "k": r["k"],
            # ets_abstention_rate alone (both reasons combined) cannot
            # support "the algorithm recognized the instance was
            # impossible" -- that claim needs the bias-floor rate
            # specifically; the timeout rate is reported alongside so a
            # reader can see when the pilot's max_rounds budget, not
            # Theorem 4's abstention condition, is what's driving
            # non-certification (PR #30's review, P2).
            "ets_abstention_rate": r["ets_abstention_rate"],
            "ets_abstention_rate_bias_floor": r["ets_abstention_rate_bias_floor"],
            "ets_abstention_rate_timeout": r["ets_abstention_rate_timeout"],
            "ets_error_rate_given_certified": r["ets_error_rate_given_certified"],
            "baseline_accuracy": r["baseline_accuracy"],
        }
        for r in reversing_cells
    ]

    payload = {
        "fit_scales": [{"n": s.n, "d": s.d} for s in _FIT_SCALES],
        "target_scale": {"n": _TARGET.n, "d": _TARGET.d},
        "sigma": _SIGMA,
        "eta_fractions": ETA_FRACTIONS,
        "scope_note": (
            "Pilot run, not the plan's literal K in {5,10,25} x delta in "
            "{0.05,0.1,0.2} x eta in {0,small,medium,large} x gap_structure in "
            "{well-separated,close top-two,reversing}, 200+ runs/cell grid -- "
            "see this file's own module docstring and docs/decisions.md for why."
        ),
        "cells": results,
        "claim1_error_rate_respects_delta_in_well_specified_regime": {
            "holds": claim1_holds,
            "violations": claim1_violations,
        },
        "claim2_compute_to_stop_note": claim2_note,
        "claim3_reversing_regime_abstention_vs_baselines": claim3_summary,
    }

    provenance.write_result(
        args.out,
        payload=payload,
        config={
            "task": "P3-04",
            "ks": args.ks,
            "deltas": args.deltas,
            "eta_levels": args.eta_levels,
            "gap_structures": args.gap_structures,
            "n_runs": args.n_runs,
            "max_rounds": args.max_rounds,
            "solver_n_iter": args.solver_n_iter,
            "min_pulls_per_pair": args.min_pulls_per_pair,
        },
    )
    print(f"wrote {args.out}")
    print(f"claim 1 (error rate respects delta, well-specified regime): holds={claim1_holds}")
    print(f"claim 2: {claim2_note}")
    print(f"claim 3 (reversing regime): {claim3_summary}")


if __name__ == "__main__":
    main()
