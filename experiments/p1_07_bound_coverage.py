"""Task P1-07: the plug-in bound and its empirical coverage.

See plan/02-phase1-datadecide.md task P1-07. Three pieces:

1. Marginal and pairwise-difference bound values (src/pdt/theory/bound.py),
   computed per (fitter, design, task) from P1-02's ground-truth gaps and
   P1-06's bootstrap bias/variance estimates.
2. The analytic delta-method v_k(C) (sandwich covariance x jacobian),
   computed by refitting each (fitter, design, task, recipe) once on its
   real (non-bootstrapped) trajectory -- reported alongside P1-06's
   bootstrap v_hat_k as a cross-check of the analytic machinery, not
   substituted into the reported bound (which uses P1-06's bootstrap
   estimates throughout, for consistency with sigma2_extrap_hat, which has
   no purely-analytic counterpart).
3. A Monte-Carlo estimate of the actual selection error (>=500 resamples,
   seed-bootstrap scheme -- see docs/decisions.md for why this scheme, not
   parametric, was chosen as the canonical one for this specific check),
   compared against both bound forms as a tightness ratio.

Writes results/p1_07_bound_coverage.json.

**If the tightness ratio for a task/design/fitter ever drops below 1**
(the bound is violated), this script does NOT silently continue past it:
it prints the offending cells explicitly and records
`any_bound_violation: true` in the summary. Per the plan, this is a
critical finding to stop and report to the PI as an unscheduled GATE, not
something to paper over.
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

import concurrent.futures  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402

from pdt import provenance  # noqa: E402
from pdt.analysis import bootstrap as bs  # noqa: E402
from pdt.analysis import decision_accuracy as da  # noqa: E402
from pdt.analysis import ground_truth as gt  # noqa: E402
from pdt.data import datadecide as dd  # noqa: E402
from pdt.data import frame as frame_mod  # noqa: E402
from pdt.scaling import fitters  # noqa: E402
from pdt.scaling.base import FitFailure, Scale  # noqa: E402
from pdt.theory import bound  # noqa: E402

_TARGET = "1B"
_METRIC = "primary_metric"
_B_MONTE_CARLO = 500
_MC_SCHEME = "seed_bootstrap"
_P1_06_RESULTS_PATH = "results/p1_06_decomposition.json"

_DESIGN_ENDPOINTS = {
    "S_fit_le_150M": "150M",
    "S_fit_le_300M": "300M",
    "S_fit_le_530M": "530M",
}

_FITTER_CLASSES = {
    "ConstantExtrapolator": fitters.ConstantExtrapolator,
    "PowerLawN": fitters.PowerLawN,
    "PowerLawC": fitters.PowerLawC,
    "ChinchillaND": fitters.ChinchillaND,
    "TwoStepLadder": fitters.TwoStepLadder,
    "LogLinear": fitters.LogLinear,
}


def _seed_for(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _sizes_ascending(long_frame) -> list[str]:
    sizes = long_frame.select(["params_str", "params_num"]).unique().sort("params_num")
    return sizes["params_str"].to_list()


def _designs(sizes_ascending: list[str]) -> dict[str, list[str]]:
    designs = {}
    for design_name, endpoint in _DESIGN_ENDPOINTS.items():
        idx = sizes_ascending.index(endpoint)
        designs[design_name] = sizes_ascending[: idx + 1]
    return designs


def _load_p1_06() -> dict:
    with open(_P1_06_RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)["data"]


# ---------------------------------------------------------------------------
# Piece 2: analytic v_k(C) via the sandwich/delta method (cheap: one fit
# per (fitter, design, task, recipe), no resampling).
# ---------------------------------------------------------------------------


def _compute_analytic_v_k(
    fitter_name: str,
    avg_trajectory: dict[str, list[tuple[Scale, float]]],
    target_scale: Scale,
) -> dict[str, dict]:
    fitter_cls = _FITTER_CLASSES[fitter_name]
    result = {}
    for recipe, trajectory in avg_trajectory.items():
        scales = [s for s, _ in trajectory]
        values = [v for _, v in trajectory]
        seed = _seed_for("analytic", fitter_name, recipe)
        try:
            model = fitter_cls(rng=np.random.default_rng(seed))
            model.fit(scales, values)
            v_k = bound.analytic_v_k(model, scales, values, target_scale)
        except (FitFailure, np.linalg.LinAlgError) as exc:
            result[recipe] = {"ok": False, "error": str(exc)}
            continue
        result[recipe] = {"ok": True, "analytic_v_k": v_k}
    return result


# ---------------------------------------------------------------------------
# Piece 3: Monte-Carlo empirical selection error (expensive: B replicates
# x 25 recipes per (fitter, design, task), seed-bootstrap scheme).
# ---------------------------------------------------------------------------


@dataclass
class _McWork:
    fitter_name: str
    design_name: str
    task: str
    target_scale: Scale
    seed_trajectory: dict[str, list[tuple[Scale, list[float]]]]
    k_star: str


def _run_mc_combo(work: _McWork) -> dict:
    fitter_cls = _FITTER_CLASSES[work.fitter_name]
    recipes = sorted(work.seed_trajectory.keys())
    first_recipe = recipes[0]

    n_correct = 0
    n_valid_replicates = 0
    n_discarded_replicates = 0
    n_fits_failed = 0

    for b in range(_B_MONTE_CARLO):
        replicate_seed = _seed_for(work.design_name, work.task, _MC_SCHEME, "mc", str(b))
        replicate_rng = np.random.default_rng(replicate_seed)
        shared_patterns = [
            bs.draw_seed_resample_pattern(replicate_rng, len(seed_values))
            for _, seed_values in work.seed_trajectory[first_recipe]
        ]

        predictions: dict[str, float] = {}
        replicate_ok = True
        for recipe in recipes:
            scales = [s for s, _ in work.seed_trajectory[recipe]]
            values = [
                bs.apply_seed_resample(pattern, seed_values)
                for (_, seed_values), pattern in zip(
                    work.seed_trajectory[recipe], shared_patterns, strict=True
                )
            ]
            fit_seed = _seed_for(
                work.fitter_name, work.design_name, work.task, "mc", recipe, str(b)
            )
            try:
                model = fitter_cls(rng=np.random.default_rng(fit_seed))
                model.fit(scales, values)
                pred = model.predict(work.target_scale)
            except FitFailure:
                n_fits_failed += 1
                replicate_ok = False
                continue
            if not math.isfinite(pred):
                n_fits_failed += 1
                replicate_ok = False
                continue
            predictions[recipe] = pred

        if not replicate_ok or len(predictions) != len(recipes):
            n_discarded_replicates += 1
            continue

        k_hat = max(predictions, key=predictions.get)
        n_valid_replicates += 1
        if k_hat == work.k_star:
            n_correct += 1

    return {
        "fitter": work.fitter_name,
        "design": work.design_name,
        "task": work.task,
        "n_valid_replicates": n_valid_replicates,
        "n_discarded_replicates": n_discarded_replicates,
        "n_fits_failed": n_fits_failed,
        "empirical_error_rate": (
            1.0 - n_correct / n_valid_replicates if n_valid_replicates else None
        ),
    }


def main() -> None:
    long_frame = frame_mod.build_frame(source="macro_avg", metrics=(_METRIC,))
    ground_truth = gt.compute_ground_truth(long_frame, _METRIC, _TARGET)
    p1_06 = _load_p1_06()

    sizes_ascending = _sizes_ascending(long_frame)
    if sizes_ascending[-1] != _TARGET:
        raise ValueError(f"expected largest size to be {_TARGET!r}, got {sizes_ascending[-1]!r}")
    designs = _designs(sizes_ascending)

    target_row = (
        long_frame.filter((long_frame["params_str"] == _TARGET) & (long_frame["is_final"]))
        .select(["params_num", "tokens"])
        .unique()
    )
    if target_row.height != 1:
        raise ValueError(f"expected exactly one (N, D) pair at {_TARGET}, got {target_row.height}")
    target_scale = Scale(n=target_row["params_num"][0], d=target_row["tokens"][0])

    tasks = sorted(ground_truth.keys())
    print(
        f"p1_07_bound_coverage: {len(tasks)} tasks, {len(designs)} designs, "
        f"{len(_FITTER_CLASSES)} fitters"
    )

    avg_traj_by_design_task: dict[tuple[str, str], dict] = {}
    seed_traj_by_design_task: dict[tuple[str, str], dict] = {}
    for design_name, design_sizes in designs.items():
        avg_by_task = da.recipe_trajectories(long_frame, _METRIC, design_sizes, seed_mode="average")
        seed_by_task = bs.recipe_seed_trajectories(long_frame, _METRIC, design_sizes)
        for task in tasks:
            avg_traj_by_design_task[(design_name, task)] = avg_by_task[task]
            seed_traj_by_design_task[(design_name, task)] = seed_by_task[task]

    # Piece 2: analytic v_k -- cheap, run inline (no process pool needed).
    print("p1_07_bound_coverage: computing analytic v_k (delta method)...")
    t0 = time.perf_counter()
    analytic_v_k_by_combo: dict[tuple[str, str, str], dict] = {}
    for fitter_name in _FITTER_CLASSES:
        for design_name in designs:
            for task in tasks:
                analytic_v_k_by_combo[(fitter_name, design_name, task)] = _compute_analytic_v_k(
                    fitter_name, avg_traj_by_design_task[(design_name, task)], target_scale
                )
    print(f"p1_07_bound_coverage: analytic v_k done in {time.perf_counter() - t0:.1f}s")

    # Piece 3: Monte-Carlo empirical selection error -- expensive, parallelized.
    mc_work_units = [
        _McWork(
            fitter_name=fitter_name,
            design_name=design_name,
            task=task,
            target_scale=target_scale,
            seed_trajectory=seed_traj_by_design_task[(design_name, task)],
            k_star=ground_truth[task]["k_star"],
        )
        for fitter_name in _FITTER_CLASSES
        for design_name in designs
        for task in tasks
    ]
    print(
        f"p1_07_bound_coverage: {len(mc_work_units)} Monte-Carlo work units, "
        f"B={_B_MONTE_CARLO}, {os.cpu_count()} CPUs available"
    )

    t0 = time.perf_counter()
    mc_results = []
    n_workers = os.cpu_count() or 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_mc_combo, w): w for w in mc_work_units}
        n_done = 0
        for future in concurrent.futures.as_completed(futures):
            mc_results.append(future.result())
            n_done += 1
            if n_done % 10 == 0 or n_done == len(mc_work_units):
                elapsed = time.perf_counter() - t0
                print(
                    f"p1_07_bound_coverage: {n_done}/{len(mc_work_units)} MC combos done "
                    f"({elapsed:.0f}s elapsed)"
                )
    print(f"p1_07_bound_coverage: Monte-Carlo done in {time.perf_counter() - t0:.0f}s")

    mc_by_combo = {(r["fitter"], r["design"], r["task"]): r for r in mc_results}

    # Assemble bounds + tightness ratios.
    by_fitter: dict[str, dict] = {}
    violations = []
    for fitter_name in _FITTER_CLASSES:
        for design_name in designs:
            for task in tasks:
                gt_task = ground_truth[task]
                k_star = gt_task["k_star"]
                for scheme in ("seed_bootstrap", "parametric_bootstrap"):
                    try:
                        cell = p1_06["by_fitter"][fitter_name][design_name][task][scheme]
                    except KeyError:
                        continue
                    marginal_terms = []
                    pairwise_terms = []
                    for recipe, gap in gt_task["gaps"].items():
                        m = cell["marginal"].get(recipe, {})
                        p = cell["pairwise"].get(recipe, {})
                        if m.get("insufficient_replicates", True) is not False:
                            continue
                        if p.get("insufficient_replicates", True) is not False:
                            continue
                        marginal_terms.append((gap, m["sigma2_extrap_hat"], m["v_hat"]))
                        pairwise_terms.append((gap, p["bias_hat"], p["v_hat"]))

                    if not marginal_terms:
                        continue

                    bound_marginal = bound.marginal_bound(marginal_terms)
                    bound_pairwise = bound.pairwise_bound(pairwise_terms)

                    mc = mc_by_combo.get((fitter_name, design_name, task))
                    empirical = mc["empirical_error_rate"] if mc else None

                    ratio_marginal = (
                        bound_marginal / empirical if empirical and empirical > 0 else None
                    )
                    ratio_pairwise = (
                        bound_pairwise / empirical if empirical and empirical > 0 else None
                    )

                    entry = {
                        "k_star": k_star,
                        "n_terms": len(marginal_terms),
                        "bound_marginal": bound_marginal,
                        "bound_pairwise": bound_pairwise,
                        "empirical_error_rate": empirical,
                        "n_valid_mc_replicates": mc["n_valid_replicates"] if mc else None,
                        "tightness_ratio_marginal": ratio_marginal,
                        "tightness_ratio_pairwise": ratio_pairwise,
                        "analytic_v_k": {
                            r: v.get("analytic_v_k")
                            for r, v in analytic_v_k_by_combo[
                                (fitter_name, design_name, task)
                            ].items()
                            if v.get("ok")
                        },
                    }
                    by_fitter.setdefault(fitter_name, {}).setdefault(design_name, {}).setdefault(
                        task, {}
                    )[scheme] = entry

                    for label, ratio in (
                        ("marginal", ratio_marginal),
                        ("pairwise", ratio_pairwise),
                    ):
                        if ratio is not None and ratio < 1.0:
                            violations.append(
                                {
                                    "fitter": fitter_name,
                                    "design": design_name,
                                    "task": task,
                                    "scheme": scheme,
                                    "bound_form": label,
                                    "tightness_ratio": ratio,
                                }
                            )

    if violations:
        print(f"p1_07_bound_coverage: *** {len(violations)} BOUND VIOLATIONS FOUND ***")
        for v in violations:
            print(f"  VIOLATION: {v}")
    else:
        print("p1_07_bound_coverage: no bound violations found (ratio >= 1 everywhere)")

    payload = {
        "target_scale": _TARGET,
        "metric": _METRIC,
        "b_monte_carlo": _B_MONTE_CARLO,
        "mc_scheme": _MC_SCHEME,
        "designs": list(designs.keys()),
        "by_fitter": by_fitter,
        "any_bound_violation": len(violations) > 0,
        "violations": violations,
        "dataset_revision_macro_avg": dd.cached_revision("macro_avg"),
    }

    provenance.write_result(
        "results/p1_07_bound_coverage.json",
        payload=payload,
        config={"task": "P1-07"},
    )
    print("wrote results/p1_07_bound_coverage.json")


if __name__ == "__main__":
    main()
