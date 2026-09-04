"""Task P1-06: the bias/variance decomposition (the core result).

See plan/02-phase1-datadecide.md task P1-06. For each (extrapolator M,
design S_fit, recipe k, task t): bootstrap-resample the fitting data (two
schemes, seed and parametric), refit, and decompose the resulting spread
in the s*-prediction into estimation variance (v_hat) and squared bias
(sigma2_extrap_hat), de-biased against the P1-05 target-truth noise. Also
computes the pairwise-difference version (D_k = mu_hat_k* - mu_hat_k from
the *same* replicate), which is what the actual decision depends on.
Writes results/p1_06_decomposition.json and docs/findings/p1_06.md.

Runs all 6 fitters (including ConstantExtrapolator, per the plan's
explicit instruction) x all 3 P1-04 designs x all 11 tasks x both
resampling schemes = 396 (fitter, design, task, scheme) work units, each
running B=200 replicates x 25 recipes. ~2M individual scaling-law fits
total -- parallelized across processes (one process pool worker per work
unit; each unit's own fits run sequentially inside that worker), since a
serial run would take on the order of a day. See docs/decisions.md for the
compute-tractability reasoning and the shared-replicate-draw design that
makes the pairwise correlation-preservation in step 4 actually work.
"""

from __future__ import annotations

import os

# Every one of this script's ~2M individual fits works on <=12 data
# points -- far too small for BLAS's own internal multi-threading to help,
# and with ~20 worker processes each spawning a full BLAS thread pool by
# default, the result is severe oversubscription (up to n_cpus^2 threads)
# that crashed a real run here with "OpenBLAS error: Memory allocation
# still failed after 10 retries, giving up" and a BrokenProcessPool. Must
# be set before numpy/scipy are imported (by this process AND by every
# spawned worker, which re-executes this module's top-level code under
# Windows' spawn-based multiprocessing) -- see docs/decisions.md.
_BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _blas_env_var in _BLAS_ENV_VARS:
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

_TARGET = "1B"
_METRIC = "primary_metric"
_B_REPLICATES = 200
_MIN_SUCCESSFUL_REPLICATES = 20  # 10% of B; below this, flag rather than report a decomposition
_P1_05_RESULTS_PATH = "results/p1_05_noise.json"
_SCHEMES = ("seed_bootstrap", "parametric_bootstrap")

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


def _load_p1_05_noise() -> dict:
    with open(_P1_05_RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)["data"]


def _params_str_to_n(p1_05_data: dict) -> dict[str, float]:
    """`params_str` -> `params_num`, from `seed_variance` (the only one of
    P1-05's three tables that carries both) -- `checkpoint_jitter` and
    `eval_sampling_noise` only carry `params_str`, so this is how their
    rows get a numeric `n` to key on."""
    return {r["params_str"]: float(r["params_num"]) for r in p1_05_data["seed_variance"]}


def _sigma2_total_lookup(p1_05_data: dict) -> dict[tuple[str, str, float], float]:
    """(recipe, task, params_num) -> sigma2_seed + sigma2_ckpt + sigma2_eval
    at that cell -- the parametric bootstrap's per-scale noise variance."""
    size_to_n = _params_str_to_n(p1_05_data)

    seed_by_key = {
        (r["recipe"], r["task"], float(r["params_num"])): r["sigma2_seed"]
        for r in p1_05_data["seed_variance"]
    }
    ckpt_by_key = {
        (r["recipe"], r["task"], size_to_n[r["params_str"]]): r["sigma2_ckpt"]
        for r in p1_05_data["checkpoint_jitter"]
    }
    eval_by_key = {
        (r["recipe"], r["task"], size_to_n[r["params_str"]]): r["sigma2_eval"]
        for r in p1_05_data["eval_sampling_noise"]
    }

    keys = set(seed_by_key) | set(ckpt_by_key) | set(eval_by_key)
    return {
        k: seed_by_key.get(k, 0.0) + ckpt_by_key.get(k, 0.0) + eval_by_key.get(k, 0.0) for k in keys
    }


def _sigma2_target_lookup(p1_05_data: dict) -> dict[tuple[str, str], float]:
    return {(r["recipe"], r["task"]): r["sigma2_target"] for r in p1_05_data["sigma2_target_at_1b"]}


@dataclass
class _ComboWork:
    fitter_name: str
    design_name: str
    task: str
    scheme: str
    target_scale: Scale
    avg_trajectory: dict[str, list[tuple[Scale, float]]]
    seed_trajectory: dict[str, list[tuple[Scale, list[float]]]] | None
    sigma2_total: dict[tuple[str, float], float]  # (recipe, n) -> noise, pre-filtered to this task
    k_star: str
    mu_true: dict[str, float]
    sigma2_target: dict[str, float]


def _shared_draws_per_scale(
    work: _ComboWork, first_recipe: str, replicate_rng: np.random.Generator
) -> list:
    """The ONE shared per-scale draw for this replicate (a resample
    pattern per scale for seed bootstrap, a standard-normal z per scale
    for parametric) -- drawn once per scale using `first_recipe` only to
    learn the scale list (every recipe shares the same scale set for a
    given design), then reused for every recipe by `_resample_recipe`.
    This is what makes the pairwise correlation-preservation in
    `bootstrap.py`'s module docstring actually happen: every recipe sees
    the identical draw for a given (scale, replicate).
    """
    if work.scheme == "parametric_bootstrap":
        return [
            bs.draw_parametric_noise_z(replicate_rng) for _ in work.avg_trajectory[first_recipe]
        ]
    return [
        bs.draw_seed_resample_pattern(replicate_rng, len(seed_values))
        for _, seed_values in work.seed_trajectory[first_recipe]
    ]


def _resample_recipe(
    work: _ComboWork, recipe: str, shared_draws: list
) -> tuple[list[Scale], list[float]]:
    """Apply this replicate's shared per-scale draws to one recipe's own
    observed values -- the recipe-specific half of the resample."""
    if work.scheme == "parametric_bootstrap":
        scales = [s for s, _ in work.avg_trajectory[recipe]]
        values = [
            bs.apply_parametric_noise(z, mu, work.sigma2_total.get((recipe, s.n), 0.0))
            for (s, mu), z in zip(work.avg_trajectory[recipe], shared_draws, strict=True)
        ]
        return scales, values
    scales = [s for s, _ in work.seed_trajectory[recipe]]
    values = [
        bs.apply_seed_resample(pattern, seed_values)
        for (_, seed_values), pattern in zip(
            work.seed_trajectory[recipe], shared_draws, strict=True
        )
    ]
    return scales, values


_ROUND_SIGFIGS = 8


def _round_sigfigs(x: float, sigfigs: int = _ROUND_SIGFIGS) -> float:
    """Round `x` to `sigfigs` significant figures -- these are noisy
    statistical estimates (bootstrap standard errors on ~200 replicates
    give roughly 2 meaningful digits at best), so full float64 precision
    (~17 digits) in the JSON is pure bloat, not information. At the full
    396-combo x ~49-recipe-entry scale this is the difference between a
    ~15MB and a ~5MB results file."""
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, sigfigs - 1 - int(math.floor(math.log10(abs(x)))))


def _compact_decomposition(decomp: dict) -> dict:
    """Drop `mean_prediction`/`n_replicates` (derivable from `bias_hat` +
    the known ground truth, and always 200 when not flagged insufficient
    -- neither is read by any downstream consumer) and round the 3 fields
    that matter to a sane precision. See `_ROUND_SIGFIGS`."""
    return {
        "v_hat": _round_sigfigs(decomp["v_hat"]),
        "bias_hat": _round_sigfigs(decomp["bias_hat"]),
        "sigma2_extrap_hat": _round_sigfigs(decomp["sigma2_extrap_hat"]),
        "insufficient_replicates": False,
    }


def _run_one_combo(work: _ComboWork) -> dict:
    """One (fitter, design, task, scheme) work unit: B replicates x every
    recipe, fit + predict, then the marginal and pairwise decompositions.
    Top-level function (not a closure) so it can be pickled to worker
    processes under Windows' spawn-based multiprocessing.
    """
    fitter_cls = _FITTER_CLASSES[work.fitter_name]
    recipes = sorted(work.mu_true.keys())
    first_recipe = recipes[0]

    replicate_predictions: dict[str, list[float]] = {r: [] for r in recipes}
    n_attempted = 0
    n_failed = 0

    for b in range(_B_REPLICATES):
        # One rng per replicate, seeded on (design, task, scheme, b) only
        # -- no fitter, no recipe -- so the resampled *data* for this
        # replicate doesn't depend on which fitter will later summarize
        # it, and every recipe sees the same shared draws below.
        replicate_seed = _seed_for(work.design_name, work.task, work.scheme, str(b))
        replicate_rng = np.random.default_rng(replicate_seed)
        shared_draws = _shared_draws_per_scale(work, first_recipe, replicate_rng)

        for recipe in recipes:
            n_attempted += 1
            scales, values = _resample_recipe(work, recipe, shared_draws)
            fit_seed = _seed_for(
                work.fitter_name, work.design_name, work.task, work.scheme, recipe, str(b)
            )
            try:
                model = fitter_cls(rng=np.random.default_rng(fit_seed))
                model.fit(scales, values)
                pred = model.predict(work.target_scale)
            except FitFailure:
                n_failed += 1
                continue
            if not math.isfinite(pred):
                n_failed += 1
                continue
            replicate_predictions[recipe].append(pred)

    marginal: dict[str, dict] = {}
    for recipe in recipes:
        preds = replicate_predictions[recipe]
        if len(preds) < _MIN_SUCCESSFUL_REPLICATES:
            marginal[recipe] = {
                "insufficient_replicates": True,
                "n_successful_replicates": len(preds),
            }
            continue
        decomp = bs.bias_variance_decomposition(
            preds, work.mu_true[recipe], work.sigma2_target.get(recipe, 0.0)
        )
        marginal[recipe] = _compact_decomposition(decomp)

    pairwise: dict[str, dict] = {}
    k_star_preds = replicate_predictions[work.k_star]
    for recipe in recipes:
        if recipe == work.k_star:
            continue
        preds = replicate_predictions[recipe]
        n_common = min(len(preds), len(k_star_preds))
        if n_common < _MIN_SUCCESSFUL_REPLICATES:
            pairwise[recipe] = {
                "insufficient_replicates": True,
                "n_successful_replicates": n_common,
            }
            continue
        d_k_replicates = [k_star_preds[i] - preds[i] for i in range(n_common)]
        true_gap = work.mu_true[work.k_star] - work.mu_true[recipe]
        pairwise_sigma2_target = work.sigma2_target.get(work.k_star, 0.0) + work.sigma2_target.get(
            recipe, 0.0
        )
        decomp = bs.bias_variance_decomposition(d_k_replicates, true_gap, pairwise_sigma2_target)
        pairwise[recipe] = _compact_decomposition(decomp)

    return {
        "fitter": work.fitter_name,
        "design": work.design_name,
        "task": work.task,
        "scheme": work.scheme,
        "n_fits_attempted": n_attempted,
        "n_fits_failed": n_failed,
        "marginal": marginal,
        "pairwise": pairwise,
    }


def main() -> None:
    long_frame = frame_mod.build_frame(source="macro_avg", metrics=(_METRIC,))
    ground_truth = gt.compute_ground_truth(long_frame, _METRIC, _TARGET)
    p1_05_data = _load_p1_05_noise()
    sigma2_total = _sigma2_total_lookup(p1_05_data)
    sigma2_target = _sigma2_target_lookup(p1_05_data)

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
        f"p1_06_decomposition: {len(tasks)} tasks, {len(designs)} designs, "
        f"{len(_FITTER_CLASSES)} fitters"
    )
    print(f"p1_06_decomposition: B={_B_REPLICATES} replicates x {len(_SCHEMES)} schemes")

    # Precompute per (design, task) trajectory data once -- shared across
    # every fitter x scheme work unit that needs it (12 per (design,task)).
    avg_traj_by_design_task: dict[tuple[str, str], dict] = {}
    seed_traj_by_design_task: dict[tuple[str, str], dict] = {}
    for design_name, design_sizes in designs.items():
        avg_by_task = da.recipe_trajectories(long_frame, _METRIC, design_sizes, seed_mode="average")
        seed_by_task = bs.recipe_seed_trajectories(long_frame, _METRIC, design_sizes)
        for task in tasks:
            avg_traj_by_design_task[(design_name, task)] = avg_by_task[task]
            seed_traj_by_design_task[(design_name, task)] = seed_by_task[task]

    # sigma2_total is keyed (recipe, task, n) globally; each ComboWork only
    # needs its own task's slice, keyed (recipe, n) -- pre-filter once per
    # task rather than re-filtering per (fitter, design, scheme) work unit.
    sigma2_total_by_task: dict[str, dict[tuple[str, float], float]] = {t: {} for t in tasks}
    for (recipe, task, n), value in sigma2_total.items():
        if task in sigma2_total_by_task:
            sigma2_total_by_task[task][(recipe, n)] = value

    work_units: list[_ComboWork] = []
    for fitter_name in _FITTER_CLASSES:
        for design_name in designs:
            for task in tasks:
                gt_task = ground_truth[task]
                for scheme in _SCHEMES:
                    work_units.append(
                        _ComboWork(
                            fitter_name=fitter_name,
                            design_name=design_name,
                            task=task,
                            scheme=scheme,
                            target_scale=target_scale,
                            avg_trajectory=avg_traj_by_design_task[(design_name, task)],
                            seed_trajectory=seed_traj_by_design_task[(design_name, task)],
                            sigma2_total=sigma2_total_by_task[task],
                            k_star=gt_task["k_star"],
                            mu_true=gt_task["mu"],
                            sigma2_target={
                                r: sigma2_target.get((r, task), 0.0) for r in gt_task["mu"]
                            },
                        )
                    )

    print(f"p1_06_decomposition: {len(work_units)} work units, {os.cpu_count()} CPUs available")

    t0 = time.perf_counter()
    results = []
    n_workers = os.cpu_count() or 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one_combo, w): w for w in work_units}
        n_done = 0
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            n_done += 1
            if n_done % 10 == 0 or n_done == len(work_units):
                elapsed = time.perf_counter() - t0
                print(
                    f"p1_06_decomposition: {n_done}/{len(work_units)} combos done "
                    f"({elapsed:.0f}s elapsed)"
                )

    total_failed = sum(r["n_fits_failed"] for r in results)
    total_attempted = sum(r["n_fits_attempted"] for r in results)
    print(
        f"p1_06_decomposition: all combos done in {time.perf_counter() - t0:.0f}s; "
        f"{total_failed}/{total_attempted} individual bootstrap fits failed"
    )

    # Keyed by (fitter, design, task, scheme); also feeds the
    # ratio-vs-compute summary below (the plan's "signature prediction":
    # sigma2_extrap_hat / v_hat should grow with compute in S_fit).
    by_fitter: dict[str, dict] = {}
    for r in results:
        by_fitter.setdefault(r["fitter"], {}).setdefault(r["design"], {}).setdefault(r["task"], {})[
            r["scheme"]
        ] = {
            "n_fits_attempted": r["n_fits_attempted"],
            "n_fits_failed": r["n_fits_failed"],
            "marginal": r["marginal"],
            "pairwise": r["pairwise"],
        }

    size_scale_lookup = {
        row["params_str"]: Scale(n=row["params_num"], d=row["tokens"])
        for row in long_frame.filter(long_frame["is_final"])
        .select(["params_str", "params_num", "tokens"])
        .unique()
        .iter_rows(named=True)
    }
    design_compute = {
        name: da.compute_cost([size_scale_lookup[s] for s in sizes])
        for name, sizes in designs.items()
    }

    ratio_vs_compute = []
    for fitter_name, by_design in by_fitter.items():
        for design_name, by_task in by_design.items():
            for scheme in _SCHEMES:
                ratios = []
                sigma2_extrap_values = []
                v_hat_values = []
                for by_scheme in by_task.values():
                    for recipe_stats in by_scheme[scheme]["marginal"].values():
                        if recipe_stats.get("insufficient_replicates"):
                            continue
                        v_hat = recipe_stats["v_hat"]
                        sigma2_extrap = recipe_stats["sigma2_extrap_hat"]
                        sigma2_extrap_values.append(sigma2_extrap)
                        v_hat_values.append(v_hat)
                        if v_hat > 0:
                            ratios.append(sigma2_extrap / v_hat)
                if ratios:
                    ratio_vs_compute.append(
                        {
                            "fitter": fitter_name,
                            "design": design_name,
                            "scheme": scheme,
                            "compute_cost": design_compute[design_name],
                            "median_ratio_sigma2_extrap_over_v": float(np.median(ratios)),
                            "median_sigma2_extrap_hat": float(np.median(sigma2_extrap_values)),
                            "median_v_hat": float(np.median(v_hat_values)),
                            "n_recipe_task_cells": len(ratios),
                        }
                    )

    payload = {
        "target_scale": _TARGET,
        "metric": _METRIC,
        "b_replicates": _B_REPLICATES,
        "schemes": list(_SCHEMES),
        "designs": {
            name: {"sizes": sizes, "compute_cost": design_compute[name]}
            for name, sizes in designs.items()
        },
        "n_fits_total_attempted": total_attempted,
        "n_fits_total_failed": total_failed,
        "by_fitter": by_fitter,
        "ratio_vs_compute": ratio_vs_compute,
        "dataset_revision_macro_avg": dd.cached_revision("macro_avg"),
    }

    provenance.write_result(
        "results/p1_06_decomposition.json",
        payload=payload,
        config={"task": "P1-06"},
    )
    print("wrote results/p1_06_decomposition.json")


if __name__ == "__main__":
    main()
