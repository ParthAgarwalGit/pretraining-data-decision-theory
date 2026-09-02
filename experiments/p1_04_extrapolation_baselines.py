"""Task P1-04: scaling-law extrapolation baselines vs. the single-scale frontier.

See plan/02-phase1-datadecide.md task P1-04. For each of the 6 extrapolators
in `pdt.scaling`, each held-out proxy design `S_fit` (all sizes <=150M,
<=300M, <=530M), and each of the 11 macro_avg tasks: fit per recipe on
`S_fit`'s trajectory, predict at the target scale (1B), and score pairwise
decision accuracy against the P1-02 ground truth. Compare each design's
macro-averaged accuracy against the single-scale baseline from P1-03 AT
MATCHED COMPUTE (log-compute interpolation of P1-03's per-size points, not
compared to the same largest size). Writes results/p1_04_extrapolation.json.

Uses the same headline definition P1-03 used (primary_metric, seed-averaged,
macro_avg source) so the two frontiers are actually comparable -- mixing
metrics or seed-handling between the extrapolation points and the
single-scale points here would silently invalidate the "at matched compute"
comparison the plan asks for.

Target to reproduce: no extrapolation method exceeds the single-scale
frontier.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from pdt import provenance
from pdt.analysis import decision_accuracy as da
from pdt.analysis import ground_truth as gt
from pdt.data import datadecide as dd
from pdt.data import frame as frame_mod
from pdt.scaling import fitters
from pdt.scaling.base import FitFailure, Scale

_TARGET = "1B"
_METRIC = "primary_metric"
_SEED_MODE = "average"
_P1_03_RESULTS_PATH = "results/p1_03_single_scale.json"
_P1_03_VARIANT_KEY = "primary_metric__average"  # must match _METRIC/_SEED_MODE above

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
    """Deterministic per-fit RNG seed, independent of iteration order --
    unlike a single shared `np.random.default_rng`, adding/removing a fit
    elsewhere can't perturb this one's random restarts. Uses sha256, not
    Python's built-in `hash()`, because str hashing is randomized per
    process (PYTHONHASHSEED) and would break the run-to-run reproducibility
    this whole project's verification discipline depends on."""
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _sizes_ascending(long_frame) -> list[str]:
    sizes = long_frame.select(["params_str", "params_num"]).unique().sort("params_num")
    return sizes["params_str"].to_list()


def _designs(sizes_ascending: list[str]) -> dict[str, list[str]]:
    designs = {}
    for design_name, endpoint in _DESIGN_ENDPOINTS.items():
        idx = sizes_ascending.index(endpoint)  # raises ValueError if the label isn't real
        designs[design_name] = sizes_ascending[: idx + 1]
    return designs


def _size_scale_lookup(long_frame) -> dict[str, Scale]:
    """One (N, D) pair per size label, read directly from final-checkpoint
    rows -- confirmed by inspection to be constant across every recipe,
    seed and task at a given size (DataDecide trains every recipe on the
    same fixed size ladder), so a single lookup is safe to reuse for every
    recipe's compute-cost accounting rather than re-deriving it per fit."""
    rows = long_frame.filter("is_final").select(["params_str", "params_num", "tokens"]).unique()
    if rows["params_str"].n_unique() != rows.height:
        raise ValueError(
            "expected exactly one (params_num, tokens) pair per size label at "
            f"is_final=True, found {rows.height} distinct pairs across "
            f"{rows['params_str'].n_unique()} labels -- DataDecide's fixed-size-ladder "
            "assumption this lookup relies on may not hold; investigate before trusting "
            "any compute_cost() figure downstream."
        )
    by_size = {
        r["params_str"]: Scale(n=r["params_num"], d=r["tokens"]) for r in rows.iter_rows(named=True)
    }
    return by_size


def _load_single_scale_frontier(size_scale: dict[str, Scale]) -> dict:
    with open(_P1_03_RESULTS_PATH, encoding="utf-8") as f:
        p1_03 = json.load(f)
    variant = p1_03["data"]["variants"][_P1_03_VARIANT_KEY]
    points = []
    for size, by_size in variant.items():
        acc = by_size["macro_avg_accuracy_including_ties"]
        if acc is None:
            continue
        points.append(
            {
                "params_str": size,
                "compute_cost": da.compute_cost([size_scale[size]]),
                "accuracy_including_ties": acc,
            }
        )
    points.sort(key=lambda p: p["compute_cost"])
    return {
        "source_file": _P1_03_RESULTS_PATH,
        "source_git_sha": p1_03["provenance"]["git_sha"],
        "source_variant": _P1_03_VARIANT_KEY,
        "points": points,
    }


def _log_interp_accuracy(points: list[dict], target_compute: float) -> tuple[float | None, bool]:
    """Linear interpolation in log10(compute) space of the single-scale
    frontier's (compute_cost, accuracy_including_ties) points. Returns
    (value, out_of_range) -- None/True when `target_compute` falls outside
    the observed single-scale range, since extrapolating the *comparison
    frontier itself* is a different, unrequested claim this task doesn't
    make (see docs/decisions.md)."""
    computes = [p["compute_cost"] for p in points]
    if target_compute < computes[0] or target_compute > computes[-1]:
        return None, True

    for i in range(len(points) - 1):
        c_lo, c_hi = computes[i], computes[i + 1]
        if c_lo <= target_compute <= c_hi:
            if c_lo == c_hi:
                return points[i]["accuracy_including_ties"], False
            a_lo = points[i]["accuracy_including_ties"]
            a_hi = points[i + 1]["accuracy_including_ties"]
            log_lo, log_hi, log_t = math.log10(c_lo), math.log10(c_hi), math.log10(target_compute)
            frac = (log_t - log_lo) / (log_hi - log_lo)
            return a_lo + frac * (a_hi - a_lo), False
    raise AssertionError("unreachable: target_compute passed the range check above")


def _fit_and_predict(
    fitter_name: str,
    design_name: str,
    task: str,
    recipe: str,
    trajectory: list[tuple[Scale, float]],
    target_scale: Scale,
) -> dict:
    scales = [s for s, _ in trajectory]
    values = [v for _, v in trajectory]
    seed = _seed_for(fitter_name, design_name, task, recipe)
    model = _FITTER_CLASSES[fitter_name](rng=np.random.default_rng(seed))
    try:
        model.fit(scales, values)
        prediction = model.predict(target_scale)
    except FitFailure as exc:
        return {"ok": False, "error": str(exc)}
    if not math.isfinite(prediction):
        return {"ok": False, "error": f"non-finite prediction: {prediction!r}"}
    diag = model.fit_diagnostics
    return {
        "ok": True,
        "prediction": prediction,
        "n_converged": diag.get("n_converged"),
        "objective_spread": diag.get("objective_spread"),
    }


def _run_fitter_design(
    fitter_name: str,
    design_sizes: list[str],
    long_frame,
    ground_truth: dict,
    target_scale: Scale,
    design_name: str,
) -> dict:
    trajectories = da.recipe_trajectories(long_frame, _METRIC, design_sizes, seed_mode=_SEED_MODE)

    per_task: dict[str, dict] = {}
    for task in sorted(ground_truth.keys()):
        recipe_trajectories_for_task = trajectories.get(task, {})
        predictions: dict[str, float] = {}
        failures: list[dict] = []
        n_converged_values: list[float] = []
        objective_spread_values: list[float] = []

        for recipe in sorted(recipe_trajectories_for_task.keys()):
            outcome = _fit_and_predict(
                fitter_name,
                design_name,
                task,
                recipe,
                recipe_trajectories_for_task[recipe],
                target_scale,
            )
            if outcome["ok"]:
                predictions[recipe] = outcome["prediction"]
                if outcome["n_converged"] is not None:
                    n_converged_values.append(outcome["n_converged"])
                if outcome["objective_spread"] is not None:
                    objective_spread_values.append(outcome["objective_spread"])
            else:
                failures.append({"recipe": recipe, "error": outcome["error"]})

        target = ground_truth[task]
        decision = da.pairwise_decision_accuracy(
            predictions, target["mu"], target["sd_seed"], target["n_seeds"]
        )

        per_task[task] = {
            "n_recipes_attempted": len(recipe_trajectories_for_task),
            "n_recipes_succeeded": len(predictions),
            "n_recipes_failed": len(failures),
            "failures": failures,
            "mean_n_converged": (
                sum(n_converged_values) / len(n_converged_values) if n_converged_values else None
            ),
            "mean_objective_spread": (
                sum(objective_spread_values) / len(objective_spread_values)
                if objective_spread_values
                else None
            ),
            "decision_accuracy": decision,
        }

    incl = [
        v["decision_accuracy"]["accuracy_including_ties"]
        for v in per_task.values()
        if v["decision_accuracy"]["accuracy_including_ties"] is not None
    ]
    excl = [
        v["decision_accuracy"]["accuracy_excluding_ties"]
        for v in per_task.values()
        if v["decision_accuracy"]["accuracy_excluding_ties"] is not None
    ]
    taus = [
        v["decision_accuracy"]["kendall_tau"]
        for v in per_task.values()
        if v["decision_accuracy"]["kendall_tau"] is not None
    ]

    return {
        "per_task": per_task,
        "n_tasks": len(per_task),
        "macro_avg_accuracy_including_ties": (sum(incl) / len(incl)) if incl else None,
        "macro_avg_accuracy_excluding_ties": (sum(excl) / len(excl)) if excl else None,
        "macro_avg_kendall_tau": (sum(taus) / len(taus)) if taus else None,
    }


def main() -> None:
    long_frame = frame_mod.build_frame(source="macro_avg", metrics=(_METRIC,))
    ground_truth = gt.compute_ground_truth(long_frame, _METRIC, _TARGET)
    print(f"p1_04_extrapolation_baselines: ground truth computed for {len(ground_truth)} tasks")

    sizes_ascending = _sizes_ascending(long_frame)
    if sizes_ascending[-1] != _TARGET:
        raise ValueError(
            f"expected the largest observed size to be the target {_TARGET!r}, "
            f"got {sizes_ascending[-1]!r} -- schema surprise, investigate before trusting "
            "any design built below."
        )
    designs = _designs(sizes_ascending)
    size_scale = _size_scale_lookup(long_frame)
    target_scale = size_scale[_TARGET]

    print("p1_04_extrapolation_baselines: designs:")
    for name, sizes in designs.items():
        cost = da.compute_cost([size_scale[s] for s in sizes])
        print(f"  {name}: {len(sizes)} sizes ({sizes[0]}..{sizes[-1]}), compute_cost={cost:.3e}")

    single_scale_frontier = _load_single_scale_frontier(size_scale)

    results_by_fitter: dict[str, dict] = {}
    for fitter_name in _FITTER_CLASSES:
        results_by_design: dict[str, dict] = {}
        for design_name, design_sizes in designs.items():
            print(f"p1_04_extrapolation_baselines: fitting {fitter_name} / {design_name} ...")
            result = _run_fitter_design(
                fitter_name, design_sizes, long_frame, ground_truth, target_scale, design_name
            )
            compute_cost = da.compute_cost([size_scale[s] for s in design_sizes])
            matched_acc, out_of_range = _log_interp_accuracy(
                single_scale_frontier["points"], compute_cost
            )
            beats = None
            if matched_acc is not None and result["macro_avg_accuracy_including_ties"] is not None:
                beats = result["macro_avg_accuracy_including_ties"] > matched_acc

            result["compute_cost"] = compute_cost
            result["matched_single_scale_accuracy_including_ties"] = matched_acc
            result["matched_compute_out_of_range"] = out_of_range
            result["beats_single_scale_at_matched_compute"] = beats
            results_by_design[design_name] = result

            acc_str = (
                f"{result['macro_avg_accuracy_including_ties']:.1%}"
                if result["macro_avg_accuracy_including_ties"] is not None
                else "n/a"
            )
            matched_str = f"{matched_acc:.1%}" if matched_acc is not None else "out of range"
            print(
                f"  {fitter_name}/{design_name}: accuracy={acc_str} vs "
                f"matched single-scale={matched_str}, beats={beats}"
            )
        results_by_fitter[fitter_name] = results_by_design

    # Internal consistency check: ConstantExtrapolator predicts the largest
    # fitted scale's own value with no fitting at all, so its accuracy for a
    # design ending at size X must exactly reproduce P1-03's single-scale
    # point at X (same filter, same seed-averaging, same groupby-mean path).
    # A mismatch here means a real bug in this script or in P1-03, not a
    # legitimate methodological difference.
    for design_name, endpoint in _DESIGN_ENDPOINTS.items():
        constant_acc = results_by_fitter["ConstantExtrapolator"][design_name][
            "macro_avg_accuracy_including_ties"
        ]
        p1_03_point = next(
            p for p in single_scale_frontier["points"] if p["params_str"] == endpoint
        )
        p1_03_acc = p1_03_point["accuracy_including_ties"]
        if constant_acc is None or abs(constant_acc - p1_03_acc) > 1e-9:
            raise AssertionError(
                f"consistency check failed: ConstantExtrapolator/{design_name} macro-avg "
                f"accuracy ({constant_acc}) should exactly equal P1-03's single-scale point "
                f"at {endpoint} ({p1_03_acc}) -- investigate before trusting any result here."
            )
    print("p1_04_extrapolation_baselines: consistency check passed (ConstantExtrapolator == P1-03)")

    winners = []
    for fitter_name, by_design in results_by_fitter.items():
        for design_name, result in by_design.items():
            if result["beats_single_scale_at_matched_compute"]:
                winners.append(
                    {
                        "fitter": fitter_name,
                        "design": design_name,
                        "margin": result["macro_avg_accuracy_including_ties"]
                        - result["matched_single_scale_accuracy_including_ties"],
                    }
                )
    winners.sort(key=lambda w: w["margin"], reverse=True)

    print(
        f"p1_04_extrapolation_baselines: HEADLINE {len(winners)} / "
        f"{len(_FITTER_CLASSES) * len(designs)} (fitter, design) combinations beat the "
        "single-scale frontier at matched compute"
    )

    payload = {
        "target_scale": _TARGET,
        "metric": _METRIC,
        "seed_mode": _SEED_MODE,
        "sizes_ascending": sizes_ascending,
        "designs": {
            name: {
                "sizes": sizes,
                "compute_cost": da.compute_cost([size_scale[s] for s in sizes]),
            }
            for name, sizes in designs.items()
        },
        "single_scale_frontier": single_scale_frontier,
        "fitters": results_by_fitter,
        "summary": {
            "n_combinations": len(_FITTER_CLASSES) * len(designs),
            "n_beat_single_scale_at_matched_compute": len(winners),
            "winners": winners,
        },
        "dataset_revision": dd.cached_revision("macro_avg"),
    }

    provenance.write_result(
        "results/p1_04_extrapolation.json",
        payload=payload,
        config={"task": "P1-04"},
    )
    print("wrote results/p1_04_extrapolation.json")


if __name__ == "__main__":
    main()
