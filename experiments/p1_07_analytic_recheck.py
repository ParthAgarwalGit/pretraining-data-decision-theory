"""Recheck of P1-07's analytic delta-method `v_k` after the covariance-conditioning fix.

Second-round review of PR #17: `sandwich_covariance` formed `pinv(J^T J)`, which
squares the design's condition number and can silently discard a weak but identified
direction (a synthetic full-rank design with cond(J) ~ 2e8 returned 1.5e-4 where the
true value is ~5e11). `analytic_v_k` now works from the SVD of the design itself and
fails closed where a target-relevant direction is below the numerical rank cutoff.

P1-07's expensive Monte-Carlo pass and every bound value are independent of
`analytic_v_k` (they use the bootstrap `v_hat` from P1-06), so the only stored fields
that could change are the per-recipe `analytic_v_k` cross-check values. This script
recomputes exactly those -- the same fits, the same designs, the same 3,300 per-recipe
values -- with the fixed code and compares them with what is stored in
`results/p1_07_bound_coverage.json`, reporting the maximum relative difference, how
many values moved by more than 1e-6, and how many recipes became unidentified. It
writes `results/p1_07_analytic_recheck.json`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

for _blas_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ.setdefault(_blas_env_var, "1")

from pdt import provenance  # noqa: E402
from pdt.analysis import decision_accuracy as da  # noqa: E402
from pdt.analysis import ground_truth as gt  # noqa: E402
from pdt.data import frame as frame_mod  # noqa: E402
from pdt.scaling.base import Scale  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "p1_07_bound_coverage", os.path.join(os.path.dirname(__file__), "p1_07_bound_coverage.py")
)
p107 = importlib.util.module_from_spec(_SPEC)
sys.modules["p1_07_bound_coverage"] = p107  # dataclasses resolve annotations via sys.modules
_SPEC.loader.exec_module(p107)

_STORED_PATH = "results/p1_07_bound_coverage.json"
_REL_TOL = 1e-6


def main() -> None:
    long_frame = frame_mod.build_frame(source="macro_avg", metrics=(p107._METRIC,))
    ground_truth = gt.compute_ground_truth(long_frame, p107._METRIC, p107._TARGET)
    sizes_ascending = p107._sizes_ascending(long_frame)
    designs = p107._designs(sizes_ascending)
    target_row = (
        long_frame.filter((long_frame["params_str"] == p107._TARGET) & (long_frame["is_final"]))
        .select(["params_num", "tokens"])
        .unique()
    )
    target_scale = Scale(n=target_row["params_num"][0], d=target_row["tokens"][0])
    tasks = sorted(ground_truth.keys())

    with open(_STORED_PATH, encoding="utf-8") as f:
        stored = json.load(f)["data"]["by_fitter"]

    n_compared = 0
    n_changed = 0
    n_new_unidentified = 0
    n_missing_stored = 0
    max_rel_diff = 0.0
    worst: dict | None = None
    by_fitter: dict[str, dict] = {}
    for design_name, design_sizes in designs.items():
        avg_by_task = da.recipe_trajectories(
            long_frame, p107._METRIC, design_sizes, seed_mode="average"
        )
        for fitter_name in p107._FITTER_CLASSES:
            for task in tasks:
                fresh = p107._compute_analytic_v_k(fitter_name, avg_by_task[task], target_scale)
                stored_cell = (
                    stored.get(fitter_name, {})
                    .get(design_name, {})
                    .get(task, {})
                    .get("seed_bootstrap")
                )
                stored_v = (stored_cell or {}).get("analytic_v_k", {})
                for recipe, res in fresh.items():
                    if not res.get("ok"):
                        if recipe in stored_v:
                            n_new_unidentified += 1
                        continue
                    if recipe not in stored_v:
                        n_missing_stored += 1
                        continue
                    old, new = stored_v[recipe], res["analytic_v_k"]
                    n_compared += 1
                    denom = max(abs(old), abs(new), 1e-300)
                    rel = abs(new - old) / denom
                    fb = by_fitter.setdefault(
                        fitter_name,
                        {
                            "n_compared": 0,
                            "n_changed": 0,
                            "n_stored_negative": 0,
                            "n_changed_by_more_than_10pct": 0,
                            "max_relative_difference": 0.0,
                        },
                    )
                    fb["n_compared"] += 1
                    fb["n_stored_negative"] += int(old < 0.0)
                    fb["max_relative_difference"] = max(fb["max_relative_difference"], rel)
                    if rel > _REL_TOL:
                        n_changed += 1
                        fb["n_changed"] += 1
                        fb["n_changed_by_more_than_10pct"] += int(rel > 0.1)
                    if rel > max_rel_diff:
                        max_rel_diff = rel
                        worst = {
                            "fitter": fitter_name,
                            "design": design_name,
                            "task": task,
                            "recipe": recipe,
                            "stored": old,
                            "fresh": new,
                        }

    print(
        f"p1_07_analytic_recheck: compared {n_compared} per-recipe analytic v_k values: "
        f"{n_changed} moved by more than {_REL_TOL:g} (relative), max relative difference "
        f"{max_rel_diff:.3e}, {n_new_unidentified} newly unidentified, "
        f"{n_missing_stored} without a stored counterpart"
    )
    payload = {
        "stored_file": _STORED_PATH,
        "relative_tolerance": _REL_TOL,
        "n_compared": n_compared,
        "n_changed_beyond_tolerance": n_changed,
        "n_newly_unidentified": n_new_unidentified,
        "n_without_stored_counterpart": n_missing_stored,
        "max_relative_difference": max_rel_diff,
        "worst_case": worst,
        # A stored variance below zero is impossible (v = sum_i g_i^2 r_i^2 >= 0): the old
        # pinv(J^T J) sandwich could return one when it truncated a weak direction.
        "by_fitter": by_fitter,
    }
    provenance.write_result(
        "results/p1_07_analytic_recheck.json",
        payload=payload,
        config={"task": "P1-07-analytic-recheck"},
    )
    print("wrote results/p1_07_analytic_recheck.json")


if __name__ == "__main__":
    main()
