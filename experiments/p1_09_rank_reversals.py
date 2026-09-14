"""Task P1-09: rank-reversal census.

See plan/02-phase1-datadecide.md task P1-09. Evidence for the
impossibility regime (Claim 2): for every task and every recipe pair
(k, k'), does the sign of mu_k(s) - mu_k'(s) genuinely flip somewhere
across the size ladder, or does it only ever look like it flips because
the gap sits within noise? Classifies every pair as stable / reversing /
within_noise (src/pdt/analysis/rank_reversal.py), reports the Kendall tau
between each size's ranking and the target ranking as a curve in s, and
names a stress-test list of the clearest reversals. Writes
results/p1_09_rank_reversals.json.

Independent of P1-06/P1-07/P1-08 -- only needs P1-01's frame and P1-05's
seed-variance component (recomputed here directly via
noise.seed_variance(), not read from P1-05's results file, since it's a
cheap direct call and this task doesn't need any other P1-05 output).

Reports classification at TWO effect-size thresholds, not one -- see
docs/decisions.md. Testing 14 sizes per pair at the project's established
`AMBIGUOUS_EFFECT_SIZE_THRESHOLD=1.0` (a per-project convention, not a
calibrated hypothesis test) without correcting for running that test 14
times inflates the apparent reversal rate: an exploratory check found it
drop from ~79% to ~9% between the uncorrected threshold and a
Bonferroni-corrected one on a 3-task sample. `bonferroni` (family-wise
alpha=0.05 across len(sizes_ascending) simultaneous per-pair tests) is
reported as the primary figure; `uncorrected_1_0` is kept for continuity
with P1-02/03's own use of the same threshold, clearly labeled as
optimistic.
"""

from __future__ import annotations

import itertools

from scipy.stats import kendalltau, norm

from pdt import provenance
from pdt.analysis import noise
from pdt.analysis import rank_reversal as rr
from pdt.data import datadecide as dd
from pdt.data import frame as frame_mod

_METRIC = "primary_metric"
_TARGET = "1B"
_N_STRESS_TEST_PAIRS = 15
_FAMILYWISE_ALPHA = 0.05


def _sizes_ascending(long_frame) -> list[str]:
    sizes = long_frame.select(["params_str", "params_num"]).unique().sort("params_num")
    return sizes["params_str"].to_list()


def _bonferroni_threshold(n_tests: int, familywise_alpha: float = _FAMILYWISE_ALPHA) -> float:
    """Two-sided z-threshold controlling family-wise error at
    `familywise_alpha` across `n_tests` simultaneous per-pair tests
    (one per size on the ladder)."""
    return float(norm.ppf(1 - (familywise_alpha / n_tests) / 2))


def _by_task_size_recipe(seed_var) -> dict[tuple[str, str], dict[str, dict]]:
    """(task, params_str) -> {recipe: {mu, sigma2_seed, n_seeds}}."""
    result: dict[tuple[str, str], dict[str, dict]] = {}
    for row in seed_var.iter_rows(named=True):
        key = (row["task"], row["params_str"])
        result.setdefault(key, {})[row["recipe"]] = {
            "mu": row["mu"],
            "sigma2_seed": row["sigma2_seed"],
            "n_seeds": row["n_seeds"],
        }
    return result


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2


def _classify_task(
    task: str,
    recipes: list[str],
    sizes_ascending: list[str],
    by_size: dict[str, dict[str, dict]],
    threshold: float,
) -> dict:
    pair_results = []
    for k, k_prime in itertools.combinations(recipes, 2):
        resolved_signs: dict[str, int] = {}
        for size in sizes_ascending:
            cell = by_size[size]
            if k not in cell or k_prime not in cell:
                continue
            a, b = cell[k], cell[k_prime]
            effect_size = rr.pair_effect_size(
                a["mu"], b["mu"], a["sigma2_seed"], b["sigma2_seed"], a["n_seeds"]
            )
            sign = rr.resolved_sign(effect_size, threshold=threshold)
            if sign is not None:
                resolved_signs[size] = sign

        classification = rr.classify_pair_trajectory(sizes_ascending, resolved_signs)

        target_cell = by_size[_TARGET]
        gap_at_target = target_cell[k]["mu"] - target_cell[k_prime]["mu"]
        effect_size_at_target = rr.pair_effect_size(
            target_cell[k]["mu"],
            target_cell[k_prime]["mu"],
            target_cell[k]["sigma2_seed"],
            target_cell[k_prime]["sigma2_seed"],
            target_cell[k]["n_seeds"],
        )

        pair_results.append(
            {
                "recipe_a": k,
                "recipe_b": k_prime,
                "classification": classification["classification"],
                "crossing_size": classification["crossing_size"],
                "n_resolved_sizes": classification["n_resolved_sizes"],
                "gap_at_target": gap_at_target,
                "effect_size_at_target": effect_size_at_target,
            }
        )

    n_stable = sum(1 for p in pair_results if p["classification"] == "stable")
    n_reversing = sum(1 for p in pair_results if p["classification"] == "reversing")
    n_within_noise = sum(1 for p in pair_results if p["classification"] == "within_noise")

    reversing_gaps = sorted(
        abs(p["gap_at_target"]) for p in pair_results if p["classification"] == "reversing"
    )
    stable_gaps = sorted(
        abs(p["gap_at_target"]) for p in pair_results if p["classification"] == "stable"
    )

    return {
        "task": task,
        "n_pairs": len(pair_results),
        "n_stable": n_stable,
        "n_reversing": n_reversing,
        "n_within_noise": n_within_noise,
        "fraction_reversing": n_reversing / len(pair_results) if pair_results else None,
        "fraction_reversing_of_resolved": (
            n_reversing / (n_stable + n_reversing) if (n_stable + n_reversing) else None
        ),
        "median_abs_gap_at_target_reversing_pairs": _median(reversing_gaps),
        "median_abs_gap_at_target_stable_pairs": _median(stable_gaps),
        "pairs": pair_results,
    }


def _kendall_curve(
    recipes: list[str], sizes_ascending: list[str], by_size: dict[str, dict[str, dict]]
) -> list[dict]:
    """Task-level, threshold-independent: Kendall tau between each size's
    recipe ranking and the target scale's ranking."""
    curve = []
    target_mus = [by_size[_TARGET][r]["mu"] for r in recipes]
    for size in sizes_ascending:
        cell = by_size[size]
        if not all(r in cell for r in recipes):
            continue
        size_mus = [cell[r]["mu"] for r in recipes]
        tau_result = kendalltau(size_mus, target_mus)
        curve.append(
            {
                "params_str": size,
                "kendall_tau": float(tau_result.statistic),
                "kendall_p_value": float(tau_result.pvalue),
            }
        )
    return curve


def _run_at_threshold(
    threshold: float,
    tasks: list[str],
    recipes: list[str],
    sizes_ascending: list[str],
    by_task_size: dict[tuple[str, str], dict[str, dict]],
    *,
    label: str,
) -> dict:
    by_task = {}
    for task in tasks:
        by_size = {size: by_task_size.get((task, size), {}) for size in sizes_ascending}
        by_task[task] = _classify_task(task, recipes, sizes_ascending, by_size, threshold)
        t = by_task[task]
        print(
            f"p1_09_rank_reversals [{label}]: {task}: {t['n_stable']} stable, "
            f"{t['n_reversing']} reversing, {t['n_within_noise']} within-noise "
            f"(of {t['n_pairs']} pairs)"
        )

    all_reversing = []
    for task in tasks:
        for pair in by_task[task]["pairs"]:
            if pair["classification"] == "reversing":
                all_reversing.append({"task": task, **pair})
    all_reversing.sort(
        key=lambda p: (
            abs(p["effect_size_at_target"]) if p["effect_size_at_target"] is not None else 0.0
        ),
        reverse=True,
    )
    stress_test_pairs = all_reversing[:_N_STRESS_TEST_PAIRS]

    total_pairs = sum(by_task[t]["n_pairs"] for t in tasks)
    total_reversing = sum(by_task[t]["n_reversing"] for t in tasks)
    total_stable = sum(by_task[t]["n_stable"] for t in tasks)
    total_within_noise = sum(by_task[t]["n_within_noise"] for t in tasks)
    print(
        f"p1_09_rank_reversals [{label}]: TOTAL: {total_stable} stable, {total_reversing} "
        f"reversing, {total_within_noise} within-noise, of {total_pairs} pairs "
        f"({100 * total_reversing / total_pairs:.1f}% reversing)"
    )

    return {
        "threshold": threshold,
        "by_task": by_task,
        "summary": {
            "total_pairs": total_pairs,
            "total_stable": total_stable,
            "total_reversing": total_reversing,
            "total_within_noise": total_within_noise,
            "fraction_reversing": total_reversing / total_pairs if total_pairs else None,
        },
        "stress_test_pairs": stress_test_pairs,
    }


def main() -> None:
    long_frame = frame_mod.build_frame(source="macro_avg", metrics=(_METRIC,))
    seed_var = noise.seed_variance(long_frame, _METRIC)
    by_task_size = _by_task_size_recipe(seed_var)

    sizes_ascending = _sizes_ascending(long_frame)
    if sizes_ascending[-1] != _TARGET:
        raise ValueError(f"expected largest size to be {_TARGET!r}, got {sizes_ascending[-1]!r}")

    tasks = sorted({t for t, _ in by_task_size})
    recipes = sorted(seed_var["recipe"].unique().to_list())
    print(
        f"p1_09_rank_reversals: {len(tasks)} tasks, {len(recipes)} recipes, "
        f"{len(sizes_ascending)} sizes"
    )

    bonferroni_threshold = _bonferroni_threshold(len(sizes_ascending))
    print(
        f"p1_09_rank_reversals: bonferroni threshold for {len(sizes_ascending)} tests, "
        f"family-wise alpha={_FAMILYWISE_ALPHA} -> z={bonferroni_threshold:.4f}"
    )

    uncorrected = _run_at_threshold(
        1.0, tasks, recipes, sizes_ascending, by_task_size, label="uncorrected_1.0"
    )
    bonferroni = _run_at_threshold(
        bonferroni_threshold,
        tasks,
        recipes,
        sizes_ascending,
        by_task_size,
        label="bonferroni",
    )

    kendall_by_task = {}
    for task in tasks:
        by_size = {size: by_task_size.get((task, size), {}) for size in sizes_ascending}
        kendall_by_task[task] = _kendall_curve(recipes, sizes_ascending, by_size)

    payload = {
        "target_scale": _TARGET,
        "metric": _METRIC,
        "sizes_ascending": sizes_ascending,
        "recipes": recipes,
        "primary_threshold_label": "bonferroni",
        "bonferroni_threshold_value": bonferroni_threshold,
        "familywise_alpha": _FAMILYWISE_ALPHA,
        "n_simultaneous_tests_per_pair": len(sizes_ascending),
        "thresholds": {
            "uncorrected_1_0": uncorrected,
            "bonferroni": bonferroni,
        },
        "kendall_tau_vs_scale_by_task": kendall_by_task,
        "dataset_revision_macro_avg": dd.cached_revision("macro_avg"),
    }

    provenance.write_result(
        "results/p1_09_rank_reversals.json",
        payload=payload,
        config={"task": "P1-09"},
    )
    print("wrote results/p1_09_rank_reversals.json")


if __name__ == "__main__":
    main()
