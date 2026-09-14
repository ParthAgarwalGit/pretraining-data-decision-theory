"""Pairwise decision accuracy: does a proxy-scale ranking predict the
target-scale winner?

See plan/02-phase1-datadecide.md task P1-03. This is the reproduction of
DataDecide's headline "~80% of comparisons correct at ~150M params"
finding -- the check that this project's whole pipeline is correct. Do not
build anything else on Phase 1 until this reproduces.
"""

from __future__ import annotations

import itertools
import math

import polars as pl
from scipy.stats import kendalltau

from pdt.analysis.ground_truth import AMBIGUOUS_EFFECT_SIZE_THRESHOLD
from pdt.scaling.base import Scale


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def recipe_means(
    long_frame: pl.DataFrame,
    metric_name: str,
    params_str: str,
    *,
    seed_mode: str = "average",
) -> dict[str, dict[str, float]]:
    """{task: {recipe: value}} at one scale, for one metric.

    `seed_mode`:
      - "average": mean over every seed present at this scale (the P1-02
        default, and the recommended choice).
      - "default_only": only the `default` seed's value, no averaging --
        one arm of P1-03's four-variant sensitivity check (does averaging
        over seeds vs. using a single run change the reproduction?).

    Only final-checkpoint rows are used; there is no variant for this (see
    the module docstring on the plan's "(a) are you using final
    checkpoints" debugging step -- there is only one right answer, this
    always uses it).
    """
    if seed_mode not in ("average", "default_only"):
        raise ValueError(f"seed_mode must be 'average' or 'default_only', got {seed_mode!r}")

    subset = long_frame.filter(
        (pl.col("metric_name") == metric_name)
        & (pl.col("params_str") == params_str)
        & (pl.col("is_final"))
    )
    if seed_mode == "default_only":
        subset = subset.filter(pl.col("seed") == "default")
        if subset.height == 0:
            raise ValueError(
                f"no 'default'-seed rows for metric_name={metric_name!r}, "
                f"params_str={params_str!r} -- check the scale label is real."
            )

    if subset.height == 0:
        raise ValueError(
            f"no rows for metric_name={metric_name!r}, params_str={params_str!r}, "
            "is_final=True -- check the metric name and scale label are real."
        )

    grouped = subset.group_by(["recipe", "task"]).agg(pl.col("metric_value").mean().alias("mu"))

    result: dict[str, dict[str, float]] = {}
    for task, recipe, mu in zip(
        grouped["task"].to_list(),
        grouped["recipe"].to_list(),
        grouped["mu"].to_list(),
        strict=True,
    ):
        result.setdefault(task, {})[recipe] = mu
    return result


def pairwise_decision_accuracy(
    proxy_means: dict[str, float],
    target_means: dict[str, float],
    target_sd_seed: dict[str, float],
    n_seeds_target: int,
    *,
    tie_effect_size_threshold: float = AMBIGUOUS_EFFECT_SIZE_THRESHOLD,
) -> dict:
    """Compare every recipe pair's sign(proxy gap) against sign(target gap).

    A pair is "tied" (excluded from the excluding-ties accuracy) when its
    target-scale standardized effect size -- the same
    delta / pooled-seed-standard-error statistic ground_truth.py uses --
    is below `tie_effect_size_threshold`. Reuses that exact threshold by
    default rather than inventing a second one, so P1-02's "ambiguous
    task" flag and P1-03's "tied pair" flag mean the same thing.

    Only recipes present in both `proxy_means` and `target_means` are
    compared (a recipe missing from one side is silently excluded from
    pairs, not treated as a wrong guess -- P1-01's coverage matrix already
    guarantees this doesn't happen for real DataDecide data, but a
    synthetic/partial input here shouldn't crash).
    """
    recipes = sorted(set(proxy_means) & set(target_means))
    pairs = list(itertools.combinations(recipes, 2))

    n_correct_including_ties = 0
    n_correct_excluding_ties = 0
    n_tied = 0

    for k, k_prime in pairs:
        target_gap = target_means[k] - target_means[k_prime]
        proxy_gap = proxy_means[k] - proxy_means[k_prime]
        is_correct = _sign(target_gap) == _sign(proxy_gap)
        if is_correct:
            n_correct_including_ties += 1

        pooled_variance = (target_sd_seed[k] ** 2 + target_sd_seed[k_prime] ** 2) / n_seeds_target
        is_tied = pooled_variance == 0 or (
            abs(target_gap) / math.sqrt(pooled_variance) < tie_effect_size_threshold
        )
        if is_tied:
            n_tied += 1
        elif is_correct:
            n_correct_excluding_ties += 1

    n_pairs = len(pairs)
    n_non_tied = n_pairs - n_tied

    tau = None
    tau_p_value = None
    if len(recipes) >= 2:
        proxy_values = [proxy_means[r] for r in recipes]
        target_values = [target_means[r] for r in recipes]
        tau_result = kendalltau(proxy_values, target_values)
        tau, tau_p_value = float(tau_result.statistic), float(tau_result.pvalue)

    return {
        "n_recipes_compared": len(recipes),
        "n_pairs": n_pairs,
        "n_tied_pairs": n_tied,
        "accuracy_including_ties": (n_correct_including_ties / n_pairs) if n_pairs else None,
        "accuracy_excluding_ties": (n_correct_excluding_ties / n_non_tied) if n_non_tied else None,
        "kendall_tau": tau,
        "kendall_p_value": tau_p_value,
    }


def recipe_trajectories(
    long_frame: pl.DataFrame,
    metric_name: str,
    proxy_sizes: list[str],
    *,
    seed_mode: str = "average",
) -> dict[str, dict[str, list[tuple[Scale, float]]]]:
    """{task: {recipe: [(Scale(n, d), value), ...]}} across `proxy_sizes`,
    each recipe's list sorted by scale ascending -- exactly the shape
    plan/02-phase1-datadecide.md P1-04's per-recipe extrapolator fitting
    needs (`Extrapolator.fit(scales, values)`).

    Unlike `recipe_means()` (one scale at a time), this carries `n`/`d`
    through so the caller doesn't need a second lookup to build `Scale`
    objects for fitting.
    """
    if seed_mode not in ("average", "default_only"):
        raise ValueError(f"seed_mode must be 'average' or 'default_only', got {seed_mode!r}")

    subset = long_frame.filter(
        (pl.col("metric_name") == metric_name)
        & (pl.col("params_str").is_in(proxy_sizes))
        & (pl.col("is_final"))
    )
    if seed_mode == "default_only":
        subset = subset.filter(pl.col("seed") == "default")

    if subset.height == 0:
        raise ValueError(
            f"no rows for metric_name={metric_name!r}, proxy_sizes={proxy_sizes!r}, "
            "is_final=True -- check the metric name and size labels are real."
        )

    grouped = subset.group_by(["recipe", "task", "params_str"]).agg(
        pl.col("metric_value").mean().alias("mu"),
        pl.col("params_num").first().alias("n"),
        pl.col("tokens").mean().alias("d"),
    )

    result: dict[str, dict[str, list[tuple[Scale, float]]]] = {}
    for row in grouped.iter_rows(named=True):
        task_dict = result.setdefault(row["task"], {})
        task_dict.setdefault(row["recipe"], []).append((Scale(n=row["n"], d=row["d"]), row["mu"]))

    for task_dict in result.values():
        for recipe, trajectory in task_dict.items():
            task_dict[recipe] = sorted(trajectory, key=lambda pair: pair[0].n)

    return result


def compute_cost(scales: list[Scale]) -> float:
    """Total FLOPs to train one recipe across every scale in `scales`
    (C ~= 6*N*D per scale, summed) -- the cost of a proxy design `S_fit`,
    per plan/02-phase1-datadecide.md P1-04. Recipe-agnostic: every recipe
    trains the same size ladder, so this depends only on the scales
    themselves, not which recipe.
    """
    return sum(scale.compute for scale in scales)
