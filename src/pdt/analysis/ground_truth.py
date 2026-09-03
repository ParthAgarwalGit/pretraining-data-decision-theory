"""Ground truth at a target scale: per-task winners, gaps, and ambiguity.

See plan/02-phase1-datadecide.md task P1-02. Shared by P1-02 (the primary
1B ground truth), P1-03 (needs it to score the single-scale baseline), and
eventually P5-03 (metric-choice ablation, different metric argument).
"""

from __future__ import annotations

import math

import polars as pl

# A task's winner is "ambiguous" when the gap to the runner-up is smaller
# than one pooled-seed standard error of that gap -- i.e. the effect size
# (see _effect_size below) is under this many "noise units". This is a
# diagnostic threshold for flagging tasks, not a calibrated hypothesis
# test (the per-recipe std is estimated from only 3 seeds, 2 degrees of
# freedom -- treat effect_size as directional, not a p-value).
AMBIGUOUS_EFFECT_SIZE_THRESHOLD = 1.0


def _effect_size(delta_min: float, sd_a: float, sd_b: float, n_seeds: int) -> float | None:
    """Standardized gap between the winner and runner-up: delta_min divided
    by the standard error of that gap, treating the two recipes as
    independent (separate training runs), each estimated from n_seeds
    replicates: SE = sqrt((sd_a^2 + sd_b^2) / n_seeds).

    Returns None if both recipes have zero seed variance (degenerate --
    cannot compute a standard error; happens only with duplicate/identical
    seed values, not expected in real data but handled rather than
    crashing).
    """
    pooled_variance = (sd_a**2 + sd_b**2) / n_seeds
    if pooled_variance == 0:
        return None
    return delta_min / math.sqrt(pooled_variance)


def compute_ground_truth(
    long_frame: pl.DataFrame, metric_name: str, target_params_str: str
) -> dict:
    """Per-task ground truth at `target_params_str` for `metric_name`.

    Uses only final-checkpoint rows, averaged over whatever seeds are
    present for that scale (P0-06 found this is always exactly 3, with the
    seed *labels* differing at 1B -- this function doesn't care what the
    labels are, only how many there are).

    Returns a dict with one entry per task: k_star, mu (per recipe), gaps
    (per non-winning recipe), delta_min, runner_up, sd_seed (per recipe),
    effect_size, and whether the task is flagged ambiguous.
    """
    subset = long_frame.filter(
        (pl.col("metric_name") == metric_name)
        & (pl.col("params_str") == target_params_str)
        & (pl.col("is_final"))
    )
    if subset.height == 0:
        raise ValueError(
            f"no rows for metric_name={metric_name!r}, params_str={target_params_str!r}, "
            "is_final=True -- check the metric name and scale label are real."
        )

    # Sorted on a fully deterministic key before the group-by/mean+std
    # reduction, with maintain_order=True on the group_by itself -- the
    # same defensive pattern noise.py uses (see docs/decisions.md
    # 2026-09-03, P1-05 Decision 5): mean()/std() are not order-associative
    # in floating point, so an unspecified row order out of a
    # parallel/chunked parquet read can in principle produce
    # last-bit-different results run to run. A dedicated audit (see
    # docs/decisions.md, the entry following this one) diffed 24
    # independent full runs of this exact function against real cached
    # data (both sources, every scale) and found zero differences here --
    # unlike decision_accuracy.recipe_means(), which the same audit found
    # DOES fail this way on the larger eval_results frame. Fixed here too,
    # for consistency, since nothing guarantees this stays safe as the
    # data or polars' internals change.
    per_recipe_task = (
        subset.sort(["recipe", "task", "seed"])
        .group_by(["recipe", "task"], maintain_order=True)
        .agg(
            pl.col("metric_value").mean().alias("mu"),
            pl.col("metric_value").std(ddof=1).fill_null(0.0).alias("sd_seed"),
            pl.col("metric_value").len().alias("n_seeds"),
        )
    )

    per_task: dict[str, dict] = {}
    for task in sorted(per_recipe_task["task"].unique().to_list()):
        # Sort by mu descending, breaking exact ties by recipe name
        # ascending. A secondary deterministic key is not optional here:
        # this project found for real that exact ties in mu are common
        # (many tasks have delta_min == 0.0 between the top recipes), and
        # without one, which tied recipe becomes k_star/runner_up varies
        # from run to run -- polars' sort has no stability guarantee across
        # ties, so two runs of the identical computation on identical data
        # silently produced different "winners". Caught by diffing two
        # clean-tree runs against each other before trusting this task's
        # own reproducibility claim.
        task_rows = per_recipe_task.filter(pl.col("task") == task).sort(
            ["mu", "recipe"], descending=[True, False]
        )
        recipes = task_rows["recipe"].to_list()
        mus = task_rows["mu"].to_list()
        sds = task_rows["sd_seed"].to_list()
        n_seeds_list = task_rows["n_seeds"].to_list()

        k_star = recipes[0]
        mu_star = mus[0]
        sd_star = sds[0]

        gaps = {recipes[i]: mu_star - mus[i] for i in range(1, len(recipes))}
        runner_up = min(gaps, key=lambda k: gaps[k])
        delta_min = gaps[runner_up]
        runner_up_idx = recipes.index(runner_up)
        sd_runner_up = sds[runner_up_idx]
        # Same grid-cell shape for every recipe here, per P1-01's coverage check.
        n_seeds = n_seeds_list[0]

        effect_size = _effect_size(delta_min, sd_star, sd_runner_up, n_seeds)
        is_ambiguous = effect_size is None or effect_size < AMBIGUOUS_EFFECT_SIZE_THRESHOLD

        per_task[task] = {
            "k_star": k_star,
            "mu": dict(zip(recipes, mus, strict=True)),
            "sd_seed": dict(zip(recipes, sds, strict=True)),
            "n_seeds": n_seeds,
            "gaps": gaps,
            "delta_min": delta_min,
            "runner_up": runner_up,
            "effect_size": effect_size,
            "is_ambiguous": is_ambiguous,
        }

    return per_task
