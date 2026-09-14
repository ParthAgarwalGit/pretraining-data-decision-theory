"""Task P1-05: noise-floor estimation.

See plan/02-phase1-datadecide.md task P1-05. Before anything can be called
"bias" (P1-06), we need to know the noise. Estimates three variance
components -- seed variance, checkpoint jitter, and eval-sampling noise --
across every (recipe, size, task), plus the target-truth noise
sigma^2_target(k, t) that P1-06 needs to de-bias its sigma^2_extrap
estimate. Writes results/p1_05_noise.json.

Uses the same headline convention as P1-02/03/04 (macro_avg source,
primary_metric) so this noise model applies to the same quantities those
tasks already established.
"""

from __future__ import annotations

import polars as pl

from pdt import provenance
from pdt.analysis import noise
from pdt.data import datadecide as dd
from pdt.data import frame as frame_mod

_METRIC = "primary_metric"
_TARGET_SIZE = "1B"

# olmes_10_macro_avg is the unweighted mean of these 10 primitive tasks'
# primary_metric values -- verified directly against the cached frame
# (16/16 sampled (recipe, size) combinations matched exactly, 0.0 diff)
# before relying on it here. See docs/decisions.md.
_OLMES_10_COMPONENTS = (
    "arc_challenge",
    "arc_easy",
    "boolq",
    "csqa",
    "hellaswag",
    "mmlu",
    "openbookqa",
    "piqa",
    "socialiqa",
    "winogrande",
)
_COMPOSITE_TASK = "olmes_10_macro_avg"


def _eval_sampling_noise_table(
    seed_var: pl.DataFrame, instance_counts: dict[str, int]
) -> pl.DataFrame:
    """Per (recipe, params_str, task): sigma^2_eval, built from
    `seed_var`'s own `mu` (the seed-averaged accuracy at that cell) so the
    `p` in `p(1-p)/n` matches exactly what every other noise component in
    this file is computed against.

    Primitive tasks (10 of the 11) get `p(1-p)/n_instances` directly.
    `olmes_10_macro_avg` gets the variance-of-a-mean combination instead,
    since it has no eval set of its own -- see
    `noise.eval_sampling_noise_of_mean`'s docstring.
    """
    rows = []
    by_cell: dict[tuple[str, str], dict[str, float]] = {}
    for row in seed_var.iter_rows(named=True):
        by_cell.setdefault((row["recipe"], row["params_str"]), {})[row["task"]] = row["mu"]

    missing_instance_counts = set()
    for (recipe, params_str), task_to_mu in by_cell.items():
        for task, mu in task_to_mu.items():
            if task == _COMPOSITE_TASK:
                continue
            if task not in instance_counts:
                missing_instance_counts.add(task)
                continue
            sigma2 = noise.eval_sampling_noise(mu, instance_counts[task])
            rows.append(
                {"recipe": recipe, "params_str": params_str, "task": task, "sigma2_eval": sigma2}
            )

        if _COMPOSITE_TASK in task_to_mu:
            missing_components = [t for t in _OLMES_10_COMPONENTS if t not in task_to_mu]
            if missing_components:
                raise ValueError(
                    f"recipe={recipe!r}, params_str={params_str!r} has "
                    f"{_COMPOSITE_TASK!r} but is missing component task(s) "
                    f"{missing_components} -- P1-01 established 100% coverage for "
                    "this frame, so this cell should never be incomplete; "
                    "investigate before trusting sigma2_eval for this task."
                )
            ps = [task_to_mu[t] for t in _OLMES_10_COMPONENTS]
            ns = [instance_counts[t] for t in _OLMES_10_COMPONENTS]
            sigma2 = noise.eval_sampling_noise_of_mean(ps, ns)
            rows.append(
                {
                    "recipe": recipe,
                    "params_str": params_str,
                    "task": _COMPOSITE_TASK,
                    "sigma2_eval": sigma2,
                }
            )

    if missing_instance_counts:
        raise ValueError(
            f"no eval-instance count for task(s) {sorted(missing_instance_counts)} -- "
            "either the eval_instance_counts loader is missing a task the analysis "
            "frame has, or a new task appeared; investigate before trusting "
            "sigma2_eval for that task."
        )

    table = pl.DataFrame(rows)
    if table.height != seed_var.height:
        raise AssertionError(
            f"eval_sampling_noise produced {table.height} rows but seed_variance has "
            f"{seed_var.height} (recipe, params_str, task) cells -- some cell was "
            "silently dropped rather than raising; investigate before trusting this table."
        )
    return table


def _noise_vs_scale_summary(
    seed_var: pl.DataFrame, ckpt_jitter_by_cell: pl.DataFrame, eval_noise: pl.DataFrame
) -> pl.DataFrame:
    """Per (params_str, task): median sigma2_seed and sigma2_ckpt across
    the 25 recipes, plus mean sigma2_eval -- a plot-ready noise-vs-scale
    table (P1-05's definition of done), and the direct answer to "does
    sigma2_seed shrink with scale" (an explicit empirical question the
    plan asks, not assumed)."""
    seed_summary = seed_var.group_by(["params_str", "task"]).agg(
        pl.col("sigma2_seed").median().alias("median_sigma2_seed"),
        pl.col("params_num").first().alias("params_num"),
        pl.len().alias("n_recipes"),
    )
    ckpt_summary = ckpt_jitter_by_cell.group_by(["params_str", "task"]).agg(
        pl.col("sigma2_ckpt").median().alias("median_sigma2_ckpt"),
    )
    eval_summary = eval_noise.group_by(["params_str", "task"]).agg(
        pl.col("sigma2_eval").mean().alias("mean_sigma2_eval"),
    )

    summary = seed_summary.join(ckpt_summary, on=["params_str", "task"], how="left")
    summary = summary.join(eval_summary, on=["params_str", "task"], how="left")
    return summary.sort(["task", "params_num"])


def main() -> None:
    long_frame = frame_mod.build_frame(source="macro_avg", metrics=(_METRIC,))

    # Component 1: seed variance, every (recipe, size, task).
    seed_var = noise.seed_variance(long_frame, _METRIC)
    print(
        f"p1_05_noise_floor: seed_variance computed for {seed_var.height} "
        "(recipe, size, task) cells"
    )

    # Component 2: checkpoint jitter, every (recipe, size, seed, task), then
    # aggregated across seeds to (recipe, size, task) to match the other
    # two components' grain.
    ckpt_jitter_raw = noise.checkpoint_jitter(long_frame, _METRIC)
    ckpt_jitter_by_cell = ckpt_jitter_raw.group_by(["recipe", "params_str", "task"]).agg(
        pl.col("sigma2_ckpt").mean().alias("sigma2_ckpt"),
        pl.col("n_checkpoints_used").mean().alias("mean_n_checkpoints_used"),
        pl.col("seed").n_unique().alias("n_seeds_used"),
    )
    print(
        f"p1_05_noise_floor: checkpoint_jitter computed for {ckpt_jitter_raw.height} "
        f"(recipe, size, seed, task) runs, {ckpt_jitter_by_cell.height} cells after "
        "averaging across seeds"
    )

    # Component 3: eval-sampling noise. Needs per-task instance counts from
    # the eval-instances repo's summary file (one targeted 269MB download,
    # not the 123GB repo -- see the module docstring in
    # src/pdt/data/datadecide.py).
    instance_counts_df = dd.load_eval_instance_counts()
    instance_counts = dict(
        zip(
            instance_counts_df["task"].to_list(),
            instance_counts_df["num_instances"].to_list(),
            strict=True,
        )
    )
    print(f"p1_05_noise_floor: eval instance counts loaded for {len(instance_counts)} tasks")
    eval_noise = _eval_sampling_noise_table(seed_var, instance_counts)
    print(f"p1_05_noise_floor: eval_sampling_noise computed for {eval_noise.height} cells")

    # sigma^2_target(k, t): the noise in P1-02's own estimator of mu_k(s*).
    # P1-02 estimates mu_k(s*) as a plain average over the 3 seeds present
    # at 1B -- no checkpoint or eval correction applied there -- so the
    # honest noise in that specific estimate is exactly the standard error
    # of that average: sigma2_seed(k, s*, t) / n_seeds(k, s*, t). See
    # docs/decisions.md for why sigma2_ckpt/sigma2_eval are reported
    # separately rather than folded in here.
    target_rows = seed_var.filter(pl.col("params_str") == _TARGET_SIZE)
    sigma2_target = target_rows.with_columns(
        (pl.col("sigma2_seed") / pl.col("n_seeds")).alias("sigma2_target")
    ).select(["recipe", "task", "sigma2_target", "sigma2_seed", "n_seeds"])
    print(
        f"p1_05_noise_floor: sigma2_target computed for {sigma2_target.height} "
        f"(recipe, task) pairs at {_TARGET_SIZE}"
    )

    # Empirical question: does sigma2_seed shrink with scale?
    seed_trend = (
        seed_var.group_by("params_str")
        .agg(
            pl.col("sigma2_seed").median().alias("median_sigma2_seed"),
            pl.col("params_num").first(),
        )
        .sort("params_num")
    )
    trend_values = seed_trend["median_sigma2_seed"].to_list()
    is_monotonic_nonincreasing = all(
        trend_values[i] >= trend_values[i + 1] - 1e-12 for i in range(len(trend_values) - 1)
    )
    rounded_trend = dict(
        zip(seed_trend["params_str"].to_list(), [round(v, 6) for v in trend_values], strict=True)
    )
    print(f"p1_05_noise_floor: median sigma2_seed by size: {rounded_trend}")
    print(f"p1_05_noise_floor: monotonic non-increasing with scale = {is_monotonic_nonincreasing}")

    noise_vs_scale = _noise_vs_scale_summary(seed_var, ckpt_jitter_by_cell, eval_noise)

    payload = {
        "metric": _METRIC,
        "target_size": _TARGET_SIZE,
        "eval_instance_counts": instance_counts,
        "eval_instance_counts_source": (
            "allenai/DataDecide-eval-instances/summary-metrics.jsonl "
            f"(revision {dd.cached_revision('eval_instance_counts')})"
        ),
        "seed_variance": seed_var.sort(["task", "params_num", "recipe"]).to_dicts(),
        "checkpoint_jitter": ckpt_jitter_by_cell.sort(["task", "params_str", "recipe"]).to_dicts(),
        "eval_sampling_noise": eval_noise.sort(["task", "params_str", "recipe"]).to_dicts(),
        "sigma2_target_at_1b": sigma2_target.sort(["task", "recipe"]).to_dicts(),
        "seed_variance_vs_scale_trend": {
            "median_by_size": dict(
                zip(seed_trend["params_str"].to_list(), trend_values, strict=True)
            ),
            "monotonically_non_increasing": is_monotonic_nonincreasing,
        },
        "noise_vs_scale_summary": noise_vs_scale.to_dicts(),
        "dataset_revision_macro_avg": dd.cached_revision("macro_avg"),
    }

    provenance.write_result(
        "results/p1_05_noise.json",
        payload=payload,
        config={"task": "P1-05"},
    )
    print("wrote results/p1_05_noise.json")


if __name__ == "__main__":
    main()
