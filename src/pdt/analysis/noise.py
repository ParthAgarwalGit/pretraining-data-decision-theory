"""Noise-floor estimation: the three variance components P1-05 requires.

See plan/02-phase1-datadecide.md task P1-05. Before anything can be called
"bias" (P1-06), we need to know how much of the observed gap between an
extrapolated prediction and the ground truth is just noise, and there are
three independent sources of it in DataDecide's data: which random seed a
run used, which nearby checkpoint happened to be evaluated, and how big the
eval set was.
"""

from __future__ import annotations

import polars as pl


def seed_variance(long_frame: pl.DataFrame, metric_name: str) -> pl.DataFrame:
    """Per (recipe, params_str, task): mean, seed variance (ddof=1), and
    seed count, from final-checkpoint rows at EVERY size (not just s*).

    P0-06/P1-01 already confirmed 3 seeds exist everywhere, including 1B
    (relabeled large-aux there, but still 3), so no extrapolated fallback
    to smaller sizes is needed -- an earlier plan draft assumed one might
    be; that assumption was wrong, see plan/02-phase1-datadecide.md P1-05.
    """
    # Sorted before the group-by/variance reduction for the same reason
    # `checkpoint_jitter` below is: fixing the input order makes the
    # (non-associative in floating point) `var()` reduction reproducible
    # run to run, regardless of what order a parallel/chunked parquet read
    # happens to hand back rows in.
    subset = long_frame.filter((pl.col("metric_name") == metric_name) & (pl.col("is_final"))).sort(
        ["recipe", "params_str", "task", "seed"]
    )
    return subset.group_by(["recipe", "params_str", "task"], maintain_order=True).agg(
        pl.col("params_num").first().alias("params_num"),
        pl.col("metric_value").mean().alias("mu"),
        pl.col("metric_value").var(ddof=1).fill_null(0.0).alias("sigma2_seed"),
        pl.col("metric_value").len().alias("n_seeds"),
    )


# The smallest run (6M params) has exactly 4 checkpoints total. A larger
# window would silently shrink to "every checkpoint" there while staying a
# genuine tail window everywhere else, making the quantity mean something
# different by size. 4 is the largest window that means the same thing --
# "variance of the last 4 checkpoints" -- at every size.
_N_LAST_CHECKPOINTS = 4


def checkpoint_jitter(
    long_frame: pl.DataFrame, metric_name: str, *, n_last_checkpoints: int = _N_LAST_CHECKPOINTS
) -> pl.DataFrame:
    """Per (recipe, params_str, seed, task): variance of the metric across
    the last `n_last_checkpoints` steps of that single run (raw variance,
    no detrending -- matches the plan's literal spec of "variance across
    the last few checkpoints").

    Available at every size and seed -- including where only 1 seed
    exists -- unlike seed variance, which needs multiple seeds. This is
    what the plan calls the fallback for when seeds are missing at s*;
    not needed at 1B itself (3 seeds are confirmed present there), but
    computed everywhere both for completeness and as a cross-check against
    seed variance's own trend across scale.
    """
    # Sorted on a fully deterministic key *before* the windowed rank and
    # the group-by/variance reduction below: `var()` is not
    # order-associative in floating point, and without a fixed input order
    # a parallel/chunked read of the cached parquet can feed the reduction
    # a different-but-equivalent row order on every run, producing
    # last-bit-different `sigma2_ckpt` values for the exact same set of 4
    # checkpoints -- caught by diffing two runs against each other, same as
    # every other reproducibility fix in this project.
    subset = long_frame.filter(pl.col("metric_name") == metric_name).sort(
        ["recipe", "params_str", "seed", "task", "step"]
    )
    ranked = subset.with_columns(
        pl.col("step")
        .rank(method="ordinal", descending=True)
        .over(["recipe", "params_str", "seed", "task"])
        .alias("_step_rank")
    )
    tail = ranked.filter(pl.col("_step_rank") <= n_last_checkpoints)
    stats = tail.group_by(["recipe", "params_str", "seed", "task"], maintain_order=True).agg(
        pl.col("params_num").first().alias("params_num"),
        pl.col("metric_value").var(ddof=1).alias("sigma2_ckpt"),
        pl.col("metric_value").len().alias("n_checkpoints_used"),
    )
    return stats.filter(pl.col("n_checkpoints_used") >= 2)


def eval_sampling_noise(p: float, n_instances: int) -> float:
    """Binomial eval-sampling variance of one task's accuracy estimate:
    `p(1-p)/n_instances`."""
    if n_instances <= 0:
        raise ValueError(f"n_instances must be positive, got {n_instances!r}")
    return p * (1.0 - p) / n_instances


def eval_sampling_noise_of_mean(component_ps: list[float], component_ns: list[int]) -> float:
    """Variance of an unweighted mean of independent binomial-noise
    accuracy estimates: `Var((1/K) sum X_i) = (1/K^2) sum Var(X_i)` for
    independent `X_i`.

    Needed for a *composite* task such as `olmes_10_macro_avg`, which this
    project verified is exactly the unweighted mean of the 10 primitive
    OLMES tasks' `primary_metric` values (see docs/decisions.md) -- it has
    no eval set of its own, so there is no single `n_instances` to plug
    into `eval_sampling_noise` directly; its noise is the noise of an
    average, not of a single binomial proportion.
    """
    if len(component_ps) != len(component_ns):
        raise ValueError("component_ps and component_ns must be the same length")
    if not component_ps:
        raise ValueError("need at least one component")
    k = len(component_ps)
    total = sum(eval_sampling_noise(p, n) for p, n in zip(component_ps, component_ns, strict=True))
    return total / (k**2)
