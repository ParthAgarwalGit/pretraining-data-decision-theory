"""The canonical tidy analysis frame for Phase 1.

See plan/02-phase1-datadecide.md task P1-01. Builds on
`pdt.data.datadecide.load_eval_results()` (already cached; already handles
metrics parsing, the step==0 filter, and final-checkpoint flagging) rather
than re-parsing the raw `metrics` column a second time -- that parse alone
costs several minutes over 1.4M rows, and P0-06 already paid it once.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pdt.data import datadecide as dd

_CACHE_PATH = Path("data/cache/pdt/frame.parquet")

# The 6 metrics P1-01 whitelists: 2 discrete-accuracy metrics used for the
# headline decision-accuracy analysis, plus 4 continuous likelihood proxies
# reserved for the P5-03 metric-choice ablation.
METRIC_WHITELIST = (
    "primary_metric",
    "acc_per_char",
    "acc_raw",
    "correct_prob_per_char",
    "norm_correct_prob",
    "bits_per_byte_corr",
)

_ID_COLS = (
    "data",
    "params",
    "params_num",
    "seed",
    "task",
    "step",
    "tokens",
    "compute",
    "is_final_checkpoint",
)

_FINAL_COLUMN_ORDER = (
    "recipe",
    "params_str",
    "params_num",
    "seed",
    "task",
    "step",
    "tokens",
    "compute",
    "metric_name",
    "metric_value",
    "is_final",
)


def build_frame(*, revision: str | None = None, force_refresh: bool = False) -> pl.DataFrame:
    """Build (or load from cache) the tidy long-format analysis frame.

    One row per (recipe, params, seed, task, step, metric_name), restricted
    to METRIC_WHITELIST. Columns exactly match the spec in
    plan/02-phase1-datadecide.md P1-01:
    recipe | params_str | params_num | seed | task | step | tokens | compute
    | metric_name | metric_value | is_final
    """
    if _CACHE_PATH.exists() and not force_refresh:
        return pl.read_parquet(_CACHE_PATH)

    wide = dd.load_eval_results(revision=revision)

    value_cols = [f"metric_{name}" for name in METRIC_WHITELIST]
    missing = [c for c in value_cols if c not in wide.columns]
    if missing:
        raise RuntimeError(
            f"METRIC_WHITELIST names not found as columns in load_eval_results(): "
            f"{missing}. The whitelist may be stale against the real schema."
        )

    long = wide.select([*_ID_COLS, *value_cols]).unpivot(
        index=list(_ID_COLS),
        on=value_cols,
        variable_name="metric_name",
        value_name="metric_value",
    )
    long = long.with_columns(pl.col("metric_name").str.replace("^metric_", ""))
    long = long.rename(
        {"data": "recipe", "params": "params_str", "is_final_checkpoint": "is_final"}
    )
    long = long.select(list(_FINAL_COLUMN_ORDER))

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    long.write_parquet(_CACHE_PATH)
    return long


def coverage_matrix(frame_or_wide: pl.DataFrame) -> dict:
    """Check every (recipe, params, seed, task) cell for a final checkpoint.

    Accepts either the long frame from build_frame() or the wide frame from
    datadecide.load_eval_results() -- only `recipe`/`data`, `params_str`/
    `params`, `seed`, `task`, `is_final`/`is_final_checkpoint` are used, so
    either shape works and the caller doesn't pay for an unnecessary melt
    just to check coverage.

    The expected grid is built from the *observed* (params, seed) pairs
    (not an assumed fixed seed list), since seed labels differ by scale
    (small aux vs large aux at 1B -- see docs/decisions.md 2026-09-02) and
    a hardcoded seed list would silently misjudge coverage at 1B.

    Returns a JSON-serializable summary: totals, a per-task and per-params
    breakdown of holes, and up to 200 example missing cells.
    """
    df = frame_or_wide
    recipe_col = "recipe" if "recipe" in df.columns else "data"
    params_col = "params_str" if "params_str" in df.columns else "params"
    final_col = "is_final" if "is_final" in df.columns else "is_final_checkpoint"

    df = df.select([recipe_col, params_col, "seed", "task", final_col]).rename(
        {recipe_col: "recipe", params_col: "params_str", final_col: "is_final"}
    )

    recipes = df.select("recipe").unique()
    params_seed_pairs = df.select(["params_str", "seed"]).unique()
    tasks = df.select("task").unique()

    expected = recipes.join(params_seed_pairs, how="cross").join(tasks, how="cross")

    present = (
        df.filter(pl.col("is_final"))
        .select(["recipe", "params_str", "seed", "task"])
        .unique()
        .with_columns(pl.lit(True).alias("_present"))
    )

    joined = expected.join(present, on=["recipe", "params_str", "seed", "task"], how="left")
    missing = joined.filter(pl.col("_present").is_null()).drop("_present")

    n_expected = expected.height
    n_missing = missing.height
    n_present = n_expected - n_missing

    missing_by_task = (
        missing.group_by("task").agg(pl.len().alias("n_missing")).sort("n_missing", descending=True)
    )
    missing_by_params = (
        missing.group_by("params_str")
        .agg(pl.len().alias("n_missing"))
        .sort("n_missing", descending=True)
    )

    return {
        "n_recipes": recipes.height,
        "n_params_seed_pairs": params_seed_pairs.height,
        "n_tasks": tasks.height,
        "n_expected_cells": n_expected,
        "n_present_cells": n_present,
        "n_missing_cells": n_missing,
        "fraction_present": (n_present / n_expected) if n_expected else 1.0,
        "missing_by_task": dict(
            zip(
                missing_by_task["task"].to_list(),
                missing_by_task["n_missing"].to_list(),
                strict=True,
            )
        ),
        "missing_by_params": dict(
            zip(
                missing_by_params["params_str"].to_list(),
                missing_by_params["n_missing"].to_list(),
                strict=True,
            )
        ),
        "missing_cells_sample": missing.head(200).to_dicts(),
        "missing_cells_sample_truncated": n_missing > 200,
    }
