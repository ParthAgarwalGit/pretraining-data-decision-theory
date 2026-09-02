"""The canonical tidy analysis frame for Phase 1.

See plan/02-phase1-datadecide.md task P1-01. Builds on
`pdt.data.datadecide`'s loaders (already cached; already handle metrics
parsing, the step==0 filter, and final-checkpoint flagging) rather than
re-parsing a raw `metrics` column a second time -- that parse alone costs
several minutes over 1.4M rows for `eval_results`, and P0-06 already paid
it once.

**Two sources, not one** (added after P1-01 first merged -- see
docs/decisions.md 2026-09-02 "P1-02: eval_results is the wrong granularity
for the headline reproduction"):

- ``"eval_results"`` -- 66 tasks, including each of the 57 individual MMLU
  *subject* splits (`mmlu_abstract_algebra`, `mmlu_marketing`, ...)
  separately. Fine-grained; many of these individual splits have small eval
  sets and produce exact ties between recipes.
- ``"macro_avg"`` -- 11 tasks: the 9 core OLMES task families plus an
  aggregated `mmlu` and `olmes_10_macro_avg`. **This is the granularity
  DataDecide's own baseline results use** (confirmed directly against
  `scaling_law_fit`'s `task` column, which contains exactly these 11
  values) -- P1-02/P1-03's headline-number reproduction must source from
  this, not `eval_results`, or it is comparing against the wrong unit of
  analysis.

The two sources don't expose the same metric columns (`macro_avg` is
missing `bits_per_byte_corr`, one of the four continuous-proxy metrics
reserved for P5-03) -- `metrics` is a parameter, not a fixed whitelist
baked into the function, precisely because of that.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from pdt.data import datadecide as dd

_CACHE_DIR = Path("data/cache/pdt")

# The 6 metrics P1-01 whitelists: 2 discrete-accuracy metrics used for the
# headline decision-accuracy analysis, plus 4 continuous likelihood proxies
# reserved for the P5-03 metric-choice ablation. Only a source's own
# available columns are used by default -- see build_frame()'s `metrics`
# parameter and its per-source validation.
METRIC_WHITELIST = (
    "primary_metric",
    "acc_per_char",
    "acc_raw",
    "correct_prob_per_char",
    "norm_correct_prob",
    "bits_per_byte_corr",
)

# Maps to attribute *names* on pdt.data.datadecide, resolved via getattr()
# at call time (not bound directly to the function objects here) so that
# monkeypatching dd.load_eval_results / dd.load_macro_avg in tests works as
# expected -- a dict of bound references would capture them at this
# module's import time instead, before any test monkeypatch runs.
_SOURCE_LOADER_NAMES = {
    "eval_results": "load_eval_results",
    "macro_avg": "load_macro_avg",
}

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


def build_frame(
    *,
    source: str = "eval_results",
    metrics: tuple[str, ...] = METRIC_WHITELIST,
    revision: str | None = None,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """Build (or load from cache) the tidy long-format analysis frame.

    One row per (recipe, params, seed, task, step, metric_name), restricted
    to `metrics`. Columns exactly match the spec in
    plan/02-phase1-datadecide.md P1-01:
    recipe | params_str | params_num | seed | task | step | tokens | compute
    | metric_name | metric_value | is_final

    `source` selects which upstream table to build from -- see the module
    docstring for why this matters and is not a single fixed choice.
    """
    if source not in _SOURCE_LOADER_NAMES:
        raise ValueError(
            f"unknown source {source!r}; expected one of {sorted(_SOURCE_LOADER_NAMES)}"
        )

    # The cache key must include `metrics`, not just `source`: two calls
    # with the same source but different metric selections build genuinely
    # different frames. Found the hard way -- an earlier version keyed the
    # cache on `source` alone, so a P1-02 run requesting only 2 metrics
    # silently poisoned the cache for P1-01's own 6-metric default request
    # made afterward (3x fewer rows, no error, no warning). A short hash of
    # the sorted metric names keeps the filename compact regardless of how
    # many metrics are requested.
    metrics_key = hashlib.sha256(",".join(sorted(metrics)).encode()).hexdigest()[:10]
    cache_path = _CACHE_DIR / f"frame_{source}__{metrics_key}.parquet"
    if cache_path.exists() and not force_refresh:
        return pl.read_parquet(cache_path)

    loader = getattr(dd, _SOURCE_LOADER_NAMES[source])
    wide = loader(revision=revision)

    value_cols = [f"metric_{name}" for name in metrics]
    missing = [c for c in value_cols if c not in wide.columns]
    if missing:
        raise RuntimeError(
            f"metrics {missing} not found as columns in load_{source}(). Either "
            f"the requested metric doesn't exist in this source, or the "
            f"whitelist is stale against the real schema -- check both."
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    long.write_parquet(cache_path)
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
