"""Acquisition and canonical loading for AI2's DataDecide artifacts.

See plan/01-phase0-setup.md task P0-06 and plan/02-phase1-datadecide.md for
the analysis this feeds. Three repos are used:

- ``allenai/DataDecide-eval-results`` (~700MB): the primary table, plus two
  bonus tables discovered while building this module that the plan didn't
  originally know existed -- see the module docstring notes below.
- ``allenai/DataDecide-ppl-results`` (~2MB): per-domain validation
  perplexities.
- ``allenai/DataDecide-data-recipes``: **NOT** downloaded as a snapshot.
  Its own README states "This HuggingFace Dataset contains the tokenized
  data used to build these recipes" -- it is 6,194 files and ~19.2 TB of
  raw preprocessed token .npy shards, not a small metadata table. What
  P0-06 actually needs (a human-readable table of what each of the 25
  named recipes is made of) lives in that repo's README.md as a markdown
  table, which is the only file this module ever fetches from it.
  ``allenai/DataDecide-eval-instances`` (~123GB) is similarly never
  touched -- per the plan, Phase 1 does not need it.

Two things about ``allenai/DataDecide-eval-results`` that the plan's
author did not know before this module was written, found by inspecting
the real files directly rather than trusting a secondhand description:

1. Beyond the four ``data/train-*.parquet`` shards (the per (recipe, size,
   seed, task, step) rows the plan describes), the repo also ships
   ``data/macro_avg-*.parquet`` (task-macro-averaged rows, including a
   precomputed ``olmes_10_macro_avg`` "task") and
   ``data/scaling_law_fit-*.parquet`` (**DataDecide's own baseline
   scaling-law fit results**, with a ``decision_acc`` column per
   task/mix/metric/setup). The latter is extremely valuable: it lets
   Phase 1 cross-check its own reproduction against the authors' own
   numbers directly, not just the ~80% figure quoted in the abstract.
2. The ``metrics`` column is **not** consistently formatted across these
   files. In the ``train`` shards it is a single-quoted Python dict repr
   (``ast.literal_eval`` required, ``json.loads`` raises
   ``JSONDecodeError``). In ``macro_avg`` it is proper double-quoted JSON.
   ``_parse_metrics`` below tries JSON first and falls back to
   ``ast.literal_eval``, handling both without the caller needing to know
   which file a row came from.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import huggingface_hub
import polars as pl

_RAW_DIR = Path("data/raw/datadecide")
_CACHE_DIR = Path("data/cache/datadecide")

_EVAL_RESULTS_REPO = "allenai/DataDecide-eval-results"
_PPL_RESULTS_REPO = "allenai/DataDecide-ppl-results"
_DATA_RECIPES_REPO = "allenai/DataDecide-data-recipes"

_PARAMS_RE = re.compile(r"^(\d+(?:\.\d+)?)([MB])$")
_PARAMS_MULTIPLIER = {"M": 1_000_000, "B": 1_000_000_000}


# ---------------------------------------------------------------------------
# Hub access
# ---------------------------------------------------------------------------


def resolve_revision(repo_id: str, repo_type: str = "dataset") -> str:  # pragma: no cover
    """Resolve `repo_id`'s current HEAD commit SHA on the Hub.

    Not unit tested: a thin pass-through to the Hub API. Verified by hand
    against the live Hub while building this module -- see the P0-06 PR
    description for the exact calls run and their output. Adding it to the
    automated suite would mean every CI run needs live network access to a
    third-party service to test one line of logic (`if not sha: raise`).
    """
    api = huggingface_hub.HfApi()
    info = api.dataset_info(repo_id) if repo_type == "dataset" else api.model_info(repo_id)
    sha = info.sha
    if not sha:
        raise RuntimeError(f"{repo_id}: Hub API returned no commit SHA")
    return sha


def download_snapshot(  # pragma: no cover -- see resolve_revision
    repo_id: str,
    *,
    allow_patterns: list[str] | str,
    revision: str | None = None,
    repo_type: str = "dataset",
) -> tuple[Path, str]:
    """Download `repo_id` at `revision` (or the current HEAD if unset).

    `allow_patterns` is required (not optional) -- there is no safe default
    for an arbitrary Hub repo, and two of the four DataDecide repos this
    module is aware of are 19TB and 123GB respectively. Every call site in
    this module passes an explicit pattern; do not add a call that omits
    one.

    Returns (local snapshot directory, resolved commit SHA). Pinning to an
    explicit SHA means a later call with the same SHA is a local no-op
    (huggingface_hub's own cache short-circuits it), and the SHA recorded
    in provenance always names an immutable snapshot.
    """
    pinned = revision or resolve_revision(repo_id, repo_type)
    local_dir = huggingface_hub.snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=pinned,
        allow_patterns=allow_patterns,
        local_dir=_RAW_DIR / repo_id.replace("/", "__"),
    )
    return Path(local_dir), pinned


def download_file(  # pragma: no cover -- see resolve_revision
    repo_id: str, filename: str, *, revision: str | None = None, repo_type: str = "dataset"
) -> tuple[Path, str]:
    """Download a single named file, pinned the same way as `download_snapshot`.

    Used for `allenai/DataDecide-data-recipes/README.md` specifically, so
    the other 6,193 files in that repo (the actual 19TB of tokenized
    corpora) are never touched.
    """
    pinned = revision or resolve_revision(repo_id, repo_type)
    local_path = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=pinned,
        filename=filename,
        local_dir=_RAW_DIR / repo_id.replace("/", "__"),
    )
    return Path(local_path), pinned


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_metrics(raw: str) -> dict[str, Any]:
    """Parse the `metrics` column: real JSON in some files, Python dict
    repr (single-quoted) in others. See module docstring point 2."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return ast.literal_eval(raw)


def parse_params(text: str) -> int:
    """Parse a params string like "10M" or "1B" to an integer count.

    Fails loudly on an unrecognised format rather than guessing -- a
    silently-wrong parse here would corrupt every scale comparison
    downstream.
    """
    match = _PARAMS_RE.match(text.strip())
    if not match:
        raise ValueError(f"unrecognised params format: {text!r}")
    number, suffix = match.groups()
    return int(float(number) * _PARAMS_MULTIPLIER[suffix])


def _explode_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Parse every row's `metrics` string and add one column per key found
    anywhere in the column (union of keys; missing keys become null for
    rows that don't have them -- different task types report different
    metric sets)."""
    parsed = [_parse_metrics(m) for m in df["metrics"].to_list()]
    all_keys: set[str] = set()
    for d in parsed:
        all_keys.update(d.keys())
    metric_cols = {
        f"metric_{key}": pl.Series([d.get(key) for d in parsed]) for key in sorted(all_keys)
    }
    return df.drop("metrics").with_columns(**metric_cols)


def _add_params_num_and_final_flag(df: pl.DataFrame) -> pl.DataFrame:
    params_num = pl.Series([parse_params(p) for p in df["params"].to_list()])
    df = df.with_columns(params_num=params_num).filter(pl.col("step") > 0)
    max_step = df.group_by(["params", "data", "seed"]).agg(pl.col("step").max().alias("_max_step"))
    df = df.join(max_step, on=["params", "data", "seed"], how="left")
    df = df.with_columns(is_final_checkpoint=(pl.col("step") == pl.col("_max_step"))).drop(
        "_max_step"
    )
    return df


# ---------------------------------------------------------------------------
# Loaders (one per logical table, each independently cached)
# ---------------------------------------------------------------------------


def _cached_or_build(
    cache_name: str, build: Callable[[], tuple[pl.DataFrame, str]]
) -> pl.DataFrame:
    """Read `cache_name`.parquet from the cache dir if present, else call
    `build()` (which returns (dataframe, resolved_revision)), write both
    the parquet and a `.revision.json` sidecar, and return the dataframe.

    This is what makes a second run of anything built on these loaders
    reproduce byte-for-byte (aside from the timestamp) without touching
    the network at all: once cached, `build` is never called again unless
    the cache is deleted.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{cache_name}.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    df, revision = build()
    df.write_parquet(cache_path)
    (_CACHE_DIR / f"{cache_name}.revision.json").write_text(
        json.dumps({"cache_name": cache_name, "revision": revision}, indent=2) + "\n",
        encoding="utf-8",
    )
    return df


def cached_revision(cache_name: str) -> str | None:
    """The resolved revision SHA recorded when `cache_name` was last built,
    or None if it hasn't been built yet."""
    sidecar = _CACHE_DIR / f"{cache_name}.revision.json"
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))["revision"]


def load_eval_results(revision: str | None = None) -> pl.DataFrame:
    """The primary per (recipe, size, seed, task, step) results table.

    `metrics` exploded into typed `metric_*` columns, `params` parsed to
    `params_num`, `step == 0` (untrained checkpoints) filtered out, and
    `is_final_checkpoint` flagging the max step within each
    (params, data, seed) group.
    """

    def build() -> tuple[pl.DataFrame, str]:  # pragma: no cover -- see download_*
        local_dir, pinned = download_snapshot(
            _EVAL_RESULTS_REPO, allow_patterns=["data/train-*.parquet"], revision=revision
        )
        shards = sorted(Path(local_dir).glob("data/train-*.parquet"))
        if not shards:
            raise RuntimeError(f"no train-*.parquet shards found under {local_dir}")
        df = pl.concat([pl.read_parquet(s) for s in shards])
        df = _explode_metrics(df)
        df = _add_params_num_and_final_flag(df)
        return df, pinned

    return _cached_or_build("eval_results", build)  # pragma: no cover -- see resolve_revision


def load_macro_avg(revision: str | None = None) -> pl.DataFrame:
    """The bonus task-macro-averaged table (see module docstring point 1),
    including the precomputed `olmes_10_macro_avg` "task"."""

    def build() -> tuple[pl.DataFrame, str]:  # pragma: no cover -- see download_*
        local_dir, pinned = download_snapshot(
            _EVAL_RESULTS_REPO, allow_patterns=["data/macro_avg-*.parquet"], revision=revision
        )
        shards = sorted(Path(local_dir).glob("data/macro_avg-*.parquet"))
        if not shards:
            raise RuntimeError(f"no macro_avg-*.parquet shards found under {local_dir}")
        df = pl.concat([pl.read_parquet(s) for s in shards])
        df = _explode_metrics(df)
        df = _add_params_num_and_final_flag(df)
        return df, pinned

    return _cached_or_build("macro_avg", build)  # pragma: no cover -- see resolve_revision


def load_scaling_law_fit(revision: str | None = None) -> pl.DataFrame:
    """DataDecide's own baseline scaling-law fit results, including a
    `decision_acc` column per (task, mix, metric, setup) -- the authors'
    own numbers to cross-check Phase 1's reproduction against."""

    def build() -> tuple[pl.DataFrame, str]:  # pragma: no cover -- see download_*
        local_dir, pinned = download_snapshot(
            _EVAL_RESULTS_REPO, allow_patterns=["data/scaling_law_fit-*.parquet"], revision=revision
        )
        shards = sorted(Path(local_dir).glob("data/scaling_law_fit-*.parquet"))
        if not shards:
            raise RuntimeError(f"no scaling_law_fit-*.parquet shards found under {local_dir}")
        df = pl.concat([pl.read_parquet(s) for s in shards])
        return df, pinned

    return _cached_or_build("scaling_law_fit", build)  # pragma: no cover -- see resolve_revision


def load_ppl_results(revision: str | None = None) -> pl.DataFrame:
    """Per-domain validation perplexities."""

    def build() -> tuple[pl.DataFrame, str]:  # pragma: no cover -- see download_*
        local_dir, pinned = download_snapshot(
            _PPL_RESULTS_REPO, allow_patterns=["data/*.parquet"], revision=revision
        )
        shards = sorted(Path(local_dir).glob("data/*.parquet"))
        if not shards:
            raise RuntimeError(f"no parquet shards found under {local_dir}")
        df = pl.concat([pl.read_parquet(s) for s in shards])
        return df, pinned

    return _cached_or_build("ppl_results", build)  # pragma: no cover -- see resolve_revision


_RECIPE_TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def _strip_markdown_links(cell: str) -> str:
    """Turn a markdown link cell into its link text, e.g.
    "[Original](https://...)" into "Original". Two of the 25 rows (the
    FineWeb variants) link their recipe name to the source dataset; every
    other cell is passed through unchanged."""
    return _MARKDOWN_LINK_RE.sub(r"\1", cell)


def _parse_recipes_markdown_table(readme_text: str) -> pl.DataFrame:
    """Parse the first markdown table in the data-recipes README (the
    "25 Data Recipes" section: Source | Recipe | Description). Stops at
    the second table (the per-size model-checkpoint link table under
    "350 Models over Differences in Data in Scale"), which is a different
    shape and not what `load_recipes` returns."""
    lines = readme_text.splitlines()
    rows: list[list[str]] = []
    in_table = False
    header: list[str] | None = None
    for line in lines:
        match = _RECIPE_TABLE_ROW_RE.match(line.strip())
        if not match:
            if in_table:
                break  # table ended
            continue
        cells = [_strip_markdown_links(c.strip()) for c in match.group(1).split("|")]
        if header is None:
            header = [c.strip("* ") for c in cells]
            in_table = True
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue  # markdown header/body separator row
        rows.append(cells)

    if header is None or not rows:
        raise RuntimeError("could not find the recipe definitions table in the README")

    columns = {header[i].lower(): [r[i] for r in rows] for i in range(len(header))}
    return pl.DataFrame(columns)


def load_recipes(revision: str | None = None) -> pl.DataFrame:
    """The 25 named data recipes with their source and human-readable
    composition, parsed from `allenai/DataDecide-data-recipes/README.md`.

    Deliberately does NOT download anything else from that repo -- see the
    module docstring. Returns columns `source`, `recipe`, `description`.
    """

    def build() -> tuple[pl.DataFrame, str]:  # pragma: no cover -- see download_*
        path, pinned = download_file(_DATA_RECIPES_REPO, "README.md", revision=revision)
        df = _parse_recipes_markdown_table(path.read_text(encoding="utf-8"))
        return df, pinned

    return _cached_or_build("recipes", build)  # pragma: no cover -- see resolve_revision
