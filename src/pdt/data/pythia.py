"""Acquisition and loading for EleutherAI's Pythia suite (secondary ladder).

See plan/02-phase1-datadecide.md task P1-10. Pythia (Biderman et al. 2023)
gives a genuine K=2 data-recipe instance -- the deduplicated versus
standard Pile training split -- across a fixed 8-size ladder, independent
of DataDecide and of anything this project has tuned against.

Unlike DataDecide, Pythia's published per-checkpoint eval results live in
EleutherAI's own GitHub repo (``EleutherAI/pythia``), not a Hugging Face
dataset -- ``evals/pythia-v1/pythia-<size>[-deduped]/zero-shot/`` has one
small JSON file per (size, variant, training step), in the standard
``lm-evaluation-harness`` output format (``{"results": {task: {metric:
value, ...}, ...}}``). Confirmed directly (not assumed) before writing
this: 7 of the 8 canonical sizes (70m, 160m, 410m, 1.4b, 2.8b, 6.9b, 12b)
have both a plain and a ``-deduped`` directory, each with the same 27
checkpoint steps (0 through 143000 -- Pythia trains every size to the same
step count and token budget, unlike DataDecide's compute-matched ladder;
worth stating explicitly in any writeup that compares the two).

**``1b`` is the one exception** -- there is no plain ``pythia-1b``
directory, only ``pythia-1b-bf16``, ``pythia-1b-0.5MtokBS``, and
``pythia-1b-deduped`` (a real naming irregularity in the source repo,
found by trying it and getting a 404, not assumed). This module does not
special-case it: `SIZES` still lists ``"1b"`` for completeness, but
`raw_url("1b", "standard", ...)` will 404. Callers needing a full ladder
should use ``"1.4b"`` as the next size up instead, which has clean
`pythia-1.4b` / `pythia-1.4b-deduped` directories -- this is what
`experiments/p1_10_secondary_ladder.py` does.

Per the plan's "use published eval results where they exist" instruction,
this module only ever reads this published table -- it does not run any
evaluation locally.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import requests

_RAW_DIR = Path("data/raw/pythia")
_CACHE_DIR = Path("data/cache/pythia")

_REPO_RAW_BASE = "https://raw.githubusercontent.com/EleutherAI/pythia/main/evals/pythia-v1"

SIZES = ("70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b")
VARIANTS = ("standard", "deduped")

_STEPS = (
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1000,
    3000,
    13000,
    23000,
    33000,
    43000,
    53000,
    63000,
    73000,
    83000,
    93000,
    103000,
    113000,
    123000,
    133000,
    143000,
)
FINAL_STEP = max(_STEPS)


def _model_dir_name(size: str, variant: str) -> str:
    return f"pythia-{size}-deduped" if variant == "deduped" else f"pythia-{size}"


def _file_name(size: str, variant: str, step: int) -> str:
    stem = f"{size}-deduped" if variant == "deduped" else size
    return f"{stem}_step{step}.json"


def raw_url(size: str, variant: str, step: int) -> str:
    """The raw.githubusercontent.com URL for one (size, variant, step)
    eval JSON -- exposed so a caller can construct/verify a URL without
    downloading, e.g. for a dry-run or an error message."""
    if size not in SIZES:
        raise ValueError(f"unknown Pythia size {size!r}; expected one of {SIZES}")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    model_dir = _model_dir_name(size, variant)
    filename = _file_name(size, variant, step)
    return f"{_REPO_RAW_BASE}/{model_dir}/zero-shot/{filename}"


def download_checkpoint_eval(  # pragma: no cover -- thin network wrapper, verified by hand
    size: str, variant: str, step: int, *, force_refresh: bool = False
) -> Path:
    """Download (or return the cached copy of) one (size, variant, step)
    eval JSON. Cached under `_RAW_DIR` keyed by the same path structure
    the source repo uses, so a second call never re-hits the network."""
    local_path = _RAW_DIR / _model_dir_name(size, variant) / _file_name(size, variant, step)
    if local_path.exists() and not force_refresh:
        return local_path

    url = raw_url(size, variant, step)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(response.content)
    return local_path


def parse_eval_json(raw: dict, *, task_whitelist: frozenset[str] | None = None) -> dict[str, dict]:
    """Extract `{task_name: {metric_name: value}}` from one checkpoint's
    raw `lm-evaluation-harness` JSON (its `"results"` key). Restricts to
    `task_whitelist` when given -- the raw file carries ~100 tasks
    (bias/fairness probes, arithmetic, etc.) most of which have no
    DataDecide counterpart and aren't needed here.
    """
    results = raw.get("results", {})
    if task_whitelist is None:
        return results
    return {task: metrics for task, metrics in results.items() if task in task_whitelist}


def build_frame(  # pragma: no cover -- orchestrates download_checkpoint_eval, see above
    *,
    tasks: frozenset[str],
    metric_name: str = "acc",
    sizes: tuple[str, ...] = SIZES,
    steps: tuple[int, ...] = _STEPS,
    force_refresh: bool = False,
) -> pl.DataFrame:
    """The tidy long-format Pythia frame: one row per (variant, size, task,
    step), restricted to `tasks` and reading `metric_name` (`"acc"` by
    default -- matches DataDecide's own accuracy-based `primary_metric`
    more closely than `"acc_norm"`, which normalizes by answer length and
    isn't what this project's own headline metric does).

    Columns: `recipe` (`"standard"` or `"deduped"`, matching this
    project's `recipe` naming elsewhere), `params_str`, `task`, `step`,
    `is_final` (`step == FINAL_STEP`), `metric_name`, `metric_value`.

    Downloads are cached per (size, variant, step) via
    `download_checkpoint_eval`; a second call with the same arguments
    touches the network only for any newly-added combination.
    """
    rows = []
    for variant in VARIANTS:
        for size in sizes:
            for step in steps:
                path = download_checkpoint_eval(size, variant, step, force_refresh=force_refresh)
                raw = json.loads(path.read_text(encoding="utf-8"))
                parsed = parse_eval_json(raw, task_whitelist=tasks)
                for task, metrics in parsed.items():
                    if metric_name not in metrics:
                        continue
                    rows.append(
                        {
                            "recipe": variant,
                            "params_str": size,
                            "task": task,
                            "step": step,
                            "is_final": step == FINAL_STEP,
                            "metric_name": metric_name,
                            "metric_value": metrics[metric_name],
                        }
                    )

    if not rows:
        raise ValueError(
            f"no rows extracted for tasks={sorted(tasks)!r}, metric_name={metric_name!r} -- "
            "check the task names match the raw JSON's own task keys exactly "
            "(lm-evaluation-harness naming, e.g. 'arc_challenge', not DataDecide's)."
        )

    df = pl.DataFrame(rows)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_CACHE_DIR / "frame.parquet")
    return df
