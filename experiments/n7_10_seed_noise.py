"""N7-10 S1: does DataDecide's 3-seed estimate of 1B seed noise agree with a 10-seed
estimate? (CONFIRMATORY check, docs/preregistration.md section 5, frozen at ff4b978.)

Signal & Noise (`allenai/signal-and-noise`, split `random_seeds`) trained 10 OLMo-2 1B-5xC
models that differ only in initialisation seed (run_type "seed"), on one data mix (the
OLMoE mix; not a DataDecide recipe). For each OLMES task shared with DataDecide, compare
the SD of the final-step scores over those 10 runs with DataDecide's pooled 3-seed
within-recipe SD at 1B, using the same metric (`acc_per_char`, available in both).
Criterion: SD ratio in [0.5, 2] for >= 7 of the shared tasks. Descriptive: the recipe,
architecture version and evaluation pipeline differ.
Writes results/n7_10_seed_noise.json.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from huggingface_hub import HfApi, hf_hub_download

from pdt import provenance
from pdt.analysis import reliability as rl
from pdt.data import datadecide as dd
from pdt.data import frame as frame_mod

_REPO = "allenai/signal-and-noise"
_FILE = "data/random_seeds-00000-of-00001.parquet"
_METRIC = "acc_per_char"
_TASKS = (
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


def main() -> None:
    revision = HfApi().dataset_info(_REPO).sha
    path = hf_hub_download(
        _REPO, _FILE, repo_type="dataset", revision=revision, cache_dir="data/cache/hf"
    )
    sn = pl.read_parquet(path).filter(
        (pl.col("run_type") == "seed") & (pl.col("metric") == _METRIC)
    )
    final = sn.group_by("run_name").agg(pl.col("step").max().alias("final"))
    sn = sn.join(final, on="run_name").filter(pl.col("step") == pl.col("final"))

    lf = frame_mod.build_frame(source="macro_avg", metrics=(_METRIC,))
    lf = lf.filter(pl.col("is_final") & (pl.col("params_str") == "1B"))
    per_task, n_in = {}, 0
    for task in _TASKS:
        vals = sn.filter(pl.col("task_name") == task).sort("run_name")["value"].to_numpy()
        cell = lf.filter(pl.col("task") == task).sort(["recipe", "seed"])
        recipes = sorted(cell["recipe"].unique().to_list())
        m = np.array(
            [
                cell.filter(pl.col("recipe") == r).sort("seed")["metric_value"].to_numpy()
                for r in recipes
            ]
        )
        if len(vals) < 2 or m.shape != (len(recipes), 3):
            per_task[task] = {"status": "missing", "n_signal_noise_runs": len(vals)}
            continue
        sd10 = float(np.std(vals, ddof=1))
        sd3 = rl.pooled_sd(m)
        ratio = sd3 / sd10 if sd10 > 0 else float("nan")
        inside = bool(0.5 <= ratio <= 2.0)
        n_in += inside
        per_task[task] = {
            "n_signal_noise_runs": int(len(vals)),
            "signal_noise_sd_10_init_seeds": sd10,
            "datadecide_pooled_3seed_sd": sd3,
            "ratio_datadecide_over_signal_noise": ratio,
            "ratio_in_0.5_2": inside,
        }
    n_shared = sum("ratio_in_0.5_2" in v for v in per_task.values())
    payload = {
        "status": "CONFIRMATORY check S1 (pre-registered; descriptive comparison)",
        "metric": _METRIC,
        "per_task": per_task,
        "n_shared_tasks": n_shared,
        "n_ratio_in_range": int(n_in),
        "criterion_ge_7": bool(n_in >= 7),
    }
    provenance.write_result(
        "results/n7_10_seed_noise.json",
        payload=payload,
        config={
            "task": "N7-10-S1",
            "signal_noise_revision": revision,
            "dataset_revision": dd.cached_revision("macro_avg"),
            "preregistration_commit": "ff4b978",
        },
    )
    print("wrote results/n7_10_seed_noise.json", n_in, "/", n_shared)


if __name__ == "__main__":
    main()
