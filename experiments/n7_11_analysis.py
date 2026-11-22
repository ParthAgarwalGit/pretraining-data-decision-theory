"""N7-11 confirmatory analysis: hypotheses C1-C5 of docs/preregistration.md section 3
(frozen at ff4b978), on DataDecide final checkpoints re-scored on SciQ, LAMBADA-openai and
BLiMP by experiments/n7_11_score.py (outputs copied to data/n7_11_outputs/).

Benchmark scores (pre-registered): SciQ = mean `correct_per_char`; LAMBADA = mean
`greedy_match`; BLiMP = mean over its 67 subsets of mean `correct_raw`.
The analysis refuses to run on an incomplete set (all 25 recipes x 3 seeds x 4 sizes).

Per benchmark b, with the 3-seed 1B target:
  C1  target_noise_err (EB fit with the 530M proxy)       mean-CI lower bound > 0.02
  C2  reorder(150M) - reorder(530M)                       mean paired CI above 0
  C3  reorder(530M)                                       mean-CI lower bound > 0
  C4  held-out top-bin (|z|>=3) agreement, 530M x3        mean point >= 0.90
  C5  acc(300M x2) - acc(530M x1) vs 2-seed held-out 1B   mean paired CI inside +-0.05
Decision rule: criterion on the 3-benchmark mean AND predicted direction on >= 2 of 3.
A benchmark whose 3-seed 1B means take fewer than 2 distinct values is dropped (decided
from the 1B scores alone). Writes results/n7_11_confirmatory.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

for _blas_env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_env_var, "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from pdt import provenance  # noqa: E402
from pdt.analysis import decomposition as dc  # noqa: E402
from pdt.analysis import reliability as rl  # noqa: E402

OUTPUTS = Path("data/n7_11_outputs")
_SIZES = ("150M", "300M", "530M", "1B")
_BENCHMARKS = ("sciq", "lambada_openai", "blimp")
_SEED = 20261125
_B = 200
_DRAWS = 300
_HYP = ("C1", "C2", "C3", "C4", "C5")


def benchmark_score(bench: str, path: Path) -> float:
    df = pl.read_parquet(path)
    if bench == "sciq":
        return float(df["correct_per_char"].mean())
    if bench == "lambada_openai":
        return float(df["greedy_match"].mean())
    if bench == "blimp":
        sub = df.with_columns(pl.col("item_id").str.split("/").list.first().alias("subset"))
        per = sub.group_by("subset").agg(pl.col("correct_raw").mean()).sort("subset")
        return float(per["correct_raw"].mean())
    raise ValueError(bench)


def load_matrices(root: Path = OUTPUTS):
    """{bench: {size: (25, 3) array}} in sorted recipe x sorted seed-label order."""
    rows = []
    for done in sorted(root.glob("*/done.json")):
        job = json.loads(done.read_text(encoding="utf-8"))["job"]
        if job["kind"] != "datadecide":
            continue
        for b in _BENCHMARKS:
            rows.append(
                (
                    b,
                    job["size"],
                    job["recipe"],
                    job["seed_label"],
                    benchmark_score(b, done.parent / f"{b}.parquet"),
                )
            )
    df = pl.DataFrame(rows, schema=["bench", "size", "recipe", "seed", "score"], orient="row")
    recipes = sorted(df["recipe"].unique().to_list())
    out, missing = {}, []
    for b in _BENCHMARKS:
        out[b] = {}
        for size in _SIZES:
            cell = df.filter((pl.col("bench") == b) & (pl.col("size") == size))
            seeds = sorted(cell["seed"].unique().to_list())
            m = np.full((len(recipes), 3), np.nan)
            for rec, seed, val in cell.select(["recipe", "seed", "score"]).iter_rows():
                m[recipes.index(rec), seeds.index(seed)] = val
            if len(recipes) != 25 or len(seeds) != 3 or np.isnan(m).any():
                missing.append(f"{b}/{size}")
            out[b][size] = m
    return out, recipes, missing


def _stats(mats, sel, fit_seed):
    """(benchmarks, 5) hypothesis statistics on recipe resample `sel`."""
    ids = np.asarray(sel)
    rows = []
    for b in _BENCHMARKS:
        m = {s: mats[b][s][sel] for s in _SIZES}
        f150 = dc.fit_decomposition(m["150M"], m["1B"], ids, n_draws=_DRAWS, seed=fit_seed)
        f530 = dc.fit_decomposition(m["530M"], m["1B"], ids, n_draws=_DRAWS, seed=fit_seed)
        rows.append(
            [
                f530["target_noise_err"],
                f150["reorder"] - f530["reorder"],
                f530["reorder"],
                rl.stratified_heldout(m["530M"], m["1B"], ids)["agreement"][3],
                rl.proxy_agreement(m["300M"], m["1B"], ids, 2)
                - rl.proxy_agreement(m["530M"], m["1B"], ids, 1),
            ]
        )
    return np.array(rows)


def _decide(point_mean, lo, hi, per_bench, kept):
    """Pre-registered criteria on the mean + direction on >= 2 of the kept benchmarks."""
    crit = {
        "C1": lo[0] > 0.02,
        "C2": lo[1] > 0.0,
        "C3": lo[2] > 0.0,
        "C4": point_mean[3] >= 0.90,
        "C5": lo[4] > -0.05 and hi[4] < 0.05,
    }
    direction = {
        "C1": [v[0] > 0.02 for v in per_bench],
        "C2": [v[1] > 0 for v in per_bench],
        "C3": [v[2] > 0 for v in per_bench],
        "C4": [v[3] >= 0.90 for v in per_bench],
        "C5": [abs(v[4]) < 0.05 for v in per_bench],
    }
    need = min(2, len(kept))
    return {
        h: {
            "mean_criterion_met": bool(crit[h]),
            "n_benchmarks_in_direction": int(sum(direction[h])),
            "supported": bool(crit[h] and sum(direction[h]) >= need),
        }
        for h in _HYP
    }


def main() -> None:
    mats, recipes, missing = load_matrices()
    if missing:
        raise SystemExit(f"incomplete outputs, refusing to analyse: {missing}")
    kept = [b for b in _BENCHMARKS if len(np.unique(np.round(mats[b]["1B"].mean(1), 12))) >= 2]
    kidx = [_BENCHMARKS.index(b) for b in kept]
    r = len(recipes)
    rng = np.random.default_rng(_SEED + 1)
    fit_seeds = rng.integers(2**31, size=_B + 1)
    point = _stats(mats, np.arange(r), int(fit_seeds[0]))
    boot = np.array(
        [
            _stats(mats, idx, int(s))
            for idx, s in zip(
                rl.cluster_bootstrap_indices(r, B=_B, seed=_SEED), fit_seeds[1:], strict=True
            )
        ]
    )
    pm = point[kidx].mean(0)
    bm = boot[:, kidx, :].mean(1)
    lo, hi = np.percentile(bm, [2.5, 97.5], axis=0)
    per_bench = {
        b: {
            h: {
                "estimate": float(point[k, n]),
                "ci95": [float(x) for x in np.percentile(boot[:, k, n], [2.5, 97.5])],
            }
            for n, h in enumerate(_HYP)
        }
        for k, b in enumerate(_BENCHMARKS)
    }
    payload = {
        "status": "CONFIRMATORY (pre-registered C1-C5, docs/preregistration.md section 3)",
        "benchmarks_kept": kept,
        "benchmarks_dropped": [b for b in _BENCHMARKS if b not in kept],
        "statistics": {
            "C1": "target_noise_err (530M fit)",
            "C2": "reorder(150M) - reorder(530M)",
            "C3": "reorder(530M)",
            "C4": "held-out top-bin agreement, 530M x3",
            "C5": "acc(300M x2) - acc(530M x1)",
        },
        "mean_over_kept": {
            h: {"estimate": float(pm[n]), "ci95": [float(lo[n]), float(hi[n])]}
            for n, h in enumerate(_HYP)
        },
        "per_benchmark": per_bench,
        "decisions": _decide(pm, lo, hi, [point[k] for k in kidx], kept),
        "benchmark_scores_1B_mean": {b: float(mats[b]["1B"].mean()) for b in _BENCHMARKS},
        "bootstrap": {"B": _B, "seed": _SEED, "unit": "recipe (cluster)"},
    }
    provenance.write_result(
        "results/n7_11_confirmatory.json",
        payload=payload,
        config={"task": "N7-11", "seed": _SEED, "preregistration_commit": "ff4b978"},
    )
    print(
        "wrote results/n7_11_confirmatory.json",
        {h: v["supported"] for h, v in payload["decisions"].items()},
    )


if __name__ == "__main__":
    main()
