"""N7-02: target test-retest, proxy agreement, resolvability strata, seeds vs scale.

EXPLORATORY: DataDecide `primary_metric` final checkpoints were already examined by an
out-of-repo probe (2026-11-20) before this analysis was written, so nothing here is a
confirmatory test (see plan/10-phase7-noise-decomposition.md).

All uncertainty is a recipe-cluster bootstrap: one (B, 25) matrix of recipe resamples is
shared by every task, size and statistic, so paired differences and 10-task means are
computed on identical resamples. `olmes_10_macro_avg` is reported separately and never
pooled with its own components. 750M is off-ladder (fewer training tokens than 530M) and
is excluded from the seeds-vs-scale frontier.

Writes results/n7_02_reliability.json.
"""

from __future__ import annotations

import os

for _blas_env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_env_var, "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from pdt import provenance  # noqa: E402
from pdt.analysis import reliability as rl  # noqa: E402
from pdt.data import datadecide as dd  # noqa: E402
from pdt.data import frame as frame_mod  # noqa: E402

_METRIC = "primary_metric"
_TARGET = "1B"
_AGGREGATE = "olmes_10_macro_avg"
_OFF_LADDER = ("750M",)
_B = 1000
_SEED = 20261121
_COMPARISONS = [
    (("8M", 3), ("20M", 1)),
    (("60M", 3), ("90M", 1)),
    (("90M", 2), ("150M", 1)),
    (("150M", 3), ("300M", 1)),
    (("300M", 2), ("530M", 1)),
    (("300M", 3), ("530M", 1)),
    (("530M", 3), (_TARGET, 1)),
]


def _load():
    lf = frame_mod.build_frame(source="macro_avg", metrics=(_METRIC,))
    lf = lf.filter(pl.col("is_final")).sort(["task", "params_num", "recipe", "seed"])
    sizes_df = (
        lf.group_by("params_str")
        .agg(
            pl.col("params_num").first(),
            pl.col("tokens").min().alias("tok_min"),
            pl.col("tokens").max().alias("tok_max"),
            pl.col("compute").min().alias("c_min"),
            pl.col("compute").max().alias("c_max"),
        )
        .sort("params_num")
    )
    assert (sizes_df["tok_min"] == sizes_df["tok_max"]).all()
    assert (sizes_df["c_min"] == sizes_df["c_max"]).all()
    sizes = sizes_df["params_str"].to_list()
    tokens = dict(zip(sizes, sizes_df["tok_max"].to_list(), strict=True))
    compute = dict(zip(sizes, sizes_df["c_max"].to_list(), strict=True))
    recipes = sorted(lf["recipe"].unique().to_list())
    tasks = sorted(lf["task"].unique().to_list())
    arr: dict[str, dict[str, np.ndarray]] = {}
    for task in tasks:
        arr[task] = {}
        for size in sizes:
            cell = lf.filter((pl.col("task") == task) & (pl.col("params_str") == size))
            labels = sorted(cell["seed"].unique().to_list())
            assert len(labels) == 3, (task, size, labels)
            m = np.full((len(recipes), 3), np.nan)
            for rec, seed, val in cell.select(["recipe", "seed", "metric_value"]).iter_rows():
                m[recipes.index(rec), labels.index(seed)] = val
            assert not np.isnan(m).any(), (task, size)
            arr[task][size] = m
    return arr, sizes, tokens, compute, recipes, tasks


def _stats(arr, tasks, proxies, idx):
    """Every bootstrapped statistic for one recipe resample `idx`, as a dict of floats."""
    out = {}
    for task in tasks:
        t = arr[task][_TARGET][idx]
        out[f"{task}|retest"] = rl.retest_agreement(t, idx)["mean"]
        for size in proxies:
            p = arr[task][size][idx]
            for k in (1, 2, 3):
                out[f"{task}|{size}|k{k}"] = rl.proxy_agreement(p, t, idx, k)
            ho = rl.stratified_heldout(p, t, idx)
            st = rl.stratified_same_target(p, t, idx)
            for b in range(len(rl.DEFAULT_BINS)):
                out[f"{task}|{size}|heldout|b{b}"] = ho["agreement"][b]
                out[f"{task}|{size}|internal|b{b}"] = ho["target_internal_agreement"][b]
                out[f"{task}|{size}|same|b{b}"] = st["agreement"][b]
    return out


def _with_means(d, tasks, proxies):
    comps = [t for t in tasks if t != _AGGREGATE]
    suffixes = ["retest"] + [
        f"{s}|{x}"
        for s in proxies
        for x in (
            ["k1", "k2", "k3"]
            + [f"{kind}|b{b}" for kind in ("heldout", "internal", "same") for b in range(4)]
        )
    ]
    for suf in suffixes:
        d[f"MEAN10|{suf}"] = float(np.nanmean([d[f"{t}|{suf}"] for t in comps]))
    return d


def main() -> None:
    arr, sizes, tokens, compute, recipes, tasks = _load()
    proxies = [s for s in sizes if s != _TARGET]
    r = len(recipes)
    full = np.arange(r)
    point = _with_means(_stats(arr, tasks, proxies, full), tasks, proxies)
    keys = sorted(point)
    boot_idx = rl.cluster_bootstrap_indices(r, B=_B, seed=_SEED)
    boot = np.asarray(
        [
            [d[k] for k in keys]
            for d in (_with_means(_stats(arr, tasks, proxies, i), tasks, proxies) for i in boot_idx)
        ]
    )
    col = {k: n for n, k in enumerate(keys)}

    def ci(key):
        lo, hi = rl.percentile_ci(boot[:, col[key]])
        return [float(lo), float(hi)]

    def entry(key):
        return {"estimate": point[key], "ci95": ci(key)}

    groups = [t for t in tasks if t != _AGGREGATE] + ["MEAN10", _AGGREGATE]
    by_task = {}
    for g in groups:
        rows = {"retest_1B": entry(f"{g}|retest"), "proxies": {}}
        for size in proxies:
            prox = {f"k{k}": entry(f"{g}|{size}|k{k}") for k in (1, 2, 3)}
            diff = boot[:, col[f"{g}|{size}|k1"]] - boot[:, col[f"{g}|retest"]]
            prox["k1_minus_retest"] = {
                "estimate": point[f"{g}|{size}|k1"] - point[f"{g}|retest"],
                "ci95": [float(x) for x in rl.percentile_ci(diff)],
            }
            prox["strata"] = {
                kind: [entry(f"{g}|{size}|{kind}|b{b}") for b in range(4)]
                for kind in ("heldout", "internal", "same")
            }
            prox["off_ladder"] = size in _OFF_LADDER
            rows["proxies"][size] = prox
        by_task[g] = rows

    # Stratum pair counts on the full sample (not bootstrapped).
    counts = {}
    for task in tasks:
        counts[task] = {}
        for size in proxies:
            p, t = arr[task][size], arr[task][_TARGET]
            counts[task][size] = {
                "heldout_pair_instances": rl.stratified_heldout(p, t, full)["n_pairs"],
                "same_target_pairs": rl.stratified_same_target(p, t, full)["n_pairs"],
            }

    # Seeds vs scale on the 10-task mean (costs relative to one 1B run).
    mean_vals = {(s, k): point[f"MEAN10|{s}|k{k}"] for s in proxies for k in (1, 2, 3)}
    pts = sorted(
        (
            {
                "size": s,
                "k": k,
                "cost": k * compute[s] / compute[_TARGET],
                "accuracy": mean_vals[(s, k)],
                "off_ladder": s in _OFF_LADDER,
            }
            for (s, k) in mean_vals
        ),
        key=lambda d: (d["cost"], d["size"], d["k"]),
    )
    best, frontier = -np.inf, []
    for pnt in pts:
        if not pnt["off_ladder"] and pnt["accuracy"] > best:
            best = pnt["accuracy"]
            frontier.append({"size": pnt["size"], "k": pnt["k"]})

    def arm_key(size, k):
        return "MEAN10|retest" if size == _TARGET else f"MEAN10|{size}|k{k}"

    comparisons = []
    for a, b in _COMPARISONS:
        ka, kb = arm_key(*a), arm_key(*b)
        diff = boot[:, col[ka]] - boot[:, col[kb]]
        comparisons.append(
            {
                "a": {"size": a[0], "k": a[1], "cost": a[1] * compute[a[0]] / compute[_TARGET]},
                "b": {"size": b[0], "k": b[1], "cost": b[1] * compute[b[0]] / compute[_TARGET]},
                "acc_a": point[ka],
                "acc_b": point[kb],
                "diff": point[ka] - point[kb],
                "diff_ci95": [float(x) for x in rl.percentile_ci(diff)],
            }
        )

    # Noise-model diagnostics per (task, size).
    diag = {}
    for task in tasks:
        diag[task] = {}
        for size in sizes:
            m = arr[task][size]
            disp = rl.log_variance_dispersion(m, n_sim=2000, seed=_SEED)
            diag[task][size] = {
                "pooled_seed_sd": rl.pooled_sd(m),
                "dispersion_ratio": disp["ratio"],
                "dispersion_p": disp["p"],
                "brown_forsythe_p": rl.brown_forsythe(m),
                **{f"anova_{k}": v for k, v in rl.seed_label_anova(m).items()},
            }
    comps = [t for t in tasks if t != _AGGREGATE]
    diag_summary = {
        size: {
            "n_tasks": len(comps),
            "dispersion_p_lt_05": sum(diag[t][size]["dispersion_p"] < 0.05 for t in comps),
            "brown_forsythe_p_lt_05": sum(diag[t][size]["brown_forsythe_p"] < 0.05 for t in comps),
            "anova_seed_p_lt_05": sum(diag[t][size]["anova_p"] < 0.05 for t in comps),
            "median_anova_seed_share": float(
                np.median([diag[t][size]["anova_seed_share"] for t in comps])
            ),
            "median_dispersion_ratio": float(
                np.median([diag[t][size]["dispersion_ratio"] for t in comps])
            ),
        }
        for size in sizes
    }

    payload = {
        "status": "EXPLORATORY -- data previously examined by an out-of-repo probe",
        "metric": _METRIC,
        "target": _TARGET,
        "n_recipes": r,
        "recipes": recipes,
        "sizes": sizes,
        "tokens_by_size": tokens,
        "compute_by_size": compute,
        "off_ladder_sizes": list(_OFF_LADDER),
        "bins_abs_z": [list(b) for b in rl.DEFAULT_BINS],
        "bootstrap": {"B": _B, "seed": _SEED, "unit": "recipe (cluster)"},
        "by_task": by_task,
        "stratum_counts": counts,
        "seeds_vs_scale_mean10": {
            "points": pts,
            "frontier": frontier,
            "target_retest": {"cost": 1.0, "accuracy": point["MEAN10|retest"]},
            "comparisons": comparisons,
        },
        "noise_diagnostics": diag,
        "noise_diagnostics_summary": diag_summary,
    }
    provenance.write_result(
        "results/n7_02_reliability.json",
        payload=payload,
        config={
            "task": "N7-02",
            "seed": _SEED,
            "bootstrap_B": _B,
            "metric": _METRIC,
            "dataset_revision": dd.cached_revision("macro_avg"),
        },
    )
    print("wrote results/n7_02_reliability.json")


if __name__ == "__main__":
    main()
