"""N7-09: item-level noise from DataDecide per-instance predictions (CONFIRMATORY; analyses
pre-registered in docs/preregistration.md section 4, frozen at commit ff4b978).

I0  sanity gate: every prefetched (recipe, size, seed) checkpoint reproduces the frame's
    primary_metric for all 10 tasks within 1e-6, under a single seed-name mapping per size,
    at the frame's final step. Any failure stops the analysis (status GATE).
I1  share of the 1B pairwise-gap variance from item sampling vs seed noise (estimation).
I2  population-level ceiling (target vs recipe means over the item population: seed +
    item noise) vs the fixed-item ceiling; paired recipe-cluster bootstrap of the
    10-task-mean difference. Criterion: CI above 0.
I3  resolved share of 1B recipe pairs, paired item-level sign test vs seed-only Welch
    (Bonferroni over the 300 pairs of a task, alpha 0.05). Criterion: 10-task mean higher
    by >= 10 percentage points.
Writes results/n7_09_instances.json. Parsed archives are cached (gitignored) under
data/cache/n7_09_parsed/ to make the determinism re-run cheap.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path

for _blas_env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_env_var, "1")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

from pdt import provenance  # noqa: E402
from pdt.analysis import decomposition as dc  # noqa: E402
from pdt.analysis import instances as inst  # noqa: E402
from pdt.analysis import reliability as rl  # noqa: E402
from pdt.data import datadecide as dd  # noqa: E402
from pdt.data import frame as frame_mod  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "n7_02_reliability", os.path.join(os.path.dirname(__file__), "n7_02_reliability.py")
)
n702 = importlib.util.module_from_spec(_SPEC)
sys.modules["n7_02_reliability"] = n702
_SPEC.loader.exec_module(n702)

_SEED = 20261124
_B = 200
_DRAWS = 300
_TOL = 1e-6
_ALPHA = 0.05
_PROXY_FOR_CEILING = "530M"
_TASKS = tuple(sorted(inst.PRIMARY_NORM))
_CACHE = Path("data/cache/n7_09_parsed")


def _parsed(slug: str, cell_key: str, rel_file: str) -> dict:
    path = _CACHE / slug / (cell_key.replace("/", "__") + ".npz")
    if not path.exists():
        cells = inst.cell_correctness(inst.INSTANCE_DIR / slug / rel_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **{f"{t}|{k}": v for t, d in cells.items() for k, v in d.items()},
        )
    z = np.load(path)
    out: dict[str, dict] = {}
    for key in z.files:
        t, k = key.split("|")
        out.setdefault(t, {})[k] = z[key]
    return out


def _frame_scores() -> dict:
    lf = frame_mod.build_frame(source="macro_avg", metrics=("primary_metric",))
    lf = lf.filter(pl.col("is_final") & (pl.col("task") != "olmes_10_macro_avg"))
    scores = {}
    steps = {}
    for rec, size, seed, task, step, val in lf.select(
        ["recipe", "params_str", "seed", "task", "step", "metric_value"]
    ).iter_rows():
        scores[(rec, size, seed, task)] = val
        steps[size] = step
    return scores, steps


def _i0(scores, steps):
    rows, mapping, ok = [], {}, True
    for slug, recipe in sorted(inst.RECIPE_SLUGS.items()):
        manifest = json.loads((inst.INSTANCE_DIR / slug / "manifest.json").read_text())
        by_size: dict[str, list] = {}
        for cell_key, cell in manifest["cells"].items():
            size, seed_dir = cell_key.split("/")
            near = cell.get("nearest")
            if near is None:
                rows.append({"recipe": recipe, "cell": cell_key, "status": "no_checkpoint"})
                ok = False
                continue
            parsed = _parsed(slug, cell_key, near["file"])
            sc = {t: inst.weighted_score(d) for t, d in parsed.items()}
            by_size.setdefault(size, []).append((seed_dir, sc, near["step"] == steps.get(size)))
        for size, cells in by_size.items():
            labels = sorted({k[2] for k in scores if k[0] == recipe and k[1] == size})
            for seed_dir, sc, step_match in cells:
                diffs = {
                    lab: max(abs(sc[t] - scores[(recipe, size, lab, t)]) for t in _TASKS)
                    for lab in labels
                }
                best = min(diffs, key=diffs.get)
                matched = diffs[best] <= _TOL
                ok &= matched
                prev = mapping.setdefault(size, {}).setdefault(seed_dir, best)
                consistent = prev == best
                ok &= consistent
                rows.append(
                    {
                        "recipe": recipe,
                        "cell": f"{size}/{seed_dir}",
                        "label": best,
                        "max_abs_diff": diffs[best],
                        "step_matches_frame_final": step_match,
                        "status": "ok" if (matched and consistent) else "MISMATCH",
                    }
                )
    return ok, mapping, rows


def _target_items(mapping):
    """Per task: (R, 3, n_items) correctness at 1B in frame seed-label order, weights, blocks."""
    labels = sorted(mapping["1B"].values())
    inv = {lab: d for d, lab in mapping["1B"].items()}
    recipes = sorted(inst.RECIPE_SLUGS.values())
    slug_of = {v: k for k, v in inst.RECIPE_SLUGS.items()}
    out = {}
    for task in _TASKS:
        mats = []
        for rec in recipes:
            slug = slug_of[rec]
            manifest = json.loads((inst.INSTANCE_DIR / slug / "manifest.json").read_text())
            seeds = []
            for lab in labels:
                key = f"1B/{inv[lab]}"
                d = _parsed(slug, key, manifest["cells"][key]["nearest"]["file"])[task]
                seeds.append(d["correct"])
                w, b = d["weight"], d["block"]
            mats.append(np.stack(seeds))
        out[task] = (np.stack(mats).astype(float), w, b)
    return recipes, out


def main() -> None:
    scores, steps = _frame_scores()
    ok, mapping, i0_rows = _i0(scores, steps)
    payload = {
        "status": "CONFIRMATORY (pre-registered N7-05, docs/preregistration.md section 4)",
        "i0": {
            "passed": ok,
            "n_cells": len(i0_rows),
            "n_ok": sum(r["status"] == "ok" for r in i0_rows),
            "max_abs_diff": max((r.get("max_abs_diff", 0.0) for r in i0_rows), default=0.0),
            "seed_mapping": mapping,
            "failures": [r for r in i0_rows if r["status"] != "ok"],
        },
    }
    if ok:
        arr, *_rest, tasks_frame = n702._load()
        recipes, items = _target_items(mapping)
        r = len(recipes)
        pairs = list(itertools.combinations(range(r), 2))
        i1, i3, extra, vmats = {}, {}, {}, {}
        for task in _TASKS:
            c, w, b = items[task]
            cbar = c.mean(1)  # seed-averaged per-item correctness, (R, n_items)
            s2 = rl.pooled_sd(arr[task]["1B"]) ** 2
            vmat = inst.pairwise_item_gap_variance(cbar, w, b)
            vmats[task] = vmat
            v_item = np.array([vmat[x, y] for x, y in pairs])
            v_seed = 2.0 * s2 / 3.0
            share = v_item / (v_item + v_seed)
            i1[task] = {
                "seed_var_of_gap": v_seed,
                "median_item_var_of_gap": float(np.median(v_item)),
                "median_item_share": float(np.median(share)),
            }
            extra[task] = float(np.mean(v_item) / 2.0)
            welch_p = np.array(
                [
                    stats.ttest_ind(arr[task]["1B"][x], arr[task]["1B"][y], equal_var=False).pvalue
                    for x, y in pairs
                ]
            )
            sign_p = np.array([inst.paired_sign_test_p(cbar[x], cbar[y]) for x, y in pairs])
            thr = _ALPHA / len(pairs)
            i3[task] = {
                "resolved_share_welch_seed": float(np.mean(np.nan_to_num(welch_p, nan=1.0) < thr)),
                "resolved_share_item_sign": float(np.mean(sign_p < thr)),
            }
        mean_i3 = {
            k: float(np.mean([v[k] for v in i3.values()]))
            for k in ("resolved_share_welch_seed", "resolved_share_item_sign")
        }
        mean_i3["difference"] = (
            mean_i3["resolved_share_item_sign"] - mean_i3["resolved_share_welch_seed"]
        )

        # I2: paired recipe bootstrap of ceiling(population) - ceiling(fixed items).
        idx = rl.cluster_bootstrap_indices(r, B=_B, seed=_SEED)
        rng = np.random.default_rng(_SEED + 1)
        fit_seeds = rng.integers(2**31, size=_B + 1)

        def ceilings(sel, fit_seed):
            out = []
            for task in _TASKS:
                p, t = arr[task][_PROXY_FOR_CEILING][sel], arr[task]["1B"][sel]
                ids = np.asarray(sel)
                xi, yi = rl.pair_index(ids)
                ex = float(np.mean(vmats[task][ids[xi], ids[yi]]) / 2.0)
                fixed = dc.fit_decomposition(p, t, ids, n_draws=_DRAWS, seed=int(fit_seed))
                pop = dc.fit_decomposition(
                    p, t, ids, n_draws=_DRAWS, seed=int(fit_seed), extra_target_noise_var=ex
                )
                out.append([fixed["ceiling"], pop["ceiling"]])
            return np.array(out)  # (tasks, 2)

        point = ceilings(np.arange(r), fit_seeds[0])
        boots = np.array([ceilings(i, s) for i, s in zip(idx, fit_seeds[1:], strict=True)])
        diff_pt = float(np.mean(point[:, 1] - point[:, 0]))
        diff_bs = np.mean(boots[:, :, 1] - boots[:, :, 0], axis=1)
        lo, hi = np.percentile(diff_bs, [2.5, 97.5])
        i2 = {
            "proxy_used_for_sigma": _PROXY_FOR_CEILING,
            "per_task": {
                t: {
                    "ceiling_fixed_items": float(point[k, 0]),
                    "ceiling_population": float(point[k, 1]),
                    "extra_target_noise_var": extra[t],
                }
                for k, t in enumerate(_TASKS)
            },
            "mean10_ceiling_fixed": float(point[:, 0].mean()),
            "mean10_ceiling_population": float(point[:, 1].mean()),
            "mean10_difference_pop_minus_fixed": diff_pt,
            "mean10_difference_ci95": [float(lo), float(hi)],
            "criterion_ci_above_0_for_fixed_minus_pop": bool(hi < 0),
        }
        payload.update(
            {
                "i1_item_vs_seed_variance": {
                    "per_task": i1,
                    "mean10_median_item_share": float(
                        np.mean([v["median_item_share"] for v in i1.values()])
                    ),
                },
                "i2_population_ceiling": i2,
                "i3_resolved_pairs": {
                    "per_task": i3,
                    "mean10": mean_i3,
                    "alpha_bonferroni": _ALPHA / len(pairs),
                    "criterion_ge_10pp": bool(mean_i3["difference"] >= 0.10),
                },
                "bootstrap": {"B": _B, "seed": _SEED, "unit": "recipe (cluster)"},
            }
        )
    provenance.write_result(
        "results/n7_09_instances.json",
        payload=payload,
        config={
            "task": "N7-09",
            "seed": _SEED,
            "dataset_revision": dd.cached_revision("macro_avg"),
            "instances_revision": (inst.INSTANCE_DIR / "source_revision.txt").read_text().strip(),
            "preregistration_commit": "ff4b978",
        },
    )
    print("wrote results/n7_09_instances.json; I0 passed:", ok)


if __name__ == "__main__":
    main()
