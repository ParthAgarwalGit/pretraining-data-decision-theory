"""N7-06: decompose DataDecide decision error into target noise, proxy noise and true
cross-scale reordering (EXPLORATORY: primary_metric data examined before this analysis).

For every task x proxy size vs the 3-seed 1B target, fits the Gaussian empirical-Bayes
model (pdt.analysis.decomposition) and reports, with recipe-cluster bootstrap CIs, the
CI-bearing estimands validated in N7-03:
  da_obs            observed decision accuracy (DataDecide's metric)
  da_latent         agreement of the proxy with the noise-free target
  ceiling           1 - target_noise_err: best agreement any proxy can show vs the target
  target_noise_err  target ranking errors caused by target seed noise
  reorder           true cross-scale reordering of recipe means
  proxy_noise_err   proxy ranking errors caused by proxy seed noise
Shares of observed error are point summaries only (N7-03: their CIs under-cover).
Both 95% and 99% intervals are stored (N7-03: target_noise_err covered at 0.91-0.93).
10-task means use the same bootstrap resamples. 750M is off-ladder and flagged.
Writes results/n7_06_decomposition.json.
"""

from __future__ import annotations

import importlib.util
import os
import sys

for _blas_env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_blas_env_var, "1")

import numpy as np  # noqa: E402

from pdt import provenance  # noqa: E402
from pdt.analysis import decomposition as dc  # noqa: E402
from pdt.analysis import reliability as rl  # noqa: E402
from pdt.data import datadecide as dd  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "n7_02_reliability", os.path.join(os.path.dirname(__file__), "n7_02_reliability.py")
)
n702 = importlib.util.module_from_spec(_SPEC)
sys.modules["n7_02_reliability"] = n702
_SPEC.loader.exec_module(n702)

_SEED = 20261123
_B = 200
_DRAWS = 300
_TARGET = "1B"
_AGG = "olmes_10_macro_avg"
_KEYS = ("da_obs", "da_latent", "ceiling", "target_noise_err", "reorder", "proxy_noise_err")


def _fit_vec(p, t, ids, seed):
    f = dc.fit_decomposition(p, t, ids, n_draws=_DRAWS, seed=seed)
    return np.array([f[k] for k in _KEYS]), f


def main() -> None:
    arr, sizes, _tok, _comp, recipes, tasks = n702._load()
    proxies = [s for s in sizes if s != _TARGET]
    comps = [t for t in tasks if t != _AGG]
    r = len(recipes)
    full = np.arange(r)
    idx = rl.cluster_bootstrap_indices(r, B=_B, seed=_SEED)
    rng = np.random.default_rng(_SEED + 1)
    seeds_point = int(rng.integers(2**31))
    boot_seeds = rng.integers(2**31, size=_B)

    point, boot, shares, clipped = {}, {}, {}, {}
    for task in tasks:
        for size in proxies:
            p, t = arr[task][size], arr[task][_TARGET]
            vec, fit = _fit_vec(p, t, full, seeds_point)
            point[(task, size)] = vec
            shares[(task, size)] = {"forward": fit["shares"], "reverse": fit["shares_reverse"]}
            clipped[(task, size)] = bool(fit["sigma_clipped"])
            boot[(task, size)] = np.array(
                [_fit_vec(p[i], t[i], i, int(s))[0] for i, s in zip(idx, boot_seeds, strict=True)]
            )

    def summarize(pt, bs):
        lo95, hi95 = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        lo99, hi99 = np.nanpercentile(bs, [0.5, 99.5], axis=0)
        return {
            k: {
                "estimate": float(pt[n]),
                "ci95": [float(lo95[n]), float(hi95[n])],
                "ci99": [float(lo99[n]), float(hi99[n])],
            }
            for n, k in enumerate(_KEYS)
        }

    by_task = {}
    for task in tasks:
        by_task[task] = {
            size: {
                **summarize(point[(task, size)], boot[(task, size)]),
                "shares_point_only": shares[(task, size)],
                "sigma_clipped": clipped[(task, size)],
                "off_ladder": size == "750M",
            }
            for size in proxies
        }
    by_task["MEAN10"] = {}
    for size in proxies:
        pt = np.mean([point[(t, size)] for t in comps], axis=0)
        bs = np.mean([boot[(t, size)] for t in comps], axis=0)
        e_obs = 1.0 - pt[0]
        by_task["MEAN10"][size] = {
            **summarize(pt, bs),
            "shares_point_only_from_means": {
                "target_noise": (e_obs - (1.0 - pt[1])) / e_obs,
                "proxy_noise": ((1.0 - pt[1]) - pt[4]) / e_obs,
                "reorder": pt[4] / e_obs,
            },
            "off_ladder": size == "750M",
        }
    payload = {
        "status": "EXPLORATORY -- data previously examined by an out-of-repo probe",
        "target": _TARGET,
        "estimands": list(_KEYS),
        "bootstrap": {"B": _B, "seed": _SEED, "unit": "recipe (cluster)"},
        "posterior_draws": _DRAWS,
        "n_sigma_clipped": int(sum(clipped.values())),
        "n_cells": len(clipped),
        "by_task": by_task,
    }
    provenance.write_result(
        "results/n7_06_decomposition.json",
        payload=payload,
        config={
            "task": "N7-06",
            "seed": _SEED,
            "dataset_revision": dd.cached_revision("macro_avg"),
        },
    )
    print("wrote results/n7_06_decomposition.json")


if __name__ == "__main__":
    main()
