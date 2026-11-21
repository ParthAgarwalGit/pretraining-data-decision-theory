"""N7-03: validate the empirical-Bayes decision-error decomposition before using it.

Part A -- simulation study. Data-generating processes are calibrated to DataDecide cells
(4 tasks x {150M, 530M} proxy vs 1B target):
  G1 gaussian   recipe means ~ N(m_hat, Sigma_hat) (the model is well specified),
                Gaussian seed noise with the pooled per-cell SD;
  G2 empirical  true recipe means = the observed 3-seed means (real, non-Gaussian recipe
                structure), Gaussian seed noise;
  G3 hetero_t   as G2, but per-recipe noise SDs vary (log-SD ~ N(0, 0.35)) and the noise
                is Student-t(5) rescaled to the same variance.
For each simulated suite: the point estimate, a recipe-cluster bootstrap 95% CI, and the
REALIZED truth computed from the true means. Reports bias and CI coverage of da_latent,
reorder, target_noise_err and the target-noise / reorder shares.

Part B -- posterior predictive checks on the real data: the observed 1B test-retest per
task, and the held-out-seed stratified agreement (150M and 530M), against their
posterior predictive distributions.

Pre-stated pass criteria (fixed before the first run): coverage >= 0.90 for every
estimand under G1 and G2 averaged over cells; the observed test-retest inside its 95%
predictive interval for >= 9 of 11 tasks. A failure is reported, not tuned away.
Writes results/n7_03_simulation.json.
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

_SEED = 20261122
_TASKS = ("arc_easy", "hellaswag", "boolq", "winogrande")
_SIZES = ("150M", "530M")
_GENERATORS = ("gaussian", "empirical", "hetero_t")
_N_SIMS = 100
_B = 100
_DRAWS = 200
_PPC_REPS = 500
_ESTIMANDS = ("da_latent", "reorder", "target_noise_err", "share_target_noise", "share_reorder")


def _signs(v, pairs):
    return rl.pair_signs(v, pairs)


def _realized_truth(mu, ybar, pairs):
    sp_o, st_o = _signs(ybar[:, 0], pairs), _signs(ybar[:, 1], pairs)
    sp_m, st_m = _signs(mu[:, 0], pairs), _signs(mu[:, 1], pairs)
    da_obs = rl.sign_agreement(sp_o, st_o)
    da_latent = rl.sign_agreement(sp_o, st_m)
    reorder = 1.0 - rl.sign_agreement(sp_m, st_m)
    e_obs = 1.0 - da_obs
    return {
        "da_latent": da_latent,
        "reorder": reorder,
        "target_noise_err": 1.0 - rl.sign_agreement(st_o, st_m),
        "share_target_noise": (e_obs - (1.0 - da_latent)) / e_obs if e_obs else np.nan,
        "share_reorder": reorder / e_obs if e_obs else np.nan,
    }


def _estimate(p, t, ids, seed):
    fit = dc.fit_decomposition(p, t, ids, n_draws=_DRAWS, seed=seed)
    return np.array(
        [
            fit["da_latent"],
            fit["reorder"],
            fit["target_noise_err"],
            fit["shares"]["target_noise"],
            fit["shares"]["reorder"],
        ]
    )


def _simulate(gen, p_obs, t_obs, rng):
    """One synthetic (P, T) suite and its true means, calibrated to an observed cell."""
    r = p_obs.shape[0]
    sd = np.array([rl.pooled_sd(p_obs), rl.pooled_sd(t_obs)])
    ybar = np.column_stack([p_obs.mean(1), t_obs.mean(1)])
    if gen == "gaussian":
        post = dc._posterior(p_obs, t_obs)
        sigma = post["Sigma"] + np.eye(2) * 1e-12 * np.trace(post["Sigma"] + 1e-30)
        mu = rng.multivariate_normal(ybar.mean(0), sigma, size=r)
    else:
        mu = ybar.copy()
    if gen == "hetero_t":
        scale = sd[None, :] * np.exp(rng.normal(0.0, 0.35, (r, 2)))
        noise = rng.standard_t(5, (r, 3, 2)) / np.sqrt(5.0 / 3.0)
    else:
        scale = np.broadcast_to(sd[None, :], (r, 2))
        noise = rng.normal(0.0, 1.0, (r, 3, 2))
    y = mu[:, None, :] + noise * scale[:, None, :]
    return y[:, :, 0], y[:, :, 1], mu


def _part_a(arr):
    rng = np.random.default_rng(_SEED)
    out = {}
    for gen in _GENERATORS:
        out[gen] = {}
        for task in _TASKS:
            for size in _SIZES:
                p_obs, t_obs = arr[task][size], arr[task]["1B"]
                r = p_obs.shape[0]
                ids = np.arange(r)
                pairs = rl.pair_index(ids)
                errs, covered = [], []
                for _ in range(_N_SIMS):
                    p, t, mu = _simulate(gen, p_obs, t_obs, rng)
                    truth = _realized_truth(mu, np.column_stack([p.mean(1), t.mean(1)]), pairs)
                    tv = np.array([truth[k] for k in _ESTIMANDS])
                    est = _estimate(p, t, ids, int(rng.integers(2**31)))
                    idx = rl.cluster_bootstrap_indices(r, B=_B, seed=int(rng.integers(2**31)))
                    boots = np.array(
                        [_estimate(p[i], t[i], i, int(rng.integers(2**31))) for i in idx]
                    )
                    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
                    errs.append(est - tv)
                    covered.append((tv >= lo) & (tv <= hi))
                errs, covered = np.array(errs), np.array(covered)
                out[gen][f"{task}|{size}"] = {
                    k: {
                        "bias": float(np.nanmean(errs[:, n])),
                        "bias_mc_se": float(np.nanstd(errs[:, n]) / np.sqrt(_N_SIMS)),
                        "rmse": float(np.sqrt(np.nanmean(errs[:, n] ** 2))),
                        "coverage": float(np.mean(covered[:, n])),
                    }
                    for n, k in enumerate(_ESTIMANDS)
                }
        out[gen]["MEAN_OVER_CELLS"] = {
            k: {
                stat: float(np.mean([v[k][stat] for c, v in out[gen].items() if "|" in c]))
                for stat in ("bias", "rmse", "coverage")
            }
            for k in _ESTIMANDS
        }
    return out


def _part_b(arr, tasks):
    rng = np.random.default_rng(_SEED + 1)
    retest, strata = {}, {}
    for task in tasks:
        t_obs = arr[task]["1B"]
        ids = np.arange(t_obs.shape[0])
        obs = rl.retest_agreement(t_obs, ids)["mean"]
        pred = dc.posterior_predictive_retest(
            arr[task]["530M"], t_obs, ids, n_rep=_PPC_REPS, seed=int(rng.integers(2**31))
        )
        lo, hi = np.percentile(pred, [2.5, 97.5])
        retest[task] = {
            "observed": obs,
            "pred_mean": float(pred.mean()),
            "pi95": [float(lo), float(hi)],
            "inside": bool(lo <= obs <= hi),
            "pred_quantile_of_observed": float(np.mean(pred <= obs)),
        }
        strata[task] = {}
        for size in ("150M", "530M"):
            p_obs = arr[task][size]
            obs_s = np.array(rl.stratified_heldout(p_obs, t_obs, ids)["agreement"])
            post = dc._posterior(p_obs, t_obs)
            sims = []
            for _ in range(200):
                mu = post["post_mean"] + rng.multivariate_normal(
                    np.zeros(2), post["post_cov"] + np.eye(2) * post["jitter"], size=len(ids)
                )
                p = mu[:, [0]] + rng.normal(0, np.sqrt(post["s2_P"]), (len(ids), 3))
                t = mu[:, [1]] + rng.normal(0, np.sqrt(post["s2_T"]), (len(ids), 3))
                sims.append(rl.stratified_heldout(p, t, ids)["agreement"])
            sims = np.array(sims, dtype=float)
            lo, hi = np.nanpercentile(sims, [2.5, 97.5], axis=0)
            strata[task][size] = {
                "observed": obs_s.tolist(),
                "pi95_lo": lo.tolist(),
                "pi95_hi": hi.tolist(),
                "inside": [bool(a <= o <= b) for o, a, b in zip(obs_s, lo, hi, strict=True)],
            }
    n_inside = sum(v["inside"] for v in retest.values())
    strata_checks = [x for t in strata.values() for s in t.values() for x in s["inside"]]
    return {
        "retest": retest,
        "retest_inside_count": n_inside,
        "retest_n": len(retest),
        "strata": strata,
        "strata_inside_fraction": float(np.mean(strata_checks)),
        "strata_n_checks": len(strata_checks),
    }


def main() -> None:
    arr, _sizes, _tokens, _compute, _recipes, tasks = n702._load()
    part_a = _part_a(arr)
    part_b = _part_b(arr, tasks)
    cov_ok = all(
        part_a[g]["MEAN_OVER_CELLS"][k]["coverage"] >= 0.90
        for g in ("gaussian", "empirical")
        for k in _ESTIMANDS
    )
    payload = {
        "status": "validation of the N7-03 estimator; exploratory data (calibration only)",
        "design": {
            "tasks": list(_TASKS),
            "proxy_sizes": list(_SIZES),
            "target": "1B",
            "generators": list(_GENERATORS),
            "n_sims": _N_SIMS,
            "bootstrap_B": _B,
            "posterior_draws": _DRAWS,
            "ppc_reps": _PPC_REPS,
            "estimands": list(_ESTIMANDS),
        },
        "pass_criteria": {
            "coverage_ge_0.90_G1_G2": cov_ok,
            "retest_inside_ge_9_of_11": part_b["retest_inside_count"] >= 9,
        },
        "simulation": part_a,
        "ppc": part_b,
    }
    provenance.write_result(
        "results/n7_03_simulation.json",
        payload=payload,
        config={
            "task": "N7-03",
            "seed": _SEED,
            "dataset_revision": dd.cached_revision("macro_avg"),
        },
    )
    print("wrote results/n7_03_simulation.json", payload["pass_criteria"])


if __name__ == "__main__":
    main()
