"""Tests for pdt.analysis.decomposition (N7-03)."""

from __future__ import annotations

import numpy as np
import pytest

from pdt.analysis import decomposition as dc
from pdt.analysis import reliability as rl

R = 25
IDS = list(range(R))
PAIRS = rl.pair_index(IDS)
ESTIMANDS = (
    "da_latent",
    "da_proxyclean",
    "reorder",
    "target_noise_err",
    "proxy_noise_err",
    "ceiling",
)


def _simulate(rng, rho=0.8, sd_between=1.0, sd_noise=1.0):
    """Well-specified truth: mu ~ N(0, Sigma), seeds = mu + N(0, sd_noise^2)."""
    cov = sd_between**2 * np.array([[1.0, rho], [rho, 1.0]])
    mu = rng.multivariate_normal([0.0, 0.0], cov, size=R)
    p = mu[:, :1] + rng.normal(0.0, sd_noise, (R, 3))
    t = mu[:, 1:] + rng.normal(0.0, sd_noise, (R, 3))
    return mu, p, t


def _realized(mu, p, t):
    """Realized (true-mu) values of the estimands, plus the naive plug-in reorder."""
    sp, st = rl.pair_signs(p.mean(1), PAIRS), rl.pair_signs(t.mean(1), PAIRS)
    mp, mt = rl.pair_signs(mu[:, 0], PAIRS), rl.pair_signs(mu[:, 1], PAIRS)
    return {
        "da_latent": rl.sign_agreement(sp, mt),
        "reorder": 1.0 - rl.sign_agreement(mp, mt),
        "target_noise_err": 1.0 - rl.sign_agreement(st, mt),
        "naive_reorder": 1.0 - rl.sign_agreement(sp, st),
    }


def test_noise_free_recovery():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=R)
    t = mu[:, None] + 1e-6 * rng.normal(size=(R, 3))
    p = mu[:, None] + 1e-6 * rng.normal(size=(R, 3))  # true mu_P == mu_T
    out = dc.fit_decomposition(p, t, IDS, n_draws=500, seed=0)
    assert out["reorder"] == pytest.approx(0.0, abs=1e-3)
    assert out["target_noise_err"] == pytest.approx(0.0, abs=1e-3)
    assert out["proxy_noise_err"] == pytest.approx(0.0, abs=1e-3)
    assert out["da_obs"] == pytest.approx(1.0)
    assert out["da_latent"] == pytest.approx(1.0, abs=1e-3)
    assert out["ceiling"] == pytest.approx(1.0, abs=1e-3)


def test_shares_sum_to_one_in_both_orders():
    _, p, t = _simulate(np.random.default_rng(1))
    out = dc.fit_decomposition(p, t, IDS, n_draws=1000, seed=3)
    assert out["e_obs"] > 0
    for key in ("shares", "shares_reverse"):
        assert sum(out[key].values()) == pytest.approx(1.0, abs=1e-9)
    assert out["shares"]["reorder"] == pytest.approx(out["reorder"] / out["e_obs"])
    assert out["shares_reverse"]["reorder"] == out["shares"]["reorder"]
    for name in ESTIMANDS:
        assert out["draws"][name].shape == (1000,)
        assert out[name] == pytest.approx(out["draws"][name].mean())


def test_well_specified_calibration():
    """EB estimates are ~unbiased for the realized truth; the naive plug-in is not.

    Tolerance per estimand: 3 * SE of the mean error over the datasets (SE computed here
    from the per-dataset errors, so it already contains Monte Carlo draw noise) plus a
    0.02 plug-in allowance. The allowance covers the bias that estimating m, Sigma and the
    noise variances from R = 25 recipes introduces (Sigma is clipped to PSD in ~12% of
    these datasets); it is not Bayes-calibrated noise and does not shrink with more
    datasets. Removing shrinkage (posterior mean = ybar) biases reorder by ~+0.1.
    """
    rng = np.random.default_rng(2027)
    keys = ("da_latent", "reorder", "target_noise_err")
    err = {k: [] for k in keys}
    naive_bias = []
    for d in range(200):
        mu, p, t = _simulate(rng)
        out = dc.fit_decomposition(p, t, IDS, n_draws=500, seed=d)
        truth = _realized(mu, p, t)
        for k in keys:
            err[k].append(out[k] - truth[k])
        naive_bias.append(truth["naive_reorder"] - truth["reorder"])
    for k in keys:
        e = np.asarray(err[k])
        se = e.std(ddof=1) / np.sqrt(len(e))
        assert abs(e.mean()) < 3.0 * se + 0.02, (k, e.mean(), se)
    nb = np.asarray(naive_bias)
    nb_se = nb.std(ddof=1) / np.sqrt(len(nb))
    assert nb.mean() > 0.05 and nb.mean() > 10.0 * nb_se, (nb.mean(), nb_se)


def test_psd_projection_when_noise_exceeds_spread():
    rng = np.random.default_rng(4)
    centre = 0.5 + 1e-3 * rng.normal(size=(R, 1))  # between-recipe sd ~1e-3
    offsets = np.array([-0.1, 0.0, 0.1])  # within-recipe sd 0.1
    p = centre + offsets + 1e-4 * rng.normal(size=(R, 3))
    t = centre[::-1] + offsets + 1e-4 * rng.normal(size=(R, 3))
    out = dc.fit_decomposition(p, t, IDS, n_draws=500, seed=0)
    assert out["sigma_clipped"]
    assert np.all(np.linalg.eigvalsh(out["Sigma"]) > 0)
    assert np.all(np.isfinite(out["post_cov"]))
    for name in ESTIMANDS:
        assert np.isfinite(out[name])
        assert np.all(np.isfinite(out["draws"][name]))
    for key in ("shares", "shares_reverse"):
        assert all(np.isfinite(v) for v in out[key].values())


def test_same_seed_is_deterministic():
    _, p, t = _simulate(np.random.default_rng(5))
    a = dc.fit_decomposition(p, t, IDS, n_draws=300, seed=9)
    b = dc.fit_decomposition(p, t, IDS, n_draws=300, seed=9)
    for name in ESTIMANDS:
        assert a[name] == b[name]
        np.testing.assert_array_equal(a["draws"][name], b["draws"][name])
    np.testing.assert_array_equal(
        dc.posterior_predictive_retest(p, t, IDS, n_rep=50, seed=2),
        dc.posterior_predictive_retest(p, t, IDS, n_rep=50, seed=2),
    )


def test_posterior_predictive_retest_covers_observed():
    rng = np.random.default_rng(11)
    n_sets, covered = 60, 0
    for d in range(n_sets):
        _, p, t = _simulate(rng)
        pp = dc.posterior_predictive_retest(p, t, IDS, n_rep=200, seed=d)
        assert pp.shape == (200,)
        lo, hi = np.percentile(pp, [2.5, 97.5])
        covered += lo <= rl.retest_agreement(t, IDS)["mean"] <= hi
    assert covered >= 0.9 * n_sets
