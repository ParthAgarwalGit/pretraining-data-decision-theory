"""Model-based decomposition of small-scale decision error (Phase 7, N7-03).

Inputs are the same ``(R, 3)`` proxy and target seed matrices ``P`` and ``T`` and the
recipe-id vector used by :mod:`pdt.analysis.reliability`. A decision is the sign of the gap
between two recipes' means over all unordered pairs with different ids; every agreement
below excludes pairs tied on either side and is NaN over zero scorable pairs.

Model (Gaussian empirical Bayes, plug-in). Each recipe ``r`` has a latent pair of true
means ``mu_r = (mu_P, mu_T)``; the observed 3-seed means are
``ybar_r = mu_r + e_r`` with ``e_r ~ N(0, N)``, ``N = diag(s2_P / 3, s2_T / 3)``, where
``s2_*`` is the pooled within-recipe seed variance. Across recipes,
``mu_r ~ N(m, Sigma)`` with ``m`` the grand mean of ``ybar`` and
``Sigma = S - N`` (``S`` the ddof=1 covariance of ``ybar``) projected to PSD. The
posterior of each ``mu_r`` is Gaussian with mean ``m + K (ybar_r - m)``,
``K = Sigma (Sigma + N)^-1``, and covariance ``Sigma - K Sigma`` (shared by all
recipes). Plug-in: uncertainty in ``m``, ``Sigma``, ``s2_*`` is ignored.

Estimands (posterior means over Monte Carlo draws of ``mu``):

- ``da_obs``: agreement of sign(ybar_P) with sign(ybar_T) -- observed decision accuracy.
- ``da_latent``: agreement of sign(ybar_P) with sign(mu_T) -- accuracy against the true
  target ranking.
- ``da_proxyclean``: agreement of sign(mu_P) with sign(ybar_T).
- ``reorder``: fraction of pairs with sign(mu_P) != sign(mu_T) -- genuine reordering
  between scales.
- ``target_noise_err``: fraction with sign(ybar_T) != sign(mu_T).
- ``proxy_noise_err``: fraction with sign(ybar_P) != sign(mu_P).
- ``ceiling = 1 - target_noise_err``: the best agreement any proxy can be expected to show
  against a 3-seed target.

The decomposition of ``E_obs = 1 - da_obs`` into target noise, proxy noise and reordering
is sequential and therefore ORDER-DEPENDENT: ``shares`` removes target noise first,
``shares_reverse`` removes proxy noise first. Within each order the three shares sum to 1
by construction; an individual share can be negative or exceed 1 (e.g. when removing
target noise lowers agreement), and they are NaN when ``E_obs == 0``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pdt.analysis.reliability import (
    _check,
    pair_index,
    pair_signs,
    pooled_sd,
    retest_agreement,
    sign_agreement,
)


def _agreement_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row sign agreement over the last axis, ties on either side excluded (NaN if none)."""
    a, b = np.broadcast_arrays(a, b)
    ok = (a != 0) & (b != 0)
    n = ok.sum(-1)
    hit = ((a == b) & ok).sum(-1)
    return np.where(n > 0, hit / np.maximum(n, 1), np.nan)


def _posterior(P: np.ndarray, T: np.ndarray) -> dict:
    """Plug-in Gaussian EB posterior of the latent (mu_P, mu_T) recipe means."""
    s2_p, s2_t = pooled_sd(P) ** 2, pooled_sd(T) ** 2
    noise = np.diag([s2_p / 3.0, s2_t / 3.0])
    ybar = np.column_stack([P.mean(1), T.mean(1)])
    m = ybar.mean(0)
    s = np.cov(ybar.T, ddof=1)
    eps = 1e-12 * float(np.trace(s))
    raw, vecs = np.linalg.eigh(s - noise)
    sigma = (vecs * np.maximum(raw, eps)) @ vecs.T
    sigma = (sigma + sigma.T) / 2.0
    gain = np.linalg.solve(sigma + noise, sigma).T  # Sigma (Sigma + N)^-1; both symmetric
    post_mean = m + (ybar - m) @ gain.T
    post_cov = sigma - gain @ sigma
    post_cov = (post_cov + post_cov.T) / 2.0
    return {
        "s2_P": float(s2_p),
        "s2_T": float(s2_t),
        "ybar": ybar,
        "Sigma": sigma,
        "sigma_eigvals_raw": raw,
        "sigma_clipped": bool(np.any(raw < eps)),
        "post_mean": post_mean,
        "post_cov": post_cov,
        "jitter": eps,
    }


def _draw_mu(post: dict, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """``(n_draws, R, 2)`` posterior draws of ``mu`` (Cholesky of ``post_cov + jitter I``)."""
    chol = np.linalg.cholesky(post["post_cov"] + post["jitter"] * np.eye(2))
    z = rng.standard_normal((n_draws, *post["post_mean"].shape))
    return post["post_mean"] + z @ chol.T


def _shares(e_obs: float, err_after_first: float, reorder: float) -> tuple[float, float, float]:
    """(first-removed, second-removed, reorder) shares of ``e_obs``; NaN if ``e_obs == 0``."""
    if not e_obs > 0:
        return float("nan"), float("nan"), float("nan")
    return (
        (e_obs - err_after_first) / e_obs,
        (err_after_first - reorder) / e_obs,
        reorder / e_obs,
    )


def fit_decomposition(
    P: np.ndarray, T: np.ndarray, ids: Sequence, n_draws: int = 2000, seed: int = 0
) -> dict:
    """Decompose proxy-vs-target decision error into target noise, proxy noise, reordering.

    Estimands are the posterior means (over ``n_draws`` draws of ``mu``, RNG
    ``default_rng(seed)``) of the per-draw pair fractions defined in the module docstring;
    ``draws`` holds the per-draw arrays (length ``n_draws``) for posterior intervals.
    ``da_obs`` is deterministic. ``shares`` (target noise removed first) is
    ``target_noise = (E_obs - (1 - da_latent)) / E_obs``,
    ``proxy_noise = ((1 - da_latent) - reorder) / E_obs``, ``reorder = reorder / E_obs``;
    ``shares_reverse`` (proxy noise removed first) uses ``da_proxyclean`` in place of
    ``da_latent``. Also returns the plug-in ``s2_P``, ``s2_T``, ``Sigma`` (after PSD
    projection), its raw eigenvalues, ``sigma_clipped`` and the shared ``post_cov``.
    """
    p, t = _check(P, ids), _check(T, ids)
    pairs = pair_index(ids)
    i, j = pairs
    post = _posterior(p, t)
    mu = _draw_mu(post, n_draws, np.random.default_rng(seed))
    ybar = post["ybar"]

    sp_obs, st_obs = pair_signs(ybar[:, 0], pairs), pair_signs(ybar[:, 1], pairs)
    sp_mu = np.sign(mu[:, i, 0] - mu[:, j, 0])
    st_mu = np.sign(mu[:, i, 1] - mu[:, j, 1])

    draws = {
        "da_latent": _agreement_rows(sp_obs, st_mu),
        "da_proxyclean": _agreement_rows(sp_mu, st_obs),
        "reorder": 1.0 - _agreement_rows(sp_mu, st_mu),
        "target_noise_err": 1.0 - _agreement_rows(st_obs, st_mu),
        "proxy_noise_err": 1.0 - _agreement_rows(sp_obs, sp_mu),
    }
    draws["ceiling"] = 1.0 - draws["target_noise_err"]
    est = {k: float(np.mean(v)) for k, v in draws.items()}

    da_obs = sign_agreement(sp_obs, st_obs)
    e_obs = 1.0 - da_obs
    tn, pn, ro = _shares(e_obs, 1.0 - est["da_latent"], est["reorder"])
    pn_r, tn_r, ro_r = _shares(e_obs, 1.0 - est["da_proxyclean"], est["reorder"])
    return {
        "da_obs": da_obs,
        **est,
        "e_obs": e_obs,
        "shares": {"target_noise": tn, "proxy_noise": pn, "reorder": ro},
        "shares_reverse": {"proxy_noise": pn_r, "target_noise": tn_r, "reorder": ro_r},
        "draws": draws,
        "n_pairs": len(i),
        "s2_P": post["s2_P"],
        "s2_T": post["s2_T"],
        "Sigma": post["Sigma"],
        "sigma_eigvals_raw": post["sigma_eigvals_raw"],
        "sigma_clipped": post["sigma_clipped"],
        "post_mean": post["post_mean"],
        "post_cov": post["post_cov"],
    }


def posterior_predictive_retest(
    P: np.ndarray, T: np.ndarray, ids: Sequence, n_rep: int = 1000, seed: int = 0
) -> np.ndarray:
    """Posterior predictive distribution of the target test-retest agreement.

    Estimand: for each of ``n_rep`` replicates, draw ``mu_T`` from the plug-in posterior,
    simulate a fresh ``(R, 3)`` target ``mu_T[:, None] + N(0, s2_T)`` per seed, and return
    ``reliability.retest_agreement(...)["mean"]`` on it. Compare with the observed
    ``retest_agreement(T, ids)["mean"]`` as a posterior predictive check.
    """
    p, t = _check(P, ids), _check(T, ids)
    post = _posterior(p, t)
    rng = np.random.default_rng(seed)
    mu_t = _draw_mu(post, n_rep, rng)[:, :, 1]
    sims = mu_t[:, :, None] + rng.normal(0.0, np.sqrt(post["s2_T"]), size=(*mu_t.shape, 3))
    return np.array([retest_agreement(sim, ids)["mean"] for sim in sims])
