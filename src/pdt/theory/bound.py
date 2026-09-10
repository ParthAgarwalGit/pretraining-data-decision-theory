"""The plug-in decision-error bound: marginal and pairwise-difference forms.

See plan/02-phase1-datadecide.md task P1-07. Both forms share the same
per-arm structure, `exp(-Delta_k^2 / (2 * total_variance))`, differing
only in which variance quantity is plugged in -- `_bound_term` implements
that shared structure once; `marginal_bound_term`/`pairwise_bound_term`
just name which variance goes where the plan asks.

**These two forms are empirical/plug-in diagnostics, not a proven
worst-case selection-error bound.** P1-07 found them never violated on
396 real DataDecide cells; P2-02's numerical certificate
(`tests/theory/test_theorem1.py`) found that this specific additive
"bias-squared plus variance in the denominator" structure *can* be
violated -- badly, not marginally -- once gaps get small enough relative
to bias for the distinction to matter, which real DataDecide data never
exercised (every real cell's bound was vacuous, `>= 1`, so a false-but-
vacuous bound is indistinguishable from a true one there). See
`docs/decisions.md`, 2026-09-10. `worst_case_marginal_bound_term` /
`worst_case_marginal_bound` below implement the corrected, numerically
verified form (bias enters as a gap reduction, not an added variance,
and the *winning* arm's own bias/variance is included) --
`paper/sections/theorem1_bound.tex` states and proves that version.
Phase 3's algorithm should build its own delta-correctness guarantee on
the worst-case functions, not the plug-in ones above; the plug-in ones
stay as-is here because P1-07/P1-08's already-reported numbers are
correct descriptions of what *this specific formula* produces on real
data, which remains a legitimate (if not fully rigorously justified)
empirical fact worth keeping.
"""

from __future__ import annotations

import math

import numpy as np

from pdt.scaling.base import Extrapolator, Scale


def _bound_term(delta_k: float, total_variance: float) -> float:
    """`exp(-Delta_k^2 / (2 * total_variance))`.

    `total_variance <= 0` is a real edge case (a perfectly-fit,
    zero-noise recipe), not just a guard against division by zero: as
    `total_variance -> 0+` with `delta_k` fixed and nonzero, the true
    limit of the exponential is 0 (a nonzero gap with vanishing noise
    means certain correct selection); at `delta_k == 0` the limit is
    genuinely 1 (an exact tie with zero noise is maximally ambiguous, not
    resolvable no matter how little noise there is). Both are returned
    directly rather than raising, since `total_variance == 0` is a valid
    (if unusual) input, not a bug.
    """
    if total_variance <= 0:
        return 1.0 if delta_k == 0 else 0.0
    return math.exp(-(delta_k**2) / (2 * total_variance))


def marginal_bound_term(delta_k: float, sigma2_extrap_k: float, v_k: float) -> float:
    """One term of the marginal-form bound: `exp(-Delta_k^2 / (2 *
    (sigma2_extrap_k + v_k)))` -- the form stated in the source document.
    """
    return _bound_term(delta_k, sigma2_extrap_k + v_k)


def marginal_bound(terms: list[tuple[float, float, float]]) -> float:
    """`sum over k != k* of marginal_bound_term(delta_k, sigma2_extrap_k,
    v_k)`. `terms` is `[(delta_k, sigma2_extrap_k, v_k), ...]`, one per
    non-winning recipe."""
    return sum(marginal_bound_term(d, s, v) for d, s, v in terms)


def pairwise_bound_term(delta_k: float, bias_d_k: float, v_d_k: float) -> float:
    """One term of the pairwise-difference form: `exp(-Delta_k^2 / (2 *
    (bias(D_k)^2 + v(D_k))))`, using the *difference* statistic's own
    bias and variance (P1-06 step 4) rather than the marginal recipe's --
    this is the form the plan says is "actually tight", since correlated
    errors between the two arms of a comparison can cancel in the
    difference in a way the marginal form can't see.
    """
    return _bound_term(delta_k, bias_d_k**2 + v_d_k)


def pairwise_bound(terms: list[tuple[float, float, float]]) -> float:
    """`sum over k != k* of pairwise_bound_term(delta_k, bias_d_k, v_d_k)`.
    `terms` is `[(delta_k, bias_d_k, v_d_k), ...]`, one per non-winning
    recipe."""
    return sum(pairwise_bound_term(d, b, v) for d, b, v in terms)


def _worst_case_bound_term(delta_k: float, bias_budget: float, total_variance: float) -> float:
    """`exp(-max(0, delta_k - bias_budget)^2 / (2 * total_variance))`.

    The mathematically correct worst-case Chernoff bound for a *fixed*
    (not zero-mean-random) bias of unknown sign, bounded in magnitude by
    `bias_budget`: the adversary picks the sign that shrinks the
    effective gap as much as possible, so the bias subtracts from the gap
    rather than adding to the variance. See
    `paper/sections/theorem1_bound.tex` Theorem 1 for the full derivation
    and `docs/decisions.md` (2026-09-10) for how the additive form above
    was found not to be a valid bound in general.
    """
    gap_adjusted = max(0.0, delta_k - bias_budget)
    if total_variance <= 0:
        return 1.0 if gap_adjusted == 0 else 0.0
    return math.exp(-(gap_adjusted**2) / (2 * total_variance))


def worst_case_marginal_bound_term(
    delta_k: float,
    sigma2_extrap_k: float,
    v_k: float,
    sigma2_extrap_kstar: float,
    v_kstar: float,
) -> float:
    """Corrected marginal-form bound term (Theorem 1): unlike
    `marginal_bound_term`, this includes the *winning* arm k*'s own bias
    and variance (the comparison `mu_hat_k(s*) >= mu_hat_{k*}(s*)`
    depends on both arms' noise, not just arm k's), and treats each
    arm's bias as an unknown-sign quantity bounded in magnitude by
    `sqrt(sigma2_extrap)`, combined via `_worst_case_bound_term` rather
    than added into the variance.
    """
    bias_budget = math.sqrt(max(sigma2_extrap_k, 0.0)) + math.sqrt(max(sigma2_extrap_kstar, 0.0))
    return _worst_case_bound_term(delta_k, bias_budget, v_k + v_kstar)


def worst_case_marginal_bound(
    terms: list[tuple[float, float, float]], sigma2_extrap_kstar: float, v_kstar: float
) -> float:
    """`sum over k != k* of worst_case_marginal_bound_term(...)`. `terms`
    is `[(delta_k, sigma2_extrap_k, v_k), ...]`, one per non-winning
    recipe; k*'s own `(sigma2_extrap, v)` is passed once since there is
    only one k* shared across every term in the union bound.
    """
    return sum(
        worst_case_marginal_bound_term(d, s, v, sigma2_extrap_kstar, v_kstar) for d, s, v in terms
    )


def sandwich_covariance(
    model: Extrapolator, scales: list[Scale], values: list[float]
) -> np.ndarray:
    """Eicker-Huber-White sandwich covariance of the fitted parameter
    vector theta, from `model`'s own residuals at its fitting data:

        Sigma_theta = (J^T J)^-1 (J^T diag(r^2) J) (J^T J)^-1

    where `J` is the n_scales x n_params jacobian of predictions with
    respect to theta (one row per scale, built by calling `model`'s own
    `jacobian(scale)` at each fitting scale -- valid because
    d(residual)/d(theta) = d(prediction)/d(theta), the observed value not
    depending on theta) and `r` are the residuals `predict(scale) -
    observed`. Heteroscedasticity-consistent (HC0): no assumption that
    residual variance is constant across scales, matching what "sandwich"
    means as opposed to the simpler `sigma^2 (J^T J)^-1` OLS covariance.

    Uses the Moore-Penrose pseudo-inverse rather than a direct inverse,
    so a near-singular `J^T J` (an under-identified or nearly-degenerate
    fit) degrades gracefully instead of raising.
    """
    predictions = np.array([model.predict(s) for s in scales])
    residuals = predictions - np.asarray(values, dtype=float)
    jacobian_rows = np.array([model.jacobian(s) for s in scales])

    jtj = jacobian_rows.T @ jacobian_rows
    jtj_inv = np.linalg.pinv(jtj)
    meat = jacobian_rows.T @ np.diag(residuals**2) @ jacobian_rows
    return jtj_inv @ meat @ jtj_inv


def analytic_v_k(
    model: Extrapolator, scales: list[Scale], values: list[float], target_scale: Scale
) -> float:
    """Delta-method estimation variance of `model`'s prediction at
    `target_scale`: `v_k = J_target^T Sigma_theta J_target`, where
    `J_target = model.jacobian(target_scale)` and `Sigma_theta` is
    `sandwich_covariance(model, scales, values)`. This is the *analytic*
    counterpart to P1-06's bootstrap `v_hat_k` -- the two should agree if
    the delta-method machinery this project's theory relies on is valid
    for these fits; P1-07's job is to check that, not assume it.
    """
    sigma_theta = sandwich_covariance(model, scales, values)
    j_target = model.jacobian(target_scale)
    return float(j_target @ sigma_theta @ j_target)
