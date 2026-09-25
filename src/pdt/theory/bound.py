"""The plug-in decision-error diagnostic: marginal and pairwise-difference forms.

See plan/02-phase1-datadecide.md task P1-07.

**This is an empirical diagnostic, not a proven statistical bound.**
PR #17's review found a real counterexample against the original additive
form `exp(-Delta_k^2 / (2*(bias^2+v)))`: a challenger with gap=5,
bias=+10, variance=.01 has near-certain decision error (the bias alone
dwarfs and reverses the apparent gap), yet that formula evaluated to
~0.8825 -- an "upper bound" smaller than the true error probability it is
supposed to bound, which is invalid as a bound. Folding a *fixed, signed*
misspecification (bias) into a variance-like term averages it in as if it
were symmetric random noise, which is the wrong treatment: a large
systematic bias in the *wrong direction* should make error probability
approach 1 (a vacuous, uninformative term), not shrink the exponential's
denominator moderately. See docs/decisions.md and paper/sections'
theorem1_bound.tex (PR #23) for the fuller proof-level treatment this
module's `_bound_term` now matches: bias is treated as a fixed,
sign-unknown shift that first eats into the gap (`effective_gap = max(0,
|delta_k| - bias_magnitude)`), and only the *remaining* gap (if any) gets
the variance-driven exponential tail treatment. This still is not a
rigorous finite-sample bound in the nonlinear case (see PR #23's own P1
finding -- smoothness/bounded-Jacobian assumptions don't establish exact
sub-Gaussian tails for a nonlinear least-squares prediction); treat every
value from this module as a diagnostic to compare against P1-07's
Monte-Carlo empirical error estimate, not as a certified guarantee.
"""

from __future__ import annotations

import math

import numpy as np

from pdt.scaling.base import Extrapolator, Scale
from pdt.theory.identifiability import prediction_influence_weights

#: Fitters whose actual estimation procedure `sandwich_covariance` cannot
#: describe -- see `analytic_v_k`'s docstring.
UNSUPPORTED_SANDWICH_ESTIMATORS = frozenset({"ConstantExtrapolator", "TwoStepLadder"})


class UnsupportedEstimatorError(ValueError):
    """Raised by `analytic_v_k` for a fitter in `UNSUPPORTED_SANDWICH_ESTIMATORS`."""


class UnidentifiedTargetError(UnsupportedEstimatorError):
    """Raised by `analytic_v_k` when the target scale's parameter Jacobian is
    not in the row space of the fitting scales' Jacobians: some direction that
    moves the target prediction is unobserved, so its variance is infinite,
    not the finite number a pseudo-inverse would report."""


def _bound_term(delta_k: float, bias_magnitude: float, variance: float) -> float:
    """`exp(-effective_gap^2 / (2 * variance))`, where `effective_gap =
    max(0, |delta_k| - bias_magnitude)`.

    A fixed, sign-unknown bias of magnitude `bias_magnitude` is treated as
    a worst-case shift that first cancels as much of the apparent gap
    `delta_k` as it can (down to 0, never flipping the sign of the
    *remaining* gap the other way -- a bias can erase your margin, and at
    that point the term is vacuous, but treating it as capable of
    "erasing more than the margin and thereby making the term small again"
    would reintroduce the original bug in the opposite direction). Only
    the surviving `effective_gap`, if any, gets the variance-driven
    exponential-tail treatment. At `bias_magnitude=0` this is exactly the
    original variance-only form, so a zero-bias input is unaffected by
    this correction.

    `variance <= 0` is a real edge case (a perfectly-fit, zero-noise
    recipe), not just a guard against division by zero: as `variance ->
    0+` with `effective_gap` fixed and nonzero, the true limit of the
    exponential is 0 (a nonzero surviving gap with vanishing noise means
    certain correct selection); at `effective_gap == 0` the limit is
    genuinely 1 (nothing left to resolve, no matter how little noise
    there is). Both are returned directly rather than raising, since
    `variance == 0` is a valid (if unusual) input, not a bug.
    """
    effective_gap = max(0.0, abs(delta_k) - bias_magnitude)
    if variance <= 0:
        return 1.0 if effective_gap == 0 else 0.0
    return math.exp(-(effective_gap**2) / (2 * variance))


def marginal_bound_term(delta_k: float, sigma2_extrap_k: float, v_k: float) -> float:
    """One term of the marginal-form diagnostic. `sigma2_extrap_k` is
    P1-06's `sigma2_extrap_hat` -- already a squared-*magnitude* estimate
    of the recipe's own extrapolation bias (sign unknown by construction,
    clamped at 0), so its square root is used directly as the worst-case
    bias magnitude fed to `_bound_term`.
    """
    bias_magnitude = math.sqrt(max(0.0, sigma2_extrap_k))
    return _bound_term(delta_k, bias_magnitude, v_k)


def marginal_bound(terms: list[tuple[float, float, float]]) -> float:
    """`sum over k != k* of marginal_bound_term(delta_k, sigma2_extrap_k,
    v_k)`. `terms` is `[(delta_k, sigma2_extrap_k, v_k), ...]`, one per
    non-winning recipe."""
    return sum(marginal_bound_term(d, s, v) for d, s, v in terms)


def pairwise_bound_term(delta_k: float, bias_d_k: float, v_d_k: float) -> float:
    """One term of the pairwise-difference form, using the *difference*
    statistic's own bias and variance (P1-06 step 4) rather than the
    marginal recipe's -- this is the form the plan says is "actually
    tight", since correlated errors between the two arms of a comparison
    can cancel in the difference in a way the marginal form can't see.
    `bias_d_k` is P1-06's signed `bias_hat` point estimate for `D_k`;
    `abs(bias_d_k)` is used as the worst-case bias magnitude fed to
    `_bound_term`, since a point estimate's own sign is itself uncertain
    at the scale that matters here (see module docstring).
    """
    return _bound_term(delta_k, abs(bias_d_k), v_d_k)


def pairwise_bound(terms: list[tuple[float, float, float]]) -> float:
    """`sum over k != k* of pairwise_bound_term(delta_k, bias_d_k, v_d_k)`.
    `terms` is `[(delta_k, bias_d_k, v_d_k), ...]`, one per non-winning
    recipe."""
    return sum(pairwise_bound_term(d, b, v) for d, b, v in terms)


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

    Computed as `J^+ diag(r^2) J^+^T` with `J^+ = pinv(J)` from the SVD of the
    (column-equilibrated) DESIGN -- algebraically identical to the formula above
    for a full-rank `J` (`(J^T J)^-1 J^T = J^+`), but it never forms `J^T J`.
    Second-round review of PR #17: `pinv(J^T J)` squares the condition number, so a
    full-rank design with `cond(J) ~ 2e8` had its weak identified direction
    silently discarded (variance 1.5e-4 reported, 5e11 true). A direction that is
    genuinely below the rank cutoff of `J` itself is still dropped here (it is
    unobserved); callers that need a target variance must use `analytic_v_k`, which
    fails closed when the target depends on such a direction.
    """
    predictions = np.array([model.predict(s) for s in scales])
    residuals = predictions - np.asarray(values, dtype=float)
    jacobian_rows = np.asarray([model.jacobian(s) for s in scales], dtype=float)

    col = np.linalg.norm(jacobian_rows, axis=0)
    col = np.where(col > 0.0, col, 1.0)
    pinv_eq = np.linalg.pinv(jacobian_rows / col)  # (p, n): J_eq^+
    sigma_eq = (pinv_eq * residuals**2) @ pinv_eq.T
    return sigma_eq / np.outer(col, col)


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

    Raises `UnsupportedEstimatorError` for any fitter in
    `UNSUPPORTED_SANDWICH_ESTIMATORS`, rather than silently computing a
    number whose statistical interpretation doesn't match how that fitter
    actually estimates its parameters. PR #17's review: `sandwich_covariance`
    assumes theta is the joint argmin of a single simultaneous
    sum-of-squared-residuals objective over `scales` (standard M-estimator
    theory) -- true for `PowerLawN`/`PowerLawC`/`ChinchillaND`/`LogLinear`,
    but not for `ConstantExtrapolator` (which only uses the observations at
    the *largest* scale, ignoring the rest -- calling `sandwich_covariance`
    with the full `scales` list would wrongly charge it "residuals" at
    scales it never fit to) or `TwoStepLadder` (fit in two *separate*
    sequential stages with different objectives, not one joint
    optimization -- the single shared jacobian/residual structure here
    doesn't represent that two-stage procedure at all). Reporting a
    fabricated-looking analytic `v_k` for either would let it be silently
    "cross-checked" against P1-06's bootstrap estimate as if they measured
    the same thing.

    Also raises `UnidentifiedTargetError` (an `UnsupportedEstimatorError`)
    when the target Jacobian is not in the row space of the fitting scales'
    Jacobians (`pdt.theory.identifiability`): `sandwich_covariance` uses a
    pseudo-inverse, which treats an unobserved direction as *zero* variance,
    so a `LogLinear` fit observed at a single N (varying only D) would report
    a small finite `v_k` for any target N -- the opposite of the truth, which
    is unbounded. Second-round review of PR #17.
    """
    fitter_name = type(model).__name__
    if fitter_name in UNSUPPORTED_SANDWICH_ESTIMATORS:
        raise UnsupportedEstimatorError(
            f"{fitter_name}: the joint-least-squares sandwich covariance does not "
            "describe this estimator's actual fitting procedure (see "
            "UNSUPPORTED_SANDWICH_ESTIMATORS docstring) -- refusing to report a "
            "fabricated analytic v_k rather than silently mismeasuring it."
        )
    j_target = np.asarray(model.jacobian(target_scale), dtype=float)
    jacobian_rows = np.array([model.jacobian(s) for s in scales], dtype=float)
    weights = prediction_influence_weights(jacobian_rows, j_target)
    if weights is None:
        raise UnidentifiedTargetError(
            f"{fitter_name}: the target scale {target_scale} is not identified by the "
            "fitting scales (its parameter Jacobian is outside their row space, or depends "
            "on a direction below the design's numerical rank cutoff), so the delta-method "
            "variance is unbounded -- refusing to report a finite (and wrong) value."
        )
    residuals = np.array([model.predict(s) for s in scales]) - np.asarray(values, dtype=float)
    # v = j^T Sigma_theta j with Sigma_theta = J^+ diag(r^2) J^+^T, i.e. sum_i g_i^2 r_i^2.
    return float(np.sum(weights**2 * residuals**2))
