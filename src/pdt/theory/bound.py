"""The plug-in decision-error diagnostic: marginal and pairwise-difference forms.

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
from collections.abc import Callable

import numpy as np

from pdt.scaling.base import Extrapolator, Scale
from pdt.theory.identifiability import target_in_row_space

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
    if not target_in_row_space(jacobian_rows, j_target):
        raise UnidentifiedTargetError(
            f"{fitter_name}: the target scale {target_scale} is not identified by the "
            "fitting scales (its parameter Jacobian is outside their row space), so the "
            "delta-method variance is unbounded -- refusing to report the pseudo-inverse's "
            "finite (and wrong) value."
        )
    sigma_theta = sandwich_covariance(model, scales, values)
    return float(j_target @ sigma_theta @ j_target)


def known_noise_v_k(
    model: Extrapolator,
    scales: list[Scale],
    sigma2: Callable[[Scale], float],
    target_scale: Scale,
) -> float:
    """Model-based variance of `model`'s prediction at `target_scale` when the
    observation noise variances `sigma2(scale)` are KNOWN (an explicit input
    assumption), not estimated from residuals.

    The fit is an ordinary least-squares problem; to first order the
    prediction is a linear functional of the observations,
    `mu_hat(s*) - E[mu_hat(s*)] = sum_i g_i eps_i` with influence weights
    `g = pinv(J)^T J_target` (`J` = the fitting-scale Jacobians, `J_target` the
    target Jacobian, both at the fitted parameters). For independent noise with
    variance (proxy) `sigma2(s_i)`:

        v = sum_i g_i^2 sigma2(s_i)  =  J_t^T (J^T J)^+ J^T diag(sigma2) J (J^T J)^+ J_t.

    This is exact for a model linear in its parameters (`LogLinear`,
    `ConstantExtrapolator`) and a first-order (delta-method) approximation for
    the nonlinear power-law fits. It is `sandwich_covariance`'s HC0 form with
    the squared residuals replaced by the known noise variances -- the
    replacement matters: HC0 uses `r_i^2` as a one-observation estimate of
    `sigma2(s_i)`, and at a high-leverage point the fitted residual is
    almost forced to zero, so HC0 deletes exactly the uncertainty that
    dominates an extrapolated prediction (second-round review of PR #29:
    a design `N=[1, 1.00001, 2]` extrapolated to `N=2.001` certified the
    wrong arm 49% of the time against a requested 1%).

    Raises `UnsupportedEstimatorError` for the same fitters as `analytic_v_k`
    and `UnidentifiedTargetError` when the target is outside the row space of
    the fitting-scale Jacobians (the variance is then unbounded).
    """
    fitter_name = type(model).__name__
    if fitter_name in UNSUPPORTED_SANDWICH_ESTIMATORS:
        raise UnsupportedEstimatorError(
            f"{fitter_name}: the joint-least-squares influence-weight variance does not "
            "describe this estimator's fitting procedure (see UNSUPPORTED_SANDWICH_ESTIMATORS)."
        )
    j_target = np.asarray(model.jacobian(target_scale), dtype=float)
    jac = np.array([model.jacobian(s) for s in scales], dtype=float)
    if not target_in_row_space(jac, j_target):
        raise UnidentifiedTargetError(
            f"{fitter_name}: the target scale {target_scale} is not identified by the "
            "fitting scales, so the prediction variance is unbounded."
        )
    noise = np.array([sigma2(s) for s in scales], dtype=float)
    if np.any(noise <= 0):
        raise ValueError("sigma2(scale) must be positive at every fitting scale")
    # Column-equilibrated pseudo-inverse: prediction = sum_i g_i y_i with
    # g = pinv(J)^T J_target, computed on J / col_norm so parameters whose
    # Jacobian entries differ by many orders of magnitude do not degrade pinv.
    col = np.linalg.norm(jac, axis=0)
    col = np.where(col > 0.0, col, 1.0)
    g = np.linalg.pinv(jac / col).T @ (j_target / col)
    return float(np.sum(g**2 * noise))
