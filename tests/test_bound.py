"""Tests for pdt.theory.bound -- see plan/02-phase1-datadecide.md P1-07."""

from __future__ import annotations

import numpy as np
import pytest

from pdt.scaling import fitters
from pdt.scaling.base import Extrapolator, Scale
from pdt.theory import bound

_RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# _bound_term / marginal_bound_term / pairwise_bound_term
# ---------------------------------------------------------------------------


def test_bound_term_matches_hand_computation():
    result = bound._bound_term(delta_k=0.1, total_variance=0.02)
    import math

    expected = math.exp(-(0.1**2) / (2 * 0.02))
    assert result == pytest.approx(expected)


def test_bound_term_zero_variance_nonzero_gap_is_zero():
    assert bound._bound_term(delta_k=0.1, total_variance=0.0) == 0.0


def test_bound_term_zero_variance_zero_gap_is_one():
    assert bound._bound_term(delta_k=0.0, total_variance=0.0) == 1.0


def test_bound_term_negative_variance_treated_as_degenerate():
    # Should never happen in real use, but must not crash (sqrt/div of a
    # negative number) if it ever does.
    assert bound._bound_term(delta_k=0.1, total_variance=-1.0) == 0.0


def test_bound_term_larger_gap_gives_smaller_term():
    small_gap = bound._bound_term(0.05, 0.01)
    large_gap = bound._bound_term(0.5, 0.01)
    assert large_gap < small_gap


def test_marginal_bound_term_sums_extrap_and_v():
    # sigma2_extrap=0.01, v=0.01 -> total_variance=0.02, same as the
    # hand-computed case above.
    result = bound.marginal_bound_term(delta_k=0.1, sigma2_extrap_k=0.01, v_k=0.01)
    assert result == pytest.approx(bound._bound_term(0.1, 0.02))


def test_pairwise_bound_term_sums_bias_squared_and_v():
    result = bound.pairwise_bound_term(delta_k=0.1, bias_d_k=0.1, v_d_k=0.01)
    assert result == pytest.approx(bound._bound_term(0.1, 0.1**2 + 0.01))


def test_marginal_bound_sums_all_terms():
    terms = [(0.1, 0.01, 0.01), (0.2, 0.02, 0.01)]
    expected = sum(bound.marginal_bound_term(*t) for t in terms)
    assert bound.marginal_bound(terms) == pytest.approx(expected)


def test_pairwise_bound_sums_all_terms():
    terms = [(0.1, 0.05, 0.01), (0.2, 0.1, 0.02)]
    expected = sum(bound.pairwise_bound_term(*t) for t in terms)
    assert bound.pairwise_bound(terms) == pytest.approx(expected)


def test_marginal_bound_empty_is_zero():
    assert bound.marginal_bound([]) == 0.0


# ---------------------------------------------------------------------------
# _worst_case_bound_term / worst_case_marginal_bound_term / _bound (P2-02
# correction -- see docs/decisions.md, 2026-09-10, and
# paper/sections/theorem1_bound.tex)
# ---------------------------------------------------------------------------


def test_worst_case_bound_term_matches_hand_computation():
    import math

    result = bound._worst_case_bound_term(delta_k=0.5, bias_budget=0.2, total_variance=0.01)
    expected = math.exp(-((0.5 - 0.2) ** 2) / (2 * 0.01))
    assert result == pytest.approx(expected)


def test_worst_case_bound_term_bias_at_least_gap_is_vacuous():
    # sqrt(sigma2_extrap) >= Delta_k: an adversary could plausibly wipe out
    # the whole true gap, so no non-trivial guarantee is possible -- the
    # bound must honestly say 1.0 (vacuous), not something misleadingly
    # small.
    assert bound._worst_case_bound_term(delta_k=0.5, bias_budget=0.5, total_variance=0.01) == 1.0
    assert bound._worst_case_bound_term(delta_k=0.5, bias_budget=0.9, total_variance=0.01) == 1.0


def test_worst_case_bound_term_zero_variance_zero_gap_is_one():
    assert bound._worst_case_bound_term(delta_k=0.1, bias_budget=0.1, total_variance=0.0) == 1.0


def test_worst_case_bound_term_zero_variance_positive_gap_is_zero():
    assert bound._worst_case_bound_term(delta_k=0.5, bias_budget=0.1, total_variance=0.0) == 0.0


def test_worst_case_marginal_bound_term_includes_both_arms():
    # bias_budget = sqrt(sigma2_extrap_k) + sqrt(sigma2_extrap_kstar);
    # total_variance = v_k + v_kstar -- both arms' own quantities, not
    # just k's.
    result = bound.worst_case_marginal_bound_term(
        delta_k=0.5, sigma2_extrap_k=0.04, v_k=0.01, sigma2_extrap_kstar=0.01, v_kstar=0.02
    )
    expected = bound._worst_case_bound_term(delta_k=0.5, bias_budget=0.2 + 0.1, total_variance=0.03)
    assert result == pytest.approx(expected)


def test_worst_case_marginal_bound_sums_all_terms():
    terms = [(0.5, 0.04, 0.01), (0.3, 0.01, 0.02)]
    expected = sum(bound.worst_case_marginal_bound_term(*t, 0.02, 0.01) for t in terms)
    assert bound.worst_case_marginal_bound(terms, 0.02, 0.01) == pytest.approx(expected)


def test_worst_case_marginal_bound_empty_is_zero():
    assert bound.worst_case_marginal_bound([], sigma2_extrap_kstar=0.01, v_kstar=0.01) == 0.0


def test_worst_case_form_corrects_a_real_violation_of_the_additive_form():
    # The concrete adversarial instance that first exposed the bug: arm k
    # has a large, effectively-fixed bias (sqrt(sigma2_extrap_k)=10) and
    # tiny variance (0.01); k* is bias-free and noise-free. The true
    # error probability is ~1 (the fixed bias alone dwarfs the gap of 5).
    # marginal_bound_term (additive, current plug-in form) claims ~0.88 --
    # a real violation, not a rounding error. worst_case_marginal_bound_term
    # (gap-reduction form) correctly reports 1.0 (vacuous / no guarantee
    # possible), which is honest given the bias alone can flip the gap.
    delta_k = 5.0
    sigma2_extrap_k = 10.0**2
    v_k = 0.01

    additive_claim = bound.marginal_bound_term(delta_k, sigma2_extrap_k, v_k)
    assert additive_claim == pytest.approx(0.8825079327617985, rel=1e-9)
    # (kept as a fixed reference value, not re-derived, precisely so a
    # future change to marginal_bound_term's formula is caught here.)

    corrected = bound.worst_case_marginal_bound_term(
        delta_k, sigma2_extrap_k, v_k, sigma2_extrap_kstar=0.0, v_kstar=0.0
    )
    assert corrected == 1.0


# ---------------------------------------------------------------------------
# sandwich_covariance() / analytic_v_k()
# ---------------------------------------------------------------------------


class _Linear(Extrapolator):
    """y = theta[0] + theta[1] * n -- a minimal hand-built Extrapolator so
    the sandwich covariance can be checked against the textbook closed-form
    OLS/White covariance for simple linear regression, independent of any
    real fitter's own optimizer."""

    n_params = 2

    def _fit_theta(self, scales, values, weights):
        ns = np.array([s.n for s in scales])
        ys = np.array(values)
        design = np.column_stack([np.ones_like(ns), ns])
        theta, *_ = np.linalg.lstsq(design, ys, rcond=None)
        return theta, {}

    def _predict_from_theta(self, theta, scale):
        return theta[0] + theta[1] * scale.n


def test_sandwich_covariance_matches_textbook_white_estimator_for_linear_regression():
    ns = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    true_a, true_b = 1.0, 2.0
    ys = true_a + true_b * ns + np.array([0.1, -0.2, 0.05, -0.1, 0.15, -0.05])
    scales = [Scale(n=n, d=1.0) for n in ns]

    model = _Linear().fit(scales, list(ys))

    sigma_theta = bound.sandwich_covariance(model, scales, list(ys))

    # Hand-computed White (HC0) sandwich covariance for OLS: (X'X)^-1
    # X' diag(r^2) X (X'X)^-1, using the exact design matrix.
    design = np.column_stack([np.ones_like(ns), ns])
    theta_hat, *_ = np.linalg.lstsq(design, ys, rcond=None)
    residuals = design @ theta_hat - ys
    xtx_inv = np.linalg.pinv(design.T @ design)
    meat = design.T @ np.diag(residuals**2) @ design
    expected = xtx_inv @ meat @ xtx_inv

    assert sigma_theta == pytest.approx(expected, abs=1e-8)


def test_sandwich_covariance_is_symmetric():
    ns = np.geomspace(1e6, 1e9, 8)
    ys = 0.5 + 0.05 * np.log(ns) + _RNG.normal(0, 0.005, size=len(ns))
    scales = [Scale(n=n, d=20 * n) for n in ns]

    model = fitters.LogLinear(rng=_RNG).fit(scales, list(ys))
    sigma_theta = bound.sandwich_covariance(model, scales, list(ys))

    assert sigma_theta == pytest.approx(sigma_theta.T, abs=1e-10)


def test_sandwich_covariance_zero_residuals_gives_zero_covariance():
    # A perfect fit (residuals exactly zero) should give a zero sandwich
    # covariance -- no uncertainty left if the fit is exact everywhere.
    ns = np.array([1.0, 2.0, 3.0, 4.0])
    true_a, true_b = 1.0, 2.0
    ys = true_a + true_b * ns
    scales = [Scale(n=n, d=1.0) for n in ns]

    model = _Linear().fit(scales, list(ys))
    sigma_theta = bound.sandwich_covariance(model, scales, list(ys))

    assert sigma_theta == pytest.approx(np.zeros((2, 2)), abs=1e-8)


def test_analytic_v_k_is_nonnegative():
    ns = np.geomspace(1e6, 1e9, 8)
    ys = 0.6 - 3.0 * ns ** (-0.3) + _RNG.normal(0, 0.005, size=len(ns))
    scales = [Scale(n=n, d=20 * n) for n in ns]

    model = fitters.PowerLawN(rng=_RNG).fit(scales, list(ys))
    v_k = bound.analytic_v_k(model, scales, list(ys), Scale(n=1e9, d=2e10))

    assert v_k >= 0.0


def test_analytic_v_k_grows_with_extrapolation_distance_for_log_linear():
    # LogLinear's own jacobian entry d(prediction)/d(b) = log(N) grows
    # without bound as N increases, so its delta-method variance should
    # too -- checked directly with LogLinear specifically (not, say,
    # PowerLawN: that model's prediction *saturates* to a constant ceiling
    # as N -> infinity, so d(prediction)/d(theta) -> a fixed bounded
    # vector and v_k saturates rather than growing further -- a real,
    # separately interesting property, but the wrong model to test
    # "variance grows with extrapolation distance" against).
    ns = np.geomspace(1e6, 1e8, 8)
    ys = 0.1 + 0.05 * np.log(ns) + _RNG.normal(0, 0.005, size=len(ns))
    scales = [Scale(n=n, d=20 * n) for n in ns]

    model = fitters.LogLinear(rng=_RNG).fit(scales, list(ys))
    v_near = bound.analytic_v_k(model, scales, list(ys), Scale(n=2e8, d=4e9))
    v_far = bound.analytic_v_k(model, scales, list(ys), Scale(n=1e11, d=2e12))

    assert v_far > v_near


def test_analytic_v_k_saturates_for_power_law_n_far_extrapolation():
    # The mirror-image property to the LogLinear test above: PowerLawN's
    # prediction converges to a constant ceiling E as N -> infinity, so
    # its jacobian converges to a fixed vector and v_k should *stop
    # growing* (saturate), not diverge, once N is far enough past the
    # fitted range that the power-law term has effectively decayed away.
    ns = np.geomspace(1e6, 1e8, 8)
    ys = 0.6 - 3.0 * ns ** (-0.3) + _RNG.normal(0, 0.005, size=len(ns))
    scales = [Scale(n=n, d=20 * n) for n in ns]

    model = fitters.PowerLawN(rng=_RNG).fit(scales, list(ys))
    v_far = bound.analytic_v_k(model, scales, list(ys), Scale(n=1e11, d=2e12))
    v_farther = bound.analytic_v_k(model, scales, list(ys), Scale(n=1e14, d=2e15))

    assert v_farther == pytest.approx(v_far, rel=1e-6)
