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
    result = bound._bound_term(delta_k=0.1, bias_magnitude=0.0, variance=0.02)
    import math

    expected = math.exp(-(0.1**2) / (2 * 0.02))
    assert result == pytest.approx(expected)


def test_bound_term_zero_variance_nonzero_gap_is_zero():
    assert bound._bound_term(delta_k=0.1, bias_magnitude=0.0, variance=0.0) == 0.0


def test_bound_term_zero_variance_zero_gap_is_one():
    assert bound._bound_term(delta_k=0.0, bias_magnitude=0.0, variance=0.0) == 1.0


def test_bound_term_negative_variance_treated_as_degenerate():
    # Should never happen in real use, but must not crash (sqrt/div of a
    # negative number) if it ever does.
    assert bound._bound_term(delta_k=0.1, bias_magnitude=0.0, variance=-1.0) == 0.0


def test_bound_term_larger_gap_gives_smaller_term():
    small_gap = bound._bound_term(0.05, 0.0, 0.01)
    large_gap = bound._bound_term(0.5, 0.0, 0.01)
    assert large_gap < small_gap


def test_bound_term_bias_eats_into_the_gap_before_the_exponential():
    # bias_magnitude=0.03 should behave exactly like reducing delta_k by
    # 0.03 first, then applying the zero-bias formula.
    with_bias = bound._bound_term(delta_k=0.1, bias_magnitude=0.03, variance=0.01)
    equivalent_no_bias = bound._bound_term(delta_k=0.07, bias_magnitude=0.0, variance=0.01)
    assert with_bias == pytest.approx(equivalent_no_bias)


def test_bound_term_bias_at_least_the_gap_gives_vacuous_term():
    # Regression for PR #17's counterexample: a bias magnitude that
    # matches or exceeds the gap must floor effective_gap at 0 (term=1,
    # a vacuous/uninformative "bound"), never let the excess bias make the
    # term small again (which would silently claim a confident, near-zero
    # error probability for a case the review showed is near-certain to
    # be wrong).
    exactly_equal = bound._bound_term(delta_k=5.0, bias_magnitude=5.0, variance=0.01)
    bias_exceeds_gap = bound._bound_term(delta_k=5.0, bias_magnitude=10.0, variance=0.01)
    assert exactly_equal == pytest.approx(1.0)
    assert bias_exceeds_gap == pytest.approx(1.0)


def test_marginal_bound_term_zero_bias_matches_variance_only_formula():
    result = bound.marginal_bound_term(delta_k=0.1, sigma2_extrap_k=0.0, v_k=0.02)
    assert result == pytest.approx(bound._bound_term(0.1, 0.0, 0.02))


def test_marginal_bound_term_uses_sqrt_sigma2_extrap_as_bias_magnitude():
    result = bound.marginal_bound_term(delta_k=0.1, sigma2_extrap_k=0.01, v_k=0.02)
    assert result == pytest.approx(bound._bound_term(0.1, 0.1, 0.02))  # sqrt(0.01) = 0.1


def test_marginal_bound_term_regression_ptr17_counterexample_is_no_longer_understated():
    # PR #17's exact reproduction: gap=5, bias=+10 (sigma2_extrap_k=100),
    # variance=.01. The old additive-denominator formula gave ~0.8825, an
    # invalid "upper bound" smaller than the true near-certain error rate.
    # The corrected formula must be (near-)vacuous, i.e. very close to 1.
    result = bound.marginal_bound_term(delta_k=5.0, sigma2_extrap_k=100.0, v_k=0.01)
    assert result == pytest.approx(1.0)


def test_pairwise_bound_term_zero_bias_matches_variance_only_formula():
    result = bound.pairwise_bound_term(delta_k=0.1, bias_d_k=0.0, v_d_k=0.02)
    assert result == pytest.approx(bound._bound_term(0.1, 0.0, 0.02))


def test_pairwise_bound_term_uses_abs_bias_d_k_as_bias_magnitude():
    positive = bound.pairwise_bound_term(delta_k=0.1, bias_d_k=0.05, v_d_k=0.02)
    negative = bound.pairwise_bound_term(delta_k=0.1, bias_d_k=-0.05, v_d_k=0.02)
    assert positive == pytest.approx(negative)
    assert positive == pytest.approx(bound._bound_term(0.1, 0.05, 0.02))


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
    #
    # "Far enough" depends on the fitted alpha: N^-alpha * ln(N) (the
    # alpha-jacobian entry's shape) decays to 0 as N -> infinity for any
    # alpha > 0, but only logarithmically slowly for small alpha, so a
    # well-identified alpha close to this curve's true 0.3 needs N far
    # beyond 1e11-1e14 to actually reach saturation (verified directly:
    # a genuine alpha~0.3 fit here is still ~40-130% different between
    # v_k(1e11) and v_k(1e14), not saturated at all -- an earlier version
    # of this test only passed because it happened to run against a fit
    # that behaved differently). 1e6-1e10 over 14 points with low noise
    # (rather than the original 8 points over 1e6-1e8) reliably identifies
    # alpha close to the true 0.3 via multi-start (checked directly across
    # 10 independent seeds); a dedicated local rng (not the file-level
    # `_RNG`) keeps this test's outcome independent of how many draws
    # earlier tests in this file happen to consume.
    rng = np.random.default_rng(1)
    ns = np.geomspace(1e6, 1e10, 14)
    ys = 0.6 - 3.0 * ns ** (-0.3) + rng.normal(0, 0.002, size=len(ns))
    scales = [Scale(n=n, d=20 * n) for n in ns]

    model = fitters.PowerLawN(rng=np.random.default_rng(1)).fit(scales, list(ys))
    v_far = bound.analytic_v_k(model, scales, list(ys), Scale(n=1e30, d=2e31))
    v_farther = bound.analytic_v_k(model, scales, list(ys), Scale(n=1e40, d=2e41))

    assert v_farther == pytest.approx(v_far, rel=1e-3)


# ---------------------------------------------------------------------------
# analytic_v_k() rejects estimators the sandwich formula doesn't describe
# ---------------------------------------------------------------------------


def test_analytic_v_k_rejects_constant_extrapolator():
    # ConstantExtrapolator only fits to the largest-scale observations;
    # the joint-least-squares sandwich formula over the FULL scales list
    # doesn't describe that procedure (PR #17's review, P2).
    ns = np.geomspace(1e6, 1e8, 6)
    ys = [0.5] * 6
    scales = [Scale(n=n, d=20 * n) for n in ns]

    model = fitters.ConstantExtrapolator().fit(scales, ys)

    with pytest.raises(bound.UnsupportedEstimatorError, match="ConstantExtrapolator"):
        bound.analytic_v_k(model, scales, ys, Scale(n=1e9, d=2e10))


def test_analytic_v_k_rejects_two_step_ladder():
    # TwoStepLadder is fit in two separate sequential stages, not one
    # joint simultaneous optimization -- the sandwich formula's
    # single-objective M-estimator assumption doesn't apply.
    ns = np.geomspace(1e6, 1e9, 10)
    ys = list(np.linspace(0.1, 0.8, 10))
    scales = [Scale(n=n, d=20 * n) for n in ns]

    model = fitters.TwoStepLadder(rng=_RNG).fit(scales, ys)

    with pytest.raises(bound.UnsupportedEstimatorError, match="TwoStepLadder"):
        bound.analytic_v_k(model, scales, ys, Scale(n=1e10, d=2e11))


def _log_linear_fit(ns, ds):
    scales = [Scale(n=n, d=d) for n, d in zip(ns, ds, strict=True)]
    ys = [0.3 + 0.02 * np.log(n) for n in ns]
    model = fitters.LogLinear(rng=np.random.default_rng(0)).fit(scales, ys)
    return model, scales, ys


@pytest.mark.parametrize("target_n", [1e9, 1e9 * np.exp(0.05), 1e9 * np.exp(1e-3), 1e12])
def test_analytic_v_k_rejects_target_outside_observed_row_space(target_n):
    # LogLinear only sees log(N): a design observed at a SINGLE N (varying only
    # D) has jacobian rows all equal to [1, log N0], so a target at any other N
    # has a jacobian component the design never observes and its variance is
    # unbounded. pinv would silently report a small finite value instead
    # (second-round review of PR #17). The check must fire even when the
    # missing component is tiny relative to ||J_target|| (target_n =
    # N0 * exp(0.05) -> 0.25% of the norm, exp(1e-3) -> 0.005%), not only when
    # the design is grossly wrong.
    model, scales, ys = _log_linear_fit([1e9] * 4, [1e10, 2e10, 4e10, 8e10])
    target = Scale(n=target_n, d=1e11)
    if target_n == 1e9:
        # Same N: target jacobian IS in the row space, so this must NOT raise.
        assert bound.analytic_v_k(model, scales, ys, target) >= 0.0
        return
    with pytest.raises(bound.UnidentifiedTargetError, match="not identified"):
        bound.analytic_v_k(model, scales, ys, target)


def test_unidentified_target_is_an_unsupported_estimator_error():
    # Callers that already catch UnsupportedEstimatorError (p1_07) keep working.
    assert issubclass(bound.UnidentifiedTargetError, bound.UnsupportedEstimatorError)


def test_analytic_v_k_accepts_identified_log_linear_design():
    ns = [1e8, 2e8, 4e8, 8e8]
    model, scales, ys = _log_linear_fit(ns, [20 * n for n in ns])
    v = bound.analytic_v_k(model, scales, ys, Scale(n=1e12, d=2e13))
    assert np.isfinite(v)
    assert v >= 0.0


def test_analytic_v_k_ill_conditioned_but_identified_design_is_not_rejected():
    # Two nearby Ns: the weighted information is badly conditioned, but the
    # design IS structurally identified -- a large finite variance is the
    # honest answer, not a rejection.
    ns = [1e9, 1e9 * (1 + 1e-4), 1e9 * (1 + 2e-4)]
    model, scales, ys = _log_linear_fit(ns, [2e10, 2e10, 2e10])
    v = bound.analytic_v_k(model, scales, ys, Scale(n=1e12, d=2e13))
    assert np.isfinite(v)


def test_unsupported_sandwich_estimators_matches_the_two_flagged_fitters():
    assert bound.UNSUPPORTED_SANDWICH_ESTIMATORS == {"ConstantExtrapolator", "TwoStepLadder"}
