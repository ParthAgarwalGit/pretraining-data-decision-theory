"""Tests for pdt.scaling -- see plan/02-phase1-datadecide.md task P1-04.

Each fitter is tested by generating a clean synthetic curve from its own
functional form (small added noise) and confirming the fit recovers
predictions close to the true underlying curve -- the real correctness
question for an extrapolator, more informative than checking parameter
values directly (which can trade off against each other in a multi-modal
fit while still predicting well).
"""

from __future__ import annotations

import numpy as np
import pytest

from pdt.scaling import fitters
from pdt.scaling.base import Extrapolator, FitFailure, Scale, multi_start_fit

_RNG = np.random.default_rng(42)


def _scales(ns: list[float], d_per_n: float = 20.0) -> list[Scale]:
    """Scale objects with D = d_per_n * N -- a fixed token/param ratio,
    similar in spirit to DataDecide's 5xC convention, so PowerLawN and
    PowerLawC are fitting genuinely different (but correlated) axes."""
    return [Scale(n=n, d=d_per_n * n) for n in ns]


# ---------------------------------------------------------------------------
# base.py
# ---------------------------------------------------------------------------


def test_predict_before_fit_raises():
    model = fitters.ConstantExtrapolator()
    with pytest.raises(FitFailure, match="predict"):
        model.predict(Scale(n=1e8, d=1e9))


def test_jacobian_before_fit_raises():
    model = fitters.ConstantExtrapolator()
    with pytest.raises(FitFailure, match="jacobian"):
        model.jacobian(Scale(n=1e8, d=1e9))


def test_fit_refuses_too_few_scales_for_param_count():
    model = fitters.PowerLawN(rng=_RNG)  # n_params=3, needs >=4 scales
    with pytest.raises(FitFailure, match="need at least 4"):
        model.fit(_scales([1e6, 1e7, 1e8]), [0.3, 0.4, 0.5])


def test_multi_start_fit_raises_when_nothing_converges():
    # An impossible residual (constant huge value, tiny bounds) should
    # never converge to something scipy calls successful within so few
    # function evaluations -- but to keep this robust rather than relying
    # on scipy internals, force it directly: bounds where lower > upper is
    # invalid and scipy will raise for every restart, none succeed.
    def bad_residual(theta):
        raise RuntimeError("deliberately broken")

    with pytest.raises(FitFailure, match="no restart converged"):
        multi_start_fit(
            bad_residual, 2, (np.array([0.0, 0.0]), np.array([1.0, 1.0])), _RNG, n_restarts=3
        )


def test_base_class_abstract_methods_raise_not_implemented():
    base = Extrapolator()
    with pytest.raises(NotImplementedError):
        base._fit_theta([], [], None)
    with pytest.raises(NotImplementedError):
        base._predict_from_theta(np.array([]), Scale(n=1.0, d=1.0))


def test_generic_jacobian_matches_a_known_linear_case():
    # For _predict_from_theta = theta[0] + theta[1]*x, d/d theta0 = 1,
    # d/d theta1 = x -- check the numerically-differentiated base-class
    # jacobian recovers this for a hand-built minimal Extrapolator.
    class Linear(Extrapolator):
        n_params = 2

        def _fit_theta(self, scales, values, weights):
            return np.array([1.0, 2.0]), {}

        def _predict_from_theta(self, theta, scale):
            return theta[0] + theta[1] * scale.n

    model = Linear()
    model.fit(_scales([1.0, 2.0, 3.0]), [3.0, 5.0, 7.0])
    jac = model.jacobian(Scale(n=5.0, d=1.0))

    assert jac == pytest.approx([1.0, 5.0], abs=1e-4)


# ---------------------------------------------------------------------------
# ConstantExtrapolator
# ---------------------------------------------------------------------------


def test_constant_extrapolator_predicts_the_largest_scale_value():
    scales = _scales([1e6, 1e7, 1e8])
    values = [0.30, 0.45, 0.60]  # largest N (1e8) -> 0.60

    model = fitters.ConstantExtrapolator().fit(scales, values)

    assert model.predict(Scale(n=1e9, d=2e10)) == pytest.approx(0.60)


def test_constant_extrapolator_ignores_scale_argument_shape():
    model = fitters.ConstantExtrapolator().fit(_scales([1e6, 1e7]), [0.1, 0.2])
    # Same prediction regardless of how far past the fitted range -- it's
    # a constant, by construction.
    assert model.predict(Scale(n=1e15, d=1e16)) == model.predict(Scale(n=1e9, d=1e10))


def test_constant_extrapolator_averages_replicates_at_the_largest_scale():
    # Regression test for a real bug found by external review: an earlier
    # version took `values[idx]` for whichever row happened to be first
    # at the largest scale, rather than averaging every replicate there --
    # order-dependent and wrong for the replicate histories (e.g. several
    # seeds at the same size) every real caller passes. Exact
    # counterexample from the review: scales [(1,1),(2,1),(2,1)],
    # y=[0,0.1,0.9] must predict the mean of the two n=2 replicates
    # (0.5), not 0.1 (the first one in this order).
    scales = [Scale(n=1, d=1), Scale(n=2, d=1), Scale(n=2, d=1)]
    values = [0.0, 0.1, 0.9]
    model = fitters.ConstantExtrapolator().fit(scales, values)
    assert model.predict(Scale(n=2, d=1)) == pytest.approx(0.5)


def test_constant_extrapolator_is_order_independent():
    scales_a = [Scale(n=1, d=1), Scale(n=2, d=1), Scale(n=2, d=1)]
    values_a = [0.0, 0.1, 0.9]
    scales_b = [Scale(n=1, d=1), Scale(n=2, d=1), Scale(n=2, d=1)]
    values_b = [0.0, 0.9, 0.1]  # the two n=2 replicates swapped

    pred_a = fitters.ConstantExtrapolator().fit(scales_a, values_a).predict(Scale(n=2, d=1))
    pred_b = fitters.ConstantExtrapolator().fit(scales_b, values_b).predict(Scale(n=2, d=1))
    assert pred_a == pytest.approx(pred_b)


def test_constant_extrapolator_respects_weights_at_the_largest_scale():
    scales = [Scale(n=1, d=1), Scale(n=2, d=1), Scale(n=2, d=1)]
    values = [0.0, 0.0, 1.0]
    model = fitters.ConstantExtrapolator().fit(scales, values, weights=[1.0, 3.0, 1.0])
    # Weighted mean of the two n=2 replicates: (3*0 + 1*1) / 4 = 0.25.
    assert model.predict(Scale(n=2, d=1)) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# PowerLawN
# ---------------------------------------------------------------------------


def test_power_law_n_recovers_a_clean_synthetic_curve():
    true_e, true_a, true_alpha = 0.90, -3.0, 0.30
    ns = np.geomspace(1e6, 1e9, 8)
    ys = true_e + true_a * ns ** (-true_alpha) + _RNG.normal(0, 0.002, size=len(ns))

    model = fitters.PowerLawN(rng=_RNG).fit(_scales(list(ns)), list(ys))

    held_out_n = 3e9
    true_pred = true_e + true_a * held_out_n ** (-true_alpha)
    assert model.predict(Scale(n=held_out_n, d=20 * held_out_n)) == pytest.approx(
        true_pred, abs=0.02
    )
    assert model.fit_diagnostics["n_converged"] >= 1


def test_power_law_n_recovers_a_small_alpha_noiseless_curve_across_seeds():
    # Regression test for a real bug found by external review: uniform
    # random-restart initialization over alpha in [1e-3, 10] spends
    # almost all its mass on alpha values where N^-alpha underflows to
    # numerically zero for this project's real parameter counts
    # (1e6-1e9) -- a flat region with no gradient signal, where
    # scipy's least_squares can still report success=True (the step
    # size, not the residual, went to zero). Exact counterexample from
    # the review: PowerLawN(rng=np.random.default_rng(1)) fit to a
    # noiseless y = 0.9 - 2*N^-0.1 curve on N from 1e6 to 1.5e8 --
    # every one of the 8 default restarts converged to the identical
    # wrong prediction at N=1e9 (0.504 instead of the true 0.648),
    # objective_spread ~1e-18 (all 8 restarts agreeing with each other
    # made this look like a converged, validated fit). Checked across
    # several seeds, not just the one in the report, since the bug was
    # about *initialization landing badly by chance*, not this one seed
    # specifically.
    true_e, true_a, true_alpha = 0.9, -2.0, 0.1
    ns = np.geomspace(1e6, 1.5e8, 10)
    ys = true_e + true_a * ns ** (-true_alpha)
    held_out = Scale(n=1e9, d=20e9)
    true_pred = true_e + true_a * held_out.n ** (-true_alpha)

    for seed in range(5):
        model = fitters.PowerLawN(rng=np.random.default_rng(seed)).fit(_scales(list(ns)), list(ys))
        assert model.predict(held_out) == pytest.approx(true_pred, abs=1e-3), f"seed={seed}"
        # best_cost (not objective_spread -- some restarts can still land
        # on a genuinely worse local optimum even with the fix; what
        # matters is that the *best* one found the true global minimum)
        # near zero confirms the global optimum was actually reached.
        assert model.fit_diagnostics["best_cost"] < 1e-6, f"seed={seed}"


def test_multi_start_fit_log_uniform_dims_avoids_the_flat_high_alpha_region():
    # A direct, model-agnostic check of the fix itself: with
    # log_uniform_dims naming the exponent parameter, random restarts on
    # the exact P1-04 counterexample recover the true curve; without it
    # (the old behavior), reproduce the original failure to confirm this
    # test would have caught it.
    true_e, true_a, true_alpha = 0.9, -2.0, 0.1
    ns = np.geomspace(1e6, 1.5e8, 10)
    ys = true_e + true_a * ns ** (-true_alpha)
    bounds = (np.array([-2.0, -10.0, 1e-3]), np.array([3.0, 10.0, 10.0]))

    def residual(theta):
        e, a, alpha = theta
        return (e + a * ns ** (-alpha)) - ys

    fixed_theta, _ = multi_start_fit(
        residual, 3, bounds, np.random.default_rng(1), log_uniform_dims=(2,)
    )
    assert fixed_theta == pytest.approx([true_e, true_a, true_alpha], abs=1e-3)

    broken_theta, _ = multi_start_fit(residual, 3, bounds, np.random.default_rng(1))
    assert broken_theta[0] != pytest.approx(true_e, abs=1e-3)  # reproduces the original bug


def test_power_law_n_jacobian_has_three_entries():
    model = fitters.PowerLawN(rng=_RNG).fit(_scales(list(np.geomspace(1e6, 1e9, 6))), [0.3] * 6)
    assert model.jacobian(Scale(n=1e8, d=1e9)).shape == (3,)


# ---------------------------------------------------------------------------
# PowerLawC
# ---------------------------------------------------------------------------


def test_power_law_c_recovers_a_clean_synthetic_curve():
    true_e, true_a, true_alpha = 0.90, -3.0, 0.30
    ns = np.geomspace(1e6, 1e9, 8)
    scales = _scales(list(ns))
    cs = np.array([s.compute for s in scales])
    ys = true_e + true_a * cs ** (-true_alpha) + _RNG.normal(0, 0.002, size=len(ns))

    model = fitters.PowerLawC(rng=_RNG).fit(scales, list(ys))

    held_out = Scale(n=3e9, d=20 * 3e9)
    true_pred = true_e + true_a * held_out.compute ** (-true_alpha)
    assert model.predict(held_out) == pytest.approx(true_pred, abs=0.02)


# ---------------------------------------------------------------------------
# ChinchillaND
# ---------------------------------------------------------------------------


def test_chinchilla_nd_recovers_a_clean_synthetic_curve():
    true_e, true_a, true_alpha, true_b, true_beta = 0.92, 2.0, 0.25, 1.5, 0.20
    ns = np.geomspace(1e6, 1e9, 10)
    ds = 20 * ns
    scales = [Scale(n=n, d=d) for n, d in zip(ns, ds, strict=True)]
    ys = (
        true_e
        - true_a * ns ** (-true_alpha)
        - true_b * ds ** (-true_beta)
        + _RNG.normal(0, 0.002, size=len(ns))
    )

    model = fitters.ChinchillaND(rng=_RNG).fit(scales, list(ys))

    held_out = Scale(n=3e9, d=20 * 3e9)
    true_pred = true_e - true_a * held_out.n ** (-true_alpha) - true_b * held_out.d ** (-true_beta)
    assert model.predict(held_out) == pytest.approx(true_pred, abs=0.03)


def test_chinchilla_nd_requires_six_scales():
    model = fitters.ChinchillaND(rng=_RNG)  # n_params=5, needs >=6
    with pytest.raises(FitFailure, match="need at least 6"):
        model.fit(_scales([1e6, 1e7, 1e8, 1e9, 1e10]), [0.1, 0.2, 0.3, 0.4, 0.5])


# ---------------------------------------------------------------------------
# LogLinear
# ---------------------------------------------------------------------------


def test_log_linear_recovers_a_clean_synthetic_curve():
    true_a, true_b = 0.1, 0.05
    ns = np.geomspace(1e6, 1e9, 6)
    ys = true_a + true_b * np.log(ns) + _RNG.normal(0, 0.001, size=len(ns))

    model = fitters.LogLinear(rng=_RNG).fit(_scales(list(ns)), list(ys))

    held_out_n = 3e9
    true_pred = true_a + true_b * np.log(held_out_n)
    assert model.predict(Scale(n=held_out_n, d=1.0)) == pytest.approx(true_pred, abs=0.02)


# ---------------------------------------------------------------------------
# TwoStepLadder
# ---------------------------------------------------------------------------


def test_two_step_ladder_recovers_a_clean_synthetic_sigmoid_curve():
    # Build data directly from the model's own two-step functional form so
    # this test checks the fitting procedure, not whether real data happens
    # to be sigmoid-shaped in compute (P1-04's actual experiment is where
    # that gets checked against real data).
    e1, a1, alpha1 = 0.5, -2.0, 0.3
    lo, hi, k, x0 = 0.05, 0.95, 15.0, 0.5

    ns = np.geomspace(1e6, 1e9, 12)
    scales = _scales(list(ns))
    cs = np.array([s.compute for s in scales])
    proxy = e1 + a1 * cs ** (-alpha1)
    ys = lo + (hi - lo) / (1.0 + np.exp(-k * (proxy - x0))) + _RNG.normal(0, 0.005, size=len(ns))

    model = fitters.TwoStepLadder(rng=_RNG).fit(scales, list(ys))

    assert model.n_params == 7
    # In-sample fit should track the (noisy) data reasonably well -- a
    # weaker check than the other fitters' held-out extrapolation check,
    # deliberately: TwoStepLadder's 7-parameter sequential fit is more
    # prone to a "good enough" local optimum that doesn't perfectly
    # recover the true generating parameters, so this checks predictive
    # fit quality rather than parameter recovery.
    preds = [model.predict(s) for s in scales]
    assert np.mean(np.abs(np.array(preds) - ys)) < 0.05


def test_two_step_ladder_requires_eight_scales():
    model = fitters.TwoStepLadder(rng=_RNG)  # n_params=7, needs >=8
    with pytest.raises(FitFailure, match="need at least 8"):
        model.fit(_scales(list(np.geomspace(1e6, 1e9, 7))), [0.1] * 7)


def test_two_step_ladder_diagnostics_include_both_steps():
    model = fitters.TwoStepLadder(rng=_RNG).fit(
        _scales(list(np.geomspace(1e6, 1e9, 10))), list(np.linspace(0.1, 0.8, 10))
    )
    assert "step1" in model.fit_diagnostics
    assert "step2" in model.fit_diagnostics


# ---------------------------------------------------------------------------
# Second-round external review: randomized log-uniform starts still failed
# on the compute-based PowerLawC (2/100 seeds) -- informative starts fix it.
# ---------------------------------------------------------------------------


def _noiseless_power_law_case(alpha: float, use_compute: bool):
    ns = np.geomspace(1e6, 1.5e8, 10)
    scales = [Scale(n=n, d=20 * n) for n in ns]
    x = np.array([s.compute for s in scales]) if use_compute else ns
    target = Scale(n=1e9, d=20e9)
    x_target = target.compute if use_compute else target.n
    return scales, list(0.9 - 2 * x ** (-alpha)), target, 0.9 - 2 * x_target ** (-alpha)


def test_power_law_c_recovers_the_reviewers_exact_failing_seeds():
    # Reproduction from the review: default_rng(30) and default_rng(54)
    # predicted 0.2463 instead of 0.4004 with every restart "converged".
    scales, ys, target, truth = _noiseless_power_law_case(0.03, use_compute=True)
    for seed in (30, 54):
        model = fitters.PowerLawC(rng=np.random.default_rng(seed)).fit(scales, ys)
        assert model.predict(target) == pytest.approx(truth, abs=1e-3)


@pytest.mark.parametrize("use_compute", [False, True], ids=["PowerLawN", "PowerLawC"])
@pytest.mark.parametrize("alpha", [0.01, 0.03, 0.1, 0.3, 0.6])
def test_power_law_recovers_noiseless_curves_across_many_seeds(alpha, use_compute):
    scales, ys, target, truth = _noiseless_power_law_case(alpha, use_compute)
    cls = fitters.PowerLawC if use_compute else fitters.PowerLawN
    for seed in range(40):
        model = cls(rng=np.random.default_rng(seed)).fit(scales, ys)
        assert model.predict(target) == pytest.approx(truth, abs=1e-3), f"seed {seed}"


def test_chinchilla_nd_recovers_a_noiseless_curve_across_seeds():
    ns = np.geomspace(1e6, 1.5e8, 10)
    scales = [Scale(n=n, d=20 * n) for n in ns]
    ys = list(0.9 - 2 * ns ** (-0.1) - 3 * (20 * ns) ** (-0.15))
    target = Scale(n=1e9, d=20e9)
    truth = 0.9 - 2 * target.n ** (-0.1) - 3 * target.d ** (-0.15)
    for seed in range(8):
        model = fitters.ChinchillaND(rng=np.random.default_rng(seed)).fit(scales, ys)
        assert model.predict(target) == pytest.approx(truth, abs=5e-3), f"seed {seed}"


def test_multi_start_fit_refines_deterministic_extra_starts_first():
    # A start placed exactly at the optimum must be refined and win even
    # when zero random restarts are requested.
    def residual(theta):
        return np.array([theta[0] - 3.0, theta[1] + 1.0])

    theta, diag = multi_start_fit(
        residual,
        2,
        (np.array([-10.0, -10.0]), np.array([10.0, 10.0])),
        np.random.default_rng(0),
        n_restarts=0,
        extra_starts=(np.array([3.0, -1.0]),),
    )
    assert theta == pytest.approx([3.0, -1.0])
    assert diag["n_restarts"] == 1
