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
