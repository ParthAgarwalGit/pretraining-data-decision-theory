"""Numerical certificate for Theorem 3 (identifiability and the minimax
rate) -- paper/sections/theorem3_identifiability.tex.

plan/03-phase2-theory.md P2-04: "vary the number and spacing of scales in
simulation; confirm the estimator error follows the predicted rate and
blows up exactly where the rank condition fails."
"""

from __future__ import annotations

import numpy as np
import pytest

from pdt.scaling.base import FitFailure, Scale
from pdt.scaling.fitters import PowerLawN
from pdt.theory.bound import analytic_v_k

_SEED = 20260911


def _jac_reduced(n: float, a: float, alpha: float) -> np.ndarray:
    """Jacobian of g(A,alpha;N)=A*N^-alpha w.r.t. (A, alpha) -- the
    2-parameter reduced power law used in Proposition
    prop:power-law-spacing (PowerLawN minus the ceiling E)."""
    return np.array([n ** (-alpha), -a * n ** (-alpha) * np.log(n)])


def _det_x_formula(n1: float, n2: float, a: float, alpha: float) -> float:
    return -a * n1 ** (-alpha) * n2 ** (-alpha) * np.log(n2 / n1)


def _v_at_target_reduced(n1: float, n2: float, n_star: float, a: float, alpha: float) -> float:
    x = np.array([_jac_reduced(n1, a, alpha), _jac_reduced(n2, a, alpha)])
    xtx_inv = np.linalg.inv(x.T @ x)
    j_star = _jac_reduced(n_star, a, alpha)
    return float(j_star @ xtx_inv @ j_star)


# ---------------------------------------------------------------------------
# (a) det(X) formula (Proposition prop:power-law-spacing)
# ---------------------------------------------------------------------------


def test_det_x_formula_matches_numeric_determinant():
    rng = np.random.default_rng(_SEED)
    for _ in range(500):
        n1 = 10 ** rng.uniform(5, 9)
        n2 = 10 ** rng.uniform(5, 9)
        a = rng.uniform(0.1, 10.0)
        alpha = rng.uniform(0.05, 2.0)
        if abs(np.log(n2 / n1)) < 1e-6:
            continue  # skip near-coincident draws; not the interesting case here

        x = np.array([_jac_reduced(n1, a, alpha), _jac_reduced(n2, a, alpha)])
        det_numeric = np.linalg.det(x)
        det_formula = _det_x_formula(n1, n2, a, alpha)
        assert det_numeric == pytest.approx(det_formula, rel=1e-8)


def test_det_x_is_zero_exactly_at_coincident_scales():
    a, alpha = 2.0, 0.3
    assert _det_x_formula(1e7, 1e7, a, alpha) == 0.0


# ---------------------------------------------------------------------------
# (b) Real PowerLawN: variance blows up under clustering; hard failure
# below p+1 distinct scales.
# ---------------------------------------------------------------------------


def test_powerlawn_variance_grows_as_scales_cluster():
    # Real production fitter, real analytic_v_k -- not the reduced 2-param
    # toy -- fit on a fixed set of scales that are either well-spread
    # (geometric) or clustered, same N_min/N_max envelope, same true
    # underlying curve and noise.
    rng = np.random.default_rng(_SEED + 1)
    true_e, true_a, true_alpha = 0.7, -3.0, 0.25
    n_min, n_max, n_star = 1e6, 1e9, 1e10

    def true_curve(n):
        return true_e + true_a * n ** (-true_alpha)

    spread_ns = np.geomspace(n_min, n_max, 6)
    clustered_ns = np.concatenate(
        [np.geomspace(n_min, n_min * 1.01, 5), [n_max]]
    )  # 5 nearly-identical points + 1 far one: formally 6 scales, p+1=4 satisfied,
    # but severely ill-conditioned relative to the spread design.

    def fit_and_get_v(ns):
        ys = [true_curve(n) + rng.normal(0, 0.005) for n in ns]
        scales = [Scale(n=n, d=20 * n) for n in ns]
        model = PowerLawN(rng=np.random.default_rng(0)).fit(scales, ys)
        return analytic_v_k(model, scales, ys, Scale(n=n_star, d=20 * n_star))

    v_spread = fit_and_get_v(spread_ns)
    v_clustered = fit_and_get_v(clustered_ns)

    assert v_clustered > v_spread, (
        f"clustered design should have much larger extrapolation variance: "
        f"spread v={v_spread}, clustered v={v_clustered}"
    )
    assert v_clustered > 10 * v_spread


def test_powerlawn_fit_fails_below_p_plus_1_scales():
    rng = np.random.default_rng(_SEED + 2)
    ns = np.geomspace(1e6, 1e9, 3)  # p=3 for PowerLawN, need >= p+1=4
    ys = [0.7 - 3.0 * n**-0.25 + rng.normal(0, 0.005) for n in ns]
    scales = [Scale(n=n, d=20 * n) for n in ns]

    with pytest.raises(FitFailure):
        PowerLawN(rng=rng).fit(scales, ys)


def test_powerlawn_fit_succeeds_at_p_plus_1_scales():
    rng = np.random.default_rng(_SEED + 3)
    ns = np.geomspace(1e6, 1e9, 4)  # exactly p+1
    ys = [0.7 - 3.0 * n**-0.25 + rng.normal(0, 0.005) for n in ns]
    scales = [Scale(n=n, d=20 * n) for n in ns]

    model = PowerLawN(rng=rng).fit(scales, ys)
    assert model.fit_diagnostics["n_converged"] >= 1


# ---------------------------------------------------------------------------
# (c) Reproduce Example ex:nonmonotone-spacing's U-shaped table exactly.
# ---------------------------------------------------------------------------


def test_reproduces_the_nonmonotone_spacing_example():
    a, alpha, n_star, n1 = 2.0, 0.3, 1e9, 1e6
    ratios = [1.01, 1.5, 2.0, 5.0, 10.0, 30.0, 100.0, 300.0, 600.0, 900.0]
    v_values = [_v_at_target_reduced(n1, n1 * r, n_star, a, alpha) for r in ratios]

    # U-shaped: strictly decreasing then strictly increasing, minimum
    # strictly interior (not at either end of the swept range).
    min_idx = int(np.argmin(v_values))
    assert 0 < min_idx < len(v_values) - 1, (
        f"expected an interior minimum, got argmin at index {min_idx} of "
        f"{len(v_values)}: {list(zip(ratios, v_values, strict=True))}"
    )
    assert all(v_values[i] > v_values[i + 1] for i in range(min_idx))
    assert all(v_values[i] < v_values[i + 1] for i in range(min_idx, len(v_values) - 1))

    # The two illustrative headline numbers quoted in the .tex file.
    assert v_values[0] == pytest.approx(1.530042e04, rel=1e-4)  # ratio=1.01
    assert min(v_values) == pytest.approx(0.52, abs=0.02)  # near the minimum
    assert v_values[-1] == pytest.approx(0.97, abs=0.02)  # ratio=900


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
