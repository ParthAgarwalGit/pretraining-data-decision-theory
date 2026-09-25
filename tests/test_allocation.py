"""Tests for pdt.bai.allocation -- plan/04-phase3-algorithm.md P3-02.

See src/pdt/bai/allocation.py's own module and solve_allocation()
docstrings, and docs/decisions.md, for the full account of what was
found and fixed (and what was NOT fully fixed -- solve_allocation is a
validated-reasonable, not certified-optimal, solver) while building
this. Tolerances below are set to match that honest status: they check
solve_allocation lands in the right regime and is internally consistent,
not that it exactly matches brute_force_allocation's best sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdt.bai.allocation import (
    _arm_rate,
    _arm_rate_and_grad,
    _fisher_information,
    _project_to_simplex,
    _solve_weighted,
    brute_force_allocation,
    solve_allocation,
)
from pdt.scaling.base import Scale
from pdt.scaling.fitters import LogLinear, PowerLawN

_SCALES = [Scale(n=1e6, d=2e7), Scale(n=1e7, d=2e8), Scale(n=1e8, d=2e9)]
_TARGET = Scale(n=1e9, d=2e10)
_SIGMA2 = 1e-4


def _model(e: float, a: float, alpha: float) -> PowerLawN:
    m = PowerLawN()
    m._theta = np.array([e, a, alpha])
    return m


def _sigma2(_scale: Scale) -> float:
    return _SIGMA2


@pytest.fixture
def small_instance():
    models = {
        "kstar": _model(0.8, 2.0, 0.3),
        "k1": _model(0.7, 2.5, 0.25),
        "k2": _model(0.6, 3.0, 0.2),
    }
    deltas = {"k1": 0.05, "k2": 0.1}
    return models, deltas


# ---------------------------------------------------------------------------
# _project_to_simplex
# ---------------------------------------------------------------------------


def test_project_to_simplex_interior_point_is_normalized():
    v = np.array([0.5, 0.5, 0.5])
    p = _project_to_simplex(v)
    assert p == pytest.approx(np.full(3, 1 / 3))
    assert p.sum() == pytest.approx(1.0)


def test_project_to_simplex_already_feasible_point_is_fixed():
    v = np.array([0.2, 0.5, 0.3])
    p = _project_to_simplex(v)
    assert p == pytest.approx(v)


def test_project_to_simplex_handles_negative_components():
    v = np.array([-1.0, 2.0, 0.5])
    p = _project_to_simplex(v)
    assert np.all(p >= 0.0)
    assert p.sum() == pytest.approx(1.0)


def test_project_to_simplex_random_points_always_feasible():
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.uniform(-5, 5, size=rng.integers(2, 10))
        p = _project_to_simplex(v)
        assert np.all(p >= -1e-12)
        assert p.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# _fisher_information / _arm_rate -- match Theorem 2 Part A's definitions
# ---------------------------------------------------------------------------


def test_fisher_information_matches_hand_formula():
    model = _model(0.7, 2.5, 0.25)
    w = np.array([1e-15, 2e-15, 0.0])
    info = _fisher_information(model, _SCALES, _sigma2, w)
    expected = sum(
        wi * np.outer(model.jacobian(s), model.jacobian(s)) / _SIGMA2
        for wi, s in zip(w, _SCALES, strict=True)
        if wi > 0
    )
    assert info == pytest.approx(expected)


def test_arm_rate_is_zero_at_zero_weight():
    model = _model(0.7, 2.5, 0.25)
    rate = _arm_rate(model, _SCALES, _sigma2, np.zeros(3), _TARGET, 0.05)
    assert rate == 0.0


def test_arm_rate_increases_with_more_weight():
    model = _model(0.7, 2.5, 0.25)
    small = _arm_rate(model, _SCALES, _sigma2, np.full(3, 1e-16), _TARGET, 0.05)
    large = _arm_rate(model, _SCALES, _sigma2, np.full(3, 1e-14), _TARGET, 0.05)
    assert large > small


def test_arm_rate_is_zero_when_target_is_structurally_unidentifiable():
    # Regression for PR #28's review, exactly as given: a LogLinear model
    # (y = a + b*log(N)) observed ONLY at N=1 has jacobian [1, log(1)] =
    # [1, 0] at every candidate scale, regardless of weight -- the slope
    # parameter b is structurally unidentifiable from this design, no
    # matter how compute is allocated among (the one) candidate scale.
    # The target at N=4 has jacobian [1, log(4)], nonzero in the missing
    # direction. Before the fix, pinv's minimum-norm convention silently
    # treated that missing direction as contributing zero variance,
    # giving a small but finite (and wrong) rate; the correct rate is
    # exactly 0 (infinite variance -- cannot be estimated at all here).
    model = LogLinear()
    model._theta = np.array([0.5, 0.1])
    scales = [Scale(n=1.0, d=1.0)]
    target = Scale(n=4.0, d=4.0)
    rate = _arm_rate(model, scales, lambda s: 1.0, np.array([1.0]), target, delta_k=0.1)
    assert rate == 0.0


def test_arm_rate_and_grad_is_zero_when_target_is_structurally_unidentifiable():
    model = LogLinear()
    model._theta = np.array([0.5, 0.1])
    scales = [Scale(n=1.0, d=1.0)]
    target = Scale(n=4.0, d=4.0)
    rate, grad = _arm_rate_and_grad(
        model, scales, lambda s: 1.0, np.array([1.0]), target, delta_k=0.1
    )
    assert rate == 0.0
    assert np.all(grad == 0.0)


def test_arm_rate_is_positive_when_target_direction_is_actually_covered():
    # Sanity check that the fix doesn't over-trigger: adding a SECOND
    # candidate scale whose jacobian has a nonzero log-N component makes
    # the slope identifiable again, and the rate should be positive.
    model = LogLinear()
    model._theta = np.array([0.5, 0.1])
    scales = [Scale(n=1.0, d=1.0), Scale(n=100.0, d=100.0)]
    target = Scale(n=4.0, d=4.0)
    rate = _arm_rate(model, scales, lambda s: 1.0, np.array([1.0, 1.0]), target, delta_k=0.1)
    assert rate > 0.0


@pytest.mark.parametrize("log_target", [0.05, 0.005, 1e-4])
def test_small_missing_component_is_still_unidentifiable(log_target):
    # Second-round review of PR #28: the first fix used a 10% relative-residual
    # tolerance, so a design observed only at N=1 (J = [1, 0]) with target
    # J_target = [1, .05] -- a 5% missing component -- slipped through and got
    # a finite variance. Structural non-identifiability can sit arbitrarily
    # close to an observed scale; it must be rejected at every such distance.
    model = LogLinear()
    model._theta = np.array([0.5, 0.1])
    scales = [Scale(n=1.0, d=1.0)]
    target = Scale(n=float(np.exp(log_target)), d=1.0)
    w = np.array([1.0])
    assert _arm_rate(model, scales, lambda s: 1.0, w, target, delta_k=0.1) == 0.0
    rate, grad = _arm_rate_and_grad(model, scales, lambda s: 1.0, w, target, delta_k=0.1)
    assert rate == 0.0
    assert np.all(grad == 0.0)


def test_brute_force_rate_is_zero_for_structurally_unidentifiable_target():
    models = {"kstar": LogLinear(), "k1": LogLinear()}
    for m in models.values():
        m._theta = np.array([0.5, 0.1])
    scales = [Scale(n=1.0, d=1.0), Scale(n=1.0, d=10.0)]  # both at N=1
    target = Scale(n=float(np.exp(0.05)), d=1.0)
    res = brute_force_allocation(
        models, "kstar", scales, target, lambda s: 1.0, {"k1": 0.1}, n_samples=200
    )
    assert res.rate == 0.0
    assert res.t_star == float("inf")


def test_ill_conditioned_but_identified_design_keeps_a_positive_rate():
    # A tight tolerance on the weighted information false-triggers here: two
    # nearby N (condition number of the weighted information ~1e12) and a
    # target far away. The design IS identified, so the honest answer is a
    # small positive rate (huge variance), not "no information".
    model = LogLinear()
    model._theta = np.array([0.5, 0.1])
    scales = [
        Scale(n=1e9, d=1.0),
        Scale(n=1e9 * (1 + 1e-5), d=1.0),
        Scale(n=1e9 * (1 + 2e-5), d=1.0),
    ]
    target = Scale(n=1e12, d=1.0)
    rate = _arm_rate(model, scales, lambda s: 1.0, np.ones(3), target, delta_k=0.1)
    assert rate > 0.0
    assert np.isfinite(rate)


def test_rate_is_invariant_to_parameter_units():
    # Rescaling a parameter (theta_2 -> 1e9 * theta_2, so its Jacobian column
    # shrinks by 1e-9) must not change the rate: equilibration removes
    # unit-induced ill-conditioning instead of leaving it to the pinv cutoff.
    class _Rescaled(LogLinear):
        def jacobian(self, scale):
            j = super().jacobian(scale)
            return j * np.array([1.0, 1e-9])

    base, rescaled = LogLinear(), _Rescaled()
    for m in (base, rescaled):
        m._theta = np.array([0.5, 0.1])
    scales = [Scale(n=1e3, d=1.0), Scale(n=1e5, d=1.0), Scale(n=1e7, d=1.0)]
    target = Scale(n=1e9, d=1.0)
    w = np.array([0.3, 0.5, 0.2])
    r1 = _arm_rate(base, scales, lambda s: 1.0, w, target, delta_k=0.1)
    r2 = _arm_rate(rescaled, scales, lambda s: 1.0, w, target, delta_k=0.1)
    assert r1 > 0.0
    assert r2 == pytest.approx(r1, rel=1e-6)


def _loglinear_two_point():
    model = LogLinear()
    model._theta = np.array([0.5, 0.1])
    scales = [Scale(n=float(np.e), d=1.0), Scale(n=float(np.e**2), d=1.0)]
    return model, scales, Scale(n=float(np.e**3), d=1.0)


@pytest.mark.parametrize("weak_weight", [1e-16, 1e-20, 1e-30])
def test_weak_but_identified_direction_keeps_its_large_variance(weak_weight):
    # Second-round review of PR #28: candidates N=e, e^2 (log N = 1, 2), target N=e^3, sigma2=1,
    # gap=.1, weights [1, w]. Direct two-point regression gives
    # Var = 1 + 4/w  =>  rate = .01 / (2 (1 + 4/w)). The old pinv(A^T A) path dropped the weak
    # eigen-direction and returned ~.00125 no matter how small w was.
    model, scales, target = _loglinear_two_point()
    w = np.array([1.0, weak_weight])
    expected = 0.01 / (2.0 * (1.0 + 4.0 / weak_weight))
    rate = _arm_rate(model, scales, lambda s: 1.0, w, target, delta_k=0.1)
    assert rate == pytest.approx(expected, rel=1e-6)
    rate_g, grad = _arm_rate_and_grad(model, scales, lambda s: 1.0, w, target, delta_k=0.1)
    assert rate_g == pytest.approx(expected, rel=1e-6)
    assert np.all(np.isfinite(grad))


def test_weak_direction_below_the_rank_cutoff_fails_closed_to_zero_information():
    # A weight so small that sqrt(w) is below the design's numerical rank cutoff: the target
    # needs that direction, so the honest answer is "no information" (rate 0), never a small
    # finite variance from a dropped direction.
    model, scales, target = _loglinear_two_point()
    w = np.array([1.0, 1e-40])
    assert _arm_rate(model, scales, lambda s: 1.0, w, target, delta_k=0.1) == 0.0
    rate, grad = _arm_rate_and_grad(model, scales, lambda s: 1.0, w, target, delta_k=0.1)
    assert rate == 0.0 and np.all(grad == 0.0)


def test_batched_solve_matches_the_scalar_path_including_weak_weights():
    # The brute-force path solves a whole stack of weighted designs at once.
    model, scales, target = _loglinear_two_point()
    j_scales = np.array([model.jacobian(s) for s in scales])
    j_star = model.jacobian(target)
    weights = np.array([[1.0, 1e-16], [1.0, 1e-20], [1.0, 1.0], [0.3, 0.7], [1.0, 1e-40]])
    stack = np.sqrt(weights)[:, :, None] * j_scales[None, :, :]
    f_batch, y_batch = _solve_weighted(stack, j_star)
    for i in range(len(weights)):
        f_one, y_one = _solve_weighted(stack[i][None, ...], j_star)
        assert f_batch[i] == pytest.approx(f_one[0], rel=1e-12)
        assert y_batch[i] == pytest.approx(y_one[0], rel=1e-9, abs=1e-30)
    assert f_batch[0] == pytest.approx(1.0 + 4.0 / 1e-16, rel=1e-6)
    assert f_batch[1] == pytest.approx(1.0 + 4.0 / 1e-20, rel=1e-6)
    assert f_batch[4] == 0.0  # below the cutoff: fails closed


def test_solve_weighted_matches_the_textbook_inverse_on_a_well_conditioned_design():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(7, 3))
    j_star = rng.normal(size=3)
    f, y = _solve_weighted(a[None, ...], j_star)
    info_inv = np.linalg.inv(a.T @ a)
    assert f[0] == pytest.approx(float(j_star @ info_inv @ j_star), rel=1e-10)
    assert y[0] == pytest.approx(info_inv @ j_star, rel=1e-10)


def test_fisher_information_rejects_nonpositive_sigma2():
    model = _model(0.7, 2.5, 0.25)

    def bad_sigma2(_scale: Scale) -> float:
        return 0.0

    with pytest.raises(ValueError):
        _fisher_information(model, _SCALES, bad_sigma2, np.array([1.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# _arm_rate_and_grad -- exact gradient vs. finite differences
# ---------------------------------------------------------------------------


def test_arm_rate_and_grad_matches_finite_differences():
    model = _model(0.7, 2.5, 0.25)
    w = np.array([1e-15, 2e-16, 3e-17])
    rate, grad = _arm_rate_and_grad(model, _SCALES, _sigma2, w, _TARGET, 0.05)
    assert rate == pytest.approx(_arm_rate(model, _SCALES, _sigma2, w, _TARGET, 0.05))

    eps_rel = 1e-4
    fd_grad = np.zeros(3)
    for i in range(3):
        w1, w2 = w.copy(), w.copy()
        w1[i] *= 1 + eps_rel
        w2[i] *= 1 - eps_rel
        r1 = _arm_rate(model, _SCALES, _sigma2, w1, _TARGET, 0.05)
        r2 = _arm_rate(model, _SCALES, _sigma2, w2, _TARGET, 0.05)
        fd_grad[i] = (r1 - r2) / (w1[i] - w2[i])
    assert grad == pytest.approx(fd_grad, rel=1e-3)


def test_arm_rate_and_grad_is_zero_at_zero_information():
    model = _model(0.7, 2.5, 0.25)
    rate, grad = _arm_rate_and_grad(model, _SCALES, _sigma2, np.zeros(3), _TARGET, 0.05)
    assert rate == 0.0
    assert np.all(grad == 0.0)


# ---------------------------------------------------------------------------
# solve_allocation -- feasibility, k* exclusion, internal consistency,
# and a generous (not exact) comparison against brute_force_allocation.
# ---------------------------------------------------------------------------


def test_solve_allocation_output_is_budget_feasible(small_instance):
    models, deltas = small_instance
    res = solve_allocation(
        models, "kstar", _SCALES, _TARGET, _sigma2, deltas, n_restarts=2, n_iter=500
    )
    total_cost = sum(w * _SCALES[i].compute for (_arm, i), w in res.weights.items())
    assert total_cost == pytest.approx(1.0, abs=1e-6)
    assert all(w >= 0.0 for w in res.weights.values())


def test_solve_allocation_rejects_unknown_k_star_or_missing_models(small_instance):
    models, deltas = small_instance
    with pytest.raises(ValueError):
        solve_allocation(models, "not_a_real_arm", _SCALES, _TARGET, _sigma2, deltas)
    with pytest.raises(ValueError):
        solve_allocation({"kstar": models["kstar"]}, "kstar", _SCALES, _TARGET, _sigma2, deltas)


def test_solve_allocation_rejects_nonpositive_cost(small_instance):
    models, deltas = small_instance

    def bad_cost(s: Scale) -> float:
        return 0.0 if s.n == _SCALES[0].n else s.compute

    with pytest.raises(ValueError):
        solve_allocation(models, "kstar", _SCALES, _TARGET, _sigma2, deltas, cost=bad_cost)


def test_solve_allocation_stops_at_a_zero_gradient_stationary_point(small_instance):
    # delta_k=0 for a real challenger makes its own rate (and gradient)
    # identically zero everywhere -- it is then always `tightest` (0 is
    # the min of any nonnegative rates), so the very first iteration's
    # full_grad is all zero and the loop must take the early-exit branch
    # rather than dividing by a zero grad_norm.
    models, _ = small_instance
    deltas = {"k1": 0.0, "k2": 0.1}
    res = solve_allocation(
        models, "kstar", _SCALES, _TARGET, _sigma2, deltas, n_restarts=1, n_iter=50
    )
    assert res.rate == 0.0
    assert res.converged


def test_solve_allocation_reports_nonconvergence_on_budget_exhaustion(small_instance):
    # Regression for PR #28's review, P2: converged=True was previously
    # unconditional even when the iteration budget ran out without the
    # subgradient norm ever settling near zero. n_iter=1 gives the search
    # essentially no chance to reach a near-stationary point on this
    # instance (delta_k > 0 for every challenger, so the gradient stays
    # informative), so it should honestly report budget exhaustion.
    models, deltas = small_instance
    res = solve_allocation(
        models, "kstar", _SCALES, _TARGET, _sigma2, deltas, n_restarts=1, n_iter=1
    )
    assert res.converged is False
    assert res.n_iterations == 1
    assert "budget exhausted" in res.message


def test_solve_allocation_rejects_deltas_naming_only_k_star(small_instance):
    models, _ = small_instance
    with pytest.raises(ValueError):
        solve_allocation(models, "kstar", _SCALES, _TARGET, _sigma2, {"kstar": 0.0})


def test_solve_allocation_report_never_allocates_to_k_star(small_instance):
    # Structural property, not just an empirical observation: k_star's
    # own weights are not decision variables at all (see the module
    # docstring) -- confirmed by checking the reported weights dict never
    # contains a k_star key.
    models, deltas = small_instance
    res = solve_allocation(
        models, "kstar", _SCALES, _TARGET, _sigma2, deltas, n_restarts=2, n_iter=500
    )
    assert all(arm != "kstar" for arm, _i in res.weights)


def test_solve_allocation_is_consistent_across_restarts(small_instance):
    # A loose reliability check, deliberately not a tight one -- see the
    # module docstring's documented limitation. Empirically, a single
    # low-iteration run can land ~4-5x worse than the rest (observed
    # directly while calibrating this bound: rates spanning
    # 3.4e-19 to 1.54e-18 across 4 seeds at n_iter=500). This test's job
    # is to catch a *catastrophic* regression (collapse to zero, or a
    # >10x spread), not to assert a reliability property the solver does
    # not actually have.
    models, deltas = small_instance
    rates = [
        solve_allocation(
            models,
            "kstar",
            _SCALES,
            _TARGET,
            _sigma2,
            deltas,
            n_restarts=2,
            n_iter=500,
            rng=np.random.default_rng(seed),
        ).rate
        for seed in range(4)
    ]
    assert min(rates) > 0
    assert max(rates) / min(rates) < 10.0, f"restarts disagree too much: {rates}"


def test_solve_allocation_beats_uniform_allocation(small_instance):
    # A weak, but honest, quality floor: solve_allocation must do
    # meaningfully better than spreading the budget uniformly across
    # every (challenger, scale) pair -- if it can't clear this very low
    # bar, something is badly broken, independent of the open global-
    # optimality question.
    models, deltas = small_instance
    res = solve_allocation(
        models, "kstar", _SCALES, _TARGET, _sigma2, deltas, n_restarts=3, n_iter=1000
    )
    costs = np.array([s.compute for s in _SCALES])
    challengers = list(deltas)
    n_dims = len(challengers) * len(_SCALES)
    uniform_p = np.full(n_dims, 1.0 / n_dims)
    uniform_rates = {}
    for i, arm in enumerate(challengers):
        w = uniform_p[i * len(_SCALES) : (i + 1) * len(_SCALES)] / costs
        uniform_rates[arm] = _arm_rate(models[arm], _SCALES, _sigma2, w, _TARGET, deltas[arm])
    uniform_rate = min(uniform_rates.values())
    assert res.rate > uniform_rate


def test_solve_allocation_within_generous_factor_of_brute_force(small_instance):
    # NOT a tight-optimality check -- see the module docstring's
    # documented limitation. This only asserts solve_allocation is in
    # the right regime (same order of magnitude), a regression guard
    # against a future change reintroducing one of the severe bugs found
    # while building this (e.g. a scaling regression that collapses the
    # solution to rate=0, which this WOULD catch).
    models, deltas = small_instance
    res = solve_allocation(
        models, "kstar", _SCALES, _TARGET, _sigma2, deltas, n_restarts=4, n_iter=1500
    )
    bf = brute_force_allocation(
        models, "kstar", _SCALES, _TARGET, _sigma2, deltas, n_samples=200_000
    )
    assert res.rate > 0
    assert res.rate >= bf.rate / 10, (
        f"solve_allocation ({res.rate}) is more than 10x worse than "
        f"brute_force_allocation ({bf.rate}) -- likely a real regression, "
        f"not the already-documented optimality gap"
    )


# ---------------------------------------------------------------------------
# brute_force_allocation -- feasibility and internal consistency of the
# vectorized batch computation against the scalar _arm_rate reference.
# ---------------------------------------------------------------------------


def test_brute_force_allocation_output_is_budget_feasible(small_instance):
    models, deltas = small_instance
    res = brute_force_allocation(models, "kstar", _SCALES, _TARGET, _sigma2, deltas, n_samples=5000)
    total_cost = sum(w * _SCALES[i].compute for (_arm, i), w in res.weights.items())
    assert total_cost == pytest.approx(1.0, abs=1e-6)


def test_brute_force_vectorized_rate_matches_scalar_arm_rate(small_instance):
    models, deltas = small_instance
    res = brute_force_allocation(
        models, "kstar", _SCALES, _TARGET, _sigma2, deltas, n_samples=10_000
    )
    for arm in deltas:
        w = np.array([res.weights[(arm, i)] for i in range(len(_SCALES))])
        expected = _arm_rate(models[arm], _SCALES, _sigma2, w, _TARGET, deltas[arm])
        assert res.arm_rates[arm] == pytest.approx(expected, rel=1e-9)


def test_brute_force_allocation_more_samples_does_not_get_worse(small_instance):
    models, deltas = small_instance
    small = brute_force_allocation(
        models,
        "kstar",
        _SCALES,
        _TARGET,
        _sigma2,
        deltas,
        n_samples=1000,
        rng=np.random.default_rng(0),
    )
    large = brute_force_allocation(
        models,
        "kstar",
        _SCALES,
        _TARGET,
        _sigma2,
        deltas,
        n_samples=50_000,
        rng=np.random.default_rng(0),
    )
    assert large.rate >= small.rate
