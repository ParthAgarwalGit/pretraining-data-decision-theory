"""Tests for pdt.bai.ets -- plan/04-phase3-algorithm.md P3-03.

Two kinds of oracle are used: `_make_synthetic` (the real, calibrated
`SyntheticOracle` from P3-01) for the "all five methods run to
completion" checks, which don't need precise control over the bias; and
`_BumpOracle` (a small hand-built test double, same `phi(x)=clip(x,0,1)`
bump construction as Theorem 2 Part B / tests/theory/test_theorem4.py)
for the stopping/abstention-rule checks, which need exact, deterministic
control over each recipe's bias at the target scale that
`SyntheticOracle`'s own random construction doesn't offer.

`LogLinear` (2 parameters, fit is a trivial linear least squares) is
used as `model_factory` throughout instead of the module's own default
`PowerLawN` (3 parameters, a genuinely nonlinear multi-start fit) --
purely a test-speed choice (ETS refits every recipe every round, so the
per-round fit cost matters for how fast these tests run), not a claim
about which model a real user should pick.
"""

from __future__ import annotations

import numpy as np
import pytest

from pdt.bai.ets import (
    SelectionResult,
    _assert_design_identified,
    _beta,
    _fit_recipe,
    _largest_observed_scale,
    extrapolation_track_and_stop,
    fixed_ladder_extrapolation,
    single_scale_recommendation,
    successive_halving_over_scales,
    uniform_allocation,
)
from pdt.bai.oracle import SyntheticOracle, _stable_seed
from pdt.scaling.base import FitFailure, Scale
from pdt.scaling.fitters import LogLinear

_FIT_SCALES = [Scale(n=n, d=20 * n) for n in [1e6, 3e6, 1e7, 3e7]]
_TARGET = Scale(n=1e9, d=20e9)
_RECIPES = ["recipe_a", "recipe_b", "recipe_c"]


def _make_synthetic(seed: int = 0) -> SyntheticOracle:
    rng = np.random.default_rng(seed)
    return SyntheticOracle(recipes=_RECIPES, scales=_FIT_SCALES, target_scale=_TARGET, rng=rng)


def _sigma2(_scale: Scale) -> float:
    return 0.01


# ---------------------------------------------------------------------------
# _BumpOracle -- deterministic, exact control over bias, for stopping/
# abstention-rule tests. Same generative shape as pdt.bai.oracle.SyntheticOracle,
# but with hand-chosen (not randomly drawn) parameters.
# ---------------------------------------------------------------------------


class _BumpOracle:
    def __init__(self, scales, target, params, sigma):
        self._scales = scales
        self._target = target
        self._params = params
        self._sigma = sigma

    def _h(self, recipe, scale):
        p = self._params[recipe]
        s_max = max(s.n for s in self._scales)
        s_star = self._target.n
        if scale.n <= s_max:
            return 0.0
        x = (scale.n - s_max) / (s_star - s_max)
        return p["bias_at_target"] * float(np.clip(x, 0.0, 1.0))

    def _true_mean(self, recipe, scale):
        p = self._params[recipe]
        return p["a"] + p["b"] * np.log(scale.n) + self._h(recipe, scale)

    def pull(self, recipe, scale, seed):
        mean = self._true_mean(recipe, scale)
        draw_rng = np.random.default_rng(_stable_seed(recipe, scale.n, scale.d, seed))
        return float(mean + draw_rng.normal(0.0, self._sigma))

    def cost(self, scale):
        return scale.compute

    def available_scales(self):
        return list(self._scales)


_BUMP_SCALES = [Scale(n=n, d=20 * n) for n in [1e6, 3e6, 1e7]]
# Deliberately close to the fit ladder (3x beyond the largest fit scale,
# not 100x) -- a far target amplifies extrapolation leverage enough that
# the stopping statistic's v_hat needs many thousands of adaptive rounds
# to shrink below any reasonable epsilon_0, found by direct probing while
# building this test (see docs/decisions.md); a closer target keeps these
# tests fast without changing what property they check.
_BUMP_TARGET = Scale(n=3e7, d=20 * 3e7)


# ---------------------------------------------------------------------------
# Baselines -- all run to completion on the real SyntheticOracle.
# ---------------------------------------------------------------------------


def test_single_scale_runs_to_completion():
    oracle = _make_synthetic(seed=1)
    res = single_scale_recommendation(oracle, _RECIPES, _FIT_SCALES[-1], n_replicates=3)
    assert isinstance(res, SelectionResult)
    assert res.outcome == "decided"
    assert res.recipe in _RECIPES
    assert res.compute_spent > 0
    assert res.n_pulls == 3 * len(_RECIPES)


def test_uniform_allocation_runs_to_completion():
    oracle = _make_synthetic(seed=2)
    budget = 50 * sum(s.compute for s in _FIT_SCALES)
    res = uniform_allocation(oracle, _RECIPES, _FIT_SCALES, _TARGET, compute_budget=budget)
    assert res.outcome == "decided"
    assert res.recipe in _RECIPES
    assert res.compute_spent >= budget


def test_uniform_allocation_covers_every_recipe_even_with_few_expensive_scales():
    # Regression test, pinning the exact scenario that broke while
    # building P3-05's real DataDecide replay: many recipes (25, like
    # real DataDecide) plus a scale ladder whose priciest scale costs
    # orders of magnitude more than the cheapest. Two compounding
    # mistakes were found there (see docs/decisions.md): the calling
    # code sized `compute_budget` as a small fixed multiple of *one*
    # recipe's own ladder cost rather than scaling it with the number of
    # recipes ("equal compute per arm" needs a per-arm-scaled budget, or
    # there just isn't enough to go around regardless of pull order --
    # fixed here by using `len(many_recipes) * ...`); and, separately,
    # `uniform_allocation`'s own (recipe, scale) pull order used to be
    # recipe-major, which for a too-small budget left every later recipe
    # with zero pulls at all (not just incomplete ones) while the first
    # few recipes used up the whole budget on their own ladders --
    # `_fit_recipe` then raised `FitFailure` instead of a usable result.
    # Now scale-major, cheapest-first: this test checks the fixed
    # function actually completes end-to-end on the real configuration
    # that used to raise.
    many_recipes = [f"r{i}" for i in range(25)]
    rng = np.random.default_rng(0)
    oracle = SyntheticOracle(
        recipes=many_recipes, scales=_FIT_SCALES, target_scale=_TARGET, rng=rng
    )
    skewed_scales = [
        Scale(n=1e6, d=2e7),
        Scale(n=1e7, d=2e8),
        Scale(n=1e8, d=2e9),
        Scale(n=1e9, d=4e11),  # far off the Chinchilla-optimal D/N ratio,
        # deliberately: this one scale costs orders of magnitude more
        # than the other three, matching what made this a real bug on
        # real DataDecide data (some of its real (N, D) pairs are
        # similarly far from compute-optimal).
    ]
    budget = 6 * len(many_recipes) * sum(s.compute for s in skewed_scales)
    res = uniform_allocation(oracle, many_recipes, skewed_scales, _TARGET, compute_budget=budget)
    assert res.outcome == "decided"
    assert res.recipe in many_recipes
    assert set(res.certificate["predictions"]) == set(many_recipes)


def test_fixed_ladder_extrapolation_runs_to_completion():
    oracle = _make_synthetic(seed=3)
    res = fixed_ladder_extrapolation(oracle, _RECIPES, _FIT_SCALES, _TARGET, n_replicates=2)
    assert res.outcome == "decided"
    assert res.recipe in _RECIPES
    assert res.n_pulls == 2 * len(_RECIPES) * len(_FIT_SCALES)
    assert set(res.certificate["predictions"]) == set(_RECIPES)


def test_successive_halving_over_scales_runs_to_completion():
    oracle = _make_synthetic(seed=4)
    res = successive_halving_over_scales(oracle, _RECIPES, _FIT_SCALES, _TARGET, n_replicates=2)
    assert res.outcome == "decided"
    assert res.recipe in _RECIPES
    assert len(res.certificate["final_survivors"]) == 1


def test_successive_halving_survives_a_single_recipe_instance():
    # len(recipes) == 1: the elimination loop's "already down to one
    # survivor" break path.
    oracle = _make_synthetic(seed=5)
    res = successive_halving_over_scales(oracle, ["recipe_a"], _FIT_SCALES, _TARGET)
    assert res.recipe == "recipe_a"
    assert res.certificate["final_survivors"] == ["recipe_a"]


def test_baselines_never_abstain():
    oracle = _make_synthetic(seed=6)
    budget = 10 * sum(s.compute for s in _FIT_SCALES)
    results = [
        single_scale_recommendation(oracle, _RECIPES, _FIT_SCALES[-1]),
        uniform_allocation(oracle, _RECIPES, _FIT_SCALES, _TARGET, compute_budget=budget),
        fixed_ladder_extrapolation(oracle, _RECIPES, _FIT_SCALES, _TARGET),
        successive_halving_over_scales(oracle, _RECIPES, _FIT_SCALES, _TARGET),
    ]
    assert all(r.outcome == "decided" for r in results)


# ---------------------------------------------------------------------------
# _beta -- the anytime-valid threshold, matching test_theorem4.py exactly.
# ---------------------------------------------------------------------------


def test_beta_matches_hand_formula():
    assert _beta(1, 0.05) == pytest.approx(0.05 / 2)
    assert _beta(10, 0.05) == pytest.approx(0.05 / 110)


# ---------------------------------------------------------------------------
# _assert_design_identified / _fit_recipe / _largest_observed_scale
# ---------------------------------------------------------------------------


def test_assert_design_identified_rejects_too_few_distinct_scales():
    with pytest.raises(FitFailure):
        _assert_design_identified(2, "LogLinear", [_BUMP_SCALES[0]])


def test_assert_design_identified_accepts_exactly_n_params_plus_one():
    _assert_design_identified(2, "LogLinear", _BUMP_SCALES[:3])  # must not raise


def test_fit_recipe_raises_via_the_same_path_solve_uses():
    # A direct check that _fit_recipe (what every method in this module
    # actually calls) surfaces the design-identifiability failure, not
    # just the standalone helper in isolation.
    with pytest.raises(FitFailure):
        _fit_recipe(LogLinear, _BUMP_SCALES[:1], [0.5], _BUMP_TARGET)


def test_largest_observed_scale_raises_when_no_scale_is_shared():
    with pytest.raises(FitFailure):
        _largest_observed_scale({"a": [_BUMP_SCALES[0]], "b": [_BUMP_SCALES[1]]}, _BUMP_SCALES)


def test_largest_observed_scale_picks_the_largest_common_one():
    scales_by_recipe = {"a": list(_BUMP_SCALES), "b": _BUMP_SCALES[:2]}
    result = _largest_observed_scale(scales_by_recipe, _BUMP_SCALES)
    assert result == _BUMP_SCALES[1]


# ---------------------------------------------------------------------------
# extrapolation_track_and_stop -- solvable regime: certifies correctly.
# ---------------------------------------------------------------------------


def test_ets_certifies_correctly_in_a_comfortably_solvable_instance():
    params = {
        "good": {"a": 0.9, "b": 0.05, "bias_at_target": 0.0},
        "bad": {"a": 0.3, "b": 0.03, "bias_at_target": 0.0},
    }
    oracle = _BumpOracle(_BUMP_SCALES, _BUMP_TARGET, params, sigma=0.05)
    eta = {"good": 0.02, "bad": 0.02}
    res = extrapolation_track_and_stop(
        oracle,
        ["good", "bad"],
        _BUMP_SCALES,
        _BUMP_TARGET,
        delta=0.05,
        eta=eta,
        sigma2=_sigma2,
        model_factory=LogLinear,
        epsilon_0=0.02,
        max_rounds=400,
        solver_n_restarts=1,
        solver_n_iter=40,
    )
    assert res.outcome == "certified"
    assert res.recipe == "good"
    assert res.compute_spent > 0
    assert res.n_pulls > 0


def test_ets_never_certifies_when_gap_is_small_and_eta_covers_it():
    # Theorem 2 Part B's exact construction: both recipes agree on every
    # observed (fit) scale; the ONLY difference is a bump visible solely
    # at the target scale, which is never observed. No amount of data can
    # resolve this -- the algorithm must not falsely certify.
    #
    # `min_pulls_per_pair=150` front-loads enough warm-up replicates that
    # v_hat is already small enough for c_t <= epsilon_0 right after
    # warm-up (round 1) -- found by direct probing while building this
    # test (see docs/decisions.md): with the default single warm-up pull,
    # v_hat's early trajectory actually *rises* for a few hundred rounds
    # before its asymptotic 1/n decay takes over (a small-sample HC0
    # sandwich-estimator artifact), making natural convergence here take
    # thousands of rounds -- true, but far too slow for a unit test, and
    # unrelated to the property this test checks (the rule never falsely
    # certifies), hence the heavier warm-up rather than a larger
    # max_rounds.
    params = {
        "a": {"a": 0.6, "b": 0.02, "bias_at_target": 0.0},
        "b": {"a": 0.6, "b": 0.02, "bias_at_target": 0.15},
    }
    oracle = _BumpOracle(_BUMP_SCALES, _BUMP_TARGET, params, sigma=0.05)
    eta = {"a": 0.15, "b": 0.15}
    res = extrapolation_track_and_stop(
        oracle,
        ["a", "b"],
        _BUMP_SCALES,
        _BUMP_TARGET,
        delta=0.05,
        eta=eta,
        sigma2=_sigma2,
        model_factory=LogLinear,
        epsilon_0=0.06,
        max_rounds=5,
        solver_n_restarts=1,
        solver_n_iter=40,
        min_pulls_per_pair=150,
    )
    assert res.outcome == "abstained"
    assert res.recipe in ("a", "b")
    assert res.certificate["reason"] == "bias floor"


def test_ets_abstention_recipe_matches_single_scale_fallback():
    params = {
        "a": {"a": 0.6, "b": 0.02, "bias_at_target": 0.0},
        "b": {"a": 0.6, "b": 0.02, "bias_at_target": 0.15},
    }
    oracle = _BumpOracle(_BUMP_SCALES, _BUMP_TARGET, params, sigma=0.05)
    eta = {"a": 0.15, "b": 0.15}
    res = extrapolation_track_and_stop(
        oracle,
        ["a", "b"],
        _BUMP_SCALES,
        _BUMP_TARGET,
        delta=0.05,
        eta=eta,
        sigma2=_sigma2,
        model_factory=LogLinear,
        epsilon_0=0.06,
        max_rounds=5,
        solver_n_restarts=1,
        min_pulls_per_pair=150,
        solver_n_iter=40,
    )
    assert res.outcome == "abstained"
    assert "fallback_scale" in res.certificate
    assert res.certificate["fallback_scale"] in _BUMP_SCALES


def test_ets_rejects_fewer_than_two_recipes():
    oracle = _BumpOracle(
        _BUMP_SCALES, _BUMP_TARGET, {"a": {"a": 0.5, "b": 0.0, "bias_at_target": 0.0}}, sigma=0.1
    )
    with pytest.raises(ValueError):
        extrapolation_track_and_stop(
            oracle,
            ["a"],
            _BUMP_SCALES,
            _BUMP_TARGET,
            delta=0.05,
            eta={"a": 0.1},
            sigma2=_sigma2,
            model_factory=LogLinear,
        )


def test_ets_rejects_too_few_candidate_scales():
    oracle = _BumpOracle(
        _BUMP_SCALES[:1],
        _BUMP_TARGET,
        {
            "a": {"a": 0.5, "b": 0.0, "bias_at_target": 0.0},
            "b": {"a": 0.4, "b": 0.0, "bias_at_target": 0.0},
        },
        sigma=0.1,
    )
    with pytest.raises(ValueError):
        extrapolation_track_and_stop(
            oracle,
            ["a", "b"],
            _BUMP_SCALES[:1],  # only 1 scale, LogLinear (n_params=2) needs >= 3
            _BUMP_TARGET,
            delta=0.05,
            eta={"a": 0.1, "b": 0.1},
            sigma2=_sigma2,
            model_factory=LogLinear,
        )


def test_ets_rejects_missing_eta_entries():
    oracle = _BumpOracle(
        _BUMP_SCALES,
        _BUMP_TARGET,
        {
            "a": {"a": 0.5, "b": 0.0, "bias_at_target": 0.0},
            "b": {"a": 0.4, "b": 0.0, "bias_at_target": 0.0},
        },
        sigma=0.1,
    )
    with pytest.raises(ValueError):
        extrapolation_track_and_stop(
            oracle,
            ["a", "b"],
            _BUMP_SCALES,
            _BUMP_TARGET,
            delta=0.05,
            eta={"a": 0.1},  # missing "b"
            sigma2=_sigma2,
            model_factory=LogLinear,
        )


def test_ets_hits_max_rounds_and_abstains_rather_than_hanging():
    # A pathologically tiny round cap forces the "give up" path: the
    # impossible-regime instance (margin always deeply negative, so
    # Certified never triggers) with an unreachable epsilon_0 (so Abstain
    # never triggers either) guarantees the round cap is what ends the
    # run -- must still return a valid, non-crashing SelectionResult
    # rather than raising or looping forever.
    params = {
        "a": {"a": 0.6, "b": 0.02, "bias_at_target": 0.0},
        "b": {"a": 0.6, "b": 0.02, "bias_at_target": 0.15},
    }
    oracle = _BumpOracle(_BUMP_SCALES, _BUMP_TARGET, params, sigma=0.05)
    eta = {"a": 0.15, "b": 0.15}
    res = extrapolation_track_and_stop(
        oracle,
        ["a", "b"],
        _BUMP_SCALES,
        _BUMP_TARGET,
        delta=0.05,
        eta=eta,
        sigma2=_sigma2,
        model_factory=LogLinear,
        epsilon_0=1e-9,  # unreachable in a handful of rounds
        max_rounds=len(_BUMP_SCALES) * 2 + 1,  # barely past warm-up
        solver_n_restarts=1,
        solver_n_iter=50,
    )
    assert res.outcome == "abstained"
    assert res.certificate["reason"] == "max_rounds exhausted without certifying or abstaining"
