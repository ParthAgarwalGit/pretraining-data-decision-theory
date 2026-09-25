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
    _certification_radius,
    _fit_recipe,
    _largest_observed_scale,
    _pair_beta,
    _unmet_support_conditions,
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
# _pair_beta / _certification_radius -- second-round review of PR #29: a
# known-variance Chernoff radius with simultaneous round AND ordered-pair
# control, replacing the invalid HC0-variance + Student-t construction.
# ---------------------------------------------------------------------------


def test_pair_beta_splits_the_round_budget_over_ordered_pairs():
    assert _pair_beta(1, 0.05, 2) == pytest.approx(_beta(1, 0.05) / 2)
    assert _pair_beta(3, 0.05, 4) == pytest.approx(_beta(3, 0.05) / 12)


def test_pair_beta_union_over_rounds_and_ordered_pairs_is_at_most_delta():
    # sum_t sum_{ordered pairs} beta = delta * sum_t 1/(t(t+1)) = delta * (1 - 1/T)
    delta, k = 0.05, 5
    total = sum(_pair_beta(t, delta, k) * k * (k - 1) for t in range(1, 100_001))
    assert total <= delta


def test_certification_radius_matches_the_chernoff_formula():
    v_sum, beta = 0.02, 0.005
    assert _certification_radius(v_sum, beta) == pytest.approx(
        float(np.sqrt(2 * v_sum * np.log(1 / beta)))
    )


def test_certification_radius_is_infinite_for_infinite_variance():
    assert _certification_radius(float("inf"), 0.005) == float("inf")


# ---------------------------------------------------------------------------
# The reviewers' counterexamples: first check (max_rounds=1), constant true
# means so the correct arm is known, 200 independent oracle seeds.
# ---------------------------------------------------------------------------


class _ConstantMeanOracle:
    """Arm means constant in scale (so a LogLinear fit is exactly specified,
    zero bias), independent N(0, sigma^2) noise, one normal variate per pull
    from the oracle's own default_rng(seed) -- the reviewer's setup."""

    def __init__(self, means, sigma, seed):
        self._means = means
        self._sigma = sigma
        self._rng = np.random.default_rng(seed)

    def pull(self, recipe, scale, seed):
        return float(self._means[recipe] + self._rng.normal(0.0, self._sigma))

    def cost(self, scale):
        return 1.0

    def available_scales(self):
        return []


def _first_check_outcomes(fit_ns, target_n, variance_mode, n_seeds=200):
    means = {"a": 0.501, "b": 0.5}  # "a" is truly (barely) better
    scales = [Scale(n=n, d=1.0) for n in fit_ns]
    outcomes = []
    for seed in range(n_seeds):
        res = extrapolation_track_and_stop(
            _ConstantMeanOracle(means, 0.05, seed),
            ["a", "b"],
            scales,
            Scale(n=target_n, d=1.0),
            delta=0.01,
            eta={"a": 0.0, "b": 0.0},
            sigma2=lambda s: 0.0025,
            model_factory=LogLinear,
            cost=lambda s: 1.0,
            max_rounds=1,
            solver_n_iter=1,
            variance_mode=variance_mode,
        )
        outcomes.append((res.outcome, res.recipe))
    return outcomes


@pytest.mark.parametrize(
    ("fit_ns", "target_n"),
    [
        pytest.param([1.0, 1.00001, 2.0], 2.001, id="high_leverage_near_target"),
        pytest.param([1.0, 2.0, 3.0], 4.0, id="evenly_spaced"),
    ],
)
def test_known_sigma2_certification_error_is_within_delta_on_the_reviewers_designs(
    fit_ns, target_n
):
    # Joint P[certified AND wrong] must be <= delta = 0.01 (the guarantee is on
    # the unconditional probability of certifying the wrong arm, not on the
    # error among certified runs). The high-leverage design certified the wrong
    # arm 98/200 times under the HC0 + Student-t rule.
    outcomes = _first_check_outcomes(fit_ns, target_n, "known_sigma2")
    n_wrong_certified = sum(1 for o, r in outcomes if o == "certified" and r != "a")
    assert n_wrong_certified <= 4  # 200 runs at delta=.01 -> mean 2 at the very worst


def test_hc0_heuristic_mode_never_reports_certified():
    outcomes = _first_check_outcomes([1.0, 1.00001, 2.0], 2.001, "hc0_heuristic", n_seeds=40)
    assert all(o != "certified" for o, _ in outcomes)
    # ...and it is heuristic for a reason: it does stop (recommends) on this design.
    assert any(o == "recommended" for o, _ in outcomes)


# ---------------------------------------------------------------------------
# Third review of PR #29: known noise alone must not enable "certified"
# ---------------------------------------------------------------------------


class _DeclaredNonlinear(LogLinear):
    """A LogLinear fit that declares itself nonlinear in its parameters (stand-in for the
    power-law fits, whose prediction is not an exactly Gaussian linear functional)."""

    linear_in_parameters = False


def _run_constant_means(gap, model_factory, certification, *, sigma=0.05, max_rounds=400):
    means = {"a": 0.5 + gap, "b": 0.5}
    scales = [Scale(n=n, d=1.0) for n in (1.0, 2.0, 3.0)]
    return extrapolation_track_and_stop(
        _ConstantMeanOracle(means, sigma, 7),
        ["a", "b"],
        scales,
        Scale(n=4.0, d=1.0),
        delta=0.1,
        eta={"a": 0.0, "b": 0.0},
        sigma2=lambda s: sigma**2,
        model_factory=model_factory,
        cost=lambda s: 1.0,
        max_rounds=max_rounds,
        solver_n_iter=5,
        certification=certification,
    )


def test_certified_is_returned_at_a_non_adaptive_check_for_a_linear_unconstrained_fit():
    res = _run_constant_means(0.9, LogLinear, "supported_only", sigma=0.01)
    assert res.outcome == "certified"
    assert res.recipe == "a"
    assert res.certificate["round"] == 1  # the first check, before any adaptive pull
    assert res.certificate["unmet_supported_conditions"] == []
    assert "proved" in res.certificate["guarantee"]
    assert [a[:2] for a in res.certificate["assumptions"]] == ["A1", "A2"]


def test_a_stop_after_adaptive_pulls_is_only_recommended_unless_the_caller_accepts_it():
    # gap .12 with sigma .05: not certifiable at the first check, so any stop comes after
    # adaptive tracking, where no adaptive confidence sequence is proved.
    supported = _run_constant_means(0.12, LogLinear, "supported_only")
    assumed = _run_constant_means(0.12, LogLinear, "assume_unproved_conditions")
    assert supported.certificate["round"] > 1
    assert supported.outcome == "recommended"
    assert any("adapted" in c for c in supported.certificate["unmet_supported_conditions"])
    assert "none" in supported.certificate["guarantee"]
    # the caller's explicit acceptance changes the LABEL, not the trajectory or the decision
    assert assumed.outcome == "certified"
    assert assumed.certificate["guarantee"].startswith("assumed_unproved")
    assert (assumed.recipe, assumed.n_pulls) == (supported.recipe, supported.n_pulls)


def test_a_nonlinear_fit_is_recommended_even_at_the_first_non_adaptive_check():
    res = _run_constant_means(0.9, _DeclaredNonlinear, "supported_only", sigma=0.01)
    assert res.certificate["round"] == 1
    assert res.outcome == "recommended"
    assert any("nonlinear" in c for c in res.certificate["unmet_supported_conditions"])
    accepted = _run_constant_means(
        0.9, _DeclaredNonlinear, "assume_unproved_conditions", sigma=0.01
    )
    assert accepted.outcome == "certified"
    assert accepted.certificate["guarantee"].startswith("assumed_unproved")


def test_unmet_conditions_names_an_active_parameter_bound_and_hc0():
    model = LogLinear()
    model._theta = np.array([0.5, 10.0])  # the slope sits on its upper bound
    assert not model.bounds_inactive()
    interior = LogLinear()
    interior._theta = np.array([0.5, 0.1])
    assert interior.bounds_inactive()
    unmet = _unmet_support_conditions({"a": model, "b": interior}, "known_sigma2", 0)
    assert len(unmet) == 1 and "bound is active" in unmet[0]
    assert _unmet_support_conditions({"b": interior}, "hc0_heuristic", 0)


def test_certification_mode_is_validated():
    with pytest.raises(ValueError, match="certification"):
        _run_constant_means(0.9, LogLinear, "bogus")


def test_variance_mode_is_validated():
    oracle = _ConstantMeanOracle({"a": 0.5, "b": 0.5}, 0.05, 0)
    with pytest.raises(ValueError, match="variance_mode"):
        extrapolation_track_and_stop(
            oracle,
            ["a", "b"],
            [Scale(n=n, d=1.0) for n in (1.0, 2.0, 3.0)],
            Scale(n=4.0, d=1.0),
            delta=0.05,
            eta={"a": 0.0, "b": 0.0},
            sigma2=lambda s: 0.0025,
            model_factory=LogLinear,
            variance_mode="bogus",
        )


# ---------------------------------------------------------------------------
# _assert_design_identified / _fit_recipe / _largest_observed_scale
# ---------------------------------------------------------------------------


def test_assert_design_identified_rejects_too_few_distinct_scales():
    with pytest.raises(FitFailure):
        _assert_design_identified(LogLinear, 2, "LogLinear", [_BUMP_SCALES[0]])


def test_assert_design_identified_accepts_exactly_n_params_plus_one():
    # must not raise -- 3 genuinely distinct N values identify LogLinear's
    # [1, log(N)] jacobian.
    _assert_design_identified(LogLinear, 2, "LogLinear", _BUMP_SCALES[:3])


def test_assert_design_identified_rejects_scales_differing_only_in_an_ignored_dim():
    # Regression for PR #29's review: LogLinear's fit ignores D entirely,
    # so 3 distinct (N, D) pairs sharing the same N pass the naive
    # count-only check while providing zero information about the slope
    # parameter -- the exact counterexample given (LogLinear observed
    # only at N=1, with varying D).
    same_n_scales = [Scale(n=1.0, d=1.0), Scale(n=1.0, d=2.0), Scale(n=1.0, d=3.0)]
    with pytest.raises(FitFailure, match="rank"):
        _assert_design_identified(LogLinear, 2, "LogLinear", same_n_scales)


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
