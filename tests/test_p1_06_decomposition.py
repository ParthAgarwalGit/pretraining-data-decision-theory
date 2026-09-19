"""Tests for the bootstrap-orchestration helpers in
experiments/p1_06_decomposition.py -- see plan/02-phase1-datadecide.md
task P1-06. These are regression tests for two bugs an external review
found in this module: forcing exact +1 correlation between recipes'
parametric bootstrap noise, and misaligning replicate ids when zipping two
recipes' successful-prediction lists by position instead of by id.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.p1_06_decomposition import (
    _ComboWork,
    _pairwise_difference_series,
    _parametric_draws_for_recipe,
)
from pdt.scaling.base import Scale


def _work_with_avg_trajectory(avg_trajectory: dict[str, list[tuple[Scale, float]]]) -> _ComboWork:
    return _ComboWork(
        fitter_name="PowerLawN",
        design_name="S_fit_le_150M",
        task="task-1",
        scheme="parametric_bootstrap",
        target_scale=Scale(n=1e9, d=2e10),
        avg_trajectory=avg_trajectory,
        seed_trajectory=None,
        sigma2_total={},
        k_star="recipe-a",
        mu_true={"recipe-a": 0.5, "recipe-b": 0.5},
        sigma2_target={"recipe-a": 0.0, "recipe-b": 0.0},
    )


# ---------------------------------------------------------------------------
# _parametric_draws_for_recipe() -- no more forced +1 correlation
# ---------------------------------------------------------------------------


def test_parametric_draws_for_recipe_are_independent_across_recipes():
    # Regression for "do not impose correlation +1": two recipes given
    # DIFFERENT rngs (as _run_one_combo now does, one per (recipe, b))
    # must not receive the same shared z sequence.
    scales = [(Scale(n=1e6, d=2e7), 0.3), (Scale(n=1e7, d=2e8), 0.4)]
    work = _work_with_avg_trajectory({"recipe-a": scales, "recipe-b": scales})

    draws_a = _parametric_draws_for_recipe(work, "recipe-a", np.random.default_rng(1))
    draws_b = _parametric_draws_for_recipe(work, "recipe-b", np.random.default_rng(2))

    assert len(draws_a) == len(draws_b) == 2
    assert draws_a != draws_b


def test_parametric_draws_for_recipe_one_per_scale():
    scales = [(Scale(n=1e6, d=2e7), 0.3), (Scale(n=1e7, d=2e8), 0.4), (Scale(n=1e8, d=2e9), 0.5)]
    work = _work_with_avg_trajectory({"recipe-a": scales})

    draws = _parametric_draws_for_recipe(work, "recipe-a", np.random.default_rng(0))

    assert len(draws) == 3


def test_parametric_draws_for_recipe_deterministic_given_same_rng_state():
    scales = [(Scale(n=1e6, d=2e7), 0.3)]
    work = _work_with_avg_trajectory({"recipe-a": scales})

    draws_1 = _parametric_draws_for_recipe(work, "recipe-a", np.random.default_rng(7))
    draws_2 = _parametric_draws_for_recipe(work, "recipe-a", np.random.default_rng(7))

    assert draws_1 == pytest.approx(draws_2)


# ---------------------------------------------------------------------------
# _pairwise_difference_series() -- align by replicate id, not list position
# ---------------------------------------------------------------------------


def test_pairwise_difference_series_matches_on_common_replicate_ids():
    k_star_preds = [(0, 1.0), (1, 2.0), (2, 3.0)]
    recipe_preds = [(0, 0.5), (1, 1.5), (2, 2.5)]

    result = _pairwise_difference_series(k_star_preds, recipe_preds)

    assert result == pytest.approx([0.5, 0.5, 0.5])


def test_pairwise_difference_series_handles_asymmetric_failures():
    # Regression for "align bootstrap IDs": k_star's fit failed at b=0
    # (absent from its list), recipe's fit failed at b=1 (absent from
    # its list). Zipping by position would pair k_star's b=1 with
    # recipe's b=0 (and k_star's b=2 with recipe's b=2, coincidentally
    # correct) -- pairing by id must use ONLY the truly common
    # replicate, b=2.
    k_star_preds = [(1, 20.0), (2, 30.0)]  # b=0 failed for k_star
    recipe_preds = [(0, 5.0), (2, 25.0)]  # b=1 failed for recipe

    result = _pairwise_difference_series(k_star_preds, recipe_preds)

    assert result == pytest.approx([30.0 - 25.0])


def test_pairwise_difference_series_empty_when_no_ids_in_common():
    result = _pairwise_difference_series([(0, 1.0)], [(1, 2.0)])
    assert result == []


# ---------------------------------------------------------------------------
# _variance_inflation() -- n/(n-1) calibration for the seed bootstrap only
# ---------------------------------------------------------------------------


def test_variance_inflation_is_n_over_n_minus_one_for_seed_bootstrap():
    from experiments.p1_06_decomposition import _variance_inflation

    seed_trajectory = {
        "recipe-a": [
            (Scale(n=1e6, d=2e7), [0.1, 0.2, 0.3]),
            (Scale(n=1e7, d=2e8), [0.2, 0.3, 0.4]),
        ],
        "recipe-b": [
            (Scale(n=1e6, d=2e7), [0.1, 0.2, 0.3]),
            (Scale(n=1e7, d=2e8), [0.2, 0.3, 0.4]),
        ],
    }
    work = _work_with_avg_trajectory({})
    work.scheme = "seed_bootstrap"
    work.seed_trajectory = seed_trajectory

    assert _variance_inflation(work) == pytest.approx(1.5)


def test_variance_inflation_is_one_for_parametric_bootstrap():
    from experiments.p1_06_decomposition import _variance_inflation

    assert _variance_inflation(_work_with_avg_trajectory({})) == 1.0


def test_variance_inflation_rejects_inconsistent_seed_counts():
    from experiments.p1_06_decomposition import _variance_inflation

    work = _work_with_avg_trajectory({})
    work.scheme = "seed_bootstrap"
    work.seed_trajectory = {
        "recipe-a": [(Scale(n=1e6, d=2e7), [0.1, 0.2, 0.3])],
        "recipe-b": [(Scale(n=1e6, d=2e7), [0.1, 0.2])],
    }
    with pytest.raises(ValueError, match="one common n"):
        _variance_inflation(work)
