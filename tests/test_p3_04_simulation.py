"""Tests for experiments/p3_04_simulation.py -- see plan/04-phase3-algorithm.md
task P3-04. Regression tests for PR #30's review: make_instance's
eta_assumed used to zero out the re-resolved winner's own bias, even when
that bias was exactly what made it the winner.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.p3_04_simulation import make_instance


def test_eta_assumed_reflects_the_winners_own_bias_when_it_flipped_the_ranking():
    # Regression for PR #30's review: reproduced with the reviewer's own
    # parameters. A large eta_level can make a recipe's own target-only
    # bump big enough to overtake the original leader and become the
    # re-resolved k_star -- eta_assumed for THAT recipe must still equal
    # its own bias magnitude, not be zeroed out just because it won.
    rng = np.random.default_rng(0)
    instance, recipes, k_star, eta_assumed = make_instance(rng, 3, "well_separated", "large")

    # Pin the exact reviewer-reported scenario: winner r2 with a
    # substantial (~0.623) own bias -- if this ever stops reproducing,
    # the assertion below would otherwise pass vacuously on a seed where
    # the winner's bias happens to be 0.
    assert k_star == "r2"
    assert eta_assumed[k_star] == pytest.approx(0.6228505534223818)
    assert eta_assumed[k_star] == abs(instance.params[k_star]["bias_at_target"])


def test_eta_assumed_matches_bias_at_target_for_every_recipe():
    rng = np.random.default_rng(1)
    instance, recipes, k_star, eta_assumed = make_instance(rng, 5, "well_separated", "medium")

    for r in recipes:
        assert eta_assumed[r] == abs(instance.params[r]["bias_at_target"])


def test_eta_assumed_is_zero_at_eta_level_zero_for_every_recipe():
    rng = np.random.default_rng(2)
    _instance, recipes, _k_star, eta_assumed = make_instance(rng, 4, "well_separated", "none")

    for r in recipes:
        assert eta_assumed[r] == 0.0


def test_eta_assumed_is_nonnegative():
    rng = np.random.default_rng(3)
    for eta_level in ("none", "small", "medium", "large"):
        _instance, recipes, _k_star, eta_assumed = make_instance(rng, 3, "reversing", eta_level)
        for r in recipes:
            assert eta_assumed[r] >= 0.0
