"""Tests for experiments/p3_06_eta_sensitivity.py -- see
plan/04-phase3-algorithm.md task P3-06. Regression test for PR #32's
review: _TwoArmOracle's observation noise used to depend only on
(recipe, scale, seed), never on which TRIAL was pulling, so "fresh"
oracle instances across trials drew identical noise.
"""

from __future__ import annotations

from experiments.p3_06_eta_sensitivity import _TARGET, _TwoArmOracle, estimate_eta_plugin


def test_two_arm_oracle_pulls_are_independent_across_trial_salts():
    a = _TwoArmOracle(trial_salt=0)
    b = _TwoArmOracle(trial_salt=1)
    assert a.pull("leader", _TARGET, seed=0) != b.pull("leader", _TARGET, seed=0)


def test_two_arm_oracle_pulls_are_deterministic_given_the_same_trial_salt():
    a = _TwoArmOracle(trial_salt=5)
    b = _TwoArmOracle(trial_salt=5)
    assert a.pull("leader", _TARGET, seed=0) == b.pull("leader", _TARGET, seed=0)


def test_eta_plugin_estimates_vary_across_independent_trials():
    # Regression for PR #32's review, reproduced in spirit: three fresh
    # oracle instances (one per trial) must NOT give identical plug-in
    # eta estimates -- an earlier version of this class made every
    # "fresh" oracle draw the exact same observation noise, so the
    # purported plug-in sampling variability this experiment exists to
    # measure was exactly zero.
    estimates = [
        estimate_eta_plugin(_TwoArmOracle(trial_salt=i), "leader", n_replicates=3) for i in range(3)
    ]
    assert len(set(estimates)) == 3


def test_two_arm_oracle_underlying_means_are_unaffected_by_trial_salt():
    # trial_salt changes the NOISE, not the true generative means -- the
    # controlled bias/gap structure this experiment depends on must stay
    # fixed regardless of which trial is running.
    a = _TwoArmOracle(trial_salt=0)
    b = _TwoArmOracle(trial_salt=7)
    assert a.true_value_at_target("leader") == b.true_value_at_target("leader")
    assert a.true_value_at_target("underdog") == b.true_value_at_target("underdog")
