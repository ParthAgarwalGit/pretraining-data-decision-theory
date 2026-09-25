"""Tests for experiments/p3_07_baseline_vs_misspecification.py -- see
plan/04-phase3-algorithm.md task P3-07. Regression test for PR #33's
review: the identical missing-trial-salt defect PR #32 fixed in
p3_06_eta_sensitivity.py's own _TwoArmOracle, here with an unused
_run_idx loop variable making every "repetition" replay the same dataset.
"""

from __future__ import annotations

from experiments.p3_07_baseline_vs_misspecification import _TARGET, _TwoArmOracle


def test_two_arm_oracle_pulls_are_independent_across_trial_salts():
    a = _TwoArmOracle(true_bias=0.1, trial_salt=0)
    b = _TwoArmOracle(true_bias=0.1, trial_salt=1)
    assert a.pull("leader", _TARGET, seed=0) != b.pull("leader", _TARGET, seed=0)


def test_two_arm_oracle_pulls_are_deterministic_given_the_same_trial_salt():
    a = _TwoArmOracle(true_bias=0.1, trial_salt=5)
    b = _TwoArmOracle(true_bias=0.1, trial_salt=5)
    assert a.pull("leader", _TARGET, seed=0) == b.pull("leader", _TARGET, seed=0)


def test_two_arm_oracle_repetitions_produce_distinct_datasets():
    # Regression, reproduced in spirit: an earlier version's caller loop
    # (`for _run_idx in range(n_runs): oracle = _TwoArmOracle(true_bias)`)
    # discarded run_idx entirely, so every one of n_runs "repetitions"
    # pulled from the identical dataset -- a 30-repetition accuracy curve
    # was really one single deterministic decision repeated 30 times, not
    # an error-rate estimate over noisy trials.
    draws = [
        _TwoArmOracle(true_bias=0.1, trial_salt=i).pull("underdog", _TARGET, seed=0)
        for i in range(10)
    ]
    assert len(set(draws)) == 10
