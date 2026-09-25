"""Tests for experiments/p3_05_replay.py -- see plan/04-phase3-algorithm.md
task P3-05. Regression tests for PR #31's review: baselines bypassing the
target-scale leakage guard, only the first _N_REAL_SEEDS indices getting
bootstrap-resampled, and _CyclingOracle silently recycling non-independent
observations.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.p3_05_replay import (
    _N_REAL_SEEDS,
    _bootstrap_baseline,
    _CyclingOracle,
    _ReplicatePoolExhaustedError,
    _ReseededOracle,
)
from pdt.bai.guarded_oracle import GuardedOracle, TargetScaleLeakageError
from pdt.scaling.base import Scale

_FIT_SCALE = Scale(n=1e6, d=2e7)
_TARGET = Scale(n=1e9, d=2e10)


class _FakeOracle:
    """A minimal PullOracle over a fixed, small real-seed pool, plus an
    unbounded "pseudo" fallback for seed indices past that pool --
    enough to exercise _ReseededOracle/_CyclingOracle without needing
    real DataDecide data."""

    def __init__(self, n_real_seeds: int = _N_REAL_SEEDS):
        self._n_real_seeds = n_real_seeds
        self.pulls: list[tuple[str, float, int]] = []

    def pull(self, recipe: str, scale: Scale, seed: int) -> float:
        self.pulls.append((recipe, scale.n, seed))
        if seed >= self._n_real_seeds:
            raise IndexError(f"seed {seed} past the real+pseudo pool")
        return float(seed) + (0.0 if scale != _TARGET else 100.0)

    def cost(self, scale: Scale) -> float:
        return scale.compute

    def available_scales(self) -> list[Scale]:
        return [_FIT_SCALE, _TARGET]


# ---------------------------------------------------------------------------
# _ReseededOracle -- must remap EVERY requested seed index, not just the
# first _N_REAL_SEEDS.
# ---------------------------------------------------------------------------


def test_reseeded_oracle_remaps_indices_beyond_n_real_seeds():
    inner = _FakeOracle()
    rng = np.random.default_rng(0)
    reseeded = _ReseededOracle(inner, rng)

    # Seed indices well past _N_REAL_SEEDS (e.g. 5) must still resolve to
    # a mapped real seed in [0, _N_REAL_SEEDS), never fall through
    # unchanged to the inner oracle (which would raise IndexError for
    # seed=5 on a 3-real-seed pool).
    value = reseeded.pull("recipe", _FIT_SCALE, seed=5)
    assert value in {0.0, 1.0, 2.0}


def test_reseeded_oracle_is_consistent_for_the_same_seed_within_one_instance():
    inner = _FakeOracle()
    rng = np.random.default_rng(0)
    reseeded = _ReseededOracle(inner, rng)

    v1 = reseeded.pull("recipe", _FIT_SCALE, seed=7)
    v2 = reseeded.pull("recipe", _FIT_SCALE, seed=7)
    assert v1 == v2


def test_reseeded_oracle_never_leaves_an_index_unmapped():
    # Regression for PR #31's review: an earlier version only pre-built a
    # map for seed in range(_N_REAL_SEEDS) and fell back to the RAW seed
    # for anything else (dict.get(seed, seed)), which for seed >=
    # _N_REAL_SEEDS reached the inner oracle's own pseudo-replicate
    # fallback -- a fixed, never-resampled value across every bootstrap
    # replicate. Every one of many distinct high seed indices used here
    # must resolve inside [0, _N_REAL_SEEDS).
    inner = _FakeOracle()
    rng = np.random.default_rng(0)
    reseeded = _ReseededOracle(inner, rng)

    for seed in range(3, 20):
        value = reseeded.pull("recipe", _FIT_SCALE, seed=seed)
        assert value in {0.0, 1.0, 2.0}


# ---------------------------------------------------------------------------
# _bootstrap_baseline -- must run against a GUARDED oracle, not the raw one.
# ---------------------------------------------------------------------------


def _target_pulling_baseline(oracle, recipes, fit_scales, target):
    # A baseline that (incorrectly) tries to look at target-scale data --
    # must be blocked by the guard before returning anything.
    oracle.pull(recipes[0], target, seed=0)
    raise AssertionError("should never reach here -- the guard must raise first")


def test_bootstrap_baseline_blocks_a_target_pulling_baseline():
    # Regression for PR #31's review, P1: _bootstrap_baseline used to be
    # called with the raw, unguarded oracle, so a baseline that leaked
    # target-scale data would silently succeed instead of raising. Pass
    # a GuardedOracle-wrapped fake explicitly, matching what
    # experiments/p3_05_replay.py's _run_task now does.
    inner = _FakeOracle()
    guarded = GuardedOracle(inner, _TARGET)
    rng = np.random.default_rng(0)

    with pytest.raises(TargetScaleLeakageError):
        _bootstrap_baseline(
            _target_pulling_baseline,
            guarded,
            ["r0"],
            [_FIT_SCALE],
            _TARGET,
            winner="r0",
            n_bootstrap=1,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# _CyclingOracle -- must raise on genuine pool exhaustion, never recycle.
# ---------------------------------------------------------------------------


def test_cycling_oracle_raises_instead_of_recycling_on_pool_exhaustion():
    # Regression for PR #31's review, P1: an earlier version caught the
    # inner oracle's IndexError and silently returned an already-observed
    # value (seed % n_real_seeds) as if it were fresh data.
    inner = _FakeOracle(n_real_seeds=3)
    cycling = _CyclingOracle(inner, n_real_seeds=3)

    with pytest.raises(_ReplicatePoolExhaustedError):
        cycling.pull("recipe", _FIT_SCALE, seed=3)


def test_cycling_oracle_passes_through_within_the_real_pool():
    inner = _FakeOracle(n_real_seeds=3)
    cycling = _CyclingOracle(inner, n_real_seeds=3)

    assert cycling.pull("recipe", _FIT_SCALE, seed=0) == 0.0
    assert cycling.pull("recipe", _FIT_SCALE, seed=2) == 2.0
