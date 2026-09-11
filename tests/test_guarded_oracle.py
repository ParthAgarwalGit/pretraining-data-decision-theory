"""Tests for pdt.bai.guarded_oracle -- plan/04-phase3-algorithm.md P3-05's
leakage guard, checked directly rather than trusted by discipline."""

from __future__ import annotations

import pytest

from pdt.bai.guarded_oracle import GuardedOracle, TargetScaleLeakageError
from pdt.bai.oracle import SyntheticOracle
from pdt.scaling.base import Scale

_FIT_SCALES = [Scale(n=n, d=20 * n) for n in [1e6, 3e6, 1e7]]
_TARGET = Scale(n=1e9, d=20e9)


def _make_guarded() -> tuple[GuardedOracle, SyntheticOracle]:
    import numpy as np

    inner = SyntheticOracle(
        recipes=["a", "b"], scales=_FIT_SCALES, target_scale=_TARGET, rng=np.random.default_rng(0)
    )
    return GuardedOracle(inner, _TARGET), inner


def test_pull_at_a_fit_scale_passes_through():
    guarded, inner = _make_guarded()
    v_guarded = guarded.pull("a", _FIT_SCALES[0], seed=0)
    v_inner = inner.pull("a", _FIT_SCALES[0], seed=0)
    assert v_guarded == v_inner


def test_pull_at_the_target_scale_raises():
    guarded, _inner = _make_guarded()
    with pytest.raises(TargetScaleLeakageError):
        guarded.pull("a", _TARGET, seed=0)


def test_cost_at_a_fit_scale_passes_through():
    guarded, inner = _make_guarded()
    assert guarded.cost(_FIT_SCALES[0]) == inner.cost(_FIT_SCALES[0])


def test_cost_at_the_target_scale_raises():
    guarded, _inner = _make_guarded()
    with pytest.raises(TargetScaleLeakageError):
        guarded.cost(_TARGET)


def test_available_scales_excludes_the_target():
    guarded, _inner = _make_guarded()
    scales = guarded.available_scales()
    assert _TARGET not in scales
    assert set(scales) == set(_FIT_SCALES)


def test_available_scales_still_works_if_inner_oracle_never_listed_target():
    # A defensive check: even if the wrapped oracle's own available_scales()
    # never included the target in the first place (the common case), the
    # filter must not error or misbehave.
    guarded, _inner = _make_guarded()
    assert guarded.available_scales() == list(_FIT_SCALES)


def test_a_real_algorithm_cannot_pull_the_target_through_the_guard():
    # Integration-style check: extrapolation_track_and_stop must run to
    # completion using only the guarded oracle's fit scales -- it never
    # even attempts the target scale as a pull target, since
    # available_scales()/candidate_scales are what the caller passes in,
    # not queried from the oracle by the algorithm itself. This test
    # exists to document and pin that contract, not because ETS was ever
    # observed to violate it.
    from pdt.bai.ets import extrapolation_track_and_stop
    from pdt.scaling.fitters import LogLinear

    guarded, _inner = _make_guarded()
    res = extrapolation_track_and_stop(
        guarded,
        ["a", "b"],
        _FIT_SCALES,
        _TARGET,
        delta=0.2,
        eta={"a": 0.05, "b": 0.05},
        sigma2=lambda _s: 1e-4,
        model_factory=LogLinear,
        max_rounds=20,
        solver_n_restarts=1,
        solver_n_iter=20,
    )
    assert res.outcome in ("certified", "abstained")
