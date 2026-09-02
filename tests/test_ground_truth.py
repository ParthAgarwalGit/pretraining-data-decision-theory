"""Tests for pdt.analysis.ground_truth -- see plan/02-phase1-datadecide.md P1-02."""

from __future__ import annotations

import math

import polars as pl
import pytest

from pdt.analysis import ground_truth as gt

_SEEDS = ("s1", "s2", "s3")


def _rows(recipe, task, values, *, params_str="1B", metric_name="primary_metric", is_final=True):
    """One row per seed in `values` (up to len(_SEEDS))."""
    return [
        {
            "recipe": recipe,
            "params_str": params_str,
            "seed": seed,
            "task": task,
            "metric_name": metric_name,
            "metric_value": value,
            "is_final": is_final,
        }
        for seed, value in zip(_SEEDS, values, strict=False)
    ]


def _row(recipe, task, seed, value, **kwargs):
    """A single explicit-seed row, for the non_final / mismatched-scale tests."""
    return _rows(recipe, task, [value], **kwargs)[0] | {"seed": seed}


def test_identifies_the_winner_by_mean_across_seeds():
    rows = [
        *_rows("recipe-a", "task-1", [0.60, 0.62, 0.61]),
        *_rows("recipe-b", "task-1", [0.50, 0.51, 0.49]),
    ]
    df = pl.DataFrame(rows)

    result = gt.compute_ground_truth(df, "primary_metric", "1B")

    assert result["task-1"]["k_star"] == "recipe-a"
    assert result["task-1"]["mu"]["recipe-a"] == pytest.approx(0.61, abs=1e-9)
    assert result["task-1"]["mu"]["recipe-b"] == pytest.approx(0.50, abs=1e-9)


def test_gaps_and_delta_min_computed_relative_to_winner():
    rows = [
        *_rows("recipe-a", "task-1", [0.60, 0.60, 0.60]),
        *_rows("recipe-b", "task-1", [0.55, 0.55, 0.55]),
        *_rows("recipe-c", "task-1", [0.40, 0.40, 0.40]),
    ]
    df = pl.DataFrame(rows)

    result = gt.compute_ground_truth(df, "primary_metric", "1B")["task-1"]

    assert result["gaps"] == pytest.approx({"recipe-b": 0.05, "recipe-c": 0.20}, abs=1e-9)
    assert result["delta_min"] == pytest.approx(0.05, abs=1e-9)
    assert result["runner_up"] == "recipe-b"


def test_flags_ambiguous_when_gap_is_within_seed_noise():
    # Winner and runner-up separated by a tiny gap relative to their
    # seed-to-seed spread -- should be flagged ambiguous.
    rows = [
        *_rows("recipe-a", "task-1", [0.50, 0.60, 0.55]),
        *_rows("recipe-b", "task-1", [0.49, 0.59, 0.54]),
    ]
    df = pl.DataFrame(rows)

    result = gt.compute_ground_truth(df, "primary_metric", "1B")["task-1"]

    assert result["is_ambiguous"] is True
    assert result["effect_size"] < gt.AMBIGUOUS_EFFECT_SIZE_THRESHOLD


def test_does_not_flag_ambiguous_when_gap_dwarfs_seed_noise():
    rows = [
        *_rows("recipe-a", "task-1", [0.90, 0.91, 0.895]),
        *_rows("recipe-b", "task-1", [0.10, 0.11, 0.095]),
    ]
    df = pl.DataFrame(rows)

    result = gt.compute_ground_truth(df, "primary_metric", "1B")["task-1"]

    assert result["is_ambiguous"] is False
    assert result["effect_size"] > gt.AMBIGUOUS_EFFECT_SIZE_THRESHOLD


def test_handles_zero_variance_without_crashing():
    # Every seed identical for both recipes -- sd_seed is 0 for both, so
    # the pooled standard error is 0 and effect_size is undefined (None),
    # which must itself count as ambiguous rather than raising or dividing
    # by zero.
    rows = [
        *_rows("recipe-a", "task-1", [0.60, 0.60, 0.60]),
        *_rows("recipe-b", "task-1", [0.55, 0.55, 0.55]),
    ]
    df = pl.DataFrame(rows)

    result = gt.compute_ground_truth(df, "primary_metric", "1B")["task-1"]

    assert result["effect_size"] is None
    assert result["is_ambiguous"] is True


def test_each_task_handled_independently():
    rows = [
        *_rows("recipe-a", "task-1", [0.60, 0.60, 0.60]),
        *_rows("recipe-b", "task-1", [0.40, 0.40, 0.40]),
        *_rows("recipe-a", "task-2", [0.30, 0.30, 0.30]),
        *_rows("recipe-b", "task-2", [0.70, 0.70, 0.70]),
    ]
    df = pl.DataFrame(rows)

    result = gt.compute_ground_truth(df, "primary_metric", "1B")

    assert result["task-1"]["k_star"] == "recipe-a"
    assert result["task-2"]["k_star"] == "recipe-b"


def test_filters_to_the_requested_metric_and_scale():
    rows = [
        *_rows("recipe-a", "task-1", [0.90, 0.90, 0.90], params_str="1B"),
        *_rows("recipe-b", "task-1", [0.10, 0.10, 0.10], params_str="1B"),
        # Different scale: opposite winner, must not leak into the 1B result.
        *_rows("recipe-a", "task-1", [0.10, 0.10, 0.10], params_str="150M"),
        *_rows("recipe-b", "task-1", [0.90, 0.90, 0.90], params_str="150M"),
        # Different metric at 1B: also must not leak in.
        *_rows("recipe-a", "task-1", [0.10, 0.10, 0.10], metric_name="acc_raw"),
        *_rows("recipe-b", "task-1", [0.90, 0.90, 0.90], metric_name="acc_raw"),
    ]
    df = pl.DataFrame(rows)

    result = gt.compute_ground_truth(df, "primary_metric", "1B")["task-1"]

    assert result["k_star"] == "recipe-a"


def test_excludes_non_final_checkpoints():
    rows = [
        *_rows("recipe-a", "task-1", [0.60, 0.60, 0.60]),
        *_rows("recipe-b", "task-1", [0.40, 0.40, 0.40]),
        # An earlier, non-final checkpoint for recipe-b with a much higher
        # value -- must not be averaged in and flip the winner.
        _row("recipe-b", "task-1", "s0", 0.99, is_final=False),
    ]
    df = pl.DataFrame(rows)

    result = gt.compute_ground_truth(df, "primary_metric", "1B")["task-1"]

    assert result["k_star"] == "recipe-a"
    assert result["mu"]["recipe-b"] == pytest.approx(0.40, abs=1e-9)


def test_raises_on_unknown_metric_or_scale():
    df = pl.DataFrame(_rows("recipe-a", "task-1", [0.5, 0.5, 0.5]))

    with pytest.raises(ValueError, match="no rows"):
        gt.compute_ground_truth(df, "not_a_real_metric", "1B")

    with pytest.raises(ValueError, match="no rows"):
        gt.compute_ground_truth(df, "primary_metric", "999Z")


def test_effect_size_formula_directly():
    # delta_min=0.1, sd_a=0.03, sd_b=0.04, n_seeds=3
    # SE = sqrt((0.03^2 + 0.04^2)/3) = sqrt(0.0025/3) = sqrt(0.0008333...)
    expected = 0.1 / math.sqrt((0.03**2 + 0.04**2) / 3)
    assert gt._effect_size(0.1, 0.03, 0.04, 3) == pytest.approx(expected)


def test_effect_size_returns_none_for_zero_pooled_variance():
    assert gt._effect_size(0.1, 0.0, 0.0, 3) is None
