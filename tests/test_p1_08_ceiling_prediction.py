"""Tests for the comparability helpers in
experiments/p1_08_ceiling_prediction.py -- see plan/02-phase1-datadecide.md
task P1-08. Regression tests for two bugs an external review found: the
predicted-vs-observed "gap" diagnostic mixed two different decision events
(best-arm selection vs all-pairs accuracy), and the single-scale baseline
used for the "beats single-scale" central claim was not compute-matched.
"""

from __future__ import annotations

import pytest

from experiments.p1_08_ceiling_prediction import (
    _PREDICTED_BUDGET_LABEL,
    _macro_average_optional,
    _matched_compute_single_scale,
    _observed_beats_matched,
    _observed_best_arm_accuracy_per_task,
)

_SCHEME = "seed_bootstrap"


# ---------------------------------------------------------------------------
# _observed_best_arm_accuracy_per_task() -- same decision event as the bound
# ---------------------------------------------------------------------------


def _p1_07_with(task_to_error_rate: dict[str, float | None]) -> dict:
    by_task = {
        task: {_SCHEME: ({"empirical_error_rate": rate} if rate is not None else {})}
        for task, rate in task_to_error_rate.items()
    }
    return {"by_fitter": {"PowerLawN": {"design-a": by_task}}}


def test_observed_best_arm_accuracy_is_one_minus_error_rate():
    p1_07 = _p1_07_with({"task-1": 0.2, "task-2": 0.5})
    result = _observed_best_arm_accuracy_per_task(
        "PowerLawN", "design-a", ["task-1", "task-2"], p1_07
    )
    assert result == pytest.approx({"task-1": 0.8, "task-2": 0.5})


def test_observed_best_arm_accuracy_none_when_empirical_error_rate_missing():
    p1_07 = _p1_07_with({"task-1": 0.2, "task-2": None})
    result = _observed_best_arm_accuracy_per_task(
        "PowerLawN", "design-a", ["task-1", "task-2"], p1_07
    )
    assert result["task-1"] == pytest.approx(0.8)
    assert result["task-2"] is None


def test_observed_best_arm_accuracy_none_when_cell_entirely_absent():
    p1_07 = _p1_07_with({"task-1": 0.2})
    result = _observed_best_arm_accuracy_per_task(
        "PowerLawN", "design-a", ["task-1", "missing-task"], p1_07
    )
    assert result["missing-task"] is None


def test_macro_average_optional_skips_nones():
    assert _macro_average_optional({"a": 0.8, "b": None, "c": 0.4}) == pytest.approx(0.6)


def test_macro_average_optional_none_when_all_missing():
    assert _macro_average_optional({"a": None, "b": None}) is None


# ---------------------------------------------------------------------------
# _matched_compute_single_scale() -- PR #18's P2: matched-compute baseline
# ---------------------------------------------------------------------------


def _p1_04_with(entry: dict) -> dict:
    return {"fitters": {"PowerLawN": {"design-a": entry}}}


def test_matched_compute_single_scale_reads_p1_04s_own_matched_baseline():
    p1_04 = _p1_04_with(
        {
            "matched_single_scale_accuracy_including_ties": 0.784,
            "matched_compute_out_of_range": False,
        }
    )
    accuracy, out_of_range = _matched_compute_single_scale("PowerLawN", "design-a", p1_04)
    assert accuracy == pytest.approx(0.784)
    assert out_of_range is False


def test_matched_compute_single_scale_flags_out_of_range_designs():
    # Regression for PR #18's P2: a design whose compute exceeds what
    # P1-03's single-scale ladder can interpolate must not silently fall
    # back to some other (compute-mismatched) baseline -- it must report
    # None and flag out_of_range so callers can't accidentally treat it
    # as a valid comparison.
    p1_04 = _p1_04_with(
        {
            "matched_single_scale_accuracy_including_ties": None,
            "matched_compute_out_of_range": True,
        }
    )
    accuracy, out_of_range = _matched_compute_single_scale("PowerLawN", "design-a", p1_04)
    assert accuracy is None
    assert out_of_range is True


# ---------------------------------------------------------------------------
# _observed_beats_matched() / budget labels -- second-round review of PR #21:
# a missing matched comparison is "unassessed", never a loss.
# ---------------------------------------------------------------------------


def test_observed_beats_matched_none_when_no_matched_baseline():
    assert _observed_beats_matched(0.9, None) is None


def test_observed_beats_matched_is_a_real_bool_otherwise():
    assert _observed_beats_matched(0.9, 0.8) is True
    assert _observed_beats_matched(0.7, 0.8) is False
    assert _observed_beats_matched(0.8, 0.8) is False  # ties are not wins


def test_predicted_comparison_budget_label_says_unmatched():
    assert _PREDICTED_BUDGET_LABEL.startswith("unmatched")
