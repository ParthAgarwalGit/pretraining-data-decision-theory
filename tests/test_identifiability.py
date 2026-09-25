"""Tests for pdt.theory.identifiability."""

from __future__ import annotations

import numpy as np
import pytest

from pdt.theory.identifiability import prediction_influence_weights, target_in_row_space


def test_target_in_span_of_rows():
    rows = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert target_in_row_space(rows, np.array([3.0, -2.0]))


def test_target_outside_span_of_rank_deficient_rows():
    rows = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
    assert not target_in_row_space(rows, np.array([1.0, 0.0]))
    assert target_in_row_space(rows, np.array([0.5, 1.0]))


def test_small_missing_component_is_still_detected():
    # The reviewer's J_target=[1, .05] against rows spanning only [1, 0]:
    # a 5% missing component, well under the old 10% relative tolerance.
    rows = np.array([[1.0, 0.0], [2.0, 0.0]])
    assert not target_in_row_space(rows, np.array([1.0, 0.05]))


def test_badly_scaled_columns_do_not_hide_or_invent_a_missing_direction():
    # Column scales differing by 1e12 (like E vs d/d alpha): identified either way.
    s = np.diag([1.0, 1e-12])
    rows = np.array([[1.0, 1e-12], [1.0, -1e-12]]) @ np.eye(2)
    assert target_in_row_space(rows, np.array([0.3, 2e-12]))
    assert target_in_row_space(s, np.array([1.0, 1e-12]))


def test_ill_conditioned_but_full_rank_rows_are_identified():
    rows = np.array([[1.0, 20.0], [1.0, 20.0 + 1e-4]])
    assert target_in_row_space(rows, np.array([1.0, 27.0]))


def test_zero_column_requires_target_free_of_that_parameter():
    rows = np.array([[1.0, 0.0], [2.0, 0.0]])
    assert target_in_row_space(rows, np.array([5.0, 0.0]))
    assert not target_in_row_space(rows, np.array([5.0, 1e-3]))


def test_zero_target_and_empty_design():
    assert target_in_row_space(np.zeros((0, 2)), np.zeros(2))
    assert not target_in_row_space(np.zeros((0, 2)), np.array([1.0, 0.0]))


def test_influence_weights_match_pinv_on_a_full_rank_design():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(6, 3))
    target = rng.normal(size=3)
    weights = prediction_influence_weights(rows, target)
    assert weights == pytest.approx(np.linalg.pinv(rows).T @ target, rel=1e-10)
    # prediction == g . y for any y consistent with the design
    theta = rng.normal(size=3)
    assert float(weights @ (rows @ theta)) == pytest.approx(float(target @ theta), rel=1e-10)


def test_influence_weights_survive_an_ill_conditioned_full_rank_design():
    xs = np.array([1.0, 1.0 + 1e-8, 1.0 + 2e-8])
    rows = np.column_stack([np.ones(3), xs])
    target = np.array([1.0, 2.0])
    weights = prediction_influence_weights(rows, target)
    assert weights is not None
    theta = np.array([0.3, -0.7])
    assert float(weights @ (rows @ theta)) == pytest.approx(float(target @ theta), rel=1e-6)


def test_influence_weights_fail_closed_when_a_target_direction_is_discarded():
    rows = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # never sees the 2nd parameter
    assert prediction_influence_weights(rows, np.array([1.0, 0.5])) is None
    assert prediction_influence_weights(rows, np.array([1.0, 0.0])) is not None
