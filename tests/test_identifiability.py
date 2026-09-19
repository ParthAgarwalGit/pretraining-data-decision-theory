"""Tests for pdt.theory.identifiability."""

from __future__ import annotations

import numpy as np

from pdt.theory.identifiability import target_in_row_space


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
