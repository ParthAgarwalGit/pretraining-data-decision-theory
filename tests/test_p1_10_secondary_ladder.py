"""Tests for the bootstrap-orchestration helpers in
experiments/p1_10_secondary_ladder.py -- see plan/02-phase1-datadecide.md
task P1-10. Regression test for the bootstrap-ID-alignment defect PR #19's
review flagged in p1_06_decomposition.py and this module repeats: zipping
two recipes' successful-prediction lists by position instead of by
replicate id silently misaligns them when fits fail asymmetrically.
"""

from __future__ import annotations

import pytest

from experiments.p1_10_secondary_ladder import _pairwise_difference_series


def test_pairwise_difference_series_matches_on_common_replicate_ids():
    k_star_preds = [(0, 1.0), (1, 2.0), (2, 3.0)]
    k_other_preds = [(0, 0.5), (1, 1.5), (2, 2.5)]

    result = _pairwise_difference_series(k_star_preds, k_other_preds)

    assert result == pytest.approx([0.5, 0.5, 0.5])


def test_pairwise_difference_series_handles_asymmetric_failures():
    # k_star's fit failed at b=0 (absent from its list), k_other's failed
    # at b=1 (absent from its list). Zipping by position would pair
    # k_star's b=1 with k_other's b=0 -- pairing by id must use ONLY the
    # truly common replicate, b=2.
    k_star_preds = [(1, 20.0), (2, 30.0)]  # b=0 failed for k_star
    k_other_preds = [(0, 5.0), (2, 25.0)]  # b=1 failed for k_other

    result = _pairwise_difference_series(k_star_preds, k_other_preds)

    assert result == pytest.approx([30.0 - 25.0])


def test_pairwise_difference_series_empty_when_no_ids_in_common():
    assert _pairwise_difference_series([(0, 1.0)], [(1, 2.0)]) == []
