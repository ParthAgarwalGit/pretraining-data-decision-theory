"""Tests for pdt.analysis.rank_reversal -- see plan/02-phase1-datadecide.md P1-09."""

from __future__ import annotations

import pytest

from pdt.analysis import rank_reversal as rr

_SIZES = ["4M", "10M", "150M", "1B"]


# ---------------------------------------------------------------------------
# pair_effect_size()
# ---------------------------------------------------------------------------


def test_pair_effect_size_matches_hand_computation():
    result = rr.pair_effect_size(mu_a=0.6, mu_b=0.5, sigma2_a=0.001, sigma2_b=0.002, n_seeds=3)
    expected = (0.6 - 0.5) / ((0.001 + 0.002) / 3) ** 0.5
    assert result == pytest.approx(expected)


def test_pair_effect_size_is_negative_when_b_is_larger():
    result = rr.pair_effect_size(mu_a=0.4, mu_b=0.6, sigma2_a=0.001, sigma2_b=0.001, n_seeds=3)
    assert result < 0


def test_pair_effect_size_returns_none_for_zero_pooled_variance():
    assert rr.pair_effect_size(mu_a=0.5, mu_b=0.5, sigma2_a=0.0, sigma2_b=0.0, n_seeds=3) is None


# ---------------------------------------------------------------------------
# resolved_sign()
# ---------------------------------------------------------------------------


def test_resolved_sign_positive_above_threshold():
    assert rr.resolved_sign(2.5) == 1


def test_resolved_sign_negative_above_threshold():
    assert rr.resolved_sign(-2.5) == -1


def test_resolved_sign_none_below_threshold():
    assert rr.resolved_sign(0.5) is None
    assert rr.resolved_sign(-0.5) is None


def test_resolved_sign_none_for_none_input():
    assert rr.resolved_sign(None) is None


def test_resolved_sign_respects_custom_threshold():
    assert rr.resolved_sign(1.5, threshold=2.0) is None
    assert rr.resolved_sign(2.5, threshold=2.0) == 1


def test_resolved_sign_exactly_at_threshold_resolves():
    assert rr.resolved_sign(1.0, threshold=1.0) == 1


# ---------------------------------------------------------------------------
# classify_pair_trajectory()
# ---------------------------------------------------------------------------


def test_classify_stable_when_all_resolved_signs_agree():
    result = rr.classify_pair_trajectory(_SIZES, {"4M": 1, "150M": 1, "1B": 1})
    assert result["classification"] == "stable"
    assert result["crossing_size"] is None
    assert result["n_resolved_sizes"] == 3


def test_classify_within_noise_when_nothing_resolves():
    result = rr.classify_pair_trajectory(_SIZES, {})
    assert result["classification"] == "within_noise"
    assert result["crossing_size"] is None
    assert result["n_resolved_sizes"] == 0


def test_classify_reversing_when_resolved_signs_disagree():
    result = rr.classify_pair_trajectory(_SIZES, {"4M": -1, "10M": -1, "1B": 1})
    assert result["classification"] == "reversing"
    assert result["n_resolved_sizes"] == 3


def test_classify_reversing_crossing_size_is_the_last_reversal_boundary():
    # Resolved at 4M(-1), 10M(-1), 150M(+1), 1B(+1) -- the reversal
    # happens between 10M and 150M, so crossing_size should be 150M (the
    # first size on the "new" side of that boundary).
    result = rr.classify_pair_trajectory(
        _SIZES + ["530M"], {"4M": -1, "10M": -1, "150M": 1, "1B": 1}
    )
    assert result["classification"] == "reversing"
    assert result["crossing_size"] == "150M"


def test_classify_ignores_unresolved_sizes():
    # 10M is absent (within noise there) -- classification should be based
    # only on 4M and 1B, which agree -> stable.
    result = rr.classify_pair_trajectory(_SIZES, {"4M": 1, "1B": 1})
    assert result["classification"] == "stable"


def test_classify_single_resolved_size_is_stable_not_reversing():
    result = rr.classify_pair_trajectory(_SIZES, {"1B": 1})
    assert result["classification"] == "stable"
    assert result["n_resolved_sizes"] == 1


def test_classify_multiple_reversals_picks_the_boundary_closest_to_the_end():
    # -1, +1, -1, +1 across the ladder -- three crossings; the function
    # should report the LAST one (closest to the target scale).
    result = rr.classify_pair_trajectory(_SIZES, {"4M": -1, "10M": 1, "150M": -1, "1B": 1})
    assert result["classification"] == "reversing"
    assert result["crossing_size"] == "1B"
