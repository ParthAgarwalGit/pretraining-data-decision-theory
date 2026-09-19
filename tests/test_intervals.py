"""Tests for pdt.analysis.intervals."""

from __future__ import annotations

import pytest

from pdt.analysis.intervals import clopper_pearson


def test_known_values():
    lo, hi = clopper_pearson(1, 20)
    assert lo == pytest.approx(0.00126, abs=2e-4)
    assert hi == pytest.approx(0.2487, abs=2e-3)


def test_edges():
    assert clopper_pearson(0, 20)[0] == 0.0
    assert clopper_pearson(20, 20)[1] == 1.0
    assert clopper_pearson(0, 20)[1] == pytest.approx(0.1684, abs=2e-3)


def test_single_wrong_certification_is_not_a_detectable_violation_of_delta_point_one():
    lo, _ = clopper_pearson(1, 20)
    assert lo < 0.1


def test_many_wrong_certifications_are_detectable():
    lo, _ = clopper_pearson(15, 20)
    assert lo > 0.1


def test_validates_inputs():
    with pytest.raises(ValueError):
        clopper_pearson(1, 0)
    with pytest.raises(ValueError):
        clopper_pearson(5, 4)
