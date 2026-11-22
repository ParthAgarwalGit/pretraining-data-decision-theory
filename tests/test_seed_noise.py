"""Tests for pdt.analysis.seed_noise (N7-10 S2): the calibrated check holds its size
under Gaussian noise and has power against outlier seeds."""

from __future__ import annotations

import numpy as np
import pytest

from pdt.analysis import seed_noise as sn


def test_subset_sd_error_known_values():
    x = np.array([[0.0, 0.0, 1.0, 1.0]])  # sd4 = 0.5774; subsets (0,0,1)/(0,1,1): sd 0.5774
    assert sn.subset_sd_error(x, k=3)[0] == pytest.approx(0.0)
    assert np.isnan(sn.subset_sd_error(np.ones((1, 5)))[0])


def test_gaussian_three_seed_error_is_about_0_39():
    # With sigma known this is ~0.39; against a 10-seed SD it is somewhat smaller.
    null = sn.gaussian_null_of_median(1, 10, n_sim=4000, seed=1)
    assert 0.25 < float(np.median(null)) < 0.40


def test_check_size_and_power():
    rng = np.random.default_rng(7)
    rejections_null = 0
    for _ in range(40):
        cells = rng.standard_normal((15, 10)) * rng.uniform(0.5, 3.0, (15, 1))
        rejections_null += not sn.calibrated_sd_check(cells, n_sim=2000, seed=3)["passed"]
    assert rejections_null <= 7  # nominal 5% of 40 = 2; allow MC slack
    # One wild outlier seed per cell inflates the 3-seed error well beyond Gaussian theory.
    cells = rng.standard_normal((15, 10))
    cells[:, 0] += 8.0
    assert not sn.calibrated_sd_check(cells, n_sim=2000, seed=3)["passed"]


def test_shapiro_check():
    rng = np.random.default_rng(0)
    assert sn.shapiro_check(rng.standard_normal((15, 10)))["passed"]
    skewed = np.tile(np.r_[np.zeros(9), 50.0], (15, 1)) + rng.normal(0, 0.01, (15, 10))
    assert not sn.shapiro_check(skewed)["passed"]
