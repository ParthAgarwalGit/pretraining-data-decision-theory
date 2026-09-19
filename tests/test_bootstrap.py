"""Tests for pdt.analysis.bootstrap -- see plan/02-phase1-datadecide.md P1-06."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pdt.analysis import bootstrap as bs
from pdt.scaling.base import Scale

_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# recipe_seed_trajectories()
# ---------------------------------------------------------------------------


def _rows(recipe, task, params_str, n, d, seed_values, *, metric_name="primary_metric"):
    seeds = ["default", "small aux 2", "small aux 3"]
    return [
        {
            "recipe": recipe,
            "task": task,
            "params_str": params_str,
            "params_num": n,
            "tokens": d,
            "seed": seed,
            "metric_name": metric_name,
            "metric_value": value,
            "is_final": True,
        }
        for seed, value in zip(seeds, seed_values, strict=True)
    ]


def test_recipe_seed_trajectories_keeps_every_seed_value():
    df = pl.DataFrame(_rows("recipe-a", "task-1", "10M", 1e7, 1e9, [0.1, 0.2, 0.3]))

    result = bs.recipe_seed_trajectories(df, "primary_metric", ["10M"])

    scale, values = result["task-1"]["recipe-a"][0]
    assert scale == Scale(n=1e7, d=1e9)
    assert sorted(values) == pytest.approx([0.1, 0.2, 0.3])


def test_recipe_seed_trajectories_sorts_by_scale_ascending():
    df = pl.DataFrame(
        [
            *_rows("recipe-a", "task-1", "150M", 1.5e8, 1e10, [0.5, 0.5, 0.5]),
            *_rows("recipe-a", "task-1", "10M", 1e7, 1e9, [0.1, 0.1, 0.1]),
        ]
    )

    result = bs.recipe_seed_trajectories(df, "primary_metric", ["10M", "150M"])

    ns = [scale.n for scale, _ in result["task-1"]["recipe-a"]]
    assert ns == sorted(ns)


def test_recipe_seed_trajectories_raises_when_no_rows_match():
    df = pl.DataFrame(_rows("recipe-a", "task-1", "10M", 1e7, 1e9, [0.1, 0.1, 0.1]))

    with pytest.raises(ValueError, match="no rows"):
        bs.recipe_seed_trajectories(df, "primary_metric", ["not_a_real_size"])


# ---------------------------------------------------------------------------
# seed-bootstrap draw/apply
# ---------------------------------------------------------------------------


def test_draw_seed_resample_pattern_has_n_seeds_indices_in_range():
    pattern = bs.draw_seed_resample_pattern(_RNG, 3)
    assert len(pattern) == 3
    assert all(0 <= i < 3 for i in pattern)


def test_apply_seed_resample_selects_by_pattern():
    pattern = np.array([2, 0, 0])
    assert bs.apply_seed_resample(pattern, [10.0, 20.0, 30.0]) == pytest.approx(
        (30.0 + 10.0 + 10.0) / 3
    )


def test_apply_seed_resample_same_pattern_shared_across_recipes():
    # The whole point of a shared pattern: two different recipes' own
    # values, resampled with the identical index pattern.
    pattern = np.array([1, 1, 2])
    recipe_a = bs.apply_seed_resample(pattern, [0.1, 0.2, 0.3])
    recipe_b = bs.apply_seed_resample(pattern, [0.4, 0.5, 0.6])
    assert recipe_a == pytest.approx((0.2 + 0.2 + 0.3) / 3)
    assert recipe_b == pytest.approx((0.5 + 0.5 + 0.6) / 3)


def test_apply_seed_resample_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="pattern has"):
        bs.apply_seed_resample(np.array([0, 1, 2]), [0.1, 0.2])


# ---------------------------------------------------------------------------
# parametric-bootstrap draw/apply
# ---------------------------------------------------------------------------


def test_apply_parametric_noise_at_z_zero_returns_mu():
    assert bs.apply_parametric_noise(0.0, mu=0.5, sigma2_total=0.01) == pytest.approx(0.5)


def test_apply_parametric_noise_scales_by_sqrt_sigma2():
    z = 2.0
    sigma2 = 0.04  # sd = 0.2
    assert bs.apply_parametric_noise(z, mu=0.5, sigma2_total=sigma2) == pytest.approx(0.5 + 0.4)


def test_apply_parametric_noise_shared_z_different_recipes_get_different_shifts():
    z = 1.5
    recipe_a = bs.apply_parametric_noise(z, mu=0.3, sigma2_total=0.01)  # sd=0.1
    recipe_b = bs.apply_parametric_noise(z, mu=0.3, sigma2_total=0.04)  # sd=0.2
    assert recipe_a == pytest.approx(0.3 + 1.5 * 0.1)
    assert recipe_b == pytest.approx(0.3 + 1.5 * 0.2)


def test_apply_parametric_noise_clamps_negative_variance_to_zero():
    # sigma2_total should never be negative in real use, but this must not
    # crash (sqrt of a negative number) if it ever is.
    assert bs.apply_parametric_noise(3.0, mu=0.5, sigma2_total=-1.0) == pytest.approx(0.5)


def test_draw_parametric_noise_z_is_standard_normal_ish():
    zs = [bs.draw_parametric_noise_z(np.random.default_rng(i)) for i in range(2000)]
    assert abs(np.mean(zs)) < 0.1
    assert abs(np.std(zs) - 1.0) < 0.1


# ---------------------------------------------------------------------------
# bias_variance_decomposition()
# ---------------------------------------------------------------------------


def test_bias_variance_decomposition_matches_hand_computation():
    preds = [0.5, 0.6, 0.7, 0.4]
    mu_true = 0.5
    sigma2_target = 0.001

    result = bs.bias_variance_decomposition(preds, mu_true, sigma2_target)

    mean_pred = sum(preds) / 4
    v_hat = sum((p - mean_pred) ** 2 for p in preds) / 3
    bias_hat = mean_pred - mu_true
    expected_sigma2_extrap = max(0.0, bias_hat**2 - v_hat - v_hat / 4 - sigma2_target)

    assert result["mean_prediction"] == pytest.approx(mean_pred)
    assert result["v_hat"] == pytest.approx(v_hat)
    assert result["bias_hat"] == pytest.approx(bias_hat)
    assert result["sigma2_extrap_hat"] == pytest.approx(expected_sigma2_extrap)
    assert result["n_replicates"] == 4


def test_bias_variance_decomposition_clamps_sigma2_extrap_at_zero():
    # No bias at all, but positive target noise -> would go negative
    # without the max(0, ...) floor.
    preds = [0.5, 0.5, 0.5, 0.5]
    result = bs.bias_variance_decomposition(preds, mu_true=0.5, sigma2_target=0.01)
    assert result["sigma2_extrap_hat"] == 0.0


def test_bias_variance_decomposition_large_consistent_bias_survives_the_floor():
    # A clear, large, consistent bias should produce a strictly positive
    # sigma2_extrap_hat even after the variance-of-the-mean and
    # target-noise corrections.
    preds = [0.9] * 50  # zero replicate variance, huge bias
    result = bs.bias_variance_decomposition(preds, mu_true=0.5, sigma2_target=1e-6)
    assert result["sigma2_extrap_hat"] == pytest.approx(0.4**2, abs=1e-4)


def test_bias_variance_decomposition_subtracts_v_hat_not_just_v_hat_over_b():
    # Regression for the squared-bias correction bug: an earlier version
    # only subtracted v_hat/B (finite-bootstrap Monte Carlo noise around
    # mu_hat_orig), leaving mu_hat_orig's OWN sampling variance (~v_hat)
    # baked into sigma2_extrap_hat even when there is no true structural
    # bias. bias_hat^2=0.01 here sits strictly between v_hat/B (~0.0002)
    # and v_hat (~0.04) -- the old formula (subtracting only v_hat/B)
    # would report a spurious ~0.0098 of "extrapolation bias"; the
    # corrected formula (subtracting v_hat too) floors it at 0.
    preds = [0.3, 0.7] * 100  # B=200, mean_pred=0.5, v_hat ~ 0.04
    result = bs.bias_variance_decomposition(preds, mu_true=0.4, sigma2_target=0.0)
    assert result["v_hat"] > 0.01  # sanity: the per-replicate variance is real
    assert result["bias_hat"] == pytest.approx(0.1)
    assert result["sigma2_extrap_hat"] == pytest.approx(0.0, abs=1e-6)


def test_bias_variance_decomposition_rejects_too_few_replicates():
    with pytest.raises(ValueError, match="at least 2"):
        bs.bias_variance_decomposition([0.5], mu_true=0.5, sigma2_target=0.0)


def test_bias_variance_decomposition_works_on_a_difference_series():
    # The pairwise use case: replicate_predictions is D_k^(b), not a raw
    # mu_hat^(b) series -- the function shouldn't care which.
    d_k_replicates = [0.05, 0.06, 0.04, 0.07, 0.03]
    true_gap = 0.05
    result = bs.bias_variance_decomposition(d_k_replicates, true_gap, sigma2_target=0.0001)
    assert result["bias_hat"] == pytest.approx(sum(d_k_replicates) / 5 - true_gap)


# ---------------------------------------------------------------------------
# Second-round review (PR #16): calibrate original-estimator variance for the
# actual n-out-of-n seed bootstrap, and validate across independent datasets.
# ---------------------------------------------------------------------------


def test_seed_bootstrap_variance_inflation_is_n_over_n_minus_one():
    assert bs.seed_bootstrap_variance_inflation(3) == pytest.approx(1.5)
    assert bs.seed_bootstrap_variance_inflation(2) == pytest.approx(2.0)
    assert bs.seed_bootstrap_variance_inflation(100) == pytest.approx(100 / 99)


def test_seed_bootstrap_variance_inflation_rejects_fewer_than_two_seeds():
    with pytest.raises(ValueError, match="at least 2 seeds"):
        bs.seed_bootstrap_variance_inflation(1)


def test_bias_variance_decomposition_rejects_nonpositive_inflation():
    with pytest.raises(ValueError, match="variance_inflation"):
        bs.bias_variance_decomposition([0.1, 0.2, 0.3], 0.2, 0.0, variance_inflation=0.0)


def test_unclipped_estimator_is_the_pre_clip_value_and_matches_hand_computation():
    preds = [0.5, 0.7, 0.3, 0.6]
    mu_true, sigma2_target, inflation = 0.4, 0.001, 1.5
    result = bs.bias_variance_decomposition(
        preds, mu_true, sigma2_target, variance_inflation=inflation
    )
    mean_pred = sum(preds) / 4
    v_hat = sum((p - mean_pred) ** 2 for p in preds) / 3
    expected = (mean_pred - mu_true) ** 2 - inflation * v_hat - v_hat / 4 - sigma2_target
    assert result["sigma2_extrap_unclipped"] == pytest.approx(expected)
    assert result["sigma2_extrap_hat"] == pytest.approx(max(0.0, expected))
    assert result["variance_inflation"] == inflation


def _mean_unclipped_over_independent_datasets(variance_inflation: float, n_datasets: int = 3000):
    """The reviewer's own validation design: many INDEPENDENT n=3
    datasets of an unbiased estimator (sample mean of N(0,1) draws) with
    exact target 0 (true squared bias 0), each bootstrapped B=200 with
    n-out-of-n resampling; returns (mean, standard error) of the unclipped
    squared-bias estimate across datasets."""
    rng = np.random.default_rng(20260918)
    values = []
    for _ in range(n_datasets):
        data = rng.normal(0.0, 1.0, size=3)
        replicate_means = data[rng.integers(0, 3, size=(200, 3))].mean(axis=1)
        result = bs.bias_variance_decomposition(
            list(replicate_means), 0.0, 0.0, variance_inflation=variance_inflation
        )
        values.append(result["sigma2_extrap_unclipped"])
    values = np.array(values)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(n_datasets))


def test_uncalibrated_bootstrap_variance_overstates_squared_bias_for_three_seeds():
    # Without the n/(n-1) calibration the estimator is biased upward by
    # about Var(orig mean)/3 = 1/9 here (reviewer: +0.106 over 20,000
    # datasets) -- the defect this whole section exists to pin down.
    mean, se = _mean_unclipped_over_independent_datasets(variance_inflation=1.0)
    assert mean > 0.08
    assert mean > 8 * se


def test_calibrated_unclipped_estimator_is_unbiased_across_independent_datasets():
    mean, se = _mean_unclipped_over_independent_datasets(
        variance_inflation=bs.seed_bootstrap_variance_inflation(3)
    )
    assert abs(mean) < 4 * se
    assert abs(mean) < 0.03


def test_clipped_value_is_a_heuristic_biased_upward_for_zero_true_bias():
    # max(0, unbiased-but-noisy X) has positive mean even when E[X]=0 --
    # the reason the unclipped field exists and must be used for averages.
    rng = np.random.default_rng(1)
    inflation = bs.seed_bootstrap_variance_inflation(3)
    clipped = []
    for _ in range(1500):
        data = rng.normal(0.0, 1.0, size=3)
        means = data[rng.integers(0, 3, size=(200, 3))].mean(axis=1)
        clipped.append(
            bs.bias_variance_decomposition(list(means), 0.0, 0.0, variance_inflation=inflation)[
                "sigma2_extrap_hat"
            ]
        )
    assert np.mean(clipped) > 0.05
