"""Tests for pdt.analysis.noise -- see plan/02-phase1-datadecide.md P1-05."""

from __future__ import annotations

import polars as pl
import pytest

from pdt.analysis import noise

# ---------------------------------------------------------------------------
# seed_variance()
# ---------------------------------------------------------------------------


def _seed_rows(recipe, params_str, task, values, *, metric_name="primary_metric", is_final=True):
    seeds = ["default", "small aux 2", "small aux 3"]
    return [
        {
            "recipe": recipe,
            "params_str": params_str,
            "params_num": 150_000_000,
            "seed": seed,
            "task": task,
            "metric_name": metric_name,
            "metric_value": value,
            "is_final": is_final,
        }
        for seed, value in zip(seeds, values, strict=True)
    ]


def test_seed_variance_computes_mean_and_variance():
    df = pl.DataFrame(_seed_rows("recipe-a", "150M", "task-1", [0.60, 0.62, 0.61]))

    result = noise.seed_variance(df, "primary_metric")

    row = result.row(0, named=True)
    assert row["mu"] == pytest.approx(0.61, abs=1e-9)
    assert row["sigma2_seed"] == pytest.approx(0.0001, abs=1e-9)
    assert row["n_seeds"] == 3


def test_seed_variance_excludes_non_final_checkpoints():
    rows = [
        *_seed_rows("recipe-a", "150M", "task-1", [0.60, 0.60, 0.60], is_final=True),
        {
            "recipe": "recipe-a",
            "params_str": "150M",
            "params_num": 150_000_000,
            "seed": "s0",
            "task": "task-1",
            "metric_name": "primary_metric",
            "metric_value": 0.99,
            "is_final": False,
        },
    ]
    df = pl.DataFrame(rows)

    result = noise.seed_variance(df, "primary_metric")

    assert result.height == 1
    assert result.row(0, named=True)["mu"] == pytest.approx(0.60, abs=1e-9)


def test_seed_variance_is_per_recipe_params_task():
    df = pl.DataFrame(
        [
            *_seed_rows("recipe-a", "150M", "task-1", [0.1, 0.1, 0.1]),
            *_seed_rows("recipe-b", "150M", "task-1", [0.2, 0.2, 0.2]),
            *_seed_rows("recipe-a", "300M", "task-1", [0.3, 0.3, 0.3]),
        ]
    )

    result = noise.seed_variance(df, "primary_metric")

    assert result.height == 3


# ---------------------------------------------------------------------------
# checkpoint_jitter()
# ---------------------------------------------------------------------------


def _ckpt_rows(recipe, params_str, seed, task, step_to_value, *, metric_name="primary_metric"):
    return [
        {
            "recipe": recipe,
            "params_str": params_str,
            "params_num": 150_000_000,
            "seed": seed,
            "task": task,
            "metric_name": metric_name,
            "metric_value": value,
            "step": step,
        }
        for step, value in step_to_value.items()
    ]


def test_checkpoint_jitter_uses_only_the_last_n_checkpoints():
    # 6 checkpoints total; with n_last_checkpoints=4, only steps 300-600
    # (values 0.5, 0.5, 0.9, 0.5) should count -- the two early outliers
    # (100, 200 -> 0.0) must not affect the variance.
    steps = {100: 0.0, 200: 0.0, 300: 0.5, 400: 0.5, 500: 0.9, 600: 0.5}
    df = pl.DataFrame(_ckpt_rows("recipe-a", "150M", "default", "task-1", steps))

    result = noise.checkpoint_jitter(df, "primary_metric", n_last_checkpoints=4)

    row = result.row(0, named=True)
    assert row["n_checkpoints_used"] == 4
    last_four = [0.5, 0.5, 0.9, 0.5]
    expected_var = pl.Series(last_four).var(ddof=1)
    assert row["sigma2_ckpt"] == pytest.approx(expected_var, abs=1e-9)


def test_checkpoint_jitter_drops_runs_with_fewer_than_two_checkpoints():
    df = pl.DataFrame(_ckpt_rows("recipe-a", "150M", "default", "task-1", {100: 0.5}))

    result = noise.checkpoint_jitter(df, "primary_metric")

    assert result.height == 0


def test_checkpoint_jitter_is_per_recipe_params_seed_task():
    df = pl.DataFrame(
        [
            *_ckpt_rows("recipe-a", "150M", "default", "task-1", {100: 0.1, 200: 0.2}),
            *_ckpt_rows("recipe-a", "150M", "small aux 2", "task-1", {100: 0.3, 200: 0.4}),
        ]
    )

    result = noise.checkpoint_jitter(df, "primary_metric")

    assert result.height == 2


def test_checkpoint_jitter_respects_custom_window():
    steps = {100: 0.0, 200: 1.0, 300: 1.0}
    df = pl.DataFrame(_ckpt_rows("recipe-a", "150M", "default", "task-1", steps))

    result = noise.checkpoint_jitter(df, "primary_metric", n_last_checkpoints=2)

    row = result.row(0, named=True)
    assert row["n_checkpoints_used"] == 2
    assert row["sigma2_ckpt"] == pytest.approx(0.0, abs=1e-9)  # last two steps are both 1.0


# ---------------------------------------------------------------------------
# eval_sampling_noise() / eval_sampling_noise_of_mean()
# ---------------------------------------------------------------------------


def test_eval_sampling_noise_matches_binomial_formula():
    assert noise.eval_sampling_noise(0.5, 1000) == pytest.approx(0.5 * 0.5 / 1000)


def test_eval_sampling_noise_is_zero_at_the_extremes():
    assert noise.eval_sampling_noise(0.0, 1000) == pytest.approx(0.0)
    assert noise.eval_sampling_noise(1.0, 1000) == pytest.approx(0.0)


def test_eval_sampling_noise_rejects_nonpositive_n_instances():
    with pytest.raises(ValueError, match="n_instances"):
        noise.eval_sampling_noise(0.5, 0)


def test_eval_sampling_noise_of_mean_matches_hand_computation():
    # Var(mean of 2 independent components) = (1/4) * (v1 + v2).
    ps = [0.5, 0.2]
    ns = [1000, 500]
    v1 = 0.5 * 0.5 / 1000
    v2 = 0.2 * 0.8 / 500
    expected = (v1 + v2) / 4

    assert noise.eval_sampling_noise_of_mean(ps, ns) == pytest.approx(expected)


def test_eval_sampling_noise_of_mean_single_component_matches_plain_formula():
    assert noise.eval_sampling_noise_of_mean([0.3], [200]) == pytest.approx(
        noise.eval_sampling_noise(0.3, 200)
    )


def test_eval_sampling_noise_of_mean_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        noise.eval_sampling_noise_of_mean([0.5, 0.5], [100])


def test_eval_sampling_noise_of_mean_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        noise.eval_sampling_noise_of_mean([], [])
