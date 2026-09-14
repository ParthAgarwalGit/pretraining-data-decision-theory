"""Tests for pdt.bai.oracle -- plan/04-phase3-algorithm.md P3-01."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pdt.bai.oracle import DataDecideOracle, LiveTrainingOracle, SyntheticOracle
from pdt.data.frame import build_frame
from pdt.scaling.base import Scale

# ---------------------------------------------------------------------------
# SyntheticOracle
# ---------------------------------------------------------------------------

_FIT_SCALES = [Scale(n=n, d=20 * n) for n in [1e6, 3e6, 1e7, 3e7]]
_TARGET = Scale(n=1e9, d=20e9)


def _make_synthetic(seed: int = 0) -> SyntheticOracle:
    rng = np.random.default_rng(seed)
    return SyntheticOracle(
        recipes=["recipe_a", "recipe_b", "recipe_c"],
        scales=_FIT_SCALES,
        target_scale=_TARGET,
        rng=rng,
    )


def test_synthetic_pull_is_deterministic():
    oracle = _make_synthetic()
    v1 = oracle.pull("recipe_a", _FIT_SCALES[0], seed=0)
    v2 = oracle.pull("recipe_a", _FIT_SCALES[0], seed=0)
    assert v1 == v2


def test_synthetic_pull_varies_with_seed():
    oracle = _make_synthetic()
    values = {oracle.pull("recipe_a", _FIT_SCALES[0], seed=s) for s in range(5)}
    assert len(values) == 5  # 5 distinct noise draws


def test_synthetic_pull_is_reproducible_across_fresh_instances():
    # Same construction seed -> same internal recipe params -> same pulls,
    # even from a brand-new SyntheticOracle instance (not just the same
    # object called twice) -- this is what "calibrated, not ad hoc" means
    # operationally: a SyntheticOracle instance is fully determined by its
    # constructor rng seed.
    o1 = _make_synthetic(seed=42)
    o2 = _make_synthetic(seed=42)
    for recipe in ["recipe_a", "recipe_b", "recipe_c"]:
        for scale in _FIT_SCALES:
            assert o1.pull(recipe, scale, seed=0) == o2.pull(recipe, scale, seed=0)


def test_synthetic_cost_matches_scale_compute():
    oracle = _make_synthetic()
    for scale in _FIT_SCALES:
        assert oracle.cost(scale) == scale.compute


def test_synthetic_available_scales_matches_construction():
    oracle = _make_synthetic()
    assert oracle.available_scales() == _FIT_SCALES


def test_synthetic_bias_is_zero_at_every_fitted_scale():
    # h_k(s) = 0 for every scale <= the largest fitted scale, by
    # construction -- checked directly against the noise-free true mean
    # (not the noisy pull) via true_value_at_target's own internal helper.
    oracle = _make_synthetic()
    for recipe in ["recipe_a", "recipe_b", "recipe_c"]:
        for scale in _FIT_SCALES:
            assert oracle._h(recipe, scale) == 0.0


def test_synthetic_bias_at_target_is_within_calibrated_range():
    oracle = _make_synthetic(seed=1)
    for recipe in ["recipe_a", "recipe_b", "recipe_c"]:
        bias = oracle._h(recipe, _TARGET)
        assert oracle._BIAS_MAGNITUDE_RANGE[0] <= abs(bias) <= oracle._BIAS_MAGNITUDE_RANGE[1]


def test_synthetic_noise_level_matches_p1_05_calibration():
    # Empirical check, not just trusting the constructor's uniform draw
    # range: repeatedly pull the same (recipe, scale) and confirm the
    # SAMPLE variance across seeds falls near the calibrated sigma2_noise
    # for that recipe (P1-05's measured ~1e-4 range).
    oracle = _make_synthetic(seed=2)
    recipe = "recipe_a"
    scale = _FIT_SCALES[0]
    draws = [oracle.pull(recipe, scale, seed=s) for s in range(500)]
    empirical_var = float(np.var(draws, ddof=1))
    true_sigma2 = oracle._params[recipe]["sigma2_noise"]
    assert empirical_var == pytest.approx(true_sigma2, rel=0.3)


def test_synthetic_unknown_recipe_raises():
    oracle = _make_synthetic()
    with pytest.raises(KeyError):
        oracle.pull("not_a_recipe", _FIT_SCALES[0], seed=0)


def test_synthetic_rejects_empty_recipes_or_scales():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        SyntheticOracle(recipes=[], scales=_FIT_SCALES, target_scale=_TARGET, rng=rng)
    with pytest.raises(ValueError):
        SyntheticOracle(recipes=["a"], scales=[], target_scale=_TARGET, rng=rng)


def test_synthetic_true_value_at_target_is_noise_free_and_matches_the_mean():
    oracle = _make_synthetic(seed=3)
    for recipe in ["recipe_a", "recipe_b", "recipe_c"]:
        # Called twice: a noise-free quantity must be exactly repeatable,
        # unlike pull() at the same args (which is also repeatable, but
        # for a different reason -- the stable-seed derivation, not the
        # absence of a noise term at all).
        v1 = oracle.true_value_at_target(recipe)
        v2 = oracle.true_value_at_target(recipe)
        assert v1 == v2
        assert v1 == oracle._true_mean(recipe, _TARGET)


# ---------------------------------------------------------------------------
# DataDecideOracle -- exercises the real cached DataDecide frame.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dd_oracle() -> DataDecideOracle:
    return DataDecideOracle(task="arc_challenge", metric_name="primary_metric")


def test_datadecide_pull_is_deterministic(dd_oracle):
    scale = dd_oracle.available_scales()[0]
    recipe = next(iter({r for r, _ in dd_oracle._primary}))
    v1 = dd_oracle.pull(recipe, scale, seed=0)
    v2 = dd_oracle.pull(recipe, scale, seed=0)
    assert v1 == v2


def test_datadecide_has_25_recipes_and_14_sizes(dd_oracle):
    # Cross-check against P0-06/P1-01's already-confirmed counts, not a
    # new discovery -- a regression guard, not a finding.
    recipes = {r for r, _ in dd_oracle._primary}
    assert len(recipes) == 25
    assert len(dd_oracle.available_scales()) == 14


def test_datadecide_seed_labels_are_discovered_not_hardcoded(dd_oracle):
    # P0-06's finding: seed labels differ at 1B ("large aux 2/3") vs below
    # ("small aux 2/3"). Confirm the oracle's own discovered labels reflect
    # this rather than assuming one fixed set across every scale.
    recipe = next(iter({r for r, _ in dd_oracle._primary}))
    scales_by_n = {s.n: s for s in dd_oracle.available_scales()}
    small_scale = scales_by_n[min(scales_by_n)]
    large_scale = scales_by_n[max(scales_by_n)]
    small_key = (recipe, dd_oracle._params_str_for(small_scale))
    large_key = (recipe, dd_oracle._params_str_for(large_scale))
    labels_small = {lbl for lbl, _ in dd_oracle._primary[small_key]}
    labels_large = {lbl for lbl, _ in dd_oracle._primary[large_key]}
    assert labels_small != labels_large, (
        "expected different seed label sets at the smallest vs. largest scale "
        "(small aux vs large aux, per P0-06) -- got the same set, which would "
        "mean this ISN'T exercising the real naming irregularity"
    )


def test_datadecide_cost_matches_6nd_to_within_measured_tolerance(dd_oracle):
    # P3-01's definition of done: verify cost() (== Scale.compute, the 6ND
    # approximation) against DataDecide's own `compute` column. Checked
    # once, in aggregate, against every distinct size in the real table --
    # see docs/decisions.md for the full ratio distribution (median 0.999,
    # range 0.85-1.10 across the 14 sizes).
    frame = build_frame(source="macro_avg", metrics=("primary_metric",))
    sizes = frame.filter(pl.col("is_final")).select(["params_num", "tokens", "compute"]).unique()
    ratios = []
    for row in sizes.iter_rows(named=True):
        scale = Scale(n=row["params_num"], d=row["tokens"])
        our = dd_oracle.cost(scale)  # the actual method under test, not a
        # hand-rolled Scale(...).compute stand-in -- exercises cost() itself.
        assert our == scale.compute  # cost() is, and must stay, exactly Scale.compute
        table = row["compute"]
        if table:
            ratios.append(our / table)
    ratios = np.array(ratios)
    assert len(ratios) == 14
    assert 0.8 <= ratios.min()
    assert ratios.max() <= 1.15
    assert abs(np.median(ratios) - 1.0) < 0.01


def test_datadecide_pull_unknown_recipe_at_a_known_scale_raises(dd_oracle):
    # Distinct from test_datadecide_unknown_scale_raises: here the SCALE is
    # real (in available_scales()), only the recipe name is bogus -- a
    # different code path (_params_str_for succeeds; the (recipe,
    # params_str) lookup into _primary is what fails).
    scale = dd_oracle.available_scales()[0]
    with pytest.raises(KeyError):
        dd_oracle.pull("not_a_real_recipe", scale, seed=0)


def test_datadecide_pull_falls_back_to_pseudo_replicates_beyond_real_seeds(dd_oracle):
    scale = dd_oracle.available_scales()[0]
    recipe = next(iter({r for r, _ in dd_oracle._primary}))
    n_real = len(dd_oracle._primary[(recipe, dd_oracle._params_str_for(scale))])
    assert n_real == 3  # P0-06/P1-01 confirmed 3 seeds everywhere

    # seed=3 (0-indexed, beyond the 3 real seeds) must not raise, and must
    # come from the pseudo-replicate pool, not silently repeat a real one.
    pseudo_value = dd_oracle.pull(recipe, scale, seed=3)
    assert isinstance(pseudo_value, float)
    # (not asserting pseudo_value differs from the 3 real-seed pulls --
    # coincidental equality is possible in principle; just confirming it
    # resolves at all, rather than raising, is the actual contract here)


def test_datadecide_raises_past_all_available_replicates(dd_oracle):
    scale = dd_oracle.available_scales()[0]
    recipe = next(iter({r for r, _ in dd_oracle._primary}))
    with pytest.raises(IndexError):
        dd_oracle.pull(recipe, scale, seed=10_000)


def test_datadecide_unknown_scale_raises(dd_oracle):
    with pytest.raises(KeyError):
        dd_oracle.pull(next(iter({r for r, _ in dd_oracle._primary})), Scale(n=1, d=1), seed=0)


def test_datadecide_unknown_task_raises():
    with pytest.raises(ValueError):
        DataDecideOracle(task="not_a_real_task", metric_name="primary_metric")


# ---------------------------------------------------------------------------
# LiveTrainingOracle
# ---------------------------------------------------------------------------


def test_live_training_oracle_stub_raises_not_implemented():
    oracle = LiveTrainingOracle()
    with pytest.raises(NotImplementedError):
        oracle.pull("recipe", Scale(n=1, d=1), seed=0)
    with pytest.raises(NotImplementedError):
        oracle.cost(Scale(n=1, d=1))
    with pytest.raises(NotImplementedError):
        oracle.available_scales()
