"""Tests for pdt.analysis.decision_accuracy -- see plan/02-phase1-datadecide.md P1-03."""

from __future__ import annotations

import polars as pl
import pytest

from pdt.analysis import decision_accuracy as da
from pdt.scaling.base import Scale

_SEEDS = ("default", "small aux 2", "small aux 3")


def _rows(
    recipe,
    task,
    values,
    *,
    params_str="150M",
    params_num=150_000_000,
    tokens=14_745_600_000,
    metric_name="primary_metric",
    is_final=True,
):
    return [
        {
            "recipe": recipe,
            "params_str": params_str,
            "params_num": params_num,
            "tokens": tokens,
            "seed": seed,
            "task": task,
            "metric_name": metric_name,
            "metric_value": value,
            "is_final": is_final,
        }
        for seed, value in zip(_SEEDS, values, strict=True)
    ]


# ---------------------------------------------------------------------------
# recipe_means()
# ---------------------------------------------------------------------------


def test_recipe_means_averages_over_seeds_by_default():
    df = pl.DataFrame(_rows("recipe-a", "task-1", [0.60, 0.62, 0.61]))

    result = da.recipe_means(df, "primary_metric", "150M")

    assert result["task-1"]["recipe-a"] == pytest.approx(0.61, abs=1e-9)


def test_recipe_means_default_only_uses_just_the_default_seed():
    df = pl.DataFrame(_rows("recipe-a", "task-1", [0.60, 0.62, 0.61]))

    result = da.recipe_means(df, "primary_metric", "150M", seed_mode="default_only")

    assert result["task-1"]["recipe-a"] == pytest.approx(0.60, abs=1e-9)


def test_recipe_means_organizes_by_task_then_recipe():
    df = pl.DataFrame(
        [
            *_rows("recipe-a", "task-1", [0.1, 0.1, 0.1]),
            *_rows("recipe-b", "task-1", [0.2, 0.2, 0.2]),
            *_rows("recipe-a", "task-2", [0.3, 0.3, 0.3]),
        ]
    )

    result = da.recipe_means(df, "primary_metric", "150M")

    assert set(result.keys()) == {"task-1", "task-2"}
    assert set(result["task-1"].keys()) == {"recipe-a", "recipe-b"}
    assert set(result["task-2"].keys()) == {"recipe-a"}


def test_recipe_means_excludes_non_final_checkpoints():
    rows = [
        *_rows("recipe-a", "task-1", [0.60, 0.60, 0.60], is_final=True),
        {
            "recipe": "recipe-a",
            "params_str": "150M",
            "params_num": 150_000_000,
            "tokens": 14_745_600_000,
            "seed": "s0",
            "task": "task-1",
            "metric_name": "primary_metric",
            "metric_value": 0.99,
            "is_final": False,
        },
    ]
    df = pl.DataFrame(rows)

    result = da.recipe_means(df, "primary_metric", "150M")

    assert result["task-1"]["recipe-a"] == pytest.approx(0.60, abs=1e-9)


def test_recipe_means_rejects_unknown_seed_mode():
    df = pl.DataFrame(_rows("recipe-a", "task-1", [0.5, 0.5, 0.5]))

    with pytest.raises(ValueError, match="seed_mode"):
        da.recipe_means(df, "primary_metric", "150M", seed_mode="not_a_mode")


def test_recipe_means_raises_when_no_rows_match_at_all():
    df = pl.DataFrame(_rows("recipe-a", "task-1", [0.5, 0.5, 0.5]))

    with pytest.raises(ValueError, match="no rows"):
        da.recipe_means(df, "primary_metric", "not_a_real_scale")


def test_recipe_means_raises_when_no_default_seed_rows_exist():
    rows = [
        {
            "recipe": "recipe-a",
            "params_str": "150M",
            "params_num": 150_000_000,
            "tokens": 14_745_600_000,
            "seed": "small aux 2",
            "task": "task-1",
            "metric_name": "primary_metric",
            "metric_value": 0.5,
            "is_final": True,
        }
    ]
    df = pl.DataFrame(rows)

    with pytest.raises(ValueError, match="default"):
        da.recipe_means(df, "primary_metric", "150M", seed_mode="default_only")


# ---------------------------------------------------------------------------
# pairwise_decision_accuracy()
# ---------------------------------------------------------------------------


def test_perfect_agreement_gives_full_accuracy():
    proxy = {"a": 0.9, "b": 0.5, "c": 0.1}
    target = {"a": 0.8, "b": 0.4, "c": 0.2}  # same ordering, different values
    sd = {"a": 0.01, "b": 0.01, "c": 0.01}

    result = da.pairwise_decision_accuracy(proxy, target, sd, n_seeds_target=3)

    assert result["n_pairs"] == 3
    assert result["accuracy_including_ties"] == 1.0
    assert result["kendall_tau"] == pytest.approx(1.0)


def test_perfect_disagreement_gives_zero_accuracy():
    proxy = {"a": 0.1, "b": 0.5, "c": 0.9}  # exactly reversed vs target
    target = {"a": 0.8, "b": 0.4, "c": 0.2}
    sd = {"a": 0.01, "b": 0.01, "c": 0.01}

    result = da.pairwise_decision_accuracy(proxy, target, sd, n_seeds_target=3)

    assert result["accuracy_including_ties"] == 0.0
    assert result["kendall_tau"] == pytest.approx(-1.0)


def test_ties_are_excluded_from_the_excluding_ties_accuracy():
    # a vs b, and b vs c: target gaps are huge relative to noise -> not
    # tied, proxy agrees on both.
    # a vs c: target gap (0.0001) is tiny relative to the pooled SE
    # (~0.0008) -> effect size ~0.12, well under the 1.0 threshold -> tied,
    # excluded from the excluding-ties accuracy regardless of what the
    # proxy says about that pair.
    proxy = {"a": 0.9, "b": 0.1, "c": 0.9005}
    target = {"a": 0.9000, "b": 0.1000, "c": 0.9001}
    sd = {"a": 0.001, "b": 0.001, "c": 0.001}

    result = da.pairwise_decision_accuracy(proxy, target, sd, n_seeds_target=3)

    assert result["n_pairs"] == 3
    assert result["n_tied_pairs"] == 1
    # 2 non-tied pairs (a,b) and (b,c); both correctly ordered by proxy.
    assert result["accuracy_excluding_ties"] == 1.0


def test_zero_variance_pair_is_always_tied():
    proxy = {"a": 0.9, "b": 0.1}
    target = {"a": 0.50, "b": 0.50}  # identical -> zero gap, zero variance
    sd = {"a": 0.0, "b": 0.0}

    result = da.pairwise_decision_accuracy(proxy, target, sd, n_seeds_target=3)

    assert result["n_tied_pairs"] == 1
    assert result["accuracy_excluding_ties"] is None


def test_only_compares_recipes_present_on_both_sides():
    proxy = {"a": 0.9, "b": 0.1, "extra_only_in_proxy": 0.5}
    target = {"a": 0.8, "b": 0.2, "extra_only_in_target": 0.5}
    sd = {"a": 0.01, "b": 0.01, "extra_only_in_target": 0.01}

    result = da.pairwise_decision_accuracy(proxy, target, sd, n_seeds_target=3)

    assert result["n_recipes_compared"] == 2
    assert result["n_pairs"] == 1


def test_single_recipe_has_no_pairs():
    result = da.pairwise_decision_accuracy({"a": 0.5}, {"a": 0.5}, {"a": 0.01}, n_seeds_target=3)

    assert result["n_pairs"] == 0
    assert result["accuracy_including_ties"] is None
    assert result["kendall_tau"] is None


def test_sign_helper():
    assert da._sign(5.0) == 1
    assert da._sign(-5.0) == -1
    assert da._sign(0.0) == 0


# ---------------------------------------------------------------------------
# recipe_trajectories()
# ---------------------------------------------------------------------------


def _multi_scale_rows(recipe, task, size_to_values, *, metric_name="primary_metric"):
    scale_specs = {
        "10M": (10_000_000, 990_707_712),
        "150M": (150_000_000, 14_745_600_000),
        "300M": (300_000_000, 29_491_200_000),
    }
    rows = []
    for size, values in size_to_values.items():
        n, d = scale_specs[size]
        rows.extend(
            _rows(
                recipe,
                task,
                values,
                params_str=size,
                params_num=n,
                tokens=d,
                metric_name=metric_name,
            )
        )
    return rows


def test_recipe_trajectories_sorts_each_recipe_by_scale_ascending():
    df = pl.DataFrame(
        _multi_scale_rows(
            "recipe-a",
            "task-1",
            {
                "300M": [0.7, 0.7, 0.7],
                "10M": [0.3, 0.3, 0.3],
                "150M": [0.5, 0.5, 0.5],
            },
        )
    )

    result = da.recipe_trajectories(df, "primary_metric", ["10M", "150M", "300M"])

    trajectory = result["task-1"]["recipe-a"]
    ns = [scale.n for scale, _ in trajectory]
    assert ns == sorted(ns)
    assert [round(v, 2) for _, v in trajectory] == [0.3, 0.5, 0.7]


def test_recipe_trajectories_carries_correct_scale_per_point():
    df = pl.DataFrame(
        _multi_scale_rows("recipe-a", "task-1", {"10M": [0.3, 0.3, 0.3], "150M": [0.5, 0.5, 0.5]})
    )

    result = da.recipe_trajectories(df, "primary_metric", ["10M", "150M"])

    trajectory = result["task-1"]["recipe-a"]
    scale_10m, value_10m = trajectory[0]
    assert scale_10m == Scale(n=10_000_000, d=990_707_712)
    assert value_10m == pytest.approx(0.3)


def test_recipe_trajectories_only_includes_requested_proxy_sizes():
    df = pl.DataFrame(
        _multi_scale_rows(
            "recipe-a",
            "task-1",
            {"10M": [0.3, 0.3, 0.3], "150M": [0.5, 0.5, 0.5], "300M": [0.7, 0.7, 0.7]},
        )
    )

    result = da.recipe_trajectories(df, "primary_metric", ["10M", "150M"])

    sizes_included = [scale.n for scale, _ in result["task-1"]["recipe-a"]]
    assert 300_000_000 not in sizes_included
    assert len(sizes_included) == 2


def test_recipe_trajectories_organizes_by_task_then_recipe():
    df = pl.DataFrame(
        [
            *_multi_scale_rows("recipe-a", "task-1", {"10M": [0.1, 0.1, 0.1]}),
            *_multi_scale_rows("recipe-b", "task-1", {"10M": [0.2, 0.2, 0.2]}),
            *_multi_scale_rows("recipe-a", "task-2", {"10M": [0.3, 0.3, 0.3]}),
        ]
    )

    result = da.recipe_trajectories(df, "primary_metric", ["10M"])

    assert set(result.keys()) == {"task-1", "task-2"}
    assert set(result["task-1"].keys()) == {"recipe-a", "recipe-b"}
    assert set(result["task-2"].keys()) == {"recipe-a"}


def test_recipe_trajectories_default_only_uses_just_the_default_seed():
    df = pl.DataFrame(_multi_scale_rows("recipe-a", "task-1", {"10M": [0.60, 0.62, 0.61]}))

    result = da.recipe_trajectories(df, "primary_metric", ["10M"], seed_mode="default_only")

    _, value = result["task-1"]["recipe-a"][0]
    assert value == pytest.approx(0.60, abs=1e-9)


def test_recipe_trajectories_rejects_unknown_seed_mode():
    df = pl.DataFrame(_multi_scale_rows("recipe-a", "task-1", {"10M": [0.5, 0.5, 0.5]}))

    with pytest.raises(ValueError, match="seed_mode"):
        da.recipe_trajectories(df, "primary_metric", ["10M"], seed_mode="not_a_mode")


def test_recipe_trajectories_raises_when_no_rows_match_at_all():
    df = pl.DataFrame(_multi_scale_rows("recipe-a", "task-1", {"10M": [0.5, 0.5, 0.5]}))

    with pytest.raises(ValueError, match="no rows"):
        da.recipe_trajectories(df, "primary_metric", ["not_a_real_scale"])


def test_recipe_trajectories_excludes_non_final_checkpoints():
    rows = [
        *_multi_scale_rows("recipe-a", "task-1", {"10M": [0.60, 0.60, 0.60]}),
        {
            "recipe": "recipe-a",
            "params_str": "10M",
            "params_num": 10_000_000,
            "tokens": 990_707_712,
            "seed": "s0",
            "task": "task-1",
            "metric_name": "primary_metric",
            "metric_value": 0.99,
            "is_final": False,
        },
    ]
    df = pl.DataFrame(rows)

    result = da.recipe_trajectories(df, "primary_metric", ["10M"])

    _, value = result["task-1"]["recipe-a"][0]
    assert value == pytest.approx(0.60, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_cost()
# ---------------------------------------------------------------------------


def test_compute_cost_sums_6nd_over_scales():
    scales = [Scale(n=1e7, d=2e8), Scale(n=1e8, d=2e9)]

    result = da.compute_cost(scales)

    expected = (6.0 * 1e7 * 2e8) + (6.0 * 1e8 * 2e9)
    assert result == pytest.approx(expected)


def test_compute_cost_of_empty_list_is_zero():
    assert da.compute_cost([]) == 0.0


def test_compute_cost_of_single_scale_matches_its_own_compute_property():
    scale = Scale(n=5e8, d=3e10)
    assert da.compute_cost([scale]) == pytest.approx(scale.compute)
