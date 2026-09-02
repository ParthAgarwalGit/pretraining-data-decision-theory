"""Tests for pdt.data.frame -- see plan/02-phase1-datadecide.md task P1-01.

build_frame() is tested against a small synthetic wide frame (the shape
datadecide.load_eval_results() actually returns), with load_eval_results
itself monkeypatched -- no network access. coverage_matrix() is tested
directly against small hand-built frames in both the wide and long column
shapes it's designed to accept.
"""

from __future__ import annotations

import polars as pl
import pytest

from pdt.data import datadecide as dd
from pdt.data import frame as frame_mod

# ---------------------------------------------------------------------------
# build_frame()
# ---------------------------------------------------------------------------


def _fake_wide_frame(n_rows: int = 2) -> pl.DataFrame:
    base = {
        "data": ["recipe-a"] * n_rows,
        "params": ["4M"] * n_rows,
        "params_num": [4_000_000] * n_rows,
        "seed": ["default"] * n_rows,
        "task": ["arc_challenge"] * n_rows,
        "step": [100] * n_rows,
        "tokens": [1000] * n_rows,
        "compute": [1.0] * n_rows,
        "is_final_checkpoint": [True] * n_rows,
        "chinchilla": ["5xC"] * n_rows,  # a real column not in the whitelist
    }
    for i, name in enumerate(frame_mod.METRIC_WHITELIST):
        base[f"metric_{name}"] = [float(i)] * n_rows
    return pl.DataFrame(base)


def test_build_frame_has_the_spec_column_order(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_mod, "_CACHE_PATH", tmp_path / "frame.parquet")
    monkeypatch.setattr(dd, "load_eval_results", lambda revision=None: _fake_wide_frame())

    result = frame_mod.build_frame()

    assert result.columns == list(frame_mod._FINAL_COLUMN_ORDER)


def test_build_frame_melts_one_row_per_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_mod, "_CACHE_PATH", tmp_path / "frame.parquet")
    monkeypatch.setattr(dd, "load_eval_results", lambda revision=None: _fake_wide_frame(n_rows=3))

    result = frame_mod.build_frame()

    assert result.height == 3 * len(frame_mod.METRIC_WHITELIST)
    assert set(result["metric_name"].unique().to_list()) == set(frame_mod.METRIC_WHITELIST)


def test_build_frame_strips_metric_prefix_and_renames_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_mod, "_CACHE_PATH", tmp_path / "frame.parquet")
    monkeypatch.setattr(dd, "load_eval_results", lambda revision=None: _fake_wide_frame())

    result = frame_mod.build_frame()

    assert result["recipe"].to_list()[0] == "recipe-a"
    assert result["params_str"].to_list()[0] == "4M"
    assert all(not name.startswith("metric_") for name in result["metric_name"].unique().to_list())


def test_build_frame_caches_and_does_not_recall_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_mod, "_CACHE_PATH", tmp_path / "frame.parquet")
    calls = []

    def fake_loader(revision=None):
        calls.append(1)
        return _fake_wide_frame()

    monkeypatch.setattr(dd, "load_eval_results", fake_loader)

    frame_mod.build_frame()
    frame_mod.build_frame()

    assert len(calls) == 1


def test_build_frame_force_refresh_recalls_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_mod, "_CACHE_PATH", tmp_path / "frame.parquet")
    calls = []

    def fake_loader(revision=None):
        calls.append(1)
        return _fake_wide_frame()

    monkeypatch.setattr(dd, "load_eval_results", fake_loader)

    frame_mod.build_frame()
    frame_mod.build_frame(force_refresh=True)

    assert len(calls) == 2


def test_build_frame_raises_if_whitelist_column_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_mod, "_CACHE_PATH", tmp_path / "frame.parquet")
    incomplete = _fake_wide_frame().drop("metric_primary_metric")
    monkeypatch.setattr(dd, "load_eval_results", lambda revision=None: incomplete)

    with pytest.raises(RuntimeError, match="metric_primary_metric"):
        frame_mod.build_frame()


# ---------------------------------------------------------------------------
# coverage_matrix()
# ---------------------------------------------------------------------------


def _complete_long_frame() -> pl.DataFrame:
    # 2 recipes x 2 (params, seed) pairs x 2 tasks, every cell present.
    rows = []
    for recipe in ("recipe-a", "recipe-b"):
        for params_str, seed in (("4M", "default"), ("4M", "small aux 2")):
            for task in ("arc_challenge", "hellaswag"):
                rows.append(
                    {
                        "recipe": recipe,
                        "params_str": params_str,
                        "seed": seed,
                        "task": task,
                        "is_final": True,
                    }
                )
    return pl.DataFrame(rows)


def test_coverage_matrix_reports_complete_grid_as_fully_present():
    result = frame_mod.coverage_matrix(_complete_long_frame())

    assert result["n_expected_cells"] == 8
    assert result["n_missing_cells"] == 0
    assert result["fraction_present"] == 1.0
    assert result["missing_cells_sample"] == []


def test_coverage_matrix_finds_a_single_missing_cell():
    df = _complete_long_frame()
    # Drop the one row for (recipe-a, 4M, default, hellaswag) -- it stays
    # part of the *expected* grid (recipe-a/4M/default/arc_challenge is
    # still present elsewhere) but is now missing.
    df = df.filter(
        ~(
            (pl.col("recipe") == "recipe-a")
            & (pl.col("seed") == "default")
            & (pl.col("task") == "hellaswag")
        )
    )

    result = frame_mod.coverage_matrix(df)

    assert result["n_missing_cells"] == 1
    assert result["missing_by_task"] == {"hellaswag": 1}
    assert result["missing_by_params"] == {"4M": 1}
    assert result["missing_cells_sample"][0]["recipe"] == "recipe-a"
    assert result["missing_cells_sample"][0]["task"] == "hellaswag"


def test_coverage_matrix_treats_is_final_false_as_missing():
    df = _complete_long_frame().with_columns(
        pl.when((pl.col("recipe") == "recipe-b") & (pl.col("task") == "arc_challenge"))
        .then(False)
        .otherwise(pl.col("is_final"))
        .alias("is_final")
    )

    result = frame_mod.coverage_matrix(df)

    assert result["n_missing_cells"] == 2  # one per (params, seed) pair


def test_coverage_matrix_accepts_wide_style_column_names():
    wide_style = _complete_long_frame().rename(
        {"recipe": "data", "params_str": "params", "is_final": "is_final_checkpoint"}
    )

    result = frame_mod.coverage_matrix(wide_style)

    assert result["n_expected_cells"] == 8
    assert result["n_missing_cells"] == 0


def test_coverage_matrix_truncates_sample_at_200():
    # 10 recipes x 1 (params, seed) x 30 tasks = 300 expected cells, all
    # missing (empty present set) -- forces the 200-cap to actually engage.
    recipes = [f"recipe-{i}" for i in range(10)]
    tasks = [f"task-{i}" for i in range(30)]
    rows = [
        {"recipe": r, "params_str": "4M", "seed": "default", "task": t, "is_final": True}
        for r in recipes
        for t in tasks
    ]
    # coverage_matrix derives the expected grid from the frame itself, so
    # give it a non-empty frame carrying the right recipes/params/tasks but
    # with is_final always False (never "present") -- every cell is then
    # both expected and missing.
    df = pl.DataFrame(rows).with_columns(pl.lit(False).alias("is_final"))

    result = frame_mod.coverage_matrix(df)

    assert result["n_expected_cells"] == 300
    assert result["n_missing_cells"] == 300
    assert len(result["missing_cells_sample"]) == 200
    assert result["missing_cells_sample_truncated"] is True
