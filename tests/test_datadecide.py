"""Tests for pdt.data.datadecide -- see plan/01-phase0-setup.md task P0-06.

Pure parsing/transformation logic and the caching engine (`_cached_or_build`,
`cached_revision`) are tested here, against small synthetic fixtures and a
fake `build` callable. The actual Hub-touching functions
(`download_snapshot`, `download_file`, `resolve_revision`, and each
`load_*`'s `build()` closure) are marked `# pragma: no cover` in the
module -- they are thin wrappers, verified by hand against the live Hub
while building this module (see the P0-06 PR description for the exact
calls run and their real output: 25 recipes, 14 sizes, 1,346,400 eval-result
rows, etc.). Exercising them in the automated suite would mean every CI run
needs live network access to a third-party service, which is the wrong
tradeoff for a unit test suite.
"""

from __future__ import annotations

import polars as pl
import pytest

from pdt.data import datadecide as dd

# ---------------------------------------------------------------------------
# parse_params
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("4M", 4_000_000),
        ("10M", 10_000_000),
        ("150M", 150_000_000),
        ("1B", 1_000_000_000),
        ("530M", 530_000_000),
    ],
)
def test_parse_params_known_formats(text, expected):
    assert dd.parse_params(text) == expected


def test_parse_params_rejects_unrecognised_format():
    with pytest.raises(ValueError, match="unrecognised params format"):
        dd.parse_params("not-a-size")


# ---------------------------------------------------------------------------
# _parse_metrics -- must handle both serializations found in the real data
# ---------------------------------------------------------------------------


def test_parse_metrics_handles_single_quoted_python_repr():
    # The format used in data/train-*.parquet.
    raw = "{'acc_raw': 0.25, 'primary_metric': 0.27}"
    assert dd._parse_metrics(raw) == {"acc_raw": 0.25, "primary_metric": 0.27}


def test_parse_metrics_handles_double_quoted_json():
    # The format used in data/macro_avg-*.parquet.
    raw = '{"acc_raw": 0.25, "primary_metric": 0.27}'
    assert dd._parse_metrics(raw) == {"acc_raw": 0.25, "primary_metric": 0.27}


# ---------------------------------------------------------------------------
# _explode_metrics
# ---------------------------------------------------------------------------


def test_explode_metrics_creates_one_column_per_key():
    df = pl.DataFrame({"metrics": ["{'a': 1, 'b': 2}", "{'a': 3, 'b': 4}"]})
    out = dd._explode_metrics(df)
    assert "metric_a" in out.columns
    assert "metric_b" in out.columns
    assert "metrics" not in out.columns
    assert out["metric_a"].to_list() == [1, 3]
    assert out["metric_b"].to_list() == [2, 4]


def test_explode_metrics_handles_rows_with_different_key_sets():
    # Different task types report different metric sets in the real data.
    df = pl.DataFrame({"metrics": ["{'a': 1}", "{'b': 2}"]})
    out = dd._explode_metrics(df)
    assert out["metric_a"].to_list() == [1, None]
    assert out["metric_b"].to_list() == [None, 2]


# ---------------------------------------------------------------------------
# _add_params_num_and_final_flag
# ---------------------------------------------------------------------------


def test_add_params_num_and_final_flag_filters_step_zero():
    df = pl.DataFrame(
        {
            "params": ["4M", "4M"],
            "data": ["recipe-a", "recipe-a"],
            "seed": ["default", "default"],
            "step": [0, 100],
        }
    )
    out = dd._add_params_num_and_final_flag(df)
    assert out.height == 1
    assert out["step"].to_list() == [100]


def test_add_params_num_and_final_flag_marks_max_step_per_group():
    df = pl.DataFrame(
        {
            "params": ["4M", "4M", "4M"],
            "data": ["recipe-a", "recipe-a", "recipe-a"],
            "seed": ["default", "default", "default"],
            "step": [100, 200, 300],
        }
    )
    out = dd._add_params_num_and_final_flag(df).sort("step")
    assert out["is_final_checkpoint"].to_list() == [False, False, True]
    assert out["params_num"].to_list() == [4_000_000, 4_000_000, 4_000_000]


def test_add_params_num_and_final_flag_groups_are_independent():
    # Two different (params, data, seed) groups must each get their own
    # final-checkpoint flag, not share one global max step.
    df = pl.DataFrame(
        {
            "params": ["4M", "6M"],
            "data": ["recipe-a", "recipe-a"],
            "seed": ["default", "default"],
            "step": [100, 50],
        }
    )
    out = dd._add_params_num_and_final_flag(df)
    assert all(out["is_final_checkpoint"].to_list())


# ---------------------------------------------------------------------------
# recipes README markdown table parsing
# ---------------------------------------------------------------------------

_FIXTURE_README = """\
# DataDecide

Some intro text.

## 25 Data Recipes

Some more text describing the table below.

| **Source**  | **Recipe**                       | **Description**   |
|-------------|-----------------------------------|--------------------|
| Dolma1.7    | Original                          | The base corpus.  |
| FineWeb-Pro | [Original](https://example.com/x) | A cleaned corpus.  |

## 350 Models over Differences in Data in Scale

| Recipe | 4M | 6M |
|---|---|---|
| Dolma1.7 | [4M](https://example.com/4M) | [6M](https://example.com/6M) |
"""


def test_parse_recipes_markdown_table_extracts_correct_row_count():
    df = dd._parse_recipes_markdown_table(_FIXTURE_README)
    assert df.height == 2
    assert set(df.columns) == {"source", "recipe", "description"}


def test_parse_recipes_markdown_table_strips_markdown_links():
    df = dd._parse_recipes_markdown_table(_FIXTURE_README)
    fineweb_row = df.filter(pl.col("source") == "FineWeb-Pro")
    assert fineweb_row["recipe"].to_list() == ["Original"]


def test_parse_recipes_markdown_table_does_not_read_the_second_table():
    df = dd._parse_recipes_markdown_table(_FIXTURE_README)
    # The second table's header ("Recipe", "4M", "6M") must not appear as
    # a row or contaminate the first table's parse.
    assert "4M" not in df["source"].to_list()


def test_parse_recipes_markdown_table_raises_if_no_table_found():
    with pytest.raises(RuntimeError, match="could not find"):
        dd._parse_recipes_markdown_table("no tables here at all")


# ---------------------------------------------------------------------------
# _strip_markdown_links
# ---------------------------------------------------------------------------


def test_strip_markdown_links_replaces_link_with_text():
    assert dd._strip_markdown_links("[Original](https://example.com)") == "Original"


def test_strip_markdown_links_passes_through_plain_text():
    assert dd._strip_markdown_links("QC 10%") == "QC 10%"


# ---------------------------------------------------------------------------
# _cached_or_build / cached_revision -- the caching engine every load_*
# function shares. No network involved: `build` is a fake, and _CACHE_DIR
# is monkeypatched to a tmp_path so nothing here touches the real cache.
# ---------------------------------------------------------------------------


def test_cached_or_build_calls_build_on_first_call(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "_CACHE_DIR", tmp_path)
    calls = []

    def fake_build():
        calls.append(1)
        return pl.DataFrame({"x": [1, 2, 3]}), "deadbeef"

    df = dd._cached_or_build("thing", fake_build)

    assert len(calls) == 1
    assert df["x"].to_list() == [1, 2, 3]
    assert (tmp_path / "thing.parquet").exists()


def test_cached_or_build_does_not_call_build_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "_CACHE_DIR", tmp_path)
    calls = []

    def fake_build():
        calls.append(1)
        return pl.DataFrame({"x": [1, 2, 3]}), "deadbeef"

    dd._cached_or_build("thing", fake_build)
    df2 = dd._cached_or_build("thing", fake_build)

    assert len(calls) == 1  # build() only ran once -- the second call was a cache hit
    assert df2["x"].to_list() == [1, 2, 3]


def test_cached_or_build_writes_revision_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "_CACHE_DIR", tmp_path)
    dd._cached_or_build("thing", lambda: (pl.DataFrame({"x": [1]}), "abc123"))

    assert dd.cached_revision("thing") == "abc123"


def test_cached_revision_returns_none_before_anything_is_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "_CACHE_DIR", tmp_path)
    assert dd.cached_revision("never_built") is None


# ---------------------------------------------------------------------------
# _parse_eval_instance_counts
# ---------------------------------------------------------------------------


def test_parse_eval_instance_counts_collapses_to_one_row_per_task():
    raw = pl.DataFrame(
        {
            "task": ["arc_challenge", "arc_challenge", "boolq", "boolq"],
            "num_instances": [1172, 1172, 3270, 3270],
        }
    )
    out = dd._parse_eval_instance_counts(raw)
    assert out.height == 2
    assert dict(zip(out["task"].to_list(), out["num_instances"].to_list(), strict=True)) == {
        "arc_challenge": 1172,
        "boolq": 3270,
    }


def test_parse_eval_instance_counts_raises_if_num_instances_varies_within_a_task():
    raw = pl.DataFrame({"task": ["arc_challenge", "arc_challenge"], "num_instances": [1172, 1173]})
    with pytest.raises(RuntimeError, match="not constant"):
        dd._parse_eval_instance_counts(raw)
