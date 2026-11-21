"""Tests for the N7-11 confirmatory analysis loader and decision rule."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_PATH = Path(__file__).resolve().parents[1] / "experiments" / "n7_11_analysis.py"
_SPEC = importlib.util.spec_from_file_location("n7_11_analysis", _PATH)
an = importlib.util.module_from_spec(_SPEC)
sys.modules["n7_11_analysis"] = an
_SPEC.loader.exec_module(an)


def _job(tmp, recipe, size, seed, scores):
    d = tmp / f"dd__{recipe}__{size}__{seed}"
    d.mkdir()
    job = {"kind": "datadecide", "recipe": recipe, "size": size, "seed_label": seed}
    (d / "done.json").write_text(json.dumps({"job": job}))
    pl.DataFrame({"correct_per_char": [scores["sciq"]], "correct_raw": [1]}).write_parquet(
        d / "sciq.parquet"
    )
    pl.DataFrame({"greedy_match": [scores["lambada_openai"]]}).write_parquet(
        d / "lambada_openai.parquet"
    )
    # BLiMP: subset a has 1 item (correct), subset b has 3 items (wrong) -> macro 0.5
    pl.DataFrame(
        {"item_id": ["a/0", "b/0", "b/1", "b/2"], "correct_raw": [1, 0, 0, 0]}
    ).write_parquet(d / "blimp.parquet")


def test_blimp_is_subset_macro_average(tmp_path):
    _job(tmp_path, "R", "1B", "default", {"sciq": 1, "lambada_openai": 0})
    assert an.benchmark_score("blimp", tmp_path / "dd__R__1B__default" / "blimp.parquet") == 0.5


def test_incomplete_outputs_are_reported(tmp_path):
    _job(tmp_path, "R", "1B", "default", {"sciq": 1, "lambada_openai": 0})
    _mats, recipes, missing = an.load_matrices(tmp_path)
    assert recipes == ["R"]
    assert "sciq/1B" in missing and "blimp/150M" in missing


def test_decision_rule_needs_mean_and_two_benchmarks():
    pm = np.array([0.05, 0.1, 0.05, 0.95, 0.0])
    lo = np.array([0.03, 0.01, 0.01, 0.9, -0.02])
    hi = np.array([0.07, 0.2, 0.1, 0.97, 0.02])
    good = np.array([0.05, 0.1, 0.05, 0.95, 0.0])
    bad = np.array([0.0, -0.1, 0.0, 0.5, 0.2])
    d = an._decide(pm, lo, hi, [good, bad, bad], ["a", "b", "c"])
    assert all(v["mean_criterion_met"] for v in d.values())
    assert not any(v["supported"] for v in d.values())  # only 1 of 3 in direction
    d2 = an._decide(pm, lo, hi, [good, good, bad], ["a", "b", "c"])
    assert all(v["supported"] for v in d2.values())
    lo_c5 = lo.copy()
    lo_c5[4] = -0.06
    assert not an._decide(pm, lo_c5, hi, [good, good, good], ["a", "b", "c"])["C5"]["supported"]


@pytest.mark.parametrize("bench", ["sciq", "lambada_openai"])
def test_simple_benchmark_scores(tmp_path, bench):
    _job(tmp_path, "R", "1B", "default", {"sciq": 1, "lambada_openai": 0})
    expected = {"sciq": 1.0, "lambada_openai": 0.0}[bench]
    path = tmp_path / "dd__R__1B__default" / f"{bench}.parquet"
    assert an.benchmark_score(bench, path) == expected
