"""Tests for pdt.cli -- plan/04-phase3-algorithm.md P3-08."""

from __future__ import annotations

import json

import pytest
import yaml

from pdt.cli import _build_oracle, _scale, main, run_selection
from pdt.scaling.base import Scale

_SYNTHETIC_CFG = {
    "recipes": ["a", "b"],
    "oracle": {
        "type": "synthetic",
        "fit_scales": [
            {"n": 1e6, "d": 2e7},
            {"n": 3e6, "d": 6e7},
            {"n": 1e7, "d": 2e8},
            {"n": 3e7, "d": 6e8},
        ],
    },
    "candidate_scales": [
        {"n": 1e6, "d": 2e7},
        {"n": 3e6, "d": 6e7},
        {"n": 1e7, "d": 2e8},
        {"n": 3e7, "d": 6e8},
    ],
    "target_scale": {"n": 1e9, "d": 2e10},
    "delta": 0.1,
    "eta": 0.03,
    "sigma2": 1e-4,
    "max_rounds": 200,
    "seed": 0,
}


def test_scale_coerces_yaml_scientific_notation_strings():
    # The exact PyYAML 1.1 gotcha found while building this CLI:
    # "1.0e6" (no exponent sign) parses as a string, not a float.
    parsed = yaml.safe_load("n: 1.0e6\nd: 2.0e7")
    scale = _scale(parsed)
    assert scale == Scale(n=1e6, d=2e7)
    assert isinstance(scale.n, float)


def test_scale_accepts_plain_floats():
    assert _scale({"n": 1000000.0, "d": 20000000.0}) == Scale(n=1e6, d=2e7)


def test_build_oracle_synthetic():
    oracle = _build_oracle(_SYNTHETIC_CFG)
    assert set(oracle.available_scales()) == {
        Scale(n=1e6, d=2e7),
        Scale(n=3e6, d=6e7),
        Scale(n=1e7, d=2e8),
        Scale(n=3e7, d=6e8),
    }


def test_build_oracle_datadecide():
    oracle = _build_oracle({"oracle": {"type": "datadecide", "task": "arc_challenge"}})
    assert len(oracle.available_scales()) == 14


def test_build_oracle_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown oracle.type"):
        _build_oracle({"oracle": {"type": "not_a_real_backend"}})


def test_run_selection_returns_a_selection_result():
    result = run_selection(_SYNTHETIC_CFG)
    assert result.outcome in ("certified", "abstained")
    assert result.recipe in ("a", "b")


def test_run_selection_eta_as_mapping():
    cfg = dict(_SYNTHETIC_CFG, eta={"a": 0.02, "b": 0.04})
    result = run_selection(cfg)
    assert result.recipe in ("a", "b")


def test_main_select_prints_json_result(tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(_SYNTHETIC_CFG), encoding="utf-8")

    main(["select", "--config", str(config_path)])

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["outcome"] in ("certified", "abstained")
    assert payload["recipe"] in ("a", "b")
    assert "certificate" in payload


def test_main_requires_a_command():
    with pytest.raises(SystemExit):
        main([])
