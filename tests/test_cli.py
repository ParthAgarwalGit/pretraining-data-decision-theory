"""Tests for pdt.cli -- plan/04-phase3-algorithm.md P3-08."""

from __future__ import annotations

import json

import pytest
import yaml

import pdt.cli as cli_module
from pdt.bai.oracle import DataExhaustedError, FiniteDataOracle
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


def test_build_oracle_datadecide_is_wrapped_in_finite_data_oracle():
    # Regression for PR #34's review, P2: the datadecide backend must be
    # protected against its own finite replicate pool, not handed to ETS
    # raw (where pool exhaustion would surface as an unhandled
    # IndexError instead of a clean, distinct result).
    oracle = _build_oracle({"oracle": {"type": "datadecide", "task": "arc_challenge"}})
    assert isinstance(oracle, FiniteDataOracle)


def test_build_oracle_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown oracle.type"):
        _build_oracle({"oracle": {"type": "not_a_real_backend"}})


def test_run_selection_returns_a_selection_result():
    result = run_selection(_SYNTHETIC_CFG)
    assert result.outcome in ("certified", "abstained")
    assert result.recipe in ("a", "b")


def test_run_selection_default_variance_mode_lists_its_assumptions_when_certified():
    result = run_selection(_SYNTHETIC_CFG)
    assert result.certificate.get("variance_mode", "known_sigma2") == "known_sigma2"
    if result.outcome == "certified":
        assert any(a.startswith("A1") for a in result.certificate["assumptions"])


def test_run_selection_hc0_heuristic_never_reports_certified():
    cfg = dict(_SYNTHETIC_CFG, variance_mode="hc0_heuristic")
    result = run_selection(cfg)
    assert result.outcome in ("recommended", "abstained")
    assert result.outcome != "certified"


def test_run_selection_rejects_unknown_variance_mode():
    with pytest.raises(ValueError, match="variance_mode"):
        run_selection(dict(_SYNTHETIC_CFG, variance_mode="bogus"))


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


class _TinyPoolOracle:
    """A fake oracle with a real+pseudo pool of exactly 1 seed per
    (recipe, scale) -- exhausts almost immediately, for a fast,
    deterministic CLI-level exhaustion test without needing a real
    adaptive run to grind through dozens of rounds first."""

    def __init__(self):
        self._scales = [Scale(n=n, d=2 * n) for n in [1e6, 3e6, 1e7, 3e7]]

    def pull(self, recipe: str, scale: Scale, seed: int) -> float:
        if seed >= 1:
            raise IndexError(f"only 1 real seed for {recipe}/{scale}")
        return 0.5

    def cost(self, scale: Scale) -> float:
        return scale.compute

    def available_scales(self) -> list[Scale]:
        return list(self._scales)


def test_main_select_reports_data_exhausted_instead_of_crashing(tmp_path, capsys, monkeypatch):
    # Regression for PR #34's review, P2: exhausting the datadecide
    # backend's finite pool must produce a clean, distinct JSON result,
    # never an unhandled crash. Monkeypatches the datadecide branch to a
    # tiny fake oracle (via FiniteDataOracle, exactly as _build_oracle
    # wires the real one) so exhaustion is fast and deterministic rather
    # than depending on a real multi-round adaptive run.
    monkeypatch.setattr(
        cli_module, "_build_oracle", lambda cfg: FiniteDataOracle(_TinyPoolOracle())
    )

    # min_pulls_per_pair=2 with a 1-real-seed fake pool: exhaustion
    # happens deterministically during the WARM-UP sweep itself (every
    # (recipe, scale) pair's second pull asks for seed=1, which the fake
    # oracle never has), independent of any certification-math edge case
    # that might otherwise make a degenerate all-identical-value instance
    # resolve before ever requesting a second replicate.
    cfg = dict(
        _SYNTHETIC_CFG,
        oracle={"type": "datadecide", "task": "irrelevant"},
        min_pulls_per_pair=2,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg), encoding="utf-8")

    main(["select", "--config", str(config_path)])

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["outcome"] == "data_exhausted"
    assert payload["recipe"] is None
    assert "certificate" in payload
    assert "reason" in payload["certificate"]


def test_finite_data_oracle_raises_data_exhausted_error_not_index_error():
    oracle = FiniteDataOracle(_TinyPoolOracle())
    scale = Scale(n=1e6, d=2e6)

    assert oracle.pull("recipe", scale, seed=0) == 0.5
    with pytest.raises(DataExhaustedError):
        oracle.pull("recipe", scale, seed=1)
