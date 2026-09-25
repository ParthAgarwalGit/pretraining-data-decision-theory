"""P1-07's refresh mode carries Monte-Carlo results over only when B and the scheme match."""

from __future__ import annotations

import json

import pytest

from experiments.p1_07_bound_coverage import (
    _B_MONTE_CARLO,
    _MC_SCHEME,
    _load_reused_monte_carlo,
)


def _write(path, *, b=_B_MONTE_CARLO, scheme=_MC_SCHEME):
    payload = {
        "provenance": {"git_sha": "abcdef1234567890"},
        "data": {
            "b_monte_carlo": b,
            "mc_scheme": scheme,
            "by_fitter": {
                "LogLinear": {
                    "S_fit_le_150M": {
                        "arc_easy": {
                            _MC_SCHEME: {
                                "empirical_error_rate": 0.25,
                                "n_valid_mc_replicates": 499,
                            },
                            "parametric_bootstrap": {
                                "empirical_error_rate": 0.25,
                                "n_valid_mc_replicates": 499,
                            },
                        }
                    }
                }
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reused_monte_carlo_is_loaded_with_its_source_sha(tmp_path):
    f = tmp_path / "old.json"
    _write(f)
    reused, sha = _load_reused_monte_carlo(str(f))
    assert sha == "abcdef1234567890"
    assert reused == {
        ("LogLinear", "S_fit_le_150M", "arc_easy"): {
            "empirical_error_rate": 0.25,
            "n_valid_replicates": 499,
        }
    }


@pytest.mark.parametrize("kwargs", [{"b": _B_MONTE_CARLO + 1}, {"scheme": "parametric_bootstrap"}])
def test_reuse_is_refused_when_monte_carlo_settings_differ(tmp_path, kwargs):
    f = tmp_path / "old.json"
    _write(f, **kwargs)
    with pytest.raises(ValueError, match="full pass is required"):
        _load_reused_monte_carlo(str(f))
