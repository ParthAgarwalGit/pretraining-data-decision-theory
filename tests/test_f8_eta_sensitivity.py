"""F8 plots the joint wrong-certification rate with exact intervals (reviews of PRs #32/#33)."""

from __future__ import annotations

from pdt.viz import f8_eta_sensitivity as f8
from pdt.viz import style


def _fake_results():
    sweep6 = [
        {
            "eta_assumed": {"leader": e, "underdog": e},
            "joint_wrong_certified_rate": r,
            "joint_wrong_certified_ci95": [lo, hi],
            "round_cap_exhausted_rate": rc,
        }
        for e, r, lo, hi, rc in [
            (0.0, 0.05, 0.001, 0.249, 0.95),
            (0.1, 0.0, 0.0, 0.168, 1.0),
        ]
    ]
    sweep7 = [
        {"true_bias": b, "accuracy": dict.fromkeys(f8._BASELINE_COLORS, a)}
        for b, a in [(0.0, 1.0), (0.1, 0.0)]
    ]
    return {"delta": 0.1, "sweep": sweep6}, {"sweep": sweep7, "observed_gap": 0.05}


def test_generate_writes_a_pdf_from_joint_rate_fields(monkeypatch, tmp_path):
    d6, d7 = _fake_results()
    monkeypatch.setattr(f8, "load", lambda path: d6 if "p3_06" in path else d7)
    monkeypatch.setattr(style, "FIGURES_DIR", tmp_path)
    path = f8.generate()
    assert path.exists() and path.suffix == ".pdf"


def test_docstring_withdraws_the_identical_noise_claim():
    assert "withdrawn" in f8.__doc__
    assert "not a statistically detectable violation" in f8.__doc__.replace(
        "NOT a ", "not a "
    ).replace("\n", " ")
