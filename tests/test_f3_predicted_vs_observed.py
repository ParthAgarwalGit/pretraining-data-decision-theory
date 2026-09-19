"""F3 keeps the two decision events apart (second review of PR #20)."""

from __future__ import annotations

import pytest

from pdt.viz import f3_predicted_vs_observed as f3
from pdt.viz import style


def _entry(best_arm=0.4, all_pairs=0.7):
    return {
        "observed_best_arm_accuracy": best_arm,
        "predicted_accuracy": 0.0,
        "counterfactual_sigma2_extrap_zero_predicted_accuracy": 0.1,
        "observed_accuracy": all_pairs,
    }


def _results(entry=None):
    entry = entry or _entry()
    return {
        "fitters": ["PowerLawN", "LogLinear"],
        "by_fitter": {
            "PowerLawN": {"S_fit_le_150M": dict(entry), "S_fit_le_300M": dict(entry)},
            "LogLinear": {"S_fit_le_150M": dict(entry)},
        },
    }


def test_panels_never_mix_the_two_events():
    a_fields = {field for field, *_ in f3._PANEL_A_SERIES}
    b_fields = {field for field, *_ in f3._PANEL_B_SERIES}
    assert "observed_accuracy" not in a_fields  # all-pairs accuracy is not plotted vs the bound
    assert b_fields == {"observed_accuracy"}
    assert "predicted_accuracy" not in b_fields  # no bound exists for the all-pairs event
    assert "observed_best_arm_accuracy" in a_fields


def test_results_predating_the_same_event_fix_are_rejected_not_silently_plotted():
    stale = _results()
    del stale["by_fitter"]["PowerLawN"]["S_fit_le_150M"]["observed_best_arm_accuracy"]
    with pytest.raises(KeyError, match="regenerate P1-08"):
        f3._require_best_arm_fields(stale)


def test_generate_writes_a_pdf_and_skips_unassessed_cells(monkeypatch, tmp_path):
    results = _results()
    results["by_fitter"]["LogLinear"]["S_fit_le_150M"]["observed_best_arm_accuracy"] = None
    monkeypatch.setattr(f3, "load", lambda _path: results)
    monkeypatch.setattr(style, "FIGURES_DIR", tmp_path)
    path = f3.generate()
    assert path.exists()
    assert path.suffix == ".pdf"
