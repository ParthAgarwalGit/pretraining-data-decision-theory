"""Tests for pdt.data.pythia -- see plan/02-phase1-datadecide.md P1-10.

Network-touching functions (`download_checkpoint_eval`, `build_frame`) are
marked `# pragma: no cover` in the module and verified by hand against the
live GitHub repo -- see the P1-10 PR description. Pure logic (URL
construction, JSON parsing) is tested here with no network access.
"""

from __future__ import annotations

import pytest

from pdt.data import pythia

# ---------------------------------------------------------------------------
# raw_url() / _model_dir_name() / _file_name()
# ---------------------------------------------------------------------------


def test_raw_url_standard_variant():
    url = pythia.raw_url("410m", "standard", 143000)
    assert url == (
        "https://raw.githubusercontent.com/EleutherAI/pythia/main/evals/pythia-v1/"
        "pythia-410m/zero-shot/410m_step143000.json"
    )


def test_raw_url_deduped_variant():
    url = pythia.raw_url("410m", "deduped", 143000)
    assert url == (
        "https://raw.githubusercontent.com/EleutherAI/pythia/main/evals/pythia-v1/"
        "pythia-410m-deduped/zero-shot/410m-deduped_step143000.json"
    )


def test_raw_url_step_zero():
    url = pythia.raw_url("70m", "standard", 0)
    assert url.endswith("70m_step0.json")


def test_raw_url_rejects_unknown_size():
    with pytest.raises(ValueError, match="unknown Pythia size"):
        pythia.raw_url("999m", "standard", 0)


def test_raw_url_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown variant"):
        pythia.raw_url("70m", "not_a_variant", 0)


def test_all_sizes_produce_valid_urls_for_both_variants():
    # A cheap end-to-end sanity check across the full declared SIZES x
    # VARIANTS grid -- every combination must build a URL without raising.
    for size in pythia.SIZES:
        for variant in pythia.VARIANTS:
            url = pythia.raw_url(size, variant, pythia.FINAL_STEP)
            assert url.startswith("https://raw.githubusercontent.com/EleutherAI/pythia/")
            assert url.endswith(".json")


# ---------------------------------------------------------------------------
# parse_eval_json()
# ---------------------------------------------------------------------------


def _fixture_raw():
    return {
        "results": {
            "arc_challenge": {"acc": 0.25, "acc_stderr": 0.01, "acc_norm": 0.27},
            "winogrande": {"acc": 0.51, "acc_stderr": 0.02},
            "crows_pairs_english_gender": {"likelihood_difference": 2.4},
        },
        "versions": {"arc_challenge": 0},
    }


def test_parse_eval_json_returns_everything_with_no_whitelist():
    parsed = pythia.parse_eval_json(_fixture_raw())
    assert set(parsed.keys()) == {"arc_challenge", "winogrande", "crows_pairs_english_gender"}


def test_parse_eval_json_filters_to_whitelist():
    parsed = pythia.parse_eval_json(
        _fixture_raw(), task_whitelist=frozenset({"arc_challenge", "winogrande"})
    )
    assert set(parsed.keys()) == {"arc_challenge", "winogrande"}
    assert parsed["arc_challenge"]["acc"] == 0.25


def test_parse_eval_json_whitelist_excludes_unlisted_tasks():
    parsed = pythia.parse_eval_json(_fixture_raw(), task_whitelist=frozenset({"arc_challenge"}))
    assert "crows_pairs_english_gender" not in parsed


def test_parse_eval_json_handles_missing_results_key():
    assert pythia.parse_eval_json({"versions": {}}) == {}


def test_parse_eval_json_empty_whitelist_returns_nothing():
    parsed = pythia.parse_eval_json(_fixture_raw(), task_whitelist=frozenset())
    assert parsed == {}


# ---------------------------------------------------------------------------
# module-level constants
# ---------------------------------------------------------------------------


def test_final_step_is_the_max_of_steps():
    assert pythia.FINAL_STEP == 143000


def test_sizes_and_variants_are_nonempty():
    assert len(pythia.SIZES) == 8
    assert pythia.VARIANTS == ("standard", "deduped")
