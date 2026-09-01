"""Tests for tools/check_no_orphan_numbers.py -- see plan/01-phase0-setup.md P0-05.

Loaded by file path (not `import tools....`) since tools/ is a directory of
standalone scripts, not an installed package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parent.parent / "tools" / "check_no_orphan_numbers.py"
_spec = importlib.util.spec_from_file_location("check_no_orphan_numbers", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
check_no_orphan_numbers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_no_orphan_numbers)


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "results").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "paper").mkdir()
    return tmp_path


def test_flags_a_bare_number_with_no_source(tmp_path):
    root = _make_repo(tmp_path)
    (root / "README.md").write_text("Our method reaches 0.83 accuracy.\n", encoding="utf-8")

    orphans = check_no_orphan_numbers.find_orphan_numbers(root)

    assert len(orphans) == 1
    assert orphans[0][0].name == "README.md"


def test_does_not_flag_a_number_present_in_results(tmp_path):
    root = _make_repo(tmp_path)
    (root / "results" / "p1_08.json").write_text('{"data": {"accuracy": 0.83}}', encoding="utf-8")
    (root / "README.md").write_text("Our method reaches 0.83 accuracy.\n", encoding="utf-8")

    orphans = check_no_orphan_numbers.find_orphan_numbers(root)

    assert orphans == []


def test_does_not_flag_a_number_ok_marked_line(tmp_path):
    root = _make_repo(tmp_path)
    (root / "README.md").write_text(
        "We use a dropout rate of 0.15.  <!-- NUMBER-OK: design choice, not a result -->\n",
        encoding="utf-8",
    )

    orphans = check_no_orphan_numbers.find_orphan_numbers(root)

    assert orphans == []


def test_does_not_flag_a_cited_prior_work_number(tmp_path):
    root = _make_repo(tmp_path)
    (root / "docs" / "related_work.md").write_text(
        "DataDecide reports 0.80 decision accuracy at 150M (Magnusson et al., 2025).\n",
        encoding="utf-8",
    )

    orphans = check_no_orphan_numbers.find_orphan_numbers(root)

    assert orphans == []


def test_does_not_flag_bare_integers_or_single_decimal_digits(tmp_path):
    root = _make_repo(tmp_path)
    (root / "README.md").write_text(
        "Trained 25 recipes across 14 sizes with a 1.5x token budget.\n", encoding="utf-8"
    )

    orphans = check_no_orphan_numbers.find_orphan_numbers(root)

    assert orphans == []


def test_does_not_scan_files_outside_the_target_globs(tmp_path):
    root = _make_repo(tmp_path)
    (root / "plan").mkdir()
    (root / "plan" / "02-phase1-datadecide.md").write_text(
        "Some pseudocode literal like 0.837 that lives in the plan, not the paper.\n",
        encoding="utf-8",
    )

    orphans = check_no_orphan_numbers.find_orphan_numbers(root)

    assert orphans == []


def test_advisory_mode_always_exits_zero(tmp_path, capsys):
    root = _make_repo(tmp_path)
    (root / "README.md").write_text("Reaches 0.83 accuracy.\n", encoding="utf-8")

    exit_code = check_no_orphan_numbers.main(["--root", str(root), "--advisory"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "0.83" in captured.out


def test_non_advisory_mode_exits_nonzero_on_a_finding(tmp_path):
    root = _make_repo(tmp_path)
    (root / "README.md").write_text("Reaches 0.83 accuracy.\n", encoding="utf-8")

    exit_code = check_no_orphan_numbers.main(["--root", str(root)])

    assert exit_code == 1


def test_clean_repo_exits_zero_even_without_advisory(tmp_path):
    root = _make_repo(tmp_path)
    (root / "README.md").write_text("Nothing but prose here.\n", encoding="utf-8")

    exit_code = check_no_orphan_numbers.main(["--root", str(root)])

    assert exit_code == 0
