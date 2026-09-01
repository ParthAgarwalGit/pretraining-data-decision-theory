"""Fabrication guard: flag numeric literals with no traceable source.

Mechanical backstop for plan/00-agent-protocol.md Rule 1 ("never invent a
number"). Scans paper/**/*.tex, README.md, and docs/**/*.md for numbers that
look like a reported result -- 2+ significant digits after a decimal point
-- and flags any that are not:

  (a) on a line explicitly marked `% NUMBER-OK` (LaTeX) or `<!-- NUMBER-OK -->`
      (Markdown), for numbers that are legitimately hand-written (dates,
      section numbers, hyperparameter choices stated as design decisions,
      version numbers, etc.), or
  (b) inside a citation-adjacent "prior work" context (a line containing
      `\\cite` in .tex, or a Markdown line citing another paper by name +
      year/arXiv id), where quoting someone else's published number is
      expected and is not a fabrication risk, or
  (c) present verbatim as a number somewhere in some file under results/.

Exit code: during Phase 0-2, this is advisory only (see the `--advisory`
flag used by ci.yml) -- it prints findings but always exits 0. From task
P6-01 onward, ci.yml drops --advisory and a finding becomes a hard CI
failure, because by then every reported number is supposed to come from a
results/*.json file.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 2+ digits after a decimal point: "0.80", "12.34", "3.14159" -- deliberately
# excludes bare integers (section numbers, years, "8 baselines") and single-
# decimal-digit numbers (too many false positives from e.g. "1.5x compute"
# design choices) to keep this guard focused on result-shaped numbers.
_NUMBER_RE = re.compile(r"(?<![\w.])\d+\.\d{2,}(?![\w.])")

_NUMBER_OK_MARKERS = ("NUMBER-OK",)

# A citation-adjacent line: LaTeX \cite-family commands, or a Markdown/plain
# citation shape like "(Author et al., 2025)" / "arXiv:1234.56789".
_CITE_RE = re.compile(
    r"\\cite\w*\{"  # \cite{...}, \citep{...}, \citet{...}, ...
    r"|arxiv\s*:\s*\d"
    r"|et al\.,?\s*\d{4}",
    re.IGNORECASE,
)

_SCAN_GLOBS = ("paper/**/*.tex", "README.md", "docs/**/*.md")


def _iter_target_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in _SCAN_GLOBS:
        files.extend(sorted(root.glob(pattern)))
    return files


def _numbers_in_results(root: Path) -> set[str]:
    """Every distinct number-shaped substring appearing in results/*.json."""
    numbers: set[str] = set()
    results_dir = root / "results"
    if not results_dir.exists():
        return numbers
    for json_path in results_dir.rglob("*.json"):
        text = json_path.read_text(encoding="utf-8")
        numbers.update(_NUMBER_RE.findall(text))
    return numbers


def find_orphan_numbers(root: Path) -> list[tuple[Path, int, str]]:
    """Return (file, line_number, line_text) for every unexplained number."""
    known_numbers = _numbers_in_results(root)
    orphans: list[tuple[Path, int, str]] = []

    for file_path in _iter_target_files(root):
        lines = file_path.read_text(encoding="utf-8").splitlines()
        for line_no, line in enumerate(lines, start=1):
            if any(marker in line for marker in _NUMBER_OK_MARKERS):
                continue
            if _CITE_RE.search(line):
                continue

            for match in _NUMBER_RE.finditer(line):
                if match.group() in known_numbers:
                    continue
                orphans.append((file_path, line_no, line.strip()))
                break  # one flag per offending line is enough

    return orphans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root to scan from (default: current directory).",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help=(
            "Print findings but always exit 0. Used in ci.yml through "
            "task P6-01; drop this flag from the workflow once the paper "
            "is being written so a finding becomes a hard CI failure."
        ),
    )
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    orphans = find_orphan_numbers(root)

    if not orphans:
        print("check_no_orphan_numbers: no unexplained numbers found.")
        return 0

    print(f"check_no_orphan_numbers: {len(orphans)} unexplained number(s) found:\n")
    for file_path, line_no, line in orphans:
        rel = file_path.relative_to(root)
        print(f"  {rel}:{line_no}: {line}")
    print(
        "\nEach number above must either come from a results/*.json file, be "
        "marked with a 'NUMBER-OK' comment (a genuine hand-written constant, "
        "not a result), or be a cited prior-work figure. See "
        "plan/00-agent-protocol.md Rule 1."
    )

    if args.advisory:
        print("\n(--advisory: not failing the build yet -- see task P6-01.)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
