"""Provenance stamping and results-directory validation.

Mechanical enforcement of plan/00-agent-protocol.md Rule 1 ("never invent a
number") and Rule 3 ("every result is reproducible"). Every JSON file
written to results/ carries a provenance block recording the exact code,
config, and environment that produced it -- from a clean (non-dirty) git
working tree. A result produced from a dirty tree is not reproducible and
`validate()` (wired into `make check`) rejects it.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Package versions worth recording on every result: the numeric/data-access
# stack whose version could plausibly change a computed number. Dev tooling
# (pytest, ruff, ...) is deliberately excluded -- it can't affect a result.
_TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "polars",
    "pyarrow",
    "huggingface_hub",
    "datasets",
    "statsmodels",
    "matplotlib",
)

_RESERVED_STAMP_KEYS = frozenset(
    {
        "git_sha",
        "git_dirty",
        "utc_timestamp",
        "python_version",
        "package_versions",
        "seed",
        "config",
    }
)


def _run_git(*args: str) -> str:
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _git_sha() -> str:
    try:
        return _run_git("rev-parse", "HEAD")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _git_dirty() -> bool:
    try:
        status = _run_git("status", "--porcelain")
        return bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Can't determine cleanliness -- fail safe as dirty rather than
        # silently claiming a result is reproducible when we don't know.
        return True


def _package_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def stamp(config: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the provenance block for a result.

    `config` is echoed verbatim under the `config` key; if it contains a
    `seed`, that value is also surfaced at the top level for easy scanning
    across many result files without digging into each config. `extra`
    merges additional fields (e.g. a dataset revision SHA) into the top
    level -- it must not collide with the reserved provenance keys, so a
    collision fails loudly rather than silently overwriting something the
    reproducibility contract depends on.
    """
    if extra:
        collisions = _RESERVED_STAMP_KEYS & extra.keys()
        if collisions:
            raise ValueError(
                f"extra field(s) {sorted(collisions)} collide with reserved "
                f"provenance keys {sorted(_RESERVED_STAMP_KEYS)}"
            )

    result: dict[str, Any] = {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "utc_timestamp": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "seed": config.get("seed"),
        "config": config,
    }
    if extra:
        result.update(extra)
    return result


class _ProvenanceEncoder(json.JSONEncoder):
    """Handles numpy/pandas scalars and arrays, and Path objects.

    Duck-types on the .tolist() protocol numpy, pandas, and similar array
    libraries all share, rather than hard-importing numpy -- so this module
    stays usable even where the `core` extra isn't installed. .tolist()
    alone (no separate .item() branch) is deliberate: numpy array types
    expose both .item() and .dtype, so a check keyed on .dtype cannot tell
    a scalar from a multi-element array -- but .tolist() handles both
    correctly (numpy scalars implement it too, returning the bare scalar).
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        tolist = getattr(o, "tolist", None)
        if callable(tolist):
            return tolist()
        return super().default(o)


def write_result(path: str | Path, payload: dict[str, Any], config: dict[str, Any]) -> None:
    """Write `{"provenance": stamp(config), "data": payload}` as pretty JSON.

    Creates parent directories. Refuses to overwrite an existing file unless
    the PDT_OVERWRITE=1 environment variable is set, so a re-run never
    silently clobbers a result someone might still be looking at.
    """
    out_path = Path(path)
    if out_path.exists() and os.environ.get("PDT_OVERWRITE") != "1":
        raise FileExistsError(
            f"{out_path} already exists. Set PDT_OVERWRITE=1 to overwrite "
            f"intentionally, or write to a new path."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    document = {"provenance": stamp(config), "data": payload}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, cls=_ProvenanceEncoder)
        f.write("\n")


def validate(results_dir: str | Path) -> int:
    """Check every *.json file under results_dir has valid provenance.

    A file passes if it parses as JSON, has a `provenance` object, that
    object has a non-empty `git_sha`, and `git_dirty` is exactly `False`.
    Prints a report and returns 0 if every file passes, 1 if any fails.
    A results_dir that doesn't exist, or has no JSON files, trivially
    passes -- there is nothing yet that could be wrong.
    """
    root = Path(results_dir)
    if not root.exists():
        print(f"pdt.provenance: {root} does not exist -- nothing to validate.")
        return 0

    json_files = sorted(root.rglob("*.json"))
    if not json_files:
        print(f"pdt.provenance: no JSON files under {root} -- nothing to validate.")
        return 0

    failures: list[tuple[Path, str]] = []
    for file_path in json_files:
        try:
            with file_path.open(encoding="utf-8") as f:
                document = json.load(f)
        except json.JSONDecodeError as exc:
            failures.append((file_path, f"invalid JSON: {exc}"))
            continue

        provenance_block = document.get("provenance") if isinstance(document, dict) else None
        if provenance_block is None:
            failures.append((file_path, "missing 'provenance' block"))
            continue
        if not isinstance(provenance_block, dict):
            failures.append((file_path, "'provenance' is not an object"))
            continue

        git_sha = provenance_block.get("git_sha")
        if not git_sha:
            failures.append((file_path, "provenance.git_sha is empty or missing"))
            continue

        git_dirty = provenance_block.get("git_dirty")
        if git_dirty is not False:
            failures.append((file_path, f"provenance.git_dirty is {git_dirty!r}, must be false"))
            continue

    for file_path, reason in failures:
        print(f"FAIL  {file_path}: {reason}", file=sys.stderr)

    passed = len(json_files) - len(failures)
    print(f"pdt.provenance: {passed}/{len(json_files)} result file(s) valid.")
    return 1 if failures else 0


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pdt.provenance")
    parser.add_argument(
        "--validate",
        metavar="DIR",
        help="Validate every JSON result file under DIR has valid provenance.",
    )
    args = parser.parse_args(argv)
    if args.validate:
        return validate(args.validate)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
