"""Publishing results to the project's Hugging Face dataset repo.

See plan/01-phase0-setup.md task P0-08 and
plan/09-review-gates.md section 5 for the workflow this enforces: every
push is treated as a release (never from a dirty tree, and the content
being published must itself already satisfy the provenance contract that
makes it worth publishing in the first place).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import huggingface_hub

from pdt import provenance


class DirtyTreeError(RuntimeError):
    """Raised when push_results is called from an uncommitted working tree."""


class InvalidResultsError(RuntimeError):
    """Raised when local_dir fails provenance validation."""


def _git_is_clean() -> bool:  # pragma: no cover -- thin wrapper, same as
    # src/pdt/provenance.py's _git_dirty(); exercised indirectly by every
    # push_results test via monkeypatching.
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        return not result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Can't confirm cleanliness -- fail safe and refuse the push, same
        # reasoning as provenance._git_dirty()'s fail-safe-dirty default.
        return False


def push_results(
    local_dir: str | Path,
    repo_id: str,
    revision_msg: str,
    *,
    repo_type: str = "dataset",
) -> str:
    """Push every file under `local_dir` to the Hub repo `repo_id`.

    Refuses to run unless:
      1. the working tree is clean (a push from uncommitted code can never
         be traced back to the code that produced it), and
      2. every JSON file under `local_dir` passes `provenance.validate`
         (publishing a result that fails its own reproducibility contract
         would defeat the entire point of stamping it in the first place).

    Returns the resulting commit's URL.
    """
    if not _git_is_clean():
        raise DirtyTreeError(
            "refusing to push: working tree is not clean. Commit or stash "
            "your changes first -- a result pushed from a dirty tree is not "
            "reproducible from the commit that (allegedly) produced it."
        )

    if provenance.validate(local_dir) != 0:
        raise InvalidResultsError(
            f"refusing to push: {local_dir} failed provenance validation. "
            "Every JSON file must carry a clean-tree provenance block "
            "before it is worth publishing -- see src/pdt/provenance.py."
        )

    commit_info = huggingface_hub.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(local_dir),
        commit_message=revision_msg,
    )
    return commit_info.commit_url
