"""Tests for pdt.provenance -- see plan/01-phase0-setup.md task P0-04."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from pdt import provenance

# ---------------------------------------------------------------------------
# stamp()
# ---------------------------------------------------------------------------


def test_stamp_has_required_fields():
    result = provenance.stamp({"seed": 42, "task": "arc_challenge"})

    for key in (
        "git_sha",
        "git_dirty",
        "utc_timestamp",
        "python_version",
        "package_versions",
        "seed",
        "config",
    ):
        assert key in result

    assert isinstance(result["git_sha"], str)
    assert isinstance(result["git_dirty"], bool)
    assert isinstance(result["package_versions"], dict)
    assert result["seed"] == 42
    assert result["config"] == {"seed": 42, "task": "arc_challenge"}


def test_stamp_seed_defaults_to_none_when_absent():
    result = provenance.stamp({"task": "arc_challenge"})
    assert result["seed"] is None


def test_stamp_extra_merges_additional_fields():
    result = provenance.stamp({}, extra={"dataset_revision": "abc123"})
    assert result["dataset_revision"] == "abc123"


def test_stamp_extra_collision_with_reserved_key_raises():
    with pytest.raises(ValueError, match="git_sha"):
        provenance.stamp({}, extra={"git_sha": "not-the-real-one"})


def test_stamp_reflects_real_git_sha_of_this_checkout():
    result = provenance.stamp({})
    real_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert result["git_sha"] == real_sha


def test_stamp_tracks_known_packages():
    result = provenance.stamp({})
    # numpy is in the `core` extra, which `make setup` always installs.
    assert result["package_versions"]["numpy"] is not None


# ---------------------------------------------------------------------------
# write_result()
# ---------------------------------------------------------------------------


def test_write_result_creates_file_and_parent_dirs(tmp_path):
    out = tmp_path / "sub" / "dir" / "result.json"
    provenance.write_result(out, payload={"accuracy": 0.8}, config={"seed": 1})

    assert out.exists()
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["data"] == {"accuracy": 0.8}
    assert document["provenance"]["seed"] == 1


def test_write_result_refuses_overwrite_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("PDT_OVERWRITE", raising=False)
    out = tmp_path / "result.json"
    provenance.write_result(out, payload={"a": 1}, config={})

    with pytest.raises(FileExistsError):
        provenance.write_result(out, payload={"a": 2}, config={})


def test_write_result_allows_overwrite_with_env_var(tmp_path, monkeypatch):
    out = tmp_path / "result.json"
    provenance.write_result(out, payload={"a": 1}, config={})

    monkeypatch.setenv("PDT_OVERWRITE", "1")
    provenance.write_result(out, payload={"a": 2}, config={})

    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["data"] == {"a": 2}


def test_write_result_serializes_numpy_scalars_and_arrays(tmp_path):
    out = tmp_path / "result.json"
    payload = {
        "mean": np.float64(0.735),
        "counts": np.array([1, 2, 3]),
    }
    provenance.write_result(out, payload=payload, config={})

    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["data"]["mean"] == pytest.approx(0.735)
    assert document["data"]["counts"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_returns_zero_for_missing_directory(tmp_path):
    assert provenance.validate(tmp_path / "does_not_exist") == 0


def test_validate_returns_zero_for_empty_directory(tmp_path):
    assert provenance.validate(tmp_path) == 0


def test_validate_passes_on_result_written_by_write_result(tmp_path, monkeypatch):
    # Force a clean-tree stamp regardless of this checkout's real state --
    # during active development the real tree is essentially never clean.
    monkeypatch.setattr(provenance, "_git_dirty", lambda: False)
    out = tmp_path / "result.json"
    provenance.write_result(out, payload={"x": 1}, config={})

    assert provenance.validate(tmp_path) == 0


def test_validate_fails_on_missing_provenance_block(tmp_path):
    hand_written = tmp_path / "hand_written.json"
    hand_written.write_text(json.dumps({"data": {"x": 1}}), encoding="utf-8")

    assert provenance.validate(tmp_path) == 1


def test_validate_fails_on_dirty_git(tmp_path):
    dirty = tmp_path / "dirty.json"
    dirty.write_text(
        json.dumps({"provenance": {"git_sha": "abc123", "git_dirty": True}, "data": {}}),
        encoding="utf-8",
    )

    assert provenance.validate(tmp_path) == 1


def test_validate_fails_on_empty_git_sha(tmp_path):
    no_sha = tmp_path / "no_sha.json"
    no_sha.write_text(
        json.dumps({"provenance": {"git_sha": "", "git_dirty": False}, "data": {}}),
        encoding="utf-8",
    )

    assert provenance.validate(tmp_path) == 1


def test_validate_fails_on_invalid_json(tmp_path):
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{not valid json", encoding="utf-8")

    assert provenance.validate(tmp_path) == 1


def test_validate_fails_directory_if_any_file_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_git_dirty", lambda: False)
    good = tmp_path / "good.json"
    provenance.write_result(good, payload={}, config={})

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"data": {}}), encoding="utf-8")

    assert provenance.validate(tmp_path) == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_validate_returns_validate_result(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_git_dirty", lambda: False)
    provenance.write_result(tmp_path / "result.json", payload={}, config={})

    assert provenance._cli(["--validate", str(tmp_path)]) == 0


def test_cli_with_no_args_returns_nonzero():
    assert provenance._cli([]) == 1
