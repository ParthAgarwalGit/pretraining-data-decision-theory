"""Tests for pdt.hub -- see plan/01-phase0-setup.md task P0-08.

`huggingface_hub.upload_folder` is always monkeypatched to raise if it's
ever actually called: the whole point of these tests is proving the two
refusal guards run *before* any network call, so a test that let a real
(mocked-to-succeed) upload happen would be testing less than it looks like
it's testing.
"""

from __future__ import annotations

import pytest

from pdt import hub, provenance


def _upload_folder_should_not_be_called(**kwargs):
    raise AssertionError(
        f"huggingface_hub.upload_folder was called with {kwargs!r} -- it "
        "should never be reached when a refusal guard should have fired first."
    )


def test_push_results_refuses_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "_git_is_clean", lambda: False)
    monkeypatch.setattr(hub.huggingface_hub, "upload_folder", _upload_folder_should_not_be_called)

    with pytest.raises(hub.DirtyTreeError):
        hub.push_results(tmp_path, "someuser/somedataset", "test push")


def test_push_results_refuses_invalid_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "_git_is_clean", lambda: True)
    monkeypatch.setattr(hub.huggingface_hub, "upload_folder", _upload_folder_should_not_be_called)

    # A hand-written file with no provenance block -- exactly what
    # provenance.validate() exists to catch.
    (tmp_path / "bad.json").write_text('{"data": {}}', encoding="utf-8")

    with pytest.raises(hub.InvalidResultsError):
        hub.push_results(tmp_path, "someuser/somedataset", "test push")


def test_push_results_refuses_dirty_tree_before_checking_provenance(tmp_path, monkeypatch):
    # Order matters for a clear error message: a dirty tree should be
    # reported as dirty, not accidentally masked by a provenance failure
    # that's really a symptom of the same uncommitted state.
    monkeypatch.setattr(hub, "_git_is_clean", lambda: False)
    monkeypatch.setattr(hub.huggingface_hub, "upload_folder", _upload_folder_should_not_be_called)
    (tmp_path / "bad.json").write_text('{"data": {}}', encoding="utf-8")

    with pytest.raises(hub.DirtyTreeError):
        hub.push_results(tmp_path, "someuser/somedataset", "test push")


def test_push_results_uploads_when_clean_and_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "_git_is_clean", lambda: True)
    monkeypatch.setattr(provenance, "_git_dirty", lambda: False)
    provenance.write_result(tmp_path / "good.json", payload={"x": 1}, config={})

    calls = []

    class FakeCommitInfo:
        commit_url = "https://huggingface.co/datasets/someuser/somedataset/commit/abc123"

    def fake_upload_folder(**kwargs):
        calls.append(kwargs)
        return FakeCommitInfo()

    monkeypatch.setattr(hub.huggingface_hub, "upload_folder", fake_upload_folder)

    result = hub.push_results(tmp_path, "someuser/somedataset", "test push", repo_type="dataset")

    assert result == "https://huggingface.co/datasets/someuser/somedataset/commit/abc123"
    assert len(calls) == 1
    assert calls[0]["repo_id"] == "someuser/somedataset"
    assert calls[0]["repo_type"] == "dataset"
    assert calls[0]["folder_path"] == str(tmp_path)
    assert calls[0]["commit_message"] == "test push"
