"""Tests for pdt.analysis.instances (N7-09)."""

from __future__ import annotations

import io
import json
import tarfile

import numpy as np
import pytest

from pdt.analysis import decomposition as dc
from pdt.analysis import instances as inst


def _tar_with(tmp_path, files: dict[str, list[dict]]):
    path = tmp_path / "step-1.tar.gz"
    with tarfile.open(path, "w:gz") as tf:
        for name, rows in files.items():
            data = "\n".join(json.dumps(r) for r in rows).encode()
            info = tarfile.TarInfo(f"step-1/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_cell_correctness_primary_norm_and_mmlu_macro(tmp_path):
    def row(i, **m):
        return {"doc_id": i, "metrics": m}

    files = {
        "boolq-predictions.jsonl": [row(1, acc_raw=0, acc_per_char=1), row(0, acc_raw=1)],
        "mmlu_a-predictions.jsonl": [row(0, acc_raw=1)],
        "mmlu_b-predictions.jsonl": [row(0, acc_raw=0), row(1, acc_raw=0), row(2, acc_raw=0)],
        "boolq-metrics.json": [{"x": 1}],
    }
    cells = inst.cell_correctness(_tar_with(tmp_path, files))
    assert cells["boolq"]["correct"].tolist() == [1, 0]  # doc order, acc_raw for boolq
    # MMLU macro: subject a = 1.0, subject b = 0.0 -> 0.5 (micro would be 0.25)
    assert inst.weighted_score(cells["mmlu"]) == pytest.approx(0.5)
    assert cells["mmlu"]["block"].tolist() == [0, 1, 1, 1]


def test_item_gap_variance_matches_var_over_n_and_matrix():
    rng = np.random.default_rng(0)
    cbar = rng.integers(0, 4, (5, 40)) / 3.0
    w = np.full(40, 1 / 40)
    block = np.zeros(40, dtype=int)
    v = inst.paired_gap_item_variance(cbar[0], cbar[1], w, block)
    assert v == pytest.approx(np.var(cbar[0] - cbar[1], ddof=1) / 40)
    blocks = np.repeat([0, 1], 20)
    w2 = np.full(40, 1 / 40)
    m = inst.pairwise_item_gap_variance(cbar, w2, blocks)
    for i in range(5):
        for j in range(5):
            assert m[i, j] == pytest.approx(
                inst.paired_gap_item_variance(cbar[i], cbar[j], w2, blocks)
            )


def test_sign_test():
    a = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0])
    b = np.zeros(11)
    assert inst.paired_sign_test_p(a, b) < 0.01
    assert inst.paired_sign_test_p(b, b) == 1.0


def test_extra_target_noise_lowers_ceiling():
    rng = np.random.default_rng(3)
    mu = rng.normal(0, 1, (25, 1))
    p = mu + rng.normal(0, 0.5, (25, 3))
    t = mu + rng.normal(0, 0.5, (25, 3))
    ids = np.arange(25)
    base = dc.fit_decomposition(p, t, ids, n_draws=500, seed=1)
    same = dc.fit_decomposition(p, t, ids, n_draws=500, seed=1, extra_target_noise_var=0.0)
    more = dc.fit_decomposition(p, t, ids, n_draws=500, seed=1, extra_target_noise_var=0.2)
    assert same["ceiling"] == base["ceiling"]
    assert more["ceiling"] < base["ceiling"]
