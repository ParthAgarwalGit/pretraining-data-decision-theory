"""DataDecide per-instance predictions (N7-09): loading, primary-metric scoring, and the
item-level quantities pre-registered in docs/preregistration.md section 4.

Archives come from `allenai/DataDecide-eval-instances` (one nested `step-N.tar.gz` per
recipe/size/seed, prefetched into `data/cache/datadecide_instances/<slug>/...` with a
manifest naming the step nearest the frame's final step). Each task's primary metric
(`primary_metric` in the frame) is one normalisation of the per-instance correctness;
MMLU is the macro average over its 57 subjects.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
from scipy import stats

INSTANCE_DIR = Path("data/cache/datadecide_instances")

RECIPE_SLUGS = {
    "c4": "C4",
    "dclm-baseline": "DCLM-Baseline",
    "dclm-baseline-top-10p": "DCLM-Baseline (QC 10%)",
    "dclm-baseline-top-20p": "DCLM-Baseline (QC 20%)",
    "dclm-baseline-top-fw2-7p": "DCLM-Baseline (QC 7%, FW2)",
    "dclm-baseline-top-fw3-7p": "DCLM-Baseline (QC 7%, FW3)",
    "dclm-baseline-top-fw-10p": "DCLM-Baseline (QC FW 10%)",
    "dclm-baseline-top-fw-3p": "DCLM-Baseline (QC FW 3%)",
    "dclm-baseline-25p-dolma1.7-75p": "DCLM-Baseline 25% / Dolma 75%",
    "dclm-baseline-50p-dolma1.7-50p": "DCLM-Baseline 50% / Dolma 50%",
    "dclm-baseline-75p-dolma1.7-25p": "DCLM-Baseline 75% / Dolma 25%",
    "dolma1.6++": "Dolma1.6++",
    "dolma1.7": "Dolma1.7",
    "dolma1.7-no-flan": "Dolma1.7 (no Flan)",
    "dolma1.7-no-reddit": "Dolma1.7 (no Reddit)",
    "dolma1.7-no-code": "Dolma1.7 (no code)",
    "dolma1.7-no-math-no-code": "Dolma1.7 (no math, code)",
    "falcon": "Falcon",
    "falcon-with-cc": "Falcon+CC",
    "falcon-with-cc-top-10p": "Falcon+CC (QC 10%)",
    "falcon-with-cc-top-20p": "Falcon+CC (QC 20%)",
    "falcon-with-cc-top-orig-10p": "Falcon+CC (QC Orig 10%)",
    "falcon-with-cc-top-tulu-10p": "Falcon+CC (QC Tulu 10%)",
    "fineweb-edu": "FineWeb-Edu",
    "fineweb-pro": "FineWeb-Pro",
}

# Normalisation that reproduces the frame's `primary_metric` per task (checked in I0).
PRIMARY_NORM = {
    "arc_challenge": "acc_uncond",
    "arc_easy": "acc_per_char",
    "boolq": "acc_raw",
    "csqa": "acc_uncond",
    "hellaswag": "acc_per_char",
    "mmlu": "acc_raw",
    "openbookqa": "acc_uncond",
    "piqa": "acc_per_char",
    "socialiqa": "acc_per_char",
    "winogrande": "acc_raw",
}


def cell_correctness(tar_path: Path) -> dict[str, dict]:
    """Per task: ``{"correct": int array over items (doc_id order), "weight": item weights
    summing to 1 (MMLU: 1/(57 * n_subject) so the mean is the subject macro average)}``.
    """
    per_task: dict[str, list] = {}
    mmlu: dict[str, list] = {}
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if not m.name.endswith("-predictions.jsonl"):
                continue
            task = m.name.rsplit("/", 1)[-1].removesuffix("-predictions.jsonl")
            base = "mmlu" if task.startswith("mmlu_") else task
            norm = PRIMARY_NORM.get(base)
            if norm is None:
                continue
            rows = []
            for line in tf.extractfile(m):
                r = json.loads(line)
                rows.append((int(r["doc_id"]), int(r["metrics"][norm])))
            rows.sort()
            target = mmlu if base == "mmlu" else per_task
            target.setdefault(task, []).extend(rows)
    out = {}
    for task, rows in per_task.items():
        c = np.array([v for _, v in rows], dtype=np.int8)
        out[task] = {
            "correct": c,
            "weight": np.full(len(c), 1.0 / len(c)),
            "block": np.zeros(len(c), dtype=np.int16),
        }
    if mmlu:
        subjects = sorted(mmlu)
        cs, ws, bs = [], [], []
        for b, s in enumerate(subjects):
            c = np.array([v for _, v in mmlu[s]], dtype=np.int8)
            cs.append(c)
            ws.append(np.full(len(c), 1.0 / (len(subjects) * len(c))))
            bs.append(np.full(len(c), b, dtype=np.int16))
        out["mmlu"] = {
            "correct": np.concatenate(cs),
            "weight": np.concatenate(ws),
            "block": np.concatenate(bs),
        }
    return out


def weighted_score(cell: dict) -> float:
    return float(np.sum(cell["correct"] * cell["weight"]))


def paired_gap_item_variance(
    c_a: np.ndarray, c_b: np.ndarray, w: np.ndarray, block: np.ndarray
) -> float:
    """Item-sampling variance of the weighted mean of per-item differences ``d = c_a - c_b``
    (``c_*`` are per-item correctness averaged over seeds; items shared by both recipes).
    Uses the with-replacement (bootstrap) variance ``sum w_i^2 (d_i - dbar)^2 * n/(n-1)``
    within each block (MMLU subject; a single block otherwise), which equals ``var(d)/n``
    for one equal-weight block."""
    d = c_a - c_b
    var = 0.0
    for b in np.unique(block):
        blk = block == b
        n = int(blk.sum())
        if n < 2:
            continue
        wv = float(w[blk][0])
        var += (wv**2) * n * np.var(d[blk], ddof=1)
    return float(var)


def pairwise_item_gap_variance(cbar: np.ndarray, w: np.ndarray, block: np.ndarray) -> np.ndarray:
    """(R, R) matrix of ``paired_gap_item_variance`` for every recipe pair (vectorised).

    ``cbar`` is (R, n_items) seed-averaged correctness; blocks must be contiguous."""
    r = cbar.shape[0]
    out = np.zeros((r, r))
    starts = np.flatnonzero(np.r_[True, block[1:] != block[:-1]])
    ends = np.r_[starts[1:], len(block)]
    for i in range(r):
        d = cbar[i][None, :] - cbar  # (R, n)
        for a, b in zip(starts, ends, strict=True):
            n = b - a
            if n < 2:
                continue
            out[i] += (w[a] ** 2) * n * np.var(d[:, a:b], axis=1, ddof=1)
    return out


def paired_sign_test_p(c_a: np.ndarray, c_b: np.ndarray) -> float:
    """Two-sided exact sign test over items of ``c_a - c_b`` (seed-averaged correctness);
    zero differences are dropped."""
    d = c_a - c_b
    pos, neg = int(np.sum(d > 0)), int(np.sum(d < 0))
    if pos + neg == 0:
        return 1.0
    return float(stats.binomtest(pos, pos + neg, 0.5).pvalue)
