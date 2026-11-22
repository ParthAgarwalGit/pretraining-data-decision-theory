"""N7-10 S2: is DataDecide's 3-seed noise model adequate? Checked on PolyPythia, which has
10 training seeds per size (14M-410M), scored on the N7-11 benchmarks by
experiments/n7_11_score.py (outputs in data/n7_11_outputs/pythia__*).

Pre-registered in docs/preregistration.md section 5, as amended on 2026-11-22
(Amendment 1, recorded before any PolyPythia output existed):
  S2a  calibrated 3-seed SD check: the median over (size, benchmark) cells of the mean
       over all 3-of-10 seed subsets of |sd3/sd10 - 1| is at most the 95th percentile of
       its exact Monte-Carlo null under i.i.d. Gaussian seed noise.
  S2b  Shapiro-Wilk p > 0.01 in at least 80% of cells.
If either fails: the noise model is reported as approximate and the non-Gaussian
sensitivity analysis (N7-03 hetero_t generator) is added. Refuses to run on an
incomplete set (5 sizes x 3 benchmarks x 10 seeds).
Writes results/n7_10_s2_polypythia.json.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

from pdt import provenance
from pdt.analysis import seed_noise as sn

_SPEC = importlib.util.spec_from_file_location(
    "n7_11_analysis", os.path.join(os.path.dirname(__file__), "n7_11_analysis.py")
)
an = importlib.util.module_from_spec(_SPEC)
sys.modules["n7_11_analysis"] = an
_SPEC.loader.exec_module(an)

_SIZES = ("14m", "31m", "70m", "160m", "410m")
_BENCHMARKS = ("sciq", "lambada_openai", "blimp")
_SEEDS = tuple(f"seed{s}" for s in range(10))
_SEED = 20261126
_N_SIM = 20000


def load_cells(root: Path = an.OUTPUTS) -> tuple[np.ndarray, list[str], list[str]]:
    scores: dict[tuple, float] = {}
    for done in sorted(root.glob("pythia__*/done.json")):
        job = json.loads(done.read_text(encoding="utf-8"))["job"]
        for b in _BENCHMARKS:
            scores[(job["size"], b, job["seed_label"])] = an.benchmark_score(
                b, done.parent / f"{b}.parquet"
            )
    labels, rows, missing = [], [], []
    for size in _SIZES:
        for b in _BENCHMARKS:
            vals = [scores.get((size, b, s)) for s in _SEEDS]
            if any(v is None for v in vals):
                missing.append(f"{size}/{b}")
            labels.append(f"{size}/{b}")
            rows.append([np.nan if v is None else v for v in vals])
    return np.array(rows), labels, missing


def main() -> None:
    cells, labels, missing = load_cells()
    if missing:
        raise SystemExit(f"incomplete PolyPythia outputs, refusing to analyse: {missing}")
    s2a = sn.calibrated_sd_check(cells, n_sim=_N_SIM, seed=_SEED)
    s2b = sn.shapiro_check(cells)
    payload = {
        "status": "CONFIRMATORY check S2 (pre-registered; Amendment 1 of 2026-11-22)",
        "cells": labels,
        "seed_scores": cells.tolist(),
        "s2a_calibrated_three_seed_sd": s2a,
        "s2b_shapiro": s2b,
        "noise_model_adequate": bool(s2a["passed"] and s2b["passed"]),
        "consequence_if_not": "report noise model as approximate; add hetero_t sensitivity",
    }
    provenance.write_result(
        "results/n7_10_s2_polypythia.json",
        payload=payload,
        config={
            "task": "N7-10-S2",
            "seed": _SEED,
            "n_sim": _N_SIM,
            "preregistration_commit": "ff4b978",
            "amendment": "docs/preregistration.md Amendment 1 (2026-11-22)",
        },
    )
    print("wrote results/n7_10_s2_polypythia.json", s2a["passed"], s2b["passed"])


if __name__ == "__main__":
    main()
