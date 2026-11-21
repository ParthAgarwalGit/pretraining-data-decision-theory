"""Build and validate the N7-11 / N7-10-S2 scoring job list -> configs/n7_11_jobs.json.

DataDecide: 25 recipes x 3 seeds x {150M, 300M, 530M, 1B} at the frame's final step
(revision `step<final>-seed-<label>`), the checkpoint set frozen in
docs/preregistration.md section 1. PolyPythia (S2): pythia-{14m,31m,70m,160m,410m}
original (seed 0) + seed1..9 at step143000. Every revision is checked to exist on the Hub
before it is written, so the GPU workers never discover a missing checkpoint mid-run.
Run in the repo env (needs the cached DataDecide frame and network).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi

from pdt.data import frame as frame_mod
from pdt.eval import scoring

_OUT = Path("configs/n7_11_jobs.json")
_SIZES = ("150M", "300M", "530M", "1B")
_BENCHMARKS = ("sciq", "lambada_openai", "blimp")
_HF_SLUG = {
    "C4": "c4",
    "DCLM-Baseline": "dclm-baseline",
    "DCLM-Baseline (QC 10%)": "dclm-baseline-qc-10p",
    "DCLM-Baseline (QC 20%)": "dclm-baseline-qc-20p",
    "DCLM-Baseline (QC 7%, FW2)": "dclm-baseline-qc-7p-fw2",
    "DCLM-Baseline (QC 7%, FW3)": "dclm-baseline-qc-7p-fw3",
    "DCLM-Baseline (QC FW 10%)": "dclm-baseline-qc-fw-10p",
    "DCLM-Baseline (QC FW 3%)": "dclm-baseline-qc-fw-3p",
    "DCLM-Baseline 25% / Dolma 75%": "dclm-baseline-25p-dolma1.7-75p",
    "DCLM-Baseline 50% / Dolma 50%": "dclm-baseline-50p-dolma1.7-50p",
    "DCLM-Baseline 75% / Dolma 25%": "dclm-baseline-75p-dolma1.7-25p",
    "Dolma1.6++": "dolma1_6plus",
    "Dolma1.7": "dolma1_7",
    "Dolma1.7 (no Flan)": "dolma1_7-no-flan",
    "Dolma1.7 (no Reddit)": "dolma1_7-no-reddit",
    "Dolma1.7 (no code)": "dolma1_7-no-code",
    "Dolma1.7 (no math, code)": "dolma1_7-no-math-code",
    "Falcon": "falcon",
    "Falcon+CC": "falcon-and-cc",
    "Falcon+CC (QC 10%)": "falcon-and-cc-qc-10p",
    "Falcon+CC (QC 20%)": "falcon-and-cc-qc-20p",
    "Falcon+CC (QC Orig 10%)": "falcon-and-cc-qc-orig-10p",
    "Falcon+CC (QC Tulu 10%)": "falcon-and-cc-qc-tulu-10p",
    "FineWeb-Edu": "fineweb-edu",
    "FineWeb-Pro": "fineweb-pro",
}
_PYTHIA_SIZES = ("14m", "31m", "70m", "160m", "410m")


def _branches(api: HfApi, repo: str) -> set[str]:
    return {b.name for b in api.list_repo_refs(repo).branches}


def main() -> None:
    api = HfApi()
    lf = frame_mod.build_frame(source="macro_avg", metrics=("primary_metric",))
    lf = lf.filter(pl.col("is_final") & pl.col("params_str").is_in(_SIZES))
    cells = (
        lf.select(["recipe", "params_str", "seed", "step"])
        .unique()
        .sort(["params_str", "recipe", "seed"])
    )
    assert set(cells["recipe"].unique().to_list()) == set(_HF_SLUG), "recipe map mismatch"
    jobs, missing, refs = [], [], {}
    for rec, size, seed, step in cells.iter_rows():
        repo = f"allenai/DataDecide-{_HF_SLUG[rec]}-{size}"
        rev = scoring.datadecide_revision(step, seed)
        refs.setdefault(repo, _branches(api, repo))
        entry = {
            "job_id": f"dd__{_HF_SLUG[rec]}__{size}__{seed.replace(' ', '-')}",
            "kind": "datadecide",
            "repo_id": repo,
            "revision": rev,
            "recipe": rec,
            "size": size,
            "seed_label": seed,
        }
        (jobs if rev in refs[repo] else missing).append(entry)
    for size in _PYTHIA_SIZES:
        for s in range(10):
            repo = f"EleutherAI/pythia-{size}" + (f"-seed{s}" if s else "")
            refs.setdefault(repo, _branches(api, repo))
            entry = {
                "job_id": f"pythia__{size}__seed{s}",
                "kind": "hf",
                "repo_id": repo,
                "revision": "step143000",
                "size": size,
                "seed_label": f"seed{s}",
            }
            (jobs if "step143000" in refs[repo] else missing).append(entry)
    payload = {
        "description": "N7-11 (DataDecide re-scoring) and N7-10 S2 (PolyPythia) scoring jobs",
        "preregistration_commit": "ff4b978",
        "benchmarks": list(_BENCHMARKS),
        "dtype": "float16",
        "n_jobs": len(jobs),
        "missing": missing,
        "jobs": jobs,
    }
    _OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {_OUT}: {len(jobs)} jobs, {len(missing)} missing")


if __name__ == "__main__":
    main()
