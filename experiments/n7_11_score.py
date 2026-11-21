"""N7-11 / N7-10-S2 GPU worker: score a shard of configs/n7_11_jobs.json on the
pre-registered benchmarks and write per-question outputs. Runs in the INFERENCE
environment (torch + transformers + datasets + pyarrow), e.g. on a Colab/Kaggle T4:

    python experiments/n7_11_score.py --shard 0 --nshards 4 --out /content/drive/MyDrive/pdt_n7_11

Resumable: a job is skipped when <out>/<job_id>/done.json exists, so a disconnected
notebook can simply be re-run. Each finished job leaves one parquet per benchmark
(per-question records, see pdt.eval.scoring.item_record) and done.json (model info,
throughput, GPU name, library versions). Downloaded weights are deleted after each job.
The analysis that consumes these outputs is pre-registered (docs/preregistration.md);
this worker only scores.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
import transformers

from pdt.eval import benchmarks, convert, run


def _write_parquet(records: list[dict], path: Path) -> None:
    pd.DataFrame(records).to_parquet(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="configs/n7_11_jobs.json")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="datadecide,hf", help="comma list: datadecide,hf")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    args = ap.parse_args()

    spec = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    kinds = set(args.kinds.split(","))
    jobs = [j for j in spec["jobs"] if j["kind"] in kinds]
    mine = [j for k, j in enumerate(jobs) if k % args.nshards == args.shard]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    todo = [j for j in mine if not (out_root / j["job_id"] / "done.json").exists()]
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} jobs, {len(todo)} to do", flush=True)
    if not todo:
        return
    items = {b: benchmarks.load_items(b) for b in spec["benchmarks"]}
    dtype = spec["dtype"]
    for job in todo:
        t0 = time.time()
        cache = tempfile.mkdtemp(prefix="pdt_hf_")
        try:
            if job["kind"] == "datadecide":
                model, tok, info = convert.load_datadecide(
                    job["repo_id"], job["revision"], dtype_name=dtype, cache_dir=cache
                )
            else:
                model, tok, info = convert.load_hf(
                    job["repo_id"], job["revision"], dtype_name=dtype, cache_dir=cache
                )
            info["load_seconds"] = time.time() - t0
            job_dir = out_root / job["job_id"]
            job_dir.mkdir(parents=True, exist_ok=True)
            stats = {}
            for name, its in items.items():
                recs, st = run.score_items(model, tok, its, batch_tokens=args.batch_tokens)
                _write_parquet(recs, job_dir / f"{name}.parquet")
                stats[name] = st
            done = {
                "job": job,
                "model": info,
                "stats": stats,
                "dtype": dtype,
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "python": platform.python_version(),
                "seconds": time.time() - t0,
            }
            (job_dir / "done.json").write_text(json.dumps(done, indent=1), encoding="utf-8")
            print(f"done {job['job_id']} in {done['seconds']:.0f}s", flush=True)
        finally:
            model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
