"""N7-11 / N7-10-S2 GPU worker: score a shard of configs/n7_11_jobs.json on the
pre-registered benchmarks and write per-question outputs. Runs in the INFERENCE
environment (torch + transformers + pyarrow), e.g. on a Colab/Kaggle T4:

    python experiments/n7_11_score.py --shard 0 --nshards 4 --out /content/drive/MyDrive/pdt_n7_11

* Benchmarks come from the frozen, hash-checked configs/n7_11_benchmarks.json.gz: no
  dataset downloads at run time (the per-subset BLiMP downloads stalled Colab workers).
* Checkpoint downloads have timeouts and up to 3 attempts; a job that still fails writes
  failed.json and the shard moves on (re-running the notebook retries it).
* Resumable: a job is skipped when <out>/<job_id>/done.json exists.
* Every stage is logged with a timestamp, so a stall is visible.
Each finished job leaves one parquet per benchmark (per-question records, see
pdt.eval.scoring.item_record) and done.json (model info, throughput, GPU, versions).
"""

from __future__ import annotations

import os

# Must be set before huggingface_hub is imported (it reads them at import time).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from pdt.eval import benchmarks, convert, run  # noqa: E402

_ATTEMPTS = 3
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:8.1f}s] {msg}", flush=True)


def _load(job: dict, dtype: str, cache: str):
    if job["kind"] == "datadecide":
        return convert.load_datadecide(
            job["repo_id"], job["revision"], dtype_name=dtype, cache_dir=cache
        )
    return convert.load_hf(job["repo_id"], job["revision"], dtype_name=dtype, cache_dir=cache)


def _run_job(
    job: dict, items: dict, dtype: str, out_root: Path, batch_tokens: int, cache: str
) -> None:
    t0 = time.time()
    job_dir = out_root / job["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    model = None
    try:
        log(f"{job['job_id']}: downloading + loading {job['repo_id']}@{job['revision']}")
        model, tok, info = _load(job, dtype, cache)
        info["load_seconds"] = time.time() - t0
        log(f"{job['job_id']}: loaded in {info['load_seconds']:.0f}s")
        stats = {}
        for name, its in items.items():
            recs, st = run.score_items(model, tok, its, batch_tokens=batch_tokens)
            pd.DataFrame(recs).to_parquet(job_dir / f"{name}.parquet", index=False)
            stats[name] = st
            log(
                f"{job['job_id']}: {name} {st['n_items']} items, "
                f"{st['tokens_per_second']:.0f} tok/s"
            )
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
        (job_dir / "failed.json").unlink(missing_ok=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shutil.rmtree(cache, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="configs/n7_11_jobs.json")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="datadecide,hf", help="comma list: datadecide,hf")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument(
        "--reverse",
        action="store_true",
        help="process this shard's jobs last-to-first (to meet another worker in the middle)",
    )
    ap.add_argument(
        "--skip-kinds",
        default="",
        help="comma list of kinds to skip AFTER sharding, e.g. 'hf' on a machine whose torch "
        "is < 2.6 (keeps shard boundaries identical to workers that run every kind)",
    )
    args = ap.parse_args()

    spec = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    kinds = set(args.kinds.split(","))
    jobs = [j for j in spec["jobs"] if j["kind"] in kinds]
    mine = [j for k, j in enumerate(jobs) if k % args.nshards == args.shard]
    skip = {k for k in args.skip_kinds.split(",") if k}
    mine = [j for j in mine if j["kind"] not in skip]
    if args.reverse:
        mine = mine[::-1]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    todo = [j for j in mine if not (out_root / j["job_id"] / "done.json").exists()]
    log(f"shard {args.shard}/{args.nshards}: {len(mine)} jobs, {len(todo)} to do")
    if not todo:
        return
    log(f"loading frozen benchmarks from {spec['benchmark_file']}")
    items = {
        b: benchmarks.load_frozen(spec["benchmark_file"], b, spec["benchmark_sha256"])
        for b in spec["benchmarks"]
    }
    log("benchmarks: " + ", ".join(f"{b}={len(v)}" for b, v in items.items()))
    failed = []
    caches = [tempfile.mkdtemp(prefix="pdt_hf_") for _ in todo]
    pool = ThreadPoolExecutor(max_workers=1)

    def prefetch(k: int):
        # Download the next checkpoint while the current one is scored; any error here is
        # ignored, because the main loop downloads again (with retries) when it loads.
        job_k = todo[k]
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(job_k["repo_id"], revision=job_k["revision"], cache_dir=caches[k])
        except Exception as exc:  # noqa: BLE001
            log(f"{job_k['job_id']}: prefetch failed ({exc!r}); will download in the main loop")

    pending = pool.submit(prefetch, 0)
    for n, job in enumerate(todo, 1):
        log(f"job {n}/{len(todo)}: {job['job_id']}")
        pending.result()
        if n < len(todo):
            pending = pool.submit(prefetch, n)
        for attempt in range(1, _ATTEMPTS + 1):
            try:
                _run_job(job, items, spec["dtype"], out_root, args.batch_tokens, caches[n - 1])
                break
            except Exception as exc:  # network stall, OOM, hub error: retry, then skip
                log(f"{job['job_id']}: attempt {attempt}/{_ATTEMPTS} failed: {exc!r}")
                if attempt == _ATTEMPTS:
                    failed.append(job["job_id"])
                    (out_root / job["job_id"] / "failed.json").write_text(
                        json.dumps({"job": job, "error": traceback.format_exc()}, indent=1),
                        encoding="utf-8",
                    )
                else:
                    time.sleep(30 * attempt)
    pool.shutdown()
    log(f"shard finished; {len(todo) - len(failed)} done, {len(failed)} failed: {failed}")


if __name__ == "__main__":
    main()
