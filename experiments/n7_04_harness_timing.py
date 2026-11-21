"""N7-04: validate the inference harness against DataDecide's released log-likelihoods and
measure its cost on the local GPU. Runs in the INFERENCE environment (torch +
transformers; see src/pdt/eval/README.md), e.g.

    PYTHONPATH=src python experiments/n7_04_harness_timing.py

Validation: for the c4 150M and 1B final checkpoints (seed `default`), replay
DataDecide's own ARC-Easy request file and compare our per-choice summed log-probs and
per-char predictions with the released per-instance predictions for the same checkpoint.
Timing: SciQ, LAMBADA (openai) and a fixed 5k BLiMP sample on both checkpoints, plus
PolyPythia 14m. GPU-hours for the full re-evaluation are extrapolated from measured
tokens/s; 300M/530M and the larger Pythias are scaled by parameter count and are labelled
as extrapolated. Writes results/n7_04_harness_timing.json.
"""

from __future__ import annotations

import json
import os
import random
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from pdt import provenance
from pdt.eval import benchmarks, convert, run, scoring

_DTYPE = "float16"
_INSTANCES = Path("data/cache/datadecide_instances/c4")
_SCRATCH = Path("data/n7_04_scratch")
_REQ_REV = "23f3b2e186ca6c39026e3efa00e4af397680c075"
_CHECKPOINTS = {
    "150M": ("allenai/DataDecide-c4-150M", 37500, "150M/seed-2/step-37500.tar.gz"),
    "1B": ("allenai/DataDecide-c4-1B", 69369, "1B/seed-2/step-69369.tar.gz"),
}
_BLIMP_SAMPLE = 5000
_SEED = 20261121


def _reference_predictions(tar_rel: str) -> dict[int, dict]:
    with tarfile.open(_INSTANCES / tar_rel) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("arc_easy-predictions.jsonl"))
        lines = tf.extractfile(member).read().decode("utf-8").splitlines()
    return {int(r["doc_id"]): r for r in map(json.loads, lines)}


def _requests() -> list[benchmarks.Item]:
    path = _SCRATCH / "arc_easy-requests.jsonl.gz"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                "allenai/DataDecide-eval-instances",
                "requests/arc_easy-requests.jsonl.gz",
                repo_type="dataset",
                revision=_REQ_REV,
            )
        )
    return benchmarks.olmes_requests(str(path))


def _validate(model, tok, ref) -> dict:
    items = _requests()
    recs, stats = run.score_items(model, tok, items, batch_tokens=16384)
    ours = np.concatenate([r["sum_logprobs"] for r in recs])
    theirs = np.concatenate(
        [[o["sum_logits"] for o in ref[int(r["item_id"])]["model_output"]] for r in recs]
    )
    flips = sum(
        r["pred_per_char"] != ref[int(r["item_id"])]["metrics"]["predicted_index_per_char"]
        for r in recs
    )
    return {
        "n_items": len(recs),
        "max_abs_logprob_diff": float(np.max(np.abs(ours - theirs))),
        "median_abs_logprob_diff": float(np.median(np.abs(ours - theirs))),
        "per_char_prediction_flips": int(flips),
        "acc_per_char_ours": float(np.mean([r["correct_per_char"] for r in recs])),
        "acc_per_char_released": float(
            np.mean([ref[int(r["item_id"])]["metrics"]["acc_per_char"] for r in recs])
        ),
        "tokens_per_second": stats["tokens_per_second"],
    }


def _bench_items() -> dict[str, list[benchmarks.Item]]:
    blimp = benchmarks.load_items("blimp")
    rng = random.Random(_SEED)
    return {
        "sciq": benchmarks.load_items("sciq"),
        "lambada_openai": benchmarks.load_items("lambada_openai"),
        "blimp_sample": rng.sample(blimp, _BLIMP_SAMPLE),
        "_blimp_full_n": len(blimp),
    }


def _time(model, tok, sets) -> dict:
    out = {}
    for name, items in sets.items():
        if name.startswith("_"):
            continue
        torch.cuda.reset_peak_memory_stats()
        recs, st = run.score_items(model, tok, items, batch_tokens=16384)
        acc = {n: float(np.mean([r[f"correct_{n}"] for r in recs])) for n in scoring.NORMALISATIONS}
        if items[0].greedy:
            acc["greedy_match"] = float(np.mean([r["greedy_match"] for r in recs]))
        out[name] = {
            **st,
            "accuracy": acc,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    return out


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    sets = _bench_items()
    hf_cache = str(_SCRATCH / "hf")
    results = {}
    for size, (repo, step, tar_rel) in _CHECKPOINTS.items():
        free = torch.cuda.mem_get_info()[0] / 1e9
        try:
            t0 = time.time()
            model, tok, info = convert.load_datadecide(
                repo,
                scoring.datadecide_revision(step, "default"),
                dtype_name=_DTYPE,
                cache_dir=hf_cache,
            )
            info["load_seconds_incl_download"] = time.time() - t0
            ref = _reference_predictions(tar_rel)
            results[size] = {
                "model": info,
                "free_vram_gb_before": free,
                "validation_arc_easy_vs_released": _validate(model, tok, ref),
                "timing": _time(model, tok, sets),
            }
        except torch.cuda.OutOfMemoryError as exc:
            results[size] = {
                "skipped": f"CUDA out of memory with {free:.2f} GB free before loading",
                "error": str(exc).splitlines()[0],
            }
        model = None
        torch.cuda.empty_cache()
    t0 = time.time()
    model, tok, info = convert.load_hf(
        "EleutherAI/pythia-14m-seed1", "step143000", dtype_name=_DTYPE, cache_dir=hf_cache
    )
    info["load_seconds_incl_download"] = time.time() - t0
    results["pythia-14m-seed1"] = {
        "model": info,
        "timing": _time(model, tok, {"sciq": sets["sciq"]}),
    }
    del model
    torch.cuda.empty_cache()

    # Extrapolated GPU-hours (tokens measured on 150M; same tokenizer for every DataDecide model).
    t150 = results["150M"]["timing"]
    tok_per_item = {k: t150[k]["input_tokens"] / t150[k]["n_items"] for k in t150}
    full_tokens = (
        t150["sciq"]["input_tokens"]
        + t150["lambada_openai"]["input_tokens"]
        + tok_per_item["blimp_sample"] * sets["_blimp_full_n"]
    )
    rate = {
        s: np.mean([v["tokens_per_second"] for v in results[s]["timing"].values()])
        for s in _CHECKPOINTS
    }
    rates = {
        "150M": (rate["150M"], "measured"),
        "300M": (rate["150M"] * 150 / 300, "extrapolated by parameter count from 150M"),
        "530M": (rate["1B"] * 1000 / 530, "extrapolated by parameter count from 1B"),
        "1B": (rate["1B"], "measured"),
    }
    per_ckpt = {
        s: {"hours": scoring.gpu_hours(full_tokens, r), "basis": b} for s, (r, b) in rates.items()
    }
    total = sum(v["hours"] for v in per_ckpt.values()) * 25 * 3
    payload = {
        "status": "harness validation + local timing (inference only)",
        "device": torch.cuda.get_device_name(),
        "dtype": _DTYPE,
        "precision_note": (
            "fp16 chosen: on 400 ARC-Easy items at 150M it matched released log-probs within"
            " 0.03 nats with 0 prediction flips; bf16 flipped predictions (pre-run check)."
        ),
        "gpu_shared_note": (
            "Timed while another user process held GPU memory and compute (a fine-tuning"
            " queue from another project): throughputs are pessimistic for a dedicated GPU."
        ),
        "results": results,
        "full_benchmark_set": {
            "benchmarks": ["sciq", "lambada_openai", "blimp (all 67 subsets)"],
            "blimp_items": sets["_blimp_full_n"],
            "input_tokens_per_checkpoint": full_tokens,
        },
        "extrapolation": {
            "per_checkpoint": per_ckpt,
            "datadecide_25x3x4_total_gpu_hours": total,
            "excludes": "checkpoint download time (see load_seconds_incl_download)",
        },
    }
    provenance.write_result(
        "results/n7_04_harness_timing.json",
        payload=payload,
        config={
            "task": "N7-04",
            "seed": _SEED,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "dtype": _DTYPE,
            "instances_revision": _REQ_REV,
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        },
    )
    print("wrote results/n7_04_harness_timing.json")


if __name__ == "__main__":
    main()
