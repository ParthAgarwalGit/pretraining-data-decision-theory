"""Freeze the N7-11 benchmark items into configs/n7_11_benchmarks.json.gz (one file, no
per-worker downloads) and record its sha256 in configs/n7_11_jobs.json.

Run once in the inference environment (needs `datasets` + network):
    PYTHONPATH=src python experiments/n7_11_freeze_benchmarks.py
The JSON is serialised deterministically (sorted keys, gzip mtime 0), so re-running on
the same dataset revisions reproduces the same sha256.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from pdt.eval import benchmarks

_OUT = Path("configs/n7_11_benchmarks.json.gz")
_JOBS = Path("configs/n7_11_jobs.json")


def main() -> None:
    spec = json.loads(_JOBS.read_text(encoding="utf-8"))
    data = {
        b: [
            {
                "item_id": it.item_id,
                "context": it.context,
                "continuations": list(it.continuations),
                "gold": it.gold,
                "greedy": it.greedy,
            }
            for it in benchmarks.load_items(b)
        ]
        for b in spec["benchmarks"]
    }
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    _OUT.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    spec["benchmark_file"] = str(_OUT).replace("\\", "/")
    spec["benchmark_sha256"] = hashlib.sha256(raw).hexdigest()
    spec["benchmark_counts"] = {b: len(v) for b, v in data.items()}
    _JOBS.write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    print("froze", spec["benchmark_counts"], spec["benchmark_sha256"])


if __name__ == "__main__":
    main()
