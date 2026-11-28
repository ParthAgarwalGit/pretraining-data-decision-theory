"""Benchmark items as (item_id, context, continuations, gold) in lm-eval's cloze formats.

Formats follow EleutherAI lm-evaluation-harness (`sciq`, `lambada_openai`, `blimp_*`);
`olmes_requests` replays DataDecide's own released OLMES request files exactly, which is
how the harness is validated against DataDecide's released per-instance log-likelihoods.
Loaders import `datasets` lazily so this module stays importable without it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    item_id: str
    context: str
    continuations: tuple[str, ...]
    gold: int
    greedy: bool = False  # single-continuation tasks scored by exact greedy match
    meta: dict = field(default_factory=dict, compare=False)


def sciq_item(doc: dict, idx: int) -> Item:
    """lm-eval `sciq`: '{support}\\nQuestion: {question}\\nAnswer:' + ' {choice}'."""
    choices = [doc["distractor1"], doc["distractor2"], doc["distractor3"], doc["correct_answer"]]
    ctx = f"{doc['support'].lstrip()}\nQuestion: {doc['question']}\nAnswer:"
    return Item(str(idx), ctx, tuple(" " + c for c in choices), 3)


def lambada_item(doc: dict, idx: int) -> Item:
    """lm-eval `lambada_openai`: predict the final word; correct iff greedy decoding of
    every continuation token matches."""
    ctx, last = doc["text"].rsplit(" ", 1)
    return Item(str(idx), ctx, (" " + last,), 0, greedy=True)


def blimp_item(doc: dict, subset: str, idx: int) -> Item:
    """lm-eval `blimp_*`: empty context; the grammatical sentence should be likelier."""
    return Item(f"{subset}/{idx}", "", (doc["sentence_good"], doc["sentence_bad"]), 0)


def load_items(benchmark: str, limit: int | None = None) -> list[Item]:  # pragma: no cover
    """Download (HF `datasets`) and format a benchmark. Network-bound; exercised by the
    N7-04 timing run rather than unit tests."""
    from datasets import get_dataset_config_names, load_dataset

    if benchmark == "sciq":
        ds = load_dataset("allenai/sciq", split="test")
        items = [sciq_item(d, i) for i, d in enumerate(ds)]
    elif benchmark == "lambada_openai":
        ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")
        items = [lambada_item(d, i) for i, d in enumerate(ds)]
    elif benchmark == "blimp":
        items = []
        for subset in sorted(get_dataset_config_names("nyu-mll/blimp")):
            ds = load_dataset("nyu-mll/blimp", subset, split="train")
            items.extend(blimp_item(d, subset, i) for i, d in enumerate(ds))
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    return items[:limit] if limit else items


def olmes_requests(path: str) -> list[Item]:
    """Items from a DataDecide `requests/<task>-requests.jsonl.gz` file: one request per
    (doc, choice), grouped back into items in doc order with the exact released context
    and continuation strings."""
    by_doc: dict[int, dict] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = by_doc.setdefault(
                r["doc_id"],
                {
                    "ctx": r["request"]["context"],
                    "conts": {},
                    "gold": r["label"],
                    "native": r.get("native_id"),
                },
            )
            if r["request"]["context"] != d["ctx"]:
                raise ValueError(f"doc {r['doc_id']}: contexts differ across choices")
            d["conts"][r["idx"]] = r["request"]["continuation"]
    items = []
    for doc_id in sorted(by_doc):
        d = by_doc[doc_id]
        conts = tuple(d["conts"][k] for k in sorted(d["conts"]))
        items.append(
            Item(str(doc_id), d["ctx"], conts, int(d["gold"]), meta={"native_id": d["native"]})
        )
    return items


def load_frozen(path: str, benchmark: str, sha256: str | None = None) -> list[Item]:
    """Items of ``benchmark`` from a frozen ``*.json.gz`` written by
    ``experiments/n7_11_freeze_benchmarks.py``. If ``sha256`` is given it must match the
    uncompressed JSON, so every worker scores exactly the same items without downloads."""
    raw = gzip.decompress(open(path, "rb").read())
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"{path}: sha256 mismatch, the frozen benchmark file was altered")
    data = json.loads(raw)[benchmark]
    return [
        Item(
            d["item_id"],
            d["context"],
            tuple(d["continuations"]),
            int(d["gold"]),
            greedy=bool(d["greedy"]),
        )
        for d in data
    ]
