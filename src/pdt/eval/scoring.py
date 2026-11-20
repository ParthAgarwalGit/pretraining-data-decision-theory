"""Torch-free scoring helpers: answer selection under OLMES/lm-eval normalisations,
DataDecide revision names, and GPU-cost extrapolation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

NORMALISATIONS = ("raw", "per_token", "per_char")


def choose(
    sum_logprobs: Sequence[float], n_tokens: Sequence[int], n_chars: Sequence[int]
) -> dict[str, int]:
    """Index of the chosen answer under each normalisation.

    ``raw`` = argmax of the summed continuation log-probability; ``per_token`` divides by
    the continuation's token count; ``per_char`` divides by its character count including
    the leading space (OLMES ``acc_per_char`` = lm-eval ``acc_norm``). Ties go to the
    lowest index (``np.argmax``), as in lm-eval.
    """
    s = np.asarray(sum_logprobs, dtype=float)
    nt = np.asarray(n_tokens, dtype=float)
    nc = np.asarray(n_chars, dtype=float)
    if not (len(s) == len(nt) == len(nc)) or len(s) == 0:
        raise ValueError("choice arrays must be non-empty and of equal length")
    if np.any(nt <= 0) or np.any(nc <= 0):
        raise ValueError("token and character counts must be positive")
    return {
        "raw": int(np.argmax(s)),
        "per_token": int(np.argmax(s / nt)),
        "per_char": int(np.argmax(s / nc)),
    }


def item_record(
    item_id: str,
    sum_logprobs: Sequence[float],
    n_tokens: Sequence[int],
    continuations: Sequence[str],
    gold: int,
) -> dict:
    """One per-question output row (choices as lists, correctness per normalisation)."""
    n_chars = [len(c) for c in continuations]
    picks = choose(sum_logprobs, n_tokens, n_chars)
    rec = {
        "item_id": str(item_id),
        "gold": int(gold),
        "sum_logprobs": [float(x) for x in sum_logprobs],
        "n_tokens": [int(x) for x in n_tokens],
        "n_chars": n_chars,
    }
    for norm, idx in picks.items():
        rec[f"pred_{norm}"] = idx
        rec[f"correct_{norm}"] = int(idx == gold)
    return rec


def datadecide_revision(step: int, seed_label: str) -> str:
    """HF branch of a DataDecide checkpoint, e.g. (37500, "small aux 2") ->
    ``step37500-seed-small-aux-2``; ``default`` -> ``step37500-seed-default``."""
    return f"step{int(step)}-seed-{seed_label.strip().replace(' ', '-')}"


def gpu_hours(n_tokens: float, tokens_per_second: float) -> float:
    """Hours to score ``n_tokens`` (non-pad input tokens) at a measured throughput."""
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    return float(n_tokens) / float(tokens_per_second) / 3600.0
