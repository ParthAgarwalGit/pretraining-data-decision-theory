"""Batched, deterministic log-likelihood scoring of benchmark items (torch required).

Request encoding follows lm-eval's ``_encode_pair``: trailing whitespace of the context
moves to the continuation; continuation tokens are the tokens of ``context +
continuation`` beyond the context's own tokens; an empty context is replaced by the EOS
token as a prefix. Sequences longer than the model window are left-truncated. Only the
hidden states at continuation positions go through the output head, and the
log-softmax is taken in float32.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from pdt.eval.benchmarks import Item
from pdt.eval.scoring import item_record


def encode_pair(tok, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """(context ids, continuation ids) exactly as lm-eval's HFLM._encode_pair."""
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    if context == "":
        return [tok.eos_token_id], tok(continuation, add_special_tokens=False).input_ids
    whole = tok(context + continuation, add_special_tokens=False).input_ids
    ctx = tok(context, add_special_tokens=False).input_ids
    return ctx, whole[len(ctx) :]


def score_requests(  # pragma: no cover -- needs torch and a model
    model,
    tok,
    pairs: Sequence[tuple[str, str]],
    max_len: int = 2048,
    batch_tokens: int = 16384,
    head_rows: int = 1024,
) -> tuple[list[tuple[float, int, bool]], dict]:
    """Sum log-prob, continuation token count and greedy-match flag for each
    (context, continuation) pair, plus timing stats (non-pad input tokens / s)."""
    import torch

    device = next(model.parameters()).device
    rows = []
    for n, (c, x) in enumerate(pairs):
        ctx, cont = encode_pair(tok, c, x)
        full = (ctx + cont)[-(max_len + 1) :]
        inp = full[:-1]
        rows.append((n, inp, cont, len(inp) - len(cont)))
    order = sorted(range(len(rows)), key=lambda r: -len(rows[r][1]))
    out: list[tuple[float, int, bool] | None] = [None] * len(rows)
    n_tok, t0 = 0, time.time()
    pos = 0
    with torch.inference_mode():
        while pos < len(order):
            width = len(rows[order[pos]][1])
            bs = max(1, batch_tokens // width)
            batch = [rows[r] for r in order[pos : pos + bs]]
            pos += len(batch)
            # Build the padded batch on the host and copy it once (per-row device copies
            # dominate the runtime for many short sequences, e.g. BLiMP).
            ids_np = np.zeros((len(batch), width), dtype=np.int64)
            mask_np = np.zeros((len(batch), width), dtype=np.int64)
            for b, (_, inp, _, _) in enumerate(batch):
                ids_np[b, : len(inp)] = inp
                mask_np[b, : len(inp)] = 1
            ids = torch.from_numpy(ids_np).to(device)
            mask = torch.from_numpy(mask_np).to(device)
            n_tok += int(mask_np.sum())
            hidden = model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
            sel, tgt, owner = [], [], []
            for b, (_, _inp, cont, start) in enumerate(batch):
                sel.append(hidden[b, start : start + len(cont)])
                tgt.extend(cont)
                owner.extend([b] * len(cont))
            # Output head + float32 log-softmax in row chunks: bounds peak memory at
            # head_rows x vocab regardless of how many continuation tokens a batch has.
            hsel = torch.cat(sel)
            tgt_t = torch.tensor(tgt, device=device)
            head = model.get_output_embeddings()
            tok_lp = torch.empty(len(tgt), device=device)
            greedy = torch.empty(len(tgt), dtype=torch.bool, device=device)
            for a in range(0, len(tgt), head_rows):
                lg = head(hsel[a : a + head_rows]).float()
                t_chunk = tgt_t[a : a + head_rows]
                tok_lp[a : a + head_rows] = torch.log_softmax(lg, dim=-1).gather(
                    1, t_chunk[:, None]
                )[:, 0]
                greedy[a : a + head_rows] = lg.argmax(-1) == t_chunk
                del lg
            owner_t = torch.tensor(owner, device=device)
            sums = torch.zeros(len(batch), device=device).index_add_(0, owner_t, tok_lp)
            allg = torch.ones(len(batch), dtype=torch.int32, device=device)
            allg.scatter_reduce_(0, owner_t, greedy.to(torch.int32), reduce="amin")
            sums_l, allg_l = sums.tolist(), allg.tolist()  # one device sync per batch
            for b, (n, _, cont, _) in enumerate(batch):
                out[n] = (float(sums_l[b]), len(cont), bool(allg_l[b]))
    if device.type == "cuda":
        torch.cuda.synchronize()
    secs = max(time.time() - t0, 1e-9)  # coarse clocks can report 0 for tiny runs
    return out, {"seconds": secs, "input_tokens": n_tok, "tokens_per_second": n_tok / secs}


def score_items(  # pragma: no cover -- needs torch and a model
    model, tok, items: Sequence[Item], **kw
) -> tuple[list[dict], dict]:
    """Per-question records (see ``scoring.item_record``) for a list of items."""
    pairs, owner = [], []
    for n, it in enumerate(items):
        for cont in it.continuations:
            pairs.append((it.context, cont))
            owner.append(n)
    scored, stats = score_requests(model, tok, pairs, **kw)
    recs, k = [], 0
    for it in items:
        m = len(it.continuations)
        part = scored[k : k + m]
        k += m
        rec = item_record(
            it.item_id, [p[0] for p in part], [p[1] for p in part], it.continuations, it.gold
        )
        if it.greedy:
            rec["greedy_match"] = int(part[0][2])
        recs.append(rec)
    stats["n_items"] = len(items)
    stats["items_per_second"] = len(items) / stats["seconds"]
    return recs, stats
