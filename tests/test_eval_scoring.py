"""Tests for the N7-04 inference harness. Torch-free tests run everywhere; torch tests run
only in the inference environment (they skip in the base/CI environment)."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from pdt.eval import benchmarks, convert, run, scoring


def test_choose_normalisations_and_ties():
    # raw prefers choice 0; per-char prefers choice 1 (longer, better per char)
    picks = scoring.choose([-4.0, -6.0], [2, 6], [4, 12])
    assert picks == {"raw": 0, "per_token": 1, "per_char": 1}
    assert scoring.choose([-1.0, -1.0], [1, 1], [1, 1])["raw"] == 0  # ties -> lowest index
    with pytest.raises(ValueError):
        scoring.choose([-1.0], [0], [1])


def test_item_record_fields():
    rec = scoring.item_record("q7", [-3.0, -1.0], [3, 1], [" abc", " d"], gold=1)
    assert rec["n_chars"] == [4, 2]
    assert rec["pred_raw"] == 1 and rec["correct_raw"] == 1
    assert rec["pred_per_char"] == 1 and rec["correct_per_char"] == 1


def test_datadecide_revision_and_gpu_hours():
    assert scoring.datadecide_revision(37500, "small aux 2") == "step37500-seed-small-aux-2"
    assert scoring.datadecide_revision(69369, "default") == "step69369-seed-default"
    assert scoring.gpu_hours(7200.0, 2.0) == pytest.approx(1.0)


def test_olmo_key_map_covers_a_block():
    names = {
        t
        for k in [
            "att_proj",
            "attn_out",
            "attn_norm",
            "ff_proj",
            "ff_out",
            "ff_norm",
        ]
        for t, _, _ in convert.olmo_key_map(f"model.transformer.blocks.3.{k}.weight")
    }
    assert names == {
        f"model.layers.3.{x}.weight"
        for x in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "input_layernorm",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
            "post_attention_layernorm",
        ]
    }
    assert convert.olmo_key_map("model.transformer.ff_out.weight")[0][0] == "lm_head.weight"
    with pytest.raises(KeyError):
        convert.olmo_key_map("model.transformer.blocks.0.mystery.weight")


def test_benchmark_formatters():
    sciq = benchmarks.sciq_item(
        {
            "support": " Plants need light.",
            "question": "What do plants need?",
            "distractor1": "rocks",
            "distractor2": "salt",
            "distractor3": "ice",
            "correct_answer": "light",
        },
        5,
    )
    assert sciq.context == "Plants need light.\nQuestion: What do plants need?\nAnswer:"
    assert sciq.continuations == (" rocks", " salt", " ice", " light") and sciq.gold == 3
    lam = benchmarks.lambada_item({"text": "the cat sat on the mat"}, 0)
    assert lam.context == "the cat sat on the" and lam.continuations == (" mat",) and lam.greedy
    bl = benchmarks.blimp_item({"sentence_good": "A b.", "sentence_bad": "B a."}, "s1", 2)
    assert bl.context == "" and bl.gold == 0 and bl.item_id == "s1/2"


def test_olmes_requests_groups_choices(tmp_path):
    rows = [
        {
            "doc_id": 1,
            "idx": 1,
            "label": 0,
            "native_id": "x",
            "request": {"context": "Q1\nAnswer:", "continuation": " b"},
        },
        {
            "doc_id": 1,
            "idx": 0,
            "label": 0,
            "native_id": "x",
            "request": {"context": "Q1\nAnswer:", "continuation": " a"},
        },
        {
            "doc_id": 0,
            "idx": 0,
            "label": 1,
            "native_id": "y",
            "request": {"context": "Q0\nAnswer:", "continuation": " c"},
        },
        {
            "doc_id": 0,
            "idx": 1,
            "label": 1,
            "native_id": "y",
            "request": {"context": "Q0\nAnswer:", "continuation": " d"},
        },
    ]
    p = tmp_path / "r.jsonl.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r) for r in rows))
    items = benchmarks.olmes_requests(str(p))
    assert [it.item_id for it in items] == ["0", "1"]
    assert items[1].continuations == (" a", " b") and items[0].gold == 1


class _CharTok:
    """Character-level fake tokenizer (ord codes) with the HF call signature."""

    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        class _Enc:
            pass

        enc = _Enc()
        enc.input_ids = [ord(c) for c in text]
        return enc


def test_encode_pair_moves_trailing_space_and_handles_empty_context():
    tok = _CharTok()
    ctx, cont = run.encode_pair(tok, "Answer: ", "yes")
    assert ctx == [ord(c) for c in "Answer:"] and cont == [ord(c) for c in " yes"]
    ctx, cont = run.encode_pair(tok, "", "Hi.")
    assert ctx == [0] and cont == [ord(c) for c in "Hi."]


# ----------------------------------------------------------------- torch-only tests


def _olmo_reference_forward(sd, ids, n_heads, eps=1e-6, theta=10000.0):
    """Plain-torch OLMo-1 forward written from the OLMo equations (not via transformers)."""
    import torch

    x = sd["model.transformer.wte.weight"][ids]
    t, d = x.shape[1], x.shape[2]
    hd = d // n_heads
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    ang = torch.outer(torch.arange(t, dtype=torch.float64), inv)
    cos = torch.cat([ang.cos(), ang.cos()], -1).to(x.dtype)
    sin = torch.cat([ang.sin(), ang.sin()], -1).to(x.dtype)

    def rms(h, w):
        return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w

    def rope(q):
        q1, q2 = q[..., : hd // 2], q[..., hd // 2 :]
        return q * cos + torch.cat([-q2, q1], -1) * sin

    n_layers = 1 + max(int(k.split(".")[3]) for k in sd if ".blocks." in k)
    mask = torch.full((t, t), float("-inf")).triu(1)
    for i in range(n_layers):
        p = f"model.transformer.blocks.{i}."
        h = rms(x, sd[p + "attn_norm.weight"])
        q, k, v = (h @ sd[p + "att_proj.weight"].T).chunk(3, -1)
        q, k, v = (z.view(1, t, n_heads, hd).transpose(1, 2) for z in (q, k, v))
        q, k = rope(q), rope(k)
        a = torch.softmax(q @ k.transpose(-1, -2) / hd**0.5 + mask, -1) @ v
        x = x + a.transpose(1, 2).reshape(1, t, d) @ sd[p + "attn_out.weight"].T
        h = rms(x, sd[p + "ff_norm.weight"])
        up, gate = (h @ sd[p + "ff_proj.weight"].T).chunk(2, -1)
        x = x + (torch.nn.functional.silu(gate) * up) @ sd[p + "ff_out.weight"].T
    x = rms(x, sd["model.transformer.ln_f.weight"])
    return x @ sd["model.transformer.ff_out.weight"].T


def test_converter_matches_olmo_equations():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    g = torch.Generator().manual_seed(0)
    d, heads, hidden, vocab, layers = 32, 4, 48, 50, 2

    def r(*s):
        return torch.randn(*s, generator=g, dtype=torch.float64) * 0.2

    sd = {
        "model.transformer.wte.weight": r(vocab, d),
        "model.transformer.ln_f.weight": 1 + r(d),
        "model.transformer.ff_out.weight": r(vocab, d),
    }
    for i in range(layers):
        p = f"model.transformer.blocks.{i}."
        sd |= {
            p + "att_proj.weight": r(3 * d, d),
            p + "attn_out.weight": r(d, d),
            p + "attn_norm.weight": 1 + r(d),
            p + "ff_norm.weight": 1 + r(d),
            p + "ff_proj.weight": r(2 * hidden, d),
            p + "ff_out.weight": r(d, hidden),
        }
    cfg = transformers.LlamaConfig(
        vocab_size=vocab,
        hidden_size=d,
        intermediate_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        max_position_embeddings=64,
    )
    model = transformers.LlamaForCausalLM(cfg).double().eval()
    state = {}
    for k, t in sd.items():
        for target, chunk, n in convert.olmo_key_map(k):
            state[target] = t if chunk is None else t.chunk(n, 0)[chunk]
    model.load_state_dict(state, strict=True)
    ids = torch.tensor([[3, 7, 1, 9, 4, 4, 2]])
    with torch.no_grad():
        ours = model(ids).logits
    ref = _olmo_reference_forward(sd, ids, heads)
    # transformers computes RoPE inverse frequencies in float32, which bounds agreement
    # at ~1e-6 even in float64; a wrong weight mapping is off by O(0.1).
    assert float((ours - ref).abs().max()) < 1e-5


def test_scoring_batch_invariant_and_deterministic():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.GPTNeoXConfig(
        vocab_size=200,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    model = transformers.GPTNeoXForCausalLM(cfg).double().eval()

    class _Tok(_CharTok):
        def __call__(self, text, add_special_tokens=False):
            enc = super().__call__(text)
            enc.input_ids = [c % 199 + 1 for c in enc.input_ids]
            return enc

    pairs = [
        ("The sky is", " blue"),
        ("Two plus two is", " four"),
        ("", "A sentence."),
        ("Q: x\nA:", " yes"),
        ("Long context " * 5, " end"),
    ]
    a, _ = run.score_requests(model, _Tok(), pairs, batch_tokens=1)  # batch size 1
    b, _ = run.score_requests(model, _Tok(), pairs, batch_tokens=10_000)  # one batch
    c, _ = run.score_requests(model, _Tok(), pairs, batch_tokens=10_000)
    assert np.allclose([x[0] for x in a], [x[0] for x in b], atol=1e-9)
    assert [x[1] for x in a] == [x[1] for x in b]
    assert a == a and b == c
