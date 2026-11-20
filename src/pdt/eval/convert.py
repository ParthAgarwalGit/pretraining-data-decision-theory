"""Load a DataDecide (``hf_olmo``) checkpoint as a stock ``transformers`` LlamaForCausalLM.

DataDecide checkpoints are OLMo-1 style: pre-norm RMSNorm with weight (no bias), SwiGLU,
RoPE, no biases, untied input/output embeddings, fused QKV (``att_proj``) and fused
up/gate (``ff_proj``, first half = up "x", second half = gate, ``silu(gate) * x``). That
is exactly Llama's computation, so the weights map one-to-one. All dimensions come from
tensor shapes, not ``config.json`` (only heads, eps and rope theta are read from it).

Weights are read tensor by tensor straight to the target device and cast there, so a 1B
fp32 checkpoint never has to fit in host RAM. Requires torch + transformers (+
safetensors, huggingface_hub): the separate inference environment.
"""

from __future__ import annotations

import json
from pathlib import Path

_PREFIX = "model.transformer."


def olmo_key_map(olmo_key: str) -> list[tuple[str, int | None, int]]:
    """Target Llama key(s) for one OLMo key: list of (llama_key, chunk_index, n_chunks).

    ``chunk_index`` None means the tensor is copied whole; otherwise it is split into
    ``n_chunks`` equal parts along dim 0 and part ``chunk_index`` is used.
    """
    if not olmo_key.startswith(_PREFIX):
        raise KeyError(olmo_key)
    k = olmo_key[len(_PREFIX) :]
    if k == "wte.weight":
        return [("model.embed_tokens.weight", None, 1)]
    if k == "ln_f.weight":
        return [("model.norm.weight", None, 1)]
    if k == "ff_out.weight":
        return [("lm_head.weight", None, 1)]
    parts = k.split(".")
    if parts[0] != "blocks" or len(parts) != 4:
        raise KeyError(olmo_key)
    pre = f"model.layers.{int(parts[1])}."
    name = parts[2]
    if name == "att_proj":
        return [
            (pre + "self_attn.q_proj.weight", 0, 3),
            (pre + "self_attn.k_proj.weight", 1, 3),
            (pre + "self_attn.v_proj.weight", 2, 3),
        ]
    if name == "ff_proj":
        return [(pre + "mlp.up_proj.weight", 0, 2), (pre + "mlp.gate_proj.weight", 1, 2)]
    simple = {
        "attn_out": "self_attn.o_proj.weight",
        "attn_norm": "input_layernorm.weight",
        "ff_out": "mlp.down_proj.weight",
        "ff_norm": "post_attention_layernorm.weight",
    }
    if name in simple:
        return [(pre + simple[name], None, 1)]
    raise KeyError(olmo_key)


def load_datadecide(  # pragma: no cover -- needs torch/transformers and a checkpoint
    repo_id: str,
    revision: str,
    device: str = "cuda",
    dtype_name: str = "bfloat16",
    cache_dir: str | None = None,
):
    """Return (model, tokenizer, info) for one DataDecide checkpoint revision."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    dtype = getattr(torch, dtype_name)
    path = Path(snapshot_download(repo_id, revision=revision, cache_dir=cache_dir))
    cfg_json = json.loads((path / "config.json").read_text(encoding="utf-8"))
    files = sorted(path.glob("*.safetensors"))
    shapes = {}
    for fpath in files:
        with safe_open(str(fpath), "pt") as f:
            for key in f.keys():
                shapes[key] = tuple(f.get_slice(key).get_shape())
    vocab, d_model = shapes[_PREFIX + "wte.weight"]
    n_layers = 1 + max(int(k.split(".")[3]) for k in shapes if ".blocks." in k)
    hidden = shapes[_PREFIX + "blocks.0.ff_out.weight"][1]
    cfg = LlamaConfig(
        vocab_size=vocab,
        hidden_size=d_model,
        intermediate_size=hidden,
        num_hidden_layers=n_layers,
        num_attention_heads=cfg_json["n_heads"],
        num_key_value_heads=cfg_json.get("n_kv_heads") or cfg_json["n_heads"],
        max_position_embeddings=cfg_json["max_sequence_length"],
        rms_norm_eps=cfg_json["layer_norm_eps"],
        rope_theta=cfg_json["rope_theta"],
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        bos_token_id=cfg_json.get("eos_token_id"),
        eos_token_id=cfg_json.get("eos_token_id"),
        pad_token_id=cfg_json.get("pad_token_id"),
    )
    if not (
        cfg_json.get("activation_type") == "swiglu"
        and cfg_json.get("layer_norm_type") == "rms"
        and cfg_json.get("rope")
        and not cfg_json.get("include_bias")
        and not cfg_json.get("weight_tying")
        and not cfg_json.get("attention_layer_norm")
        and not cfg_json.get("clip_qkv")
        and not cfg_json.get("norm_after")
    ):
        raise ValueError(f"{repo_id}@{revision}: architecture is not Llama-equivalent")
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)  # build the skeleton directly in the target dtype
    try:
        with torch.device(device):
            model = LlamaForCausalLM(cfg)
    finally:
        torch.set_default_dtype(prev_dtype)
    state = {}
    for fpath in files:
        with safe_open(str(fpath), "pt", device=device) as f:
            for key in f.keys():
                t = f.get_tensor(key).to(dtype)
                for target, chunk, n in olmo_key_map(key):
                    state[target] = t if chunk is None else t.chunk(n, dim=0)[chunk].clone()
                del t
    model.load_state_dict(state, strict=True)
    model.eval()
    # The repo names a custom OLMoTokenizer class; the underlying tokenizer.json is a
    # standard fast tokenizer (GPT-NeoX/OLMo BPE), so load it directly.
    tok = PreTrainedTokenizerFast(tokenizer_file=str(path / "tokenizer.json"))
    special = json.loads((path / "special_tokens_map.json").read_text(encoding="utf-8"))
    for name in ("eos_token", "pad_token", "bos_token", "unk_token"):
        val = special.get(name)
        if val is not None:
            setattr(tok, name, val["content"] if isinstance(val, dict) else val)
    if tok.eos_token_id != cfg_json.get("eos_token_id"):
        raise ValueError(f"eos mismatch: tokenizer {tok.eos_token_id} vs config")
    info = {
        "repo_id": repo_id,
        "revision": revision,
        "snapshot": path.name,
        "dtype": dtype_name,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "download_bytes": int(sum(f.stat().st_size for f in files)),
    }
    return model, tok, info


def load_hf(  # pragma: no cover -- needs torch/transformers and a checkpoint
    repo_id: str,
    revision: str,
    device: str = "cuda",
    dtype_name: str = "float16",
    cache_dir: str | None = None,
):
    """Return (model, tokenizer, info) for a stock-transformers checkpoint (PolyPythia)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id, revision=revision, torch_dtype=dtype, cache_dir=cache_dir
    ).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(repo_id, revision=revision, cache_dir=cache_dir)
    info = {
        "repo_id": repo_id,
        "revision": revision,
        "dtype": dtype_name,
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    return model, tok, info
