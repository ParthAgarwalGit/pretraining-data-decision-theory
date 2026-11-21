# `pdt.eval`: inference harness (Phase 7)

`pdt.eval.scoring` and `pdt.eval.benchmarks` are torch-free and run in the repo's uv
environment and in CI. Model loading (`convert`) and scoring (`run`) need an
**inference environment**: Python 3.12 with `torch` (CUDA), `transformers` (4.57+),
`safetensors`, `huggingface_hub`, `datasets`, `numpy` and `pytest`. Torch is deliberately
not a project extra, because CI installs all extras.

## Validation (N7-04)

- **DataDecide checkpoints.** These are OLMo-1 (`hf_olmo`). `convert.load_datadecide`
  maps them one-to-one onto `LlamaForCausalLM`.
- **Validation method.** DataDecide's own ARC-Easy request file was replayed for the same
  checkpoint and compared with DataDecide's released per-instance log-likelihoods
  (`allenai/DataDecide-eval-instances`).
- **Result.** In fp32 the outputs agree to within 1e-4 nats. In **fp16**, the precision
  used for all Phase 7 runs, they agree to within a few hundredths of a nat with 0
  per-char prediction flips. bf16 flips some predictions, so it is not used.
  See `results/n7_04_harness_timing.json`.

## Laptop (Windows, RTX 4060)

```bash
PYTHONPATH=src HF_HUB_CACHE=data/n7_04_scratch/hf python experiments/n7_04_harness_timing.py
PYTHONPATH=src python -m pytest tests/test_eval_scoring.py -o addopts="" -q
```

## Colab / Kaggle (T4: fp16 only, which is what we use)

```python
!git clone <repo> && cd pretraining-data-decision-theory && git checkout <branch>
!pip install -q datasets safetensors   # torch and transformers are preinstalled
import sys; sys.path.insert(0, "src")
from pdt.eval import convert, run, benchmarks, scoring
model, tok, info = convert.load_datadecide(
    "allenai/DataDecide-c4-1B", scoring.datadecide_revision(69369, "default"),
    dtype_name="float16")
records, stats = run.score_items(model, tok, benchmarks.load_items("sciq"))
```

## Final checkpoints

A "final" checkpoint is the frame's `is_final` step for the size. Pass it with the
seed label, e.g. `datadecide_revision(37500, "small aux 2")`. Do NOT take the latest
branch: some aux seeds have later branches from longer runs.
