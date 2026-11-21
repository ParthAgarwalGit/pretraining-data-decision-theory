"""Phase 7 inference harness: score released checkpoints on multiple-choice benchmarks.

`pdt.eval.scoring` and `pdt.eval.benchmarks` are torch-free (importable in the base
environment and in CI). `pdt.eval.convert` and `pdt.eval.run` need torch + transformers
and run only in the separate inference environment documented in `src/pdt/eval/README.md`.
"""
