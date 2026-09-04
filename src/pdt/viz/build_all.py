"""Regenerates every Phase 1 figure (F1-F5) from results/*.json, with no
manual steps -- `make figures` runs `python -m pdt.viz.build_all`.
"""

from __future__ import annotations

from pdt.viz import (
    f1_accuracy_vs_compute,
    f2_bias_variance_vs_compute,
    f3_predicted_vs_observed,
    f4_rank_reversal,
    f5_bound_tightness,
)

_GENERATORS = (
    f1_accuracy_vs_compute,
    f2_bias_variance_vs_compute,
    f3_predicted_vs_observed,
    f4_rank_reversal,
    f5_bound_tightness,
)


def main() -> None:
    for module in _GENERATORS:
        path = module.generate()
        print(f"build_all: wrote {path}")


if __name__ == "__main__":
    main()
