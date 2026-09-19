"""F6 -- compute to a decision, by method, by task, on the real
DataDecide replay (results/p3_05_replay.json).

ETS never certified on any of the 4 tasks tested (see docs/decisions.md,
P3-05 entries) -- shown here honestly, not smoothed over. Where it hit the
round cap its bar (compute spent before falling back) is drawn hatched; where the
finite DataDecide replicate pool ran out first ("pool exhausted") no compute
is recorded and the panel carries a text note instead of a bar -- never a
zero-height bar that could be read as "free". Both are distinct from the
baselines' real compute-to-a-decision bars. Baseline accuracy (bootstrapped
over the 3 real seeds, with a 95% CI) is shown as the marker's fill: solid green outline for high
accuracy, hollow/red for low -- deliberately not a full second panel,
since accuracy-vs-compute is already F1's job; this figure's own point
is compute, with accuracy as context.
"""

from __future__ import annotations

import numpy as np

from pdt.viz import style
from pdt.viz.data import load

_P3_05_PATH = "results/p3_05_replay.json"
_METHODS = [
    "SingleScale",
    "FixedLadderExtrapolation",
    "UniformAllocation",
    "SuccessiveHalvingOverScales",
    "ExtrapolationTrackAndStop",
]
_METHOD_LABELS = {
    "SingleScale": "Single-\nscale",
    "FixedLadderExtrapolation": "Fixed\nladder",
    "UniformAllocation": "Uniform",
    "SuccessiveHalvingOverScales": "Succ.\nhalving",
    "ExtrapolationTrackAndStop": "ETS\n(ours)",
}


def generate():
    d = load(_P3_05_PATH)
    cells = d["cells"]
    fig, axes = style.new_figure_grid(1, len(cells), width_cm=13.0, height_cm=5.2)

    for ax, cell in zip(axes, cells, strict=True):
        xs = np.arange(len(_METHODS))
        computes = []
        accuracies = []
        is_ets = []
        for m in _METHODS:
            if m == "ExtrapolationTrackAndStop":
                run = cell["ets_single_run"]
                # pool_exhausted runs record no compute or correctness: plot NaN
                # (no bar) and annotate below, rather than fabricate a value.
                computes.append(
                    run["compute_spent"] if run["compute_spent"] is not None else np.nan
                )
                accuracies.append(1.0 if run["correct"] else 0.0)
                is_ets.append(True)
            else:
                computes.append(cell["baselines"][m]["mean_compute"])
                accuracies.append(cell["baselines"][m]["accuracy"])
                is_ets.append(False)

        colors = [
            style.COLORS["bluish_green"] if a >= 0.5 else style.COLORS["vermillion"]
            for a in accuracies
        ]
        bars = ax.bar(
            xs,
            computes,
            color=[
                style.COLORS["black"] if ets else c for ets, c in zip(is_ets, colors, strict=True)
            ],
            hatch=["//" if ets else None for ets in is_ets],
            edgecolor="white",
            linewidth=0.3,
        )
        for bar, ets in zip(bars, is_ets, strict=True):
            if ets:
                bar.set_alpha(0.35)

        ets_run = cell["ets_single_run"]
        if ets_run["compute_spent"] is None:
            ax.text(
                len(_METHODS) - 1,
                np.nanmin(computes) * 3,
                "pool\nexhausted\n(no compute\nrecorded)",
                ha="center",
                va="bottom",
                fontsize=3.6,
            )
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([_METHOD_LABELS[m] for m in _METHODS], fontsize=4.2)
        ax.set_title(cell["task"], fontsize=6)
        if ax is axes[0]:
            ax.set_ylabel("Compute (FLOPs)")

    fig.suptitle(
        "Compute per method, by task (real DataDecide replay) -- ETS (hatched) never "
        "certified: round cap or replicate pool exhausted first",
        fontsize=5.5,
    )
    return style.save(fig, "f6_compute_by_method")
