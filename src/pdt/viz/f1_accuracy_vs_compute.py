"""F1 -- decision accuracy versus compute: single-scale frontier against
every extrapolation method. The reproduction of DataDecide's central
finding (P1-03/P1-04): no extrapolation method beats the single-scale
frontier at matched compute.
"""

from __future__ import annotations

from pdt.viz import style
from pdt.viz.data import load

_P1_04_PATH = "results/p1_04_extrapolation.json"


def generate():
    d = load(_P1_04_PATH)
    fig, ax = style.new_figure()

    frontier = sorted(d["single_scale_frontier"]["points"], key=lambda p: p["compute_cost"])
    ax.plot(
        [p["compute_cost"] for p in frontier],
        [p["accuracy_including_ties"] for p in frontier],
        color=style.COLORS["black"],
        linewidth=1.2,
        marker="o",
        markersize=3,
        label="Single-scale",
        zorder=3,
    )

    for fitter, by_design in sorted(d["fitters"].items()):
        xs = []
        ys = []
        for design in sorted(by_design, key=lambda name: by_design[name]["compute_cost"]):
            entry = by_design[design]
            if entry["macro_avg_accuracy_including_ties"] is None:
                continue
            xs.append(entry["compute_cost"])
            ys.append(entry["macro_avg_accuracy_including_ties"])
        ax.plot(
            xs,
            ys,
            color=style.FITTER_COLORS[fitter],
            marker=style.FITTER_MARKERS[fitter],
            markersize=3,
            linewidth=0.9,
            linestyle="--",
            label=fitter,
            alpha=0.85,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Fitting compute, $C = 6ND$ (FLOPs)")
    ax.set_ylabel("Decision accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Accuracy vs. compute")
    ax.legend(fontsize=4.5, loc="lower right", ncol=2, frameon=False)

    return style.save(fig, "f1_accuracy_vs_compute")


if __name__ == "__main__":
    print(generate())
