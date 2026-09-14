"""F3 -- predicted versus observed decision accuracy, with the
sigma2_extrap=0 counterfactual as a third series. Shows P1-08's headline
finding plainly: predicted (clipped) sits at 0% everywhere (a vacuous
plug-in bound -- see docs/decisions.md), while observed clusters 60-85%
and the counterfactual sits in between for most fitters.
"""

from __future__ import annotations

from pdt.viz import style
from pdt.viz.data import load

_P1_08_PATH = "results/p1_08_ceiling_prediction.json"

# Wider than the default 6cm panel -- 18 (fitter, design) categories need
# more horizontal room per tick than a single-series plot; matches the
# 12cm precedent already used for F4's multi-panel figure. Taller too, to
# fit the full rotated fitter names (not truncated -- an earlier version
# cut every label to 4 characters, which made "PowerLawC" and "PowerLawN"
# both read "Powe", indistinguishable -- plus a legend row below the axis.
_WIDTH_CM = 12.0
_HEIGHT_CM = 8.0


def generate():
    d = load(_P1_08_PATH)
    fig, ax = style.new_figure(width_cm=_WIDTH_CM, height_cm=_HEIGHT_CM)
    fig.subplots_adjust(bottom=0.30, top=0.90)

    series = [
        ("observed_accuracy", "Observed (P1-03/04)", style.COLORS["black"], "o"),
        ("predicted_accuracy", "Predicted (plug-in bound)", style.COLORS["vermillion"], "x"),
        (
            "counterfactual_sigma2_extrap_zero_predicted_accuracy",
            r"Predicted, $\sigma^2_{extrap}=0$",
            style.COLORS["blue"],
            "^",
        ),
    ]

    fitter_order = {f: i for i, f in enumerate(sorted(style.FITTER_COLORS))}
    x_positions = []
    x_labels = []
    x = 0
    for fitter in sorted(d["fitters"], key=lambda f: fitter_order[f]):
        for design in sorted(d["by_fitter"][fitter]):
            x_positions.append(x)
            x_labels.append(f"{fitter}\n{design.replace('S_fit_le_', '')}")
            x += 1

    for field, label, color, marker in series:
        ys = []
        for fitter in sorted(d["fitters"], key=lambda f: fitter_order[f]):
            for design in sorted(d["by_fitter"][fitter]):
                ys.append(d["by_fitter"][fitter][design][field])
        ax.scatter(x_positions, ys, color=color, marker=marker, s=10, label=label, zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=4, rotation=90)
    ax.set_ylabel("Decision accuracy")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title("Predicted vs. observed accuracy")
    ax.axhline(0.0, color=style.COLORS["black"], linewidth=0.4, alpha=0.4)

    # Below the axis, not in-plot: the three series occupy most of the
    # y-range between them (observed ~0.6-0.85, the two predicted series
    # ~0.0-0.4), leaving no band wide enough to drop a 3-row legend into
    # without it sitting on top of real points -- confirmed by reading the
    # first render, where "Observed (P1-03/04)" landed right at the same
    # height as the actual observed cluster it was labelling.
    fig.legend(loc="lower center", bbox_to_anchor=(0.56, 0.005), ncol=3, frameon=False, fontsize=6)

    return style.save(fig, "f3_predicted_vs_observed")


if __name__ == "__main__":
    print(generate())
