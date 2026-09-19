"""F3 -- the bound versus what actually happened, with the two DECISION EVENTS
kept apart.

Second review of PR #20: the original F3 plotted the plug-in bound (a lower bound
on the probability of selecting the single best arm) and the observed fraction of
correctly ordered recipe PAIRS on one "Decision accuracy" axis as if they were
predicted-versus-observed values of the same quantity. They are different events
and cannot be compared. The figure now has two panels:

- **Panel A -- best-arm selection (the event the bound is about).** The plug-in
  lower bound (clipped to [0, 1]), the same bound with `sigma2_extrap = 0`, and
  P1-07's own Monte-Carlo estimate of P(argmax recipe == k*) for the same fitting
  procedure. Only these three are comparable with each other.
- **Panel B -- all-pairs ordering accuracy (a different event).** The observed
  fraction of correctly ordered pairs from P1-03/04. There is no bound for this
  event, so nothing is "predicted" here; the panel exists to show it is a
  different, generally larger, number than panel A and must not be read against it.

Requires the corrected `results/p1_08_ceiling_prediction.json`
(`observed_best_arm_accuracy`); absent fields mean the file predates the fix and
must be regenerated, not silently plotted.
"""

from __future__ import annotations

from pdt.viz import style
from pdt.viz.data import load

_P1_08_PATH = "results/p1_08_ceiling_prediction.json"

_WIDTH_CM = 12.0
_HEIGHT_CM = 8.0

_PANEL_A_SERIES = [
    (
        "observed_best_arm_accuracy",
        "Observed best-arm selection (P1-07 Monte Carlo)",
        "black",
        "o",
    ),
    ("predicted_accuracy", "Plug-in lower bound on best-arm selection", "vermillion", "x"),
    (
        "counterfactual_sigma2_extrap_zero_predicted_accuracy",
        r"Same bound with $\sigma^2_{extrap}=0$",
        "blue",
        "^",
    ),
]
_PANEL_B_SERIES = [
    ("observed_accuracy", "Observed all-pairs ordering accuracy (P1-03/04)", "black", "s"),
]


def _require_best_arm_fields(d: dict) -> None:
    for fitter in d["fitters"]:
        for design, entry in d["by_fitter"][fitter].items():
            if "observed_best_arm_accuracy" not in entry:
                raise KeyError(
                    f"{_P1_08_PATH} lacks 'observed_best_arm_accuracy' for {fitter}/{design}: "
                    "it predates the same-event fix (PR #18) -- regenerate P1-08 first."
                )


def _categories(d: dict) -> tuple[list[tuple[str, str]], list[str]]:
    fitter_order = {f: i for i, f in enumerate(sorted(style.FITTER_COLORS))}
    pairs = [
        (fitter, design)
        for fitter in sorted(d["fitters"], key=lambda f: fitter_order[f])
        for design in sorted(d["by_fitter"][fitter])
    ]
    labels = [f"{fitter}\n{design.replace('S_fit_le_', '')}" for fitter, design in pairs]
    return pairs, labels


def _plot_panel(ax, d, pairs, labels, series, title):
    xs = list(range(len(pairs)))
    for field, label, color_key, marker in series:
        pts = [(x, d["by_fitter"][f][dz][field]) for x, (f, dz) in zip(xs, pairs, strict=True)]
        # A missing value (None) is an unassessed cell, not a zero: leave it out.
        pts = [(x, y) for x, y in pts if y is not None]
        if pts:
            px, py = zip(*pts, strict=True)
            ax.scatter(
                px, py, color=style.COLORS[color_key], marker=marker, s=10, label=label, zorder=3
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=4, rotation=90)
    ax.set_ylim(-0.05, 1.0)
    ax.set_title(title)
    ax.axhline(0.0, color=style.COLORS["black"], linewidth=0.4, alpha=0.4)
    ax.legend(loc="upper center", frameon=False, fontsize=4.5)


def generate():
    d = load(_P1_08_PATH)
    _require_best_arm_fields(d)
    pairs, labels = _categories(d)
    fig, axes = style.new_figure_grid(1, 2, width_cm=_WIDTH_CM * 1.6, height_cm=_HEIGHT_CM)
    fig.subplots_adjust(bottom=0.30, top=0.90, wspace=0.25)

    _plot_panel(
        axes[0],
        d,
        pairs,
        labels,
        _PANEL_A_SERIES,
        "A: best-arm selection (bound vs. observed)",
    )
    axes[0].set_ylabel("P(select the best recipe)")
    _plot_panel(
        axes[1], d, pairs, labels, _PANEL_B_SERIES, "B: all-pairs ordering (a different event)"
    )
    axes[1].set_ylabel("Fraction of recipe pairs correctly ordered")

    return style.save(fig, "f3_predicted_vs_observed")


if __name__ == "__main__":
    print(generate())
