"""F8 -- error rate against misspecification level, ours against
baselines, and the crossing point where baselines become confidently
wrong (results/p3_06_eta_sensitivity.json,
results/p3_07_baseline_vs_misspecification.json).

Two different x-axes are reported honestly rather than forced onto one
literal scale: the left panel sweeps baselines' *true* bias magnitude
(they have no eta input at all) and shows the crossing point where the
true winner flips and baseline accuracy falls off a cliff. The right
panel sweeps ETS's *assumed* eta at a fixed true bias: only the eta=0
point is a genuine, uncaveated result (100% wrong when certified); every
non-zero eta point never reached a genuine resolution within this
session's round budget (round-cap exhaustion, not genuine abstention --
see docs/decisions.md) and is marked as such rather than plotted as if
it were a clean "0% error" data point.
"""

from __future__ import annotations

from pdt.viz import style
from pdt.viz.data import load

_P3_06_PATH = "results/p3_06_eta_sensitivity.json"
_P3_07_PATH = "results/p3_07_baseline_vs_misspecification.json"

_BASELINE_COLORS = {
    "SingleScale": style.COLORS["black"],
    "FixedLadderExtrapolation": style.COLORS["orange"],
    "UniformAllocation": style.COLORS["sky_blue"],
    "SuccessiveHalvingOverScales": style.COLORS["vermillion"],
}


def generate():
    d6 = load(_P3_06_PATH)
    d7 = load(_P3_07_PATH)
    fig, (ax_left, ax_right) = style.new_figure_grid(1, 2, width_cm=12.0, height_cm=5.0)

    for name, color in _BASELINE_COLORS.items():
        xs = [c["true_bias"] for c in d7["sweep"]]
        ys = [c["accuracy"][name] for c in d7["sweep"]]
        ax_left.plot(xs, ys, color=color, marker="o", markersize=2.5, linewidth=0.9, label=name)
    ax_left.axvline(
        d7["observed_gap"],
        color=style.COLORS["black"],
        linewidth=0.6,
        linestyle=":",
        label="crossing point",
    )
    ax_left.set_xlabel("True bias magnitude")
    ax_left.set_ylabel("Baseline accuracy")
    ax_left.set_ylim(-0.05, 1.05)
    ax_left.set_title("Baselines: no eta input", fontsize=5.5)
    ax_left.legend(fontsize=3.0, loc="center left", frameon=False)

    etas = [c["eta_assumed"]["leader"] for c in d6["sweep"]]
    errors = [c["error_rate_given_certified"] for c in d6["sweep"]]
    round_cap = [c["round_cap_exhausted_rate"] for c in d6["sweep"]]
    resolved_x = [e for e, err in zip(etas, errors, strict=True) if err is not None]
    resolved_y = [err for err in errors if err is not None]
    unresolved_x = [e for e, rc in zip(etas, round_cap, strict=True) if rc == 1.0]
    ax_right.plot(
        resolved_x,
        resolved_y,
        color=style.COLORS["vermillion"],
        marker="o",
        markersize=4,
        linewidth=0,
        label="error rate (certified)",
        zorder=3,
    )
    ax_right.scatter(
        unresolved_x,
        [0.5] * len(unresolved_x),
        color=style.COLORS["black"],
        marker="x",
        s=12,
        label="round-cap exhausted\n(no genuine resolution)",
        zorder=3,
    )
    ax_right.set_xlabel("Assumed eta")
    ax_right.set_ylabel("Error rate given certified")
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.set_title("ETS (ours): eta input", fontsize=5.5)
    ax_right.legend(fontsize=3.0, loc="center right", frameon=False)

    fig.suptitle(
        "Misspecification sensitivity: baselines (true bias) vs. ETS (assumed eta)",
        fontsize=5.2,
    )
    return style.save(fig, "f8_eta_sensitivity")
