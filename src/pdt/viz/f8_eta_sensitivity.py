"""F8 -- error rate against misspecification level, ours against
baselines (results/p3_06_eta_sensitivity.json,
results/p3_07_baseline_vs_misspecification.json).

Two different x-axes are reported honestly rather than forced onto one
literal scale: the left panel sweeps baselines' *true* bias magnitude
(they have no eta input at all; 30 independent noise realizations per point) and shows
where the true winner flips and baseline accuracy falls off a cliff. The right
panel sweeps ETS's *assumed* eta at a fixed true bias and plots the JOINT rate
P[certified AND wrong] -- the quantity delta-correctness bounds -- with its exact
95% (Clopper-Pearson) interval, against the requested delta. With only 20 independent
runs per point the intervals are wide: at eta = 0 and 0.25x the true bias one run of
twenty certified (wrongly), which is NOT a statistically detectable violation of
delta. Every eta >= 0.5x the true bias ended at the round cap without certifying (marked
with an x, y-position arbitrary), so this panel says nothing about calibration in the
well-specified regime either. (The earlier version of this figure reported a
"100% wrong when certified" point that came from 20 copies of one noise realization; it
is withdrawn -- see docs/decisions.md.)
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

    delta = d6["delta"]
    etas = [c["eta_assumed"]["leader"] for c in d6["sweep"]]
    joint = [c["joint_wrong_certified_rate"] for c in d6["sweep"]]
    lo = [c["joint_wrong_certified_ci95"][0] for c in d6["sweep"]]
    hi = [c["joint_wrong_certified_ci95"][1] for c in d6["sweep"]]
    round_cap = [c["round_cap_exhausted_rate"] for c in d6["sweep"]]
    ax_right.errorbar(
        etas,
        joint,
        yerr=[
            [m - low for m, low in zip(joint, lo, strict=True)],
            [high - m for m, high in zip(joint, hi, strict=True)],
        ],
        color=style.COLORS["vermillion"],
        marker="o",
        markersize=3,
        linewidth=0,
        elinewidth=0.7,
        capsize=1.5,
        label="P[certified and wrong], exact 95% CI",
        zorder=3,
    )
    ax_right.axhline(
        delta, color=style.COLORS["black"], linewidth=0.6, linestyle="--", label="delta"
    )
    unresolved_x = [e for e, rc in zip(etas, round_cap, strict=True) if rc == 1.0]
    ax_right.scatter(
        unresolved_x,
        [0.6] * len(unresolved_x),
        color=style.COLORS["black"],
        marker="x",
        s=12,
        label="all runs hit round cap\n(none certified)",
        zorder=3,
    )
    ax_right.set_xlabel("Assumed eta")
    ax_right.set_ylabel("P[certified and wrong]")
    ax_right.set_ylim(-0.05, 1.05)
    ax_right.set_title("ETS (ours): eta input, 20 runs/point", fontsize=5.5)
    ax_right.legend(fontsize=3.0, loc="center right", frameon=False)

    fig.suptitle(
        "Misspecification sensitivity: baselines (true bias) vs. ETS (assumed eta)",
        fontsize=5.2,
    )
    return style.save(fig, "f8_eta_sensitivity")
