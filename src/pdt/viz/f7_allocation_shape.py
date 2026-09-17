"""F7 -- the shape of the COMPUTED challenger allocation across scales,
under different gap regimes (results/p3_07_allocation_shape.json).

**Not a certified-optimal allocation -- PR #33's review.** The
underlying data is `solve_allocation`'s own heuristic solution
(src/pdt/bai/allocation.py's own docstring: "validated-reasonable, not
certified-optimal" -- every restart lands at a locally max-min-consistent
point, not verified to reach the global optimum, and a real instance was
found where `brute_force_allocation` finds a strictly better feasible
point). It also only ever covers the CHALLENGER arms by construction
(Theorem 2 Part A's change-of-measure program never perturbs k*'s own
distribution, so k*'s allocation share -- handled by a separate,
unrelated heuristic in `extrapolation_track_and_stop` -- never appears
here at all). This figure shows what `solve_allocation` actually
computes for the challengers, honestly labeled as such, not evidence of
a provably optimal full-instance BAI allocation.

Not "as delta varies" literally either -- solve_allocation's shape is
provably delta-independent (Track-and-Stop's classical property,
Garivier & Kaufmann 2016); see that result's own module docstring and
docs/decisions.md for the full account. This shows the shape across
gap regimes (what the T* program actually optimizes over) instead, plus
a small inset of the separately delta-dependent quantity (total
required compute, at one fixed shape).
"""

from __future__ import annotations

import numpy as np

from pdt.viz import style
from pdt.viz.data import load

_PATH = "results/p3_07_allocation_shape.json"
_ARM_COLORS = {"k1": style.COLORS["blue"], "k2": style.COLORS["vermillion"]}


def generate():
    d = load(_PATH)
    regimes = d["regimes"]
    regime_names = list(regimes.keys())
    fig, axes = style.new_figure_grid(1, len(regime_names), width_cm=12.0, height_cm=5.0)

    scale_ns = sorted(
        float(n) for n in next(iter(regimes.values()))["compute_share_by_scale"]["k1"]
    )
    scale_labels = [f"{n:.0e}" for n in scale_ns]

    for ax, regime_name in zip(axes, regime_names, strict=True):
        regime = regimes[regime_name]
        arms = list(regime["deltas"].keys())
        width = 0.8 / len(arms)
        x = np.arange(len(scale_ns))
        for i, arm in enumerate(arms):
            shares = regime["compute_share_by_scale"][arm]
            ys = [shares[str(n)] for n in scale_ns]
            ax.bar(
                x + i * width - width / 2,
                ys,
                width=width,
                color=_ARM_COLORS.get(arm, style.COLORS["black"]),
                label=f"{arm} ($\\Delta$={regime['deltas'][arm]:.2g})",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(scale_labels, fontsize=4, rotation=45)
        ax.set_title(regime_name.replace("_", " "), fontsize=5.5)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=3.3, loc="upper left", frameon=False)
        if ax is axes[0]:
            ax.set_ylabel("Compute share")
        ax.set_xlabel("Scale $N$")

    fig.suptitle(
        "Computed challenger-allocation compute share by scale, across gap regimes\n"
        "(solve_allocation's heuristic solution, restricted to challenger arms -- "
        "NOT a certified-optimal BAI allocation, and excludes k*'s own share; "
        "shape is delta-independent -- see docs/decisions.md)",
        fontsize=4.6,
    )
    return style.save(fig, "f7_allocation_shape")
