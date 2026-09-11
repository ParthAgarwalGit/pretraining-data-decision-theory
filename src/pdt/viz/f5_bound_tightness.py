"""F5 -- bound tightness: empirical error against bound, per task, on a
log-log axis with the y=x line. Shows P1-07's finding directly: every
point sits at or above y=x (the bound is never violated), and the
pairwise form (filled) sits closer to the line than the marginal form
(open) -- the plan's predicted "pairwise is tighter" result.
"""

from __future__ import annotations

from pdt.viz import style
from pdt.viz.data import load

_P1_07_PATH = "results/p1_07_bound_coverage.json"
_SCHEME = "seed_bootstrap"


def generate():
    d = load(_P1_07_PATH)
    fig, ax = style.new_figure()

    marginal_x, marginal_y = [], []
    pairwise_x, pairwise_y = [], []
    for by_design in d["by_fitter"].values():
        for by_task in by_design.values():
            for by_scheme in by_task.values():
                entry = by_scheme.get(_SCHEME)
                if not entry or not entry["empirical_error_rate"]:
                    continue
                err = entry["empirical_error_rate"]
                marginal_x.append(err)
                marginal_y.append(entry["bound_marginal"])
                pairwise_x.append(err)
                pairwise_y.append(entry["bound_pairwise"])

    ax.scatter(
        marginal_x,
        marginal_y,
        s=5,
        facecolors="none",
        edgecolors=style.COLORS["sky_blue"],
        linewidths=0.5,
        label="Marginal form",
        alpha=0.7,
    )
    ax.scatter(
        pairwise_x,
        pairwise_y,
        s=5,
        color=style.COLORS["vermillion"],
        marker="x",
        linewidths=0.5,
        label="Pairwise form",
        alpha=0.7,
    )

    lims = [min(marginal_x + pairwise_x) * 0.5, max(marginal_y + pairwise_y) * 2]
    ax.plot(lims, lims, color=style.COLORS["black"], linewidth=0.6, linestyle="-", label="$y=x$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Empirical selection-error rate (Monte-Carlo)")
    ax.set_ylabel("Plug-in bound")
    ax.set_title("Bound tightness")
    # Lower-right, not upper-left (matplotlib's default best-corner guess):
    # every point has bound > 1 and empirical rate <= 1 (the bound is never
    # violated), so the whole scatter sits in the upper-left half of the
    # square axes -- an upper-left legend landed directly on top of dense
    # data there, confirmed by reading the render. Lower-right is provably
    # empty (checked against the underlying data: no point has x>2 or
    # y<1.16), and only the y=x line passes near it, at very different
    # (x, y) than where the legend box sits.
    ax.legend(fontsize=5, loc="lower right", frameon=False)

    return style.save(fig, "f5_bound_tightness")


if __name__ == "__main__":
    print(generate())
