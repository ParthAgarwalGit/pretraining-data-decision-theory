"""F2 -- sigma2_extrap and v versus fitting compute, log axis. "The
paper's key figure" per the plan. Plots what P1-06 actually found: both
components shrink with compute, but sigma2_extrap shrinks *faster* than
v (the opposite of the plan's own stated signature prediction that v
should shrink while sigma2_extrap stays flat) -- this figure shows the
real data, not the hypothesised pattern; see docs/decisions.md.
"""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.ticker as mticker

from pdt.viz import style
from pdt.viz.data import load

_P1_06_PATH = "results/p1_06_decomposition.json"
_SCHEME = "seed_bootstrap"

# Taller than the default 6cm-wide panel so a legend fits below the axes
# without covering data -- an early version put 12 in-plot legend entries
# (fitter x quantity) directly over the lines they described; fixed by
# looking at the rendered PDF, not assumed fine from the plotting code.
_HEIGHT_CM = 8.4


def generate():
    d = load(_P1_06_PATH)
    fig, ax = style.new_figure(height_cm=_HEIGHT_CM)
    fig.subplots_adjust(bottom=0.38, top=0.90)

    for fitter in sorted(style.FITTER_COLORS):
        points = sorted(
            (r for r in d["ratio_vs_compute"] if r["fitter"] == fitter and r["scheme"] == _SCHEME),
            key=lambda r: r["compute_cost"],
        )
        if not points:
            continue
        xs = [p["compute_cost"] for p in points]
        color = style.FITTER_COLORS[fitter]
        marker = style.FITTER_MARKERS[fitter]
        ax.plot(
            xs,
            [p["median_sigma2_extrap_hat"] for p in points],
            color=color,
            marker=marker,
            markersize=3,
            linewidth=1.0,
            linestyle="-",
        )
        ax.plot(
            xs,
            [p["median_v_hat"] for p in points],
            color=color,
            marker=marker,
            markersize=3,
            linewidth=1.0,
            linestyle=":",
            alpha=0.7,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Fitting compute, $C = 6ND$ (FLOPs)")
    ax.set_ylabel("Median value (accuracy$^2$ units)")
    ax.set_title(r"$\sigma^2_{extrap}$ (solid) vs. $v$ (dotted)")

    # Only 3 distinct compute costs exist in the data (one per design), but
    # the default log-scale locator adds minor ticks (2x, 3x, 4x, 6x) at
    # every decade regardless -- with all 3 real points inside one decade,
    # that crowded the axis into an illegible smear. Ticking exactly the
    # real x-values instead, with minor ticks off, is both more legible and
    # more honest about what was actually measured. Labels are built by
    # hand rather than via LogFormatterMathtext, which rendered these
    # non-decade values as malformed fractional exponents ("10^19.33") and
    # silently dropped the middle tick's label -- caught the same way as
    # every other figure bug this phase: by reading the actual render.
    all_xs = sorted({p["compute_cost"] for p in d["ratio_vs_compute"] if p["scheme"] == _SCHEME})

    def _sci_label(x: float) -> str:
        exponent = int(f"{x:e}".split("e")[1])
        mantissa = x / 10**exponent
        return rf"${mantissa:.1f}\times 10^{{{exponent}}}$"

    ax.xaxis.set_major_locator(mticker.FixedLocator(all_xs))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter([_sci_label(x) for x in all_xs]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Two compact legends instead of one 12-entry legend: colour -> fitter
    # (6 entries, a single representative solid line each) and
    # linestyle -> quantity (2 entries), stacked below the axes. ncol=2
    # (not 3) keeps each column narrow enough for "ConstantExtrapolator" to
    # fit inside the 6cm figure width -- a 3-column layout ran the longer
    # fitter names past the right edge (confirmed via a debug PNG render,
    # since bbox_inches="tight" alone did not rescue an artist added via
    # ax.add_artist() rather than being the axes' own tracked legend).
    #
    # Placed with fig.legend() (figure-fraction coordinates) rather than
    # ax.legend(bbox_to_anchor=..., transAxes) -- the axes-fraction version
    # put the "Fitter" legend title directly on top of the xlabel, since
    # its offset was relative to the axes' own height and didn't account
    # for the xlabel's position below it. Figure-fraction coordinates are
    # absolute, so xlabel/legend1/legend2 can be stacked without that
    # coupling; x=0.58 (not 0.5) centres under the axes, not the whole
    # canvas, since the y-axis label eats the left ~20% of the figure
    # (new_figure() sets left=0.20, right=0.96). No "Fitter" title on the
    # first legend -- its entries (the fitter names) are self-explanatory,
    # and dropping it removes the last thing that was crowding the xlabel.
    fitter_handles = [
        mlines.Line2D(
            [],
            [],
            color=style.FITTER_COLORS[f],
            marker=style.FITTER_MARKERS[f],
            markersize=3,
            label=f,
        )
        for f in sorted(style.FITTER_COLORS)
    ]
    quantity_handles = [
        mlines.Line2D(
            [], [], color=style.COLORS["black"], linestyle="-", label=r"$\sigma^2_{extrap}$"
        ),
        mlines.Line2D([], [], color=style.COLORS["black"], linestyle=":", label="$v$"),
    ]
    fig.legend(
        handles=fitter_handles,
        loc="lower center",
        bbox_to_anchor=(0.58, 0.10),
        ncol=2,
        frameon=False,
        fontsize=5,
    )
    fig.legend(
        handles=quantity_handles,
        loc="lower center",
        bbox_to_anchor=(0.58, 0.01),
        ncol=2,
        frameon=False,
        fontsize=5,
    )

    return style.save(fig, "f2_bias_variance_vs_compute")


if __name__ == "__main__":
    print(generate())
