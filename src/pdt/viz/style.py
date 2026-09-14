"""Shared figure style: colourblind-safe palette, sizing, and a common
`new_figure()` entry point every F1-F5 module uses.

Style rules from plan/02-phase1-datadecide.md P1-11: colourblind-safe
palette, no red/green pairing, legible at 6cm wide, vector PDF output,
every axis labelled with units, no chart junk.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # headless: these scripts never open an interactive window

# Set once, globally, via rcParams rather than per-Text-object calls like
# `ax.title.set_size(...)` -- those only affect a Text object that already
# exists at call time, and `ax.set_title(...)` (called later, in every F1-F5
# module, after new_figure() returns) creates a *new* Text object that does
# not inherit an earlier per-instance size call. Found by actually looking
# at a rendered figure (F1's title came out oversized and clipped), not
# assumed to work from the plotting code alone.
plt.rcParams.update(
    {
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5,
        "figure.titlesize": 8,
    }
)

FIGURES_DIR = Path("paper/figures")

# Okabe & Ito (2008) -- the standard colourblind-safe qualitative palette,
# distinguishable under the common forms of colour vision deficiency and
# containing no red/green pair.
COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# One consistent colour per fitter across every figure that shows more
# than one, so a reader doesn't have to re-learn the mapping per plot.
FITTER_COLORS = {
    "ConstantExtrapolator": COLORS["black"],
    "PowerLawN": COLORS["blue"],
    "PowerLawC": COLORS["sky_blue"],
    "ChinchillaND": COLORS["bluish_green"],
    "TwoStepLadder": COLORS["vermillion"],
    "LogLinear": COLORS["orange"],
}
FITTER_MARKERS = {
    "ConstantExtrapolator": "o",
    "PowerLawN": "s",
    "PowerLawC": "^",
    "ChinchillaND": "D",
    "TwoStepLadder": "v",
    "LogLinear": "P",
}

# 6cm wide, per the plan's own legibility requirement; a slightly taller
# aspect ratio reads better for most of these plots than a square one.
WIDTH_CM = 6.0
HEIGHT_CM = 4.6
_CM_TO_IN = 1 / 2.54


def new_figure(width_cm: float = WIDTH_CM, height_cm: float = HEIGHT_CM):
    """A blank figure/axes pair sized per the plan's 6cm-wide rule (font
    sizes come from the module-level rcParams above, which correctly
    apply even to titles/labels set later by the calling module)."""
    fig, ax = plt.subplots(figsize=(width_cm * _CM_TO_IN, height_cm * _CM_TO_IN))
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.20, top=0.88)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig, ax


def new_figure_grid(nrows: int, ncols: int, width_cm: float = 12.0, height_cm: float = 5.0):
    """A row/grid of panels sharing one figure -- for F4, which shows a
    small handful of individual case-study plots side by side rather than
    one combined axes. Wider than `new_figure()`'s single-panel default
    since it holds `ncols` panels, each individually legible."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(width_cm * _CM_TO_IN, height_cm * _CM_TO_IN))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.22, top=0.80, wspace=0.35)
    axes_flat = axes.flatten() if nrows * ncols > 1 else [axes]
    for ax in axes_flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    return fig, axes_flat


def save(fig, name: str) -> Path:
    """Vector PDF output, per the plan's rule -- always to `paper/figures/`.

    `bbox_inches="tight"` matters here: `subplots_adjust()` only controls
    the axes' position *within* the figure's fixed canvas, so an artist
    placed outside that canvas (e.g. a legend anchored below the axes via
    `bbox_to_anchor` with a negative y) is silently cropped at save time
    otherwise -- found by reading the rendered F2 PDF and seeing its
    below-axes legend labels truncated ("PowerLa", "TwoStep") rather than
    assuming the layout code was correct.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path
