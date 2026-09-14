"""F4 -- rank-reversal illustration: a handful of recipe curves that
cross, drawn from P1-09's named stress-test pairs (the Bonferroni-
corrected list, not the inflated uncorrected one -- see docs/decisions.md).

P1-09's own results file stores classifications and gaps, not full
per-size trajectories, so this recomputes just the 2 illustrated pairs'
mu_k(s) curves directly from the cached frame (the same source P1-09
itself used) rather than storing the full trajectory for all 500
reversing pairs, which no other figure or table needs.
"""

from __future__ import annotations

from pdt.analysis import noise
from pdt.data import frame as frame_mod
from pdt.viz import style
from pdt.viz.data import load

_P1_09_PATH = "results/p1_09_rank_reversals.json"
_METRIC = "primary_metric"
_N_PAIRS_TO_SHOW = 2


def generate():
    d = load(_P1_09_PATH)
    stress_pairs = d["thresholds"]["bonferroni"]["stress_test_pairs"][:_N_PAIRS_TO_SHOW]
    sizes_ascending = d["sizes_ascending"]

    long_frame = frame_mod.build_frame(source="macro_avg", metrics=(_METRIC,))
    seed_var = noise.seed_variance(long_frame, _METRIC)

    fig, axes = style.new_figure_grid(1, len(stress_pairs))

    for ax, pair in zip(axes, stress_pairs, strict=True):
        task, recipe_a, recipe_b = pair["task"], pair["recipe_a"], pair["recipe_b"]
        for recipe, color in (
            (recipe_a, style.COLORS["blue"]),
            (recipe_b, style.COLORS["vermillion"]),
        ):
            rows = seed_var.filter(
                (seed_var["recipe"] == recipe) & (seed_var["task"] == task)
            ).sort("params_num")
            by_size = dict(zip(rows["params_str"].to_list(), rows["mu"].to_list(), strict=True))
            xs = [s for s in sizes_ascending if s in by_size]
            ys = [by_size[s] for s in xs]
            ns = [rows.filter(rows["params_str"] == s)["params_num"][0] for s in xs]
            ax.plot(ns, ys, color=color, marker="o", markersize=2, linewidth=1.0, label=recipe)

        ax.set_xscale("log")
        ax.set_xlabel("Parameters $N$")
        if ax is axes[0]:
            ax.set_ylabel("Accuracy")
        ax.set_title(f"{task}", fontsize=6)
        ax.legend(fontsize=3.5, loc="best", frameon=False)
        if pair["crossing_size"] and pair["crossing_size"] in xs:
            crossing_n = rows.filter(rows["params_str"] == pair["crossing_size"])["params_num"][0]
            ax.axvline(
                crossing_n, color=style.COLORS["black"], linewidth=0.4, linestyle=":", alpha=0.6
            )

    fig.suptitle("Rank reversals", fontsize=8)

    return style.save(fig, "f4_rank_reversal")


if __name__ == "__main__":
    print(generate())
