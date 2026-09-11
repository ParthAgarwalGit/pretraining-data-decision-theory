"""Task P1-03: reproduce DataDecide's single-scale baseline.

See plan/02-phase1-datadecide.md task P1-03. THE reproduction check for
this project's whole pipeline -- target is ~80% pairwise decision accuracy
at ~150M params. Writes results/p1_03_single_scale.json.

Uses `macro_avg` only (11 tasks), not `eval_results` -- P1-02 established
this is the granularity DataDecide's own baseline results use; comparing
against `eval_results`' 66-task breakdown (57 of which are individual MMLU
subject splits) would not be reproducing the same quantity at all.

Per the plan's explicit instruction: if the ~80% figure doesn't reproduce,
do NOT adjust anything until running the four documented debugging
variants (metric x seed-handling, both computed here regardless of
whether the primary variant reproduces) and reporting all four honestly.
The published number is a target, not a requirement.
"""

from __future__ import annotations

from pdt import provenance
from pdt.analysis import decision_accuracy as da
from pdt.analysis import ground_truth as gt
from pdt.data import datadecide as dd
from pdt.data import frame as frame_mod

_TARGET = "1B"
_METRICS = ("primary_metric", "acc_per_char")
_SEED_MODES = ("average", "default_only")
_HIGHLIGHT_SIZE = "150M"
_PUBLISHED_FIGURE = 0.80  # DataDecide's reported ~80% at ~150M -- NUMBER-OK: cited prior work


def _proxy_sizes_below_target(long_frame) -> list[str]:
    sizes = long_frame.select(["params_str", "params_num"]).unique()
    below = sizes.filter(sizes["params_num"] < 1_000_000_000).sort("params_num")
    return below["params_str"].to_list()


def _run_variant(
    long_frame, ground_truth_per_task: dict, metric: str, seed_mode: str, proxy_sizes: list[str]
) -> dict:
    """Per-proxy-size decision accuracy (per task + macro-average)."""
    by_size: dict[str, dict] = {}
    for proxy_size in proxy_sizes:
        proxy_by_task = da.recipe_means(long_frame, metric, proxy_size, seed_mode=seed_mode)

        per_task: dict[str, dict] = {}
        for task, proxy_means in proxy_by_task.items():
            if task not in ground_truth_per_task:
                continue
            target = ground_truth_per_task[task]
            per_task[task] = da.pairwise_decision_accuracy(
                proxy_means, target["mu"], target["sd_seed"], target["n_seeds"]
            )

        incl = [
            v["accuracy_including_ties"]
            for v in per_task.values()
            if v["accuracy_including_ties"] is not None
        ]
        excl = [
            v["accuracy_excluding_ties"]
            for v in per_task.values()
            if v["accuracy_excluding_ties"] is not None
        ]
        taus = [v["kendall_tau"] for v in per_task.values() if v["kendall_tau"] is not None]

        by_size[proxy_size] = {
            "per_task": per_task,
            "n_tasks": len(per_task),
            "macro_avg_accuracy_including_ties": (sum(incl) / len(incl)) if incl else None,
            "macro_avg_accuracy_excluding_ties": (sum(excl) / len(excl)) if excl else None,
            "macro_avg_kendall_tau": (sum(taus) / len(taus)) if taus else None,
        }
    return by_size


def main() -> None:
    long_frame = frame_mod.build_frame(source="macro_avg", metrics=_METRICS)
    proxy_sizes = _proxy_sizes_below_target(long_frame)
    print(f"p1_03_single_scale: {len(proxy_sizes)} proxy sizes below {_TARGET}: {proxy_sizes}")

    # Ground truth per metric (reused across seed_mode variants -- the
    # target scale's ground truth doesn't depend on how a *proxy* scale's
    # seeds are handled).
    ground_truth_by_metric = {m: gt.compute_ground_truth(long_frame, m, _TARGET) for m in _METRICS}

    # The four-variant sensitivity table, computed at every proxy size (not
    # just the highlighted one) -- cheap, and the by-size table is more
    # useful with all four variants available throughout, not just at 150M.
    variants: dict[str, dict] = {}
    for metric in _METRICS:
        for seed_mode in _SEED_MODES:
            key = f"{metric}__{seed_mode}"
            variants[key] = _run_variant(
                long_frame, ground_truth_by_metric[metric], metric, seed_mode, proxy_sizes
            )

    # Headline variant: primary_metric, seed-averaged -- DataDecide's own
    # reporting convention.
    headline_key = "primary_metric__average"
    headline_150m = variants[headline_key][_HIGHLIGHT_SIZE]
    headline_acc = headline_150m["macro_avg_accuracy_including_ties"]

    print(
        f"p1_03_single_scale: HEADLINE {_HIGHLIGHT_SIZE} decision accuracy "
        f"(primary_metric, seed-averaged, including ties) = "
        f"{headline_acc:.1%} vs DataDecide's published ~{_PUBLISHED_FIGURE:.0%}"
    )
    for key, by_size in variants.items():
        acc = by_size[_HIGHLIGHT_SIZE]["macro_avg_accuracy_including_ties"]
        if acc is not None:
            print(f"p1_03_single_scale: {_HIGHLIGHT_SIZE} / {key} = {acc:.1%}")

    payload = {
        "target_scale": _TARGET,
        "highlight_proxy_size": _HIGHLIGHT_SIZE,
        "published_figure": _PUBLISHED_FIGURE,
        "headline_variant": headline_key,
        "headline_accuracy_at_highlight_size": headline_acc,
        "proxy_sizes": proxy_sizes,
        "variants": variants,
        "dataset_revision": dd.cached_revision("macro_avg"),
    }

    provenance.write_result(
        "results/p1_03_single_scale.json",
        payload=payload,
        config={"task": "P1-03"},
    )
    print("wrote results/p1_03_single_scale.json")


if __name__ == "__main__":
    main()
