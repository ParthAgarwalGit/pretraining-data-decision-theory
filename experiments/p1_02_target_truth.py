"""Task P1-02: ground truth at the target scale, and the gaps.

See plan/02-phase1-datadecide.md task P1-02. Writes results/p1_02_target.json.

Computed from **both** sources frame.py supports, not just one -- discovered
while building this task (see docs/decisions.md 2026-09-02):

- `macro_avg` (11 tasks: 9 core OLMES families + aggregated `mmlu` +
  `olmes_10_macro_avg`) is the granularity DataDecide's own baseline
  results use -- confirmed directly against `scaling_law_fit`'s `task`
  column. **This is the source P1-03's headline-number reproduction must
  use.**
- `eval_results` (66 tasks, including each of the 57 individual MMLU
  *subject* splits separately) is kept too, as a diagnostic: many
  individual MMLU-subject splits have small eval sets and produce exact
  ties between recipes, which is *why* the macro-averaged `mmlu` task
  exists in the first place. Useful for understanding tie-formation, not
  for the headline comparison.

Both discrete-accuracy metrics (primary_metric, acc_per_char) are computed
for each source -- P1-03 needs ground truth under both to run its own
metric-variant sensitivity check. The 4 continuous-proxy metrics are left
for P5-03, which is the task that actually needs them (and which does not
have bits_per_byte_corr available from macro_avg regardless -- see
frame.py's module docstring).

Primary target s* = 1B. Secondary target: the largest size below 1B that
actually exists in the data -- the plan's own text names "530M" as an
example but explicitly defers to "or the largest size below 1B present";
P0-06 found the real ladder includes 750M, which is larger and closer to
1B, so 750M is used here.
"""

from __future__ import annotations

from pdt import provenance
from pdt.analysis import ground_truth as gt
from pdt.data import datadecide as dd
from pdt.data import frame as frame_mod

_METRICS = ("primary_metric", "acc_per_char")
_PRIMARY_TARGET = "1B"

# macro_avg is the granularity that matches DataDecide's own baseline
# results; eval_results is kept as a fine-grained diagnostic. See module
# docstring.
_SOURCES = ("macro_avg", "eval_results")


def _secondary_target(long_frame) -> str:
    """The largest params_str strictly below 1B actually present in the data."""
    sizes = long_frame.select(["params_str", "params_num"]).unique()
    below_target = sizes.filter(sizes["params_num"] < 1_000_000_000).sort(
        "params_num", descending=True
    )
    if below_target.height == 0:
        raise RuntimeError("no sizes found below the 1B target scale")
    return below_target["params_str"].to_list()[0]


def _run_for_source(source: str) -> dict:
    long_frame = frame_mod.build_frame(source=source, metrics=_METRICS)
    secondary_target = _secondary_target(long_frame)
    print(f"p1_02_target_truth: [{source}] secondary target scale is {secondary_target}")

    results_by_metric: dict[str, dict] = {}
    for metric in _METRICS:
        primary = gt.compute_ground_truth(long_frame, metric, _PRIMARY_TARGET)
        secondary = gt.compute_ground_truth(long_frame, metric, secondary_target)

        ambiguous_tasks = sorted(t for t, v in primary.items() if v["is_ambiguous"])
        n_tasks = len(primary)
        print(
            f"p1_02_target_truth: [{source}] metric={metric}: "
            f"{len(ambiguous_tasks)}/{n_tasks} tasks flagged ambiguous at {_PRIMARY_TARGET} "
            f"(effect_size < {gt.AMBIGUOUS_EFFECT_SIZE_THRESHOLD})"
        )

        results_by_metric[metric] = {
            "target_scale": _PRIMARY_TARGET,
            "per_task": primary,
            "n_tasks": n_tasks,
            "ambiguous_tasks": ambiguous_tasks,
            "n_ambiguous": len(ambiguous_tasks),
            "secondary_target_scale": secondary_target,
            "per_task_secondary": secondary,
        }

    return {
        "secondary_target_scale": secondary_target,
        "metrics": results_by_metric,
        "dataset_revision": dd.cached_revision(source),
    }


def main() -> None:
    by_source = {source: _run_for_source(source) for source in _SOURCES}

    payload = {
        "primary_target_scale": _PRIMARY_TARGET,
        "ambiguous_effect_size_threshold": gt.AMBIGUOUS_EFFECT_SIZE_THRESHOLD,
        "headline_source": "macro_avg",
        "headline_source_note": (
            "macro_avg is the task granularity DataDecide's own baseline "
            "results use (11 tasks: 9 core OLMES families + aggregated mmlu "
            "+ olmes_10_macro_avg), confirmed against scaling_law_fit's task "
            "column. eval_results (66 tasks, including 57 individual MMLU "
            "subject splits) is kept as a fine-grained diagnostic only -- "
            "P1-03's headline reproduction should read from 'macro_avg' here."
        ),
        "by_source": by_source,
    }

    provenance.write_result(
        "results/p1_02_target.json",
        payload=payload,
        config={"task": "P1-02"},
    )
    print("wrote results/p1_02_target.json")


if __name__ == "__main__":
    main()
