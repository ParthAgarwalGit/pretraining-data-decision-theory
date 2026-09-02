"""Task P1-01: build the canonical analysis frame and its coverage matrix.

See plan/02-phase1-datadecide.md task P1-01. Writes results/p1_01_frame.json.
"""

from __future__ import annotations

from pdt import provenance
from pdt.data import datadecide as dd
from pdt.data import frame as frame_mod


def main() -> None:
    long_frame = frame_mod.build_frame()

    # Coverage is computed on the wide frame (already loaded/cached by
    # build_frame() via load_eval_results()) rather than the long frame --
    # coverage doesn't depend on metric_name, so using the long frame would
    # mean redundantly deduplicating 6x the rows for no benefit.
    wide_frame = dd.load_eval_results()
    coverage = frame_mod.coverage_matrix(wide_frame)

    if coverage["n_missing_cells"] > 0:
        print(
            f"p1_01_build_frame: {coverage['n_missing_cells']} missing "
            f"(recipe, params, seed, task) cells found -- see "
            f"missing_by_task / missing_by_params / missing_cells_sample "
            f"in results/p1_01_frame.json."
        )
    else:
        print("p1_01_build_frame: coverage matrix is complete, no missing cells.")

    payload = {
        "frame_shape": {"n_rows": long_frame.height, "n_columns": long_frame.width},
        "metric_whitelist": list(frame_mod.METRIC_WHITELIST),
        "coverage": coverage,
        "dataset_revision": dd.cached_revision("eval_results"),
    }

    provenance.write_result(
        "results/p1_01_frame.json",
        payload=payload,
        config={"task": "P1-01"},
    )
    print("wrote results/p1_01_frame.json")


if __name__ == "__main__":
    main()
