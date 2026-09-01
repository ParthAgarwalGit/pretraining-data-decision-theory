"""Task P0-06: DataDecide data inventory.

Acquires (or reads from cache) DataDecide's eval-results, ppl-results, and
recipes tables and records their real shape into
results/p0_06_inventory.json. See plan/01-phase0-setup.md task P0-06.

Per that task's step 3: the expected counts (25 recipes, 14 sizes, 3
seeds) are checked and reported, but a mismatch does NOT abort this
script -- the point is to report reality to the PI, not to force the
assumption to hold. See plan/00-agent-protocol.md Rule 1/2.
"""

from __future__ import annotations

from pdt import provenance
from pdt.data import datadecide as dd

_EXPECTED = {"n_recipes": 25, "n_sizes": 14, "n_seeds": 3}


def main() -> None:
    eval_results = dd.load_eval_results()
    macro_avg = dd.load_macro_avg()
    scaling_law_fit = dd.load_scaling_law_fit()
    ppl_results = dd.load_ppl_results()
    recipes = dd.load_recipes()

    recipe_names = sorted(eval_results["data"].unique().to_list())
    params_labels = sorted(eval_results["params"].unique().to_list(), key=dd.parse_params)
    params_sizes_numeric = [dd.parse_params(p) for p in params_labels]
    seeds = sorted(eval_results["seed"].unique().to_list())
    tasks = sorted(eval_results["task"].unique().to_list())
    chinchilla_values = sorted(eval_results["chinchilla"].unique().to_list())

    # Per-size seed availability: cheap to compute here, and directly
    # informs GATE-1 / P1-01's seed-handling concerns, without building
    # the full (recipe, params, seed, task) coverage matrix -- that four-way
    # breakdown is P1-01's own, more expensive job.
    seeds_by_size = {
        size: sorted(eval_results.filter(eval_results["params"] == size)["seed"].unique().to_list())
        for size in params_labels
    }

    actual = {
        "n_recipes": len(recipe_names),
        "n_sizes": len(params_labels),
        "n_seeds": len(seeds),
    }
    matches_expected = {key: actual[key] == value for key, value in _EXPECTED.items()}

    if not all(matches_expected.values()):
        mismatches = {
            key: {"expected": value, "actual": actual[key]}
            for key, value in _EXPECTED.items()
            if not matches_expected[key]
        }
        print(f"p0_06_data_inventory: expected-vs-actual mismatch(es): {mismatches}")
        print(
            "Not adjusting the expectation to match -- recording the real counts as "
            "data per plan/01-phase0-setup.md P0-06 step 3."
        )

    payload = {
        "eval_results": {"n_rows": eval_results.height, "n_columns": eval_results.width},
        "macro_avg": {"n_rows": macro_avg.height, "n_columns": macro_avg.width},
        "scaling_law_fit": {"n_rows": scaling_law_fit.height, "n_columns": scaling_law_fit.width},
        "ppl_results": {"n_rows": ppl_results.height, "n_columns": ppl_results.width},
        "recipes": {"n_rows": recipes.height},
        "recipe_names": recipe_names,
        "params_labels": params_labels,
        "params_sizes_numeric": params_sizes_numeric,
        "seeds": seeds,
        "tasks": tasks,
        "n_tasks": len(tasks),
        "chinchilla_values": chinchilla_values,
        "seeds_by_size": seeds_by_size,
        "expected_vs_actual": {
            "expected": _EXPECTED,
            "actual": actual,
            "matches_expected": matches_expected,
        },
        "dataset_revisions": {
            "eval_results": dd.cached_revision("eval_results"),
            "macro_avg": dd.cached_revision("macro_avg"),
            "scaling_law_fit": dd.cached_revision("scaling_law_fit"),
            "ppl_results": dd.cached_revision("ppl_results"),
            "recipes": dd.cached_revision("recipes"),
        },
    }

    provenance.write_result(
        "results/p0_06_inventory.json",
        payload=payload,
        config={"task": "P0-06"},
    )
    print("wrote results/p0_06_inventory.json")


if __name__ == "__main__":
    main()
