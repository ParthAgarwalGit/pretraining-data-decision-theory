"""Task P1-10: secondary ladder replication on Pythia.

See plan/02-phase1-datadecide.md task P1-10. Reruns P1-06's bias/variance
decomposition on Pythia (Biderman et al. 2023) -- a genuine K=2 data-recipe
instance (deduplicated vs. standard Pile) this project has not tuned
anything on -- and states whether the sigma2_extrap/v ratio-vs-compute
finding replicates. Writes results/p1_10_secondary_ladder.json.

**Scope, capped per the plan's own "if this task balloons... cap it"
instruction, for reasons discovered while building this, not chosen in
advance:**

- **Tasks**: only 4 of DataDecide's 11 headline tasks have a same-named
  counterpart in Pythia's own published eval set (`arc_challenge`,
  `arc_easy`, `piqa`, `winogrande`) -- confirmed directly against the raw
  JSON's task keys, not assumed. A 5th, `mmlu`, is reconstructed here as
  the unweighted mean of Pythia's 57 `hendrycksTest-*` subtask scores,
  the same convention DataDecide's own `macro_avg` table uses for `mmlu`
  (verified in P1-05/P1-06's own work). `boolq`/`csqa`/`hellaswag`/
  `openbookqa`/`socialiqa` are not in Pythia's public eval set at all, so
  `olmes_10_macro_avg` (which needs all of them) cannot be reconstructed.
- **Fitters**: Pythia has only 3 published sizes below the target
  (`70m`/`160m`/`410m` -- `1b` itself turned out to have no plain,
  non-deduped directory published, a real naming irregularity; `1.4b` is
  used as the target instead, see `src/pdt/data/pythia.py`). 4 of P1-04's
  6 fitters need more fitted scales than that to identify their
  parameters at all (`PowerLawN`/`PowerLawC` need >=4, `ChinchillaND`
  needs >=6, `TwoStepLadder` needs >=8) -- this is a hard data
  constraint, not a choice. Only `ConstantExtrapolator` (needs >=2, fits
  at both designs below) and `LogLinear` (needs exactly 3, fits only at
  the larger design) can run at all.
- **Bootstrap scheme**: Pythia's public evals publish exactly one seed per
  (size, variant, step) -- no seed-resampling is possible. Uses parametric
  bootstrap only, with the noise variance estimated the same way P1-05
  estimated checkpoint jitter (variance across the last 4 of each run's
  27 published checkpoints -- the only noise source available here, since
  there is no second seed to compare against).
"""

from __future__ import annotations

import hashlib
import statistics

import numpy as np
import polars as pl

from pdt import provenance
from pdt.analysis import bootstrap as bs
from pdt.data import pythia
from pdt.scaling import fitters
from pdt.scaling.base import FitFailure, Scale

_DIRECT_TASKS = ("arc_challenge", "arc_easy", "piqa", "winogrande")
_MMLU_SUBTASK_PREFIX = "hendrycksTest-"
_METRIC = "acc"
_TARGET = "1.4b"
_PROXY_SIZES = ("70m", "160m", "410m")
_N_LAST_CHECKPOINTS = 4  # matches P1-05's choice, and is Pythia's own minimum-per-run headroom
_B_REPLICATES = 200
_MIN_SUCCESSFUL_REPLICATES = 20

_DESIGNS = {
    "le_160m": {"sizes": ("70m", "160m"), "fitters": ("ConstantExtrapolator",)},
    "le_410m": {"sizes": ("70m", "160m", "410m"), "fitters": ("ConstantExtrapolator", "LogLinear")},
}
_FITTER_CLASSES = {
    "ConstantExtrapolator": fitters.ConstantExtrapolator,
    "LogLinear": fitters.LogLinear,
}

_RECIPES = ("standard", "deduped")


def _seed_for(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _load_frame_with_mmlu() -> pl.DataFrame:
    """The Pythia frame for the 4 direct tasks plus a derived `mmlu` row
    (mean of the 57 `hendrycksTest-*` subtasks, per-recipe/size/step)."""
    all_sizes = (*_PROXY_SIZES, _TARGET)
    direct = pythia.build_frame(
        tasks=frozenset(_DIRECT_TASKS), metric_name=_METRIC, sizes=all_sizes
    )

    mmlu_tasks = frozenset(
        f"{_MMLU_SUBTASK_PREFIX}{subject}"
        for subject in [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ]
    )
    mmlu_raw = pythia.build_frame(tasks=mmlu_tasks, metric_name=_METRIC, sizes=all_sizes)
    print(f"p1_10_secondary_ladder: mmlu reconstructed from {mmlu_raw['task'].n_unique()} subtasks")
    mmlu_derived = (
        mmlu_raw.group_by(["recipe", "params_str", "step", "is_final"])
        .agg(pl.col("metric_value").mean().alias("metric_value"))
        .with_columns(task=pl.lit("mmlu"), metric_name=pl.lit(_METRIC))
        .select(direct.columns)
    )

    return pl.concat([direct, mmlu_derived])


def _checkpoint_jitter(long_frame: pl.DataFrame) -> dict[tuple[str, str, str], float]:
    """(recipe, params_str, task) -> variance across the last
    `_N_LAST_CHECKPOINTS` published steps -- the only noise source
    available (no second seed to resample from). Mirrors
    `noise.checkpoint_jitter()`'s method but Pythia's frame has no `seed`
    column to group by."""
    sorted_frame = long_frame.sort(["recipe", "params_str", "task", "step"])
    ranked = sorted_frame.with_columns(
        pl.col("step")
        .rank(method="ordinal", descending=True)
        .over(["recipe", "params_str", "task"])
        .alias("_rank")
    )
    tail = ranked.filter(pl.col("_rank") <= _N_LAST_CHECKPOINTS)
    stats = tail.group_by(["recipe", "params_str", "task"], maintain_order=True).agg(
        pl.col("metric_value").var(ddof=1).alias("sigma2_ckpt"),
        pl.col("metric_value").len().alias("n_used"),
    )
    return {
        (row["recipe"], row["params_str"], row["task"]): row["sigma2_ckpt"]
        for row in stats.filter(pl.col("n_used") >= 2).iter_rows(named=True)
    }


def _final_values(long_frame: pl.DataFrame) -> dict[tuple[str, str, str], float]:
    finals = long_frame.filter(pl.col("is_final"))
    return {
        (row["recipe"], row["params_str"], row["task"]): row["metric_value"]
        for row in finals.iter_rows(named=True)
    }


def main() -> None:
    long_frame = _load_frame_with_mmlu()
    tasks = (*_DIRECT_TASKS, "mmlu")
    print(
        f"p1_10_secondary_ladder: {len(tasks)} tasks, {len(_DESIGNS)} designs, "
        f"{len(_RECIPES)} recipes"
    )

    jitter = _checkpoint_jitter(long_frame)
    final_values = _final_values(long_frame)

    by_design: dict[str, dict] = {}
    for design_name, design in _DESIGNS.items():
        sizes = design["sizes"]
        by_task: dict[str, dict] = {}
        for task in tasks:
            true_values = {r: final_values.get((r, _TARGET, task)) for r in _RECIPES}
            if any(v is None for v in true_values.values()):
                continue
            k_star = max(true_values, key=true_values.get)
            k_other = "deduped" if k_star == "standard" else "standard"
            true_gap = true_values[k_star] - true_values[k_other]

            by_fitter: dict[str, dict] = {}
            for fitter_name in design["fitters"]:
                fitter_cls = _FITTER_CLASSES[fitter_name]
                replicate_predictions: dict[str, list[float]] = {r: [] for r in _RECIPES}

                for b in range(_B_REPLICATES):
                    replicate_seed = _seed_for(design_name, task, str(b))
                    replicate_rng = np.random.default_rng(replicate_seed)
                    shared_zs = [bs.draw_parametric_noise_z(replicate_rng) for _ in sizes]

                    for recipe in _RECIPES:
                        scales = []
                        values = []
                        for size, z in zip(sizes, shared_zs, strict=True):
                            mu = final_values.get((recipe, size, task))
                            sigma2 = jitter.get((recipe, size, task), 0.0)
                            if mu is None:
                                continue
                            n = {"70m": 70e6, "160m": 160e6, "410m": 410e6}[size]
                            # d is an unused placeholder: neither
                            # ConstantExtrapolator nor LogLinear reads
                            # scale.d -- see docs/decisions.md.
                            scales.append(Scale(n=n, d=20 * n))
                            values.append(bs.apply_parametric_noise(z, mu, sigma2))
                        if len(scales) < fitter_cls.n_params + 1:
                            continue
                        fit_seed = _seed_for(fitter_name, design_name, task, recipe, str(b))
                        try:
                            model = fitter_cls(rng=np.random.default_rng(fit_seed))
                            model.fit(scales, values)
                            target_n = 1.4e9
                            pred = model.predict(Scale(n=target_n, d=20 * target_n))
                        except FitFailure:
                            continue
                        replicate_predictions[recipe].append(pred)

                marginal = {}
                for recipe in _RECIPES:
                    preds = replicate_predictions[recipe]
                    if len(preds) < _MIN_SUCCESSFUL_REPLICATES:
                        marginal[recipe] = {"insufficient_replicates": True}
                        continue
                    marginal[recipe] = bs.bias_variance_decomposition(
                        preds, true_values[recipe], 0.0
                    )
                    marginal[recipe]["insufficient_replicates"] = False

                n_common = min(
                    len(replicate_predictions[k_star]), len(replicate_predictions[k_other])
                )
                if n_common >= _MIN_SUCCESSFUL_REPLICATES:
                    d_k = [
                        replicate_predictions[k_star][i] - replicate_predictions[k_other][i]
                        for i in range(n_common)
                    ]
                    pairwise = bs.bias_variance_decomposition(d_k, true_gap, 0.0)
                    pairwise["insufficient_replicates"] = False
                else:
                    pairwise = {"insufficient_replicates": True}

                by_fitter[fitter_name] = {"marginal": marginal, "pairwise": pairwise}
                k_star_ok = (
                    not marginal[k_star].get("insufficient_replicates")
                    and marginal[k_star]["v_hat"] > 0
                )
                if k_star_ok:
                    ratio = marginal[k_star]["sigma2_extrap_hat"] / marginal[k_star]["v_hat"]
                    print(
                        f"p1_10_secondary_ladder: {fitter_name}/{design_name}/{task}: "
                        f"sigma2_extrap={marginal[k_star]['sigma2_extrap_hat']:.4f} "
                        f"v={marginal[k_star]['v_hat']:.4f} ratio={ratio:.2f}"
                    )

            by_task[task] = {
                "k_star": k_star,
                "true_gap": true_gap,
                "by_fitter": by_fitter,
            }
        by_design[design_name] = {"sizes": list(sizes), "by_task": by_task}

    # Does the ratio-vs-compute finding replicate? Only ConstantExtrapolator
    # has data at both designs.
    ratios_by_design: dict[str, list[float]] = {"le_160m": [], "le_410m": []}
    for design_name in ("le_160m", "le_410m"):
        for task_entry in by_design[design_name]["by_task"].values():
            entry = task_entry["by_fitter"].get("ConstantExtrapolator")
            if not entry:
                continue
            for recipe_stats in entry["marginal"].values():
                if recipe_stats.get("insufficient_replicates") or recipe_stats["v_hat"] <= 0:
                    continue
                ratios_by_design[design_name].append(
                    recipe_stats["sigma2_extrap_hat"] / recipe_stats["v_hat"]
                )

    median_ratio_le_160m = (
        statistics.median(ratios_by_design["le_160m"]) if ratios_by_design["le_160m"] else None
    )
    median_ratio_le_410m = (
        statistics.median(ratios_by_design["le_410m"]) if ratios_by_design["le_410m"] else None
    )
    replicates_p1_06_direction = (
        median_ratio_le_160m is not None
        and median_ratio_le_410m is not None
        and median_ratio_le_410m < median_ratio_le_160m
    )
    print(
        f"p1_10_secondary_ladder: ConstantExtrapolator median ratio le_160m="
        f"{median_ratio_le_160m} le_410m={median_ratio_le_410m} "
        f"replicates_p1_06_decreasing_direction={replicates_p1_06_direction}"
    )

    payload = {
        "target_scale": _TARGET,
        "metric": _METRIC,
        "proxy_sizes": list(_PROXY_SIZES),
        "tasks": list(tasks),
        "recipes": list(_RECIPES),
        "b_replicates": _B_REPLICATES,
        "by_design": by_design,
        "ratio_vs_compute_replication": {
            "median_ratio_le_160m": median_ratio_le_160m,
            "median_ratio_le_410m": median_ratio_le_410m,
            "replicates_p1_06_decreasing_direction": replicates_p1_06_direction,
        },
    }

    provenance.write_result(
        "results/p1_10_secondary_ladder.json",
        payload=payload,
        config={"task": "P1-10"},
    )
    print("wrote results/p1_10_secondary_ladder.json")


if __name__ == "__main__":
    main()
