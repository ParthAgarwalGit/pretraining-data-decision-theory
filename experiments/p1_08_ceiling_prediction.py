"""Task P1-08: does the bound predict the 80% ceiling?

See plan/02-phase1-datadecide.md task P1-08 -- "the paper's money question:
our theory explains DataDecide." Plugs P1-06/P1-07's bound estimates into
a predicted decision accuracy for (a) single-scale (ConstantExtrapolator,
which P1-04's own consistency check already established reproduces P1-03's
single-scale points exactly) and (b) each extrapolation method at matched
compute, and compares against the actual observed accuracy from P1-03/04.
Also runs the sigma2_extrap=0 counterfactual: if removing the bias term
alone would make extrapolation's predicted accuracy beat single-scale's,
that demonstrates the bias term is exactly what causes the observed
failure ("extrapolation does not win").

Writes results/p1_08_ceiling_prediction.json.

"Predicted accuracy" here means `max(0, 1 - bound)` (the plug-in bound
used as a point estimate, clipped to a valid probability -- the raw,
unclipped `1 - bound` is reported too, since P1-07 already found
`bound_pairwise > 1` in every one of its 396 cells: a union bound summed
over ~24 mostly-near-tied comparisons per task routinely exceeds 1, most
directly because P1-02 already found 9 of 11 tasks have no statistically
resolvable winner at all. `max(0, ...)` is the standard, correct way to
read a probability bound that overshoots 1 -- it means "no informative
lower bound on accuracy," not literally negative accuracy. See
docs/decisions.md for the full reasoning, including why comparing this
construct against P1-03/04's different "fraction of all 300 pairs
correct" statistic is still the comparison the plan asks for.
"""

from __future__ import annotations

import json

from pdt import provenance
from pdt.analysis import ground_truth as gt
from pdt.data import frame as frame_mod
from pdt.theory import bound

_TARGET = "1B"
_METRIC = "primary_metric"
_SCHEME = "seed_bootstrap"
_DESIGNS = ("S_fit_le_150M", "S_fit_le_300M", "S_fit_le_530M")
_FITTERS = (
    "ConstantExtrapolator",
    "PowerLawN",
    "PowerLawC",
    "ChinchillaND",
    "TwoStepLadder",
    "LogLinear",
)
_EXTRAPOLATION_FITTERS = tuple(f for f in _FITTERS if f != "ConstantExtrapolator")

_P1_03_PATH = "results/p1_03_single_scale.json"
_P1_04_PATH = "results/p1_04_extrapolation.json"
_P1_06_PATH = "results/p1_06_decomposition.json"
_P1_07_PATH = "results/p1_07_bound_coverage.json"


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["data"]


def _observed_accuracy(fitter: str, design: str, p1_03: dict, p1_04: dict) -> float:
    """The real, already-computed macro-averaged accuracy for this
    (fitter, design), sourced from P1-03 for ConstantExtrapolator (whose
    per-design accuracy P1-04's own consistency check proved exactly
    equals the matching P1-03 single-scale point) and from P1-04 for
    every other fitter."""
    if fitter == "ConstantExtrapolator":
        endpoint = {"S_fit_le_150M": "150M", "S_fit_le_300M": "300M", "S_fit_le_530M": "530M"}[
            design
        ]
        return p1_03["variants"]["primary_metric__average"][endpoint][
            "macro_avg_accuracy_including_ties"
        ]
    return p1_04["fitters"][fitter][design]["macro_avg_accuracy_including_ties"]


def _clip_probability(x: float) -> float:
    """A probability bound can mathematically overshoot 1 (a union bound
    over many comparisons routinely does when many are near-tied) -- the
    correct reading of `1 - bound` when `bound > 1` is "no informative
    lower bound on accuracy" (i.e. 0), not a negative number."""
    return max(0.0, x)


def _predicted_accuracy_per_task(
    fitter: str, design: str, tasks: list[str], p1_07: dict
) -> dict[str, dict[str, float] | None]:
    """{"raw": 1 - bound_pairwise, "clipped": max(0, raw)} per task,
    straight from P1-07's already-computed bound (no recomputation needed
    for the non-counterfactual case)."""
    result = {}
    for task in tasks:
        try:
            entry = p1_07["by_fitter"][fitter][design][task][_SCHEME]
        except KeyError:
            result[task] = None
            continue
        raw = 1.0 - entry["bound_pairwise"]
        result[task] = {"raw": raw, "clipped": _clip_probability(raw)}
    return result


def _macro_average_field(values: dict[str, dict | None], field: str) -> float | None:
    present = [v[field] for v in values.values() if v is not None]
    return sum(present) / len(present) if present else None


def _counterfactual_predicted_accuracy_per_task(
    fitter: str,
    design: str,
    tasks: list[str],
    ground_truth: dict,
    p1_06: dict,
) -> dict[str, dict[str, float] | None]:
    """Recompute the pairwise-form bound per task with bias(D_k) forced to
    0 for every recipe (the pairwise-form counterfactual analogue of the
    plan's "set sigma2_extrap = 0" -- bias(D_k) is the pairwise
    decomposition's own bias term, playing the same role sigma2_extrap
    plays in the marginal form). Uses P1-02's real Delta_k gaps and
    P1-06's real v_hat(D_k) -- only the bias term is zeroed, nothing else
    about the data changes. Returns {"raw": ..., "clipped": ...} per task,
    same as `_predicted_accuracy_per_task`.
    """
    result = {}
    for task in tasks:
        gt_task = ground_truth[task]
        try:
            cell = p1_06["by_fitter"][fitter][design][task][_SCHEME]
        except KeyError:
            result[task] = None
            continue
        terms = []
        for recipe, gap in gt_task["gaps"].items():
            p = cell["pairwise"].get(recipe, {})
            if p.get("insufficient_replicates", True) is not False:
                continue
            terms.append((gap, 0.0, p["v_hat"]))  # bias forced to 0.0
        if not terms:
            result[task] = None
            continue
        raw = 1.0 - bound.pairwise_bound(terms)
        result[task] = {"raw": raw, "clipped": _clip_probability(raw)}
    return result


def main() -> None:
    long_frame = frame_mod.build_frame(source="macro_avg", metrics=(_METRIC,))
    ground_truth = gt.compute_ground_truth(long_frame, _METRIC, _TARGET)
    tasks = sorted(ground_truth.keys())

    p1_03 = _load(_P1_03_PATH)
    p1_04 = _load(_P1_04_PATH)
    p1_06 = _load(_P1_06_PATH)
    p1_07 = _load(_P1_07_PATH)

    print(
        f"p1_08_ceiling_prediction: {len(tasks)} tasks, {len(_DESIGNS)} designs, "
        f"{len(_FITTERS)} fitters"
    )

    by_fitter: dict[str, dict] = {}
    for fitter in _FITTERS:
        for design in _DESIGNS:
            observed = _observed_accuracy(fitter, design, p1_03, p1_04)
            predicted_per_task = _predicted_accuracy_per_task(fitter, design, tasks, p1_07)
            predicted_raw = _macro_average_field(predicted_per_task, "raw")
            predicted_clipped = _macro_average_field(predicted_per_task, "clipped")
            counterfactual_per_task = _counterfactual_predicted_accuracy_per_task(
                fitter, design, tasks, ground_truth, p1_06
            )
            counterfactual_raw = _macro_average_field(counterfactual_per_task, "raw")
            counterfactual_clipped = _macro_average_field(counterfactual_per_task, "clipped")

            by_fitter.setdefault(fitter, {})[design] = {
                "observed_accuracy": observed,
                "predicted_accuracy_raw": predicted_raw,
                "predicted_accuracy": predicted_clipped,
                "gap_predicted_minus_observed": (
                    predicted_clipped - observed if predicted_clipped is not None else None
                ),
                "counterfactual_sigma2_extrap_zero_predicted_accuracy_raw": counterfactual_raw,
                "counterfactual_sigma2_extrap_zero_predicted_accuracy": counterfactual_clipped,
                "predicted_accuracy_per_task": predicted_per_task,
            }
            pred_str = f"{predicted_clipped:.1%}" if predicted_clipped is not None else "n/a"
            cf_str = (
                f"{counterfactual_clipped:.1%}" if counterfactual_clipped is not None else "n/a"
            )
            raw_str = f"{predicted_raw:.1%}" if predicted_raw is not None else "n/a"
            print(
                f"p1_08_ceiling_prediction: {fitter}/{design}: observed={observed:.1%} "
                f"predicted={pred_str} (raw={raw_str}) counterfactual={cf_str}"
            )

    # The central falsifiable claim (plan step 3): single-scale vs each
    # extrapolation method at matched compute (same design -> same
    # compute, since every fitter shares the same 3 designs). Uses the
    # CLIPPED predicted accuracy for "beats" comparisons -- raw values
    # aren't valid probabilities to compare once bound > 1.
    central_claims = []
    for design in _DESIGNS:
        single_scale_observed = by_fitter["ConstantExtrapolator"][design]["observed_accuracy"]
        single_scale_predicted = by_fitter["ConstantExtrapolator"][design]["predicted_accuracy"]
        single_scale_counterfactual = by_fitter["ConstantExtrapolator"][design][
            "counterfactual_sigma2_extrap_zero_predicted_accuracy"
        ]
        for fitter in _EXTRAPOLATION_FITTERS:
            entry = by_fitter[fitter][design]
            extrap_counterfactual = entry["counterfactual_sigma2_extrap_zero_predicted_accuracy"]
            # (A) extrapolation's bias-free version vs single-scale's REAL
            # (bias-included) predicted accuracy -- the plan's literal
            # wording ("exceeds single-scale"). Since single-scale's own
            # predicted accuracy is 0.0% everywhere too (see Decision in
            # docs/decisions.md), this comparison is close to trivial: any
            # positive counterfactual "wins".
            counterfactual_beats_single_scale_real = (
                extrap_counterfactual is not None
                and single_scale_predicted is not None
                and extrap_counterfactual > single_scale_predicted
            )
            # (B) extrapolation's bias-free version vs single-scale's OWN
            # bias-free version -- controls for single-scale ALSO carrying
            # removable bias (P1-04's own docstring: single-scale has
            # "large level-bias"), isolating whether the METHODS differ on
            # variance alone once bias is out of the picture for both.
            counterfactual_beats_single_scale_counterfactual = (
                extrap_counterfactual is not None
                and single_scale_counterfactual is not None
                and extrap_counterfactual > single_scale_counterfactual
            )
            actual_beats_single_scale = (
                entry["predicted_accuracy"] is not None
                and single_scale_predicted is not None
                and entry["predicted_accuracy"] > single_scale_predicted
            )
            central_claims.append(
                {
                    "design": design,
                    "extrapolation_fitter": fitter,
                    "single_scale_predicted": single_scale_predicted,
                    "single_scale_observed": single_scale_observed,
                    "single_scale_counterfactual": single_scale_counterfactual,
                    "extrapolation_predicted": entry["predicted_accuracy"],
                    "extrapolation_observed": entry["observed_accuracy"],
                    "extrapolation_beats_single_scale_predicted": actual_beats_single_scale,
                    "extrapolation_beats_single_scale_observed": entry["observed_accuracy"]
                    > single_scale_observed,
                    "counterfactual_extrapolation_predicted": extrap_counterfactual,
                    "counterfactual_beats_single_scale_real": (
                        counterfactual_beats_single_scale_real
                    ),
                    "counterfactual_beats_single_scale_counterfactual": (
                        counterfactual_beats_single_scale_counterfactual
                    ),
                }
            )

    n_counterfactual_flips_real = sum(
        1
        for c in central_claims
        if c["counterfactual_beats_single_scale_real"]
        and not c["extrapolation_beats_single_scale_predicted"]
    )
    n_counterfactual_flips_apples_to_apples = sum(
        1
        for c in central_claims
        if c["counterfactual_beats_single_scale_counterfactual"]
        and not c["extrapolation_beats_single_scale_predicted"]
    )
    print(
        f"p1_08_ceiling_prediction: {n_counterfactual_flips_real}/{len(central_claims)} pairs flip "
        f"vs single-scale's real predicted accuracy; "
        f"{n_counterfactual_flips_apples_to_apples}/{len(central_claims)} flip vs single-scale's "
        "OWN bias-free counterfactual (apples-to-apples) under the sigma2_extrap=0 counterfactual"
    )

    payload = {
        "target_scale": _TARGET,
        "metric": _METRIC,
        "scheme": _SCHEME,
        "designs": list(_DESIGNS),
        "fitters": list(_FITTERS),
        "by_fitter": by_fitter,
        "central_claims": central_claims,
        "summary": {
            "n_central_claims": len(central_claims),
            "n_counterfactual_flips_vs_single_scale_real": n_counterfactual_flips_real,
            "n_counterfactual_flips_vs_single_scale_counterfactual": (
                n_counterfactual_flips_apples_to_apples
            ),
        },
    }

    provenance.write_result(
        "results/p1_08_ceiling_prediction.json",
        payload=payload,
        config={"task": "P1-08"},
    )
    print("wrote results/p1_08_ceiling_prediction.json")


if __name__ == "__main__":
    main()
