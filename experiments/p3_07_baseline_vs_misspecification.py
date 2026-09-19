"""Task P3-07 (figure F8 data): baseline accuracy against misspecification
level, and where baselines become confidently wrong.

See plan/04-phase3-algorithm.md P3-07 (F8 asks for "error rate and
abstention rate against misspecification level eta, ours against
baselines; the crossing point where baselines become confidently
wrong"). Writes results/p3_07_baseline_vs_misspecification.json.

**Scope note:** P3-06's own sweep (results/p3_06_eta_sensitivity.json)
gives ETS's side of this story, with independent trials: at eta=0 and
0.25x true bias one run of 20 certified (wrongly) -- not a statistically
detectable violation of delta at n=20 -- and at eta>=0.5x true bias nothing
certified (round-cap-limited within this session's compute budget, the
same finding as P3-05). An earlier version of that file reported 20/20 wrong
certifications; that came from 20 copies of one noise realization and is
withdrawn (see docs/decisions.md). F8 additionally needs baselines' own accuracy as the *true*
misspecification (bias) magnitude grows, to show the "crossing point"
-- baselines don't take an eta input at all (they have no notion of a
bias budget), so what varies for them is the actual bias in the
generating instance, not an assumed eta. Baselines are cheap
(no adaptive loop), so this sweep -- unlike P3-04/05/06's ETS-side
sweeps -- runs in seconds, not tens of minutes.
"""

from __future__ import annotations

import os

for _blas_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_blas_env_var, "1")

import argparse  # noqa: E402

import numpy as np  # noqa: E402

from pdt import provenance  # noqa: E402
from pdt.bai.ets import (  # noqa: E402
    fixed_ladder_extrapolation,
    single_scale_recommendation,
    successive_halving_over_scales,
    uniform_allocation,
)
from pdt.bai.oracle import _stable_seed  # noqa: E402
from pdt.scaling.base import Scale  # noqa: E402
from pdt.scaling.fitters import LogLinear  # noqa: E402

_FIT_SCALES = [Scale(n=n, d=20 * n) for n in [1e6, 3e6, 1e7]]
_TARGET = Scale(n=3e7, d=20 * 3e7)
_SIGMA = 0.05
_OBSERVED_GAP = 0.05  # matches p3_06_eta_sensitivity.py's own instance exactly
_DEFAULT_BIAS_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]  # x _OBSERVED_GAP
_DEFAULT_N_RUNS = 30


class _TwoArmOracle:
    """Identical construction to p3_06_eta_sensitivity.py's own
    `_TwoArmOracle`, except the bias magnitude (`true_bias`) is a
    constructor argument here rather than a fixed module constant --
    this script sweeps the *true* bias directly, not an assumed eta."""

    def __init__(self, true_bias: float, trial_salt: int, sigma: float = _SIGMA):
        # PR #33's review (the identical defect PR #32 fixed in
        # p3_06_eta_sensitivity.py's own _TwoArmOracle): pull()'s noise
        # seed used to depend only on (recipe, scale, seed), never on
        # which repetition was calling it, and the caller's own
        # `_run_idx` loop variable was unused -- every one of the 30
        # "repetitions" per bias level replayed the exact same dataset
        # and therefore the exact same deterministic decision, so a
        # reported "100% -> 0%" accuracy curve was never actually an
        # error-rate ESTIMATE over noisy trials. `trial_salt` (the
        # caller's own run index) is now required and folded into every
        # pull()'s seed below.
        self._trial_salt = trial_salt
        self._sigma = sigma
        self._params = {
            "leader": {"a": 0.6 + _OBSERVED_GAP, "b": 0.02, "bias_at_target": 0.0},
            "underdog": {"a": 0.6, "b": 0.02, "bias_at_target": true_bias},
        }

    def _h(self, recipe: str, scale: Scale) -> float:
        p = self._params[recipe]
        s_max = max(s.n for s in _FIT_SCALES)
        s_star = _TARGET.n
        if scale.n <= s_max:
            return 0.0
        x = (scale.n - s_max) / (s_star - s_max)
        return p["bias_at_target"] * float(np.clip(x, 0.0, 1.0))

    def _true_mean(self, recipe: str, scale: Scale) -> float:
        p = self._params[recipe]
        return p["a"] + p["b"] * np.log(scale.n) + self._h(recipe, scale)

    def pull(self, recipe: str, scale: Scale, seed: int) -> float:
        mean = self._true_mean(recipe, scale)
        rng = np.random.default_rng(_stable_seed(self._trial_salt, recipe, scale.n, scale.d, seed))
        return float(mean + rng.normal(0.0, self._sigma))

    def cost(self, scale: Scale) -> float:
        return scale.compute

    def available_scales(self) -> list[Scale]:
        return list(_FIT_SCALES)

    def true_value_at_target(self, recipe: str) -> float:
        return self._true_mean(recipe, _TARGET)


_BASELINES = {
    "SingleScale": lambda oracle: single_scale_recommendation(
        oracle, ["leader", "underdog"], _FIT_SCALES[-1], n_replicates=3
    ),
    # model_factory=LogLinear (not the module default PowerLawN): the
    # 3-scale ladder here (matching p3_06_eta_sensitivity.py's own,
    # deliberately small for speed) has too few distinct scales to
    # identify PowerLawN's 3 parameters (needs >= 4).
    "FixedLadderExtrapolation": lambda oracle: fixed_ladder_extrapolation(
        oracle,
        ["leader", "underdog"],
        _FIT_SCALES,
        _TARGET,
        n_replicates=3,
        model_factory=LogLinear,
    ),
    "UniformAllocation": lambda oracle: uniform_allocation(
        oracle,
        ["leader", "underdog"],
        _FIT_SCALES,
        _TARGET,
        compute_budget=2 * 3 * sum(s.compute for s in _FIT_SCALES),
        model_factory=LogLinear,
    ),
    "SuccessiveHalvingOverScales": lambda oracle: successive_halving_over_scales(
        oracle,
        ["leader", "underdog"],
        _FIT_SCALES,
        _TARGET,
        n_replicates=3,
        model_factory=LogLinear,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bias-multipliers", type=float, nargs="+", default=_DEFAULT_BIAS_MULTIPLIERS
    )
    parser.add_argument("--n-runs", type=int, default=_DEFAULT_N_RUNS)
    parser.add_argument("--out", default="results/p3_07_baseline_vs_misspecification.json")
    args = parser.parse_args()

    sweep = []
    for mult in args.bias_multipliers:
        true_bias = mult * _OBSERVED_GAP
        true_winner = "underdog" if true_bias > _OBSERVED_GAP else "leader"
        correct = dict.fromkeys(_BASELINES, 0)
        for run_idx in range(args.n_runs):
            oracle = _TwoArmOracle(true_bias, trial_salt=run_idx)
            for name, fn in _BASELINES.items():
                res = fn(oracle)
                correct[name] += int(res.recipe == true_winner)
        sweep.append(
            {
                "bias_multiplier": mult,
                "true_bias": true_bias,
                "true_winner": true_winner,
                "accuracy": {name: correct[name] / args.n_runs for name in _BASELINES},
            }
        )
        print(
            f"multiplier={mult} true_bias={true_bias:.4f} true_winner={true_winner} "
            f"accuracy={sweep[-1]['accuracy']}"
        )

    # The crossing point: the bias multiplier at which the true winner
    # flips from leader to underdog (bias_multiplier == 1.0, since
    # true_bias == _OBSERVED_GAP exactly there) -- baselines never adapt
    # to this since they carry no notion of bias at all, so their
    # accuracy vs. bias_multiplier crossing *through* 1.0 without any
    # change in behavior is itself the "confidently wrong" finding.
    payload = {
        "observed_gap": _OBSERVED_GAP,
        "crossing_point_bias_multiplier": 1.0,
        "sweep": sweep,
        "note": (
            "Baselines have no eta/bias-budget input at all -- what's swept here is the "
            "TRUE bias magnitude in the generating instance, not an assumed eta (contrast "
            "results/p3_06_eta_sensitivity.json, which sweeps ETS's assumed eta at a FIXED "
            "true bias). true_winner flips from leader to underdog at bias_multiplier=1.0; "
            "a baseline whose accuracy does not correspondingly flip is confidently wrong "
            "past that point, exactly the failure mode ETS's eta mechanism exists to avoid."
        ),
    }

    provenance.write_result(
        args.out,
        payload=payload,
        config={
            "task": "P3-07-F8",
            "bias_multipliers": args.bias_multipliers,
            "n_runs": args.n_runs,
        },
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
