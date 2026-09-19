"""Task P3-06: sensitivity to the bias-floor estimate eta.

See plan/04-phase3-algorithm.md P3-06. Writes results/p3_06_eta_sensitivity.json.

Theorem 4's delta-correctness guarantee is *conditional* on `eta_k >=
sqrt(sigma2_extrap_k)` for every arm -- a practitioner supplies `eta` as
an input, and gets no guarantee at all if it happens to be wrong in the
dangerous (under-estimating) direction. This script:

1. Sweeps the *assumed* `eta` from 0 to 3x the true bias magnitude baked
   into a controlled synthetic instance, and reports empirical error
   rate, abstention rate, and mean compute at each multiple.
2. Contrasts under-estimation (multiplier < 1, `eta` too small -- the
   dangerous direction: overconfident, possibly-wrong certifications)
   against over-estimation (multiplier > 1 -- safe but wasteful, more
   abstention/compute).
3. Proposes and tests a practical plug-in: `eta_hat_k`, estimated from
   arm k's own fit residuals on the observed scales (`rmse * sqrt(n /
   (n - p))`, a standard small-sample inflation of the in-sample RMSE,
   analogous to the classic unbiased-variance correction) via one
   inexpensive pre-fit pass, then used as a *fixed* input to
   `extrapolation_track_and_stop` exactly like any hand-chosen eta.

**Scope, stated honestly (see docs/decisions.md), consistent with P3-04/
P3-05's findings:** a real error-rate sweep needs many independent ETS
runs per eta value, and P3-04/P3-05 already established that
`extrapolation_track_and_stop` is too expensive to run at real
(K=25, PowerLawN) scale for many repeated trials. This sweep uses a
small, fast synthetic instance (K=2, `LogLinear`, matching P3-03's own
test-speed convention) with a *known, controlled* true bias magnitude,
rather than real DataDecide data. The reference bias magnitude
(`_TRUE_BIAS = 0.1`) is grounded in P1-06's real measured range (median
sigma2_extrap_hat 0.0023-0.030, i.e. bias magnitude ~0.048-0.17 --
results/p1_06_decomposition.json, docs/decisions.md 2026-09-11 P3-01
entry) -- a representative point in that range, not an arbitrary guess,
but the sweep itself runs on synthetic, not real, data.
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
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

from pdt import provenance  # noqa: E402
from pdt.analysis.intervals import clopper_pearson  # noqa: E402
from pdt.bai.ets import extrapolation_track_and_stop  # noqa: E402
from pdt.bai.oracle import _stable_seed  # noqa: E402
from pdt.scaling.base import Scale  # noqa: E402
from pdt.scaling.fitters import LogLinear  # noqa: E402

_FIT_SCALES = [Scale(n=n, d=20 * n) for n in [1e6, 3e6, 1e7]]
_TARGET = Scale(n=3e7, d=20 * 3e7)
_SIGMA = 0.05
_TRUE_BIAS = 0.1  # see module docstring -- grounded in P1-06's real range
_DELTA = 0.1
_DEFAULT_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
_DEFAULT_N_RUNS = 20
_DEFAULT_MAX_ROUNDS = 300
_DEFAULT_SOLVER_N_ITER = 30
_DEFAULT_MIN_PULLS_PER_PAIR = 8  # see P3-04's decisions.md entry: avoids the
# small-sample HC0 false-certification artifact found there.


class _TwoArmOracle:
    """`leader` is ahead on the *observed* (fit-scale, bias-blind)
    trend by a fixed structural gap `_OBSERVED_GAP`; `underdog` carries
    an invisible Theorem-2-Part-B `phi(x)=clip(x,0,1)` bump of
    `_TRUE_BIAS` -- zero on every fit scale, full strength only at the
    (never-observed) target -- and `_TRUE_BIAS > _OBSERVED_GAP`, so
    `underdog` is the *true* winner at the target despite trailing on
    every scale ever observed. This is deliberate, not an accident: it
    is exactly the configuration needed to demonstrate under-estimated
    eta's danger (a confident, wrong certification of `leader`), which
    an earlier version of this script did not have -- its `leader` had
    no genuine rival at any eta setting (the chosen gap made `leader`
    and `underdog` exactly tied at eta=0, not a case where under-
    estimating eta could cause a wrong-but-confident answer). See
    docs/decisions.md."""

    _OBSERVED_GAP = 0.05  # < _TRUE_BIAS=0.1, so underdog truly wins at target

    def __init__(self, trial_salt: int, sigma: float = _SIGMA):
        # PR #32's review: pull()'s noise seed used to depend only on
        # (recipe, scale, seed), never on which TRIAL was calling it --
        # means and noise levels here are hardcoded (not drawn from any
        # instance rng, unlike SyntheticOracle), so a "fresh" oracle
        # instance across trials drew the EXACT SAME observation noise
        # every time. Reproduced exactly as given: 3 fresh oracles
        # produced eta_hat(leader)=0.04207552584962325 identically --
        # zero sampling variability where there should have been real
        # variability across "independent" trials. `trial_salt` (the
        # caller's own run_idx-derived seed) is required and folded into
        # every pull()'s seed below, fixing this the same way PR #27's
        # review fixed the identical defect in SyntheticOracle.
        self._trial_salt = trial_salt
        self._sigma = sigma
        self._params = {
            "leader": {"a": 0.6 + self._OBSERVED_GAP, "b": 0.02, "bias_at_target": 0.0},
            "underdog": {"a": 0.6, "b": 0.02, "bias_at_target": _TRUE_BIAS},
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


def estimate_eta_plugin(
    oracle: _TwoArmOracle, recipe: str, n_replicates: int, model_factory=LogLinear
) -> float:
    """The practical plug-in: fit `recipe`'s model on `n_replicates`
    pulls per fit scale, then `eta_hat = rmse(residuals) * sqrt(n / (n -
    p))` -- the classic small-sample inflation of an in-sample RMSE
    (the same factor that turns a biased-low maximum-likelihood variance
    estimate into a less-biased one), applied here to a *bias* budget
    rather than a variance, which is a heuristic, not a proven-unbiased
    estimator of sigma2_extrap -- exactly why P3-06 exists to test how
    well it actually preserves the guarantee, not to assume it does.
    """
    scales_data = []
    values_data = []
    for scale in _FIT_SCALES:
        for seed in range(n_replicates):
            scales_data.append(scale)
            values_data.append(oracle.pull(recipe, scale, seed))
    model = model_factory()
    model.fit(scales_data, values_data)
    residuals = np.array(
        [model.predict(s) - v for s, v in zip(scales_data, values_data, strict=True)]
    )
    n = len(residuals)
    p = model.n_params
    rmse = float(np.sqrt(np.mean(residuals**2)))
    correction = float(np.sqrt(n / max(n - p, 1)))
    return rmse * correction


def _run_trials(
    eta_assumed: dict[str, float],
    n_runs: int,
    max_rounds: int,
    solver_n_iter: int,
    min_pulls_per_pair: int,
    seed_prefix: str,
) -> dict:
    outcomes: Counter[str] = Counter()
    n_correct_given_certified = 0
    n_hit_round_cap = 0
    compute_all = []
    compute_certified = []
    for run_idx in range(n_runs):
        # trial_salt is `run_idx` alone -- deliberately NOT including
        # `seed_prefix` (which varies by eta multiplier in the caller) --
        # so every eta value's sweep reuses the SAME underlying
        # observation noise realization for a given run_idx (common
        # random numbers across eta values, per PR #32's review: this
        # sharpens the comparison BETWEEN eta settings by holding the
        # random trial fixed), while still drawing genuinely fresh,
        # independent noise across different run_idx values (the actual
        # bug being fixed: pull() used to have no run_idx-dependence at
        # all, so every "trial" reused identical noise).
        oracle = _TwoArmOracle(trial_salt=run_idx)
        res = extrapolation_track_and_stop(
            oracle,
            ["leader", "underdog"],
            _FIT_SCALES,
            _TARGET,
            delta=_DELTA,
            eta=eta_assumed,
            sigma2=lambda _s: _SIGMA**2,
            model_factory=LogLinear,
            epsilon_0=0.03,
            max_rounds=max_rounds,
            solver_n_restarts=1,
            solver_n_iter=solver_n_iter,
            min_pulls_per_pair=min_pulls_per_pair,
            rng=np.random.default_rng(_stable_seed(seed_prefix, run_idx)),
        )
        outcomes[res.outcome] += 1
        compute_all.append(res.compute_spent)
        if res.outcome == "certified":
            compute_certified.append(res.compute_spent)
            if res.recipe == "underdog":
                n_correct_given_certified += 1
        # "abstained" conflates a genuine Theorem-4 Abstain event with
        # simply running out of max_rounds -- found to matter a great
        # deal in practice for P3-05, and it matters here too: see
        # docs/decisions.md.
        if res.certificate.get("reason", "").startswith("max_rounds"):
            n_hit_round_cap += 1

    n_certified = outcomes["certified"]
    return {
        "eta_assumed": eta_assumed,
        "n_runs": n_runs,
        "outcomes": dict(outcomes),
        "error_rate_given_certified": (
            1.0 - n_correct_given_certified / n_certified if n_certified > 0 else None
        ),
        # The guarantee bounds the JOINT rate P[certified AND wrong]; the conditional
        # rate above can rest on a single certified run. Violations are judged on the
        # joint rate's exact 95% interval (see `main`).
        "n_certified": n_certified,
        "n_wrong_certified": n_certified - n_correct_given_certified,
        "joint_wrong_certified_rate": (n_certified - n_correct_given_certified) / n_runs,
        "joint_wrong_certified_ci95": list(
            clopper_pearson(n_certified - n_correct_given_certified, n_runs)
        ),
        "abstention_rate": outcomes["abstained"] / n_runs,
        "genuine_abstention_rate": (outcomes["abstained"] - n_hit_round_cap) / n_runs,
        "round_cap_exhausted_rate": n_hit_round_cap / n_runs,
        "mean_compute_given_certified": (
            float(np.mean(compute_certified)) if compute_certified else None
        ),
        "mean_compute_overall": float(np.mean(compute_all)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multipliers", type=float, nargs="+", default=_DEFAULT_MULTIPLIERS)
    parser.add_argument("--n-runs", type=int, default=_DEFAULT_N_RUNS)
    parser.add_argument("--max-rounds", type=int, default=_DEFAULT_MAX_ROUNDS)
    parser.add_argument("--solver-n-iter", type=int, default=_DEFAULT_SOLVER_N_ITER)
    parser.add_argument("--min-pulls-per-pair", type=int, default=_DEFAULT_MIN_PULLS_PER_PAIR)
    parser.add_argument("--plugin-n-replicates", type=int, default=8)
    parser.add_argument("--out", default="results/p3_06_eta_sensitivity.json")
    args = parser.parse_args()

    t_start = time.time()
    sweep = []
    for i, mult in enumerate(args.multipliers):
        eta_assumed = {"leader": mult * _TRUE_BIAS, "underdog": mult * _TRUE_BIAS}
        cell = _run_trials(
            eta_assumed,
            args.n_runs,
            args.max_rounds,
            args.solver_n_iter,
            args.min_pulls_per_pair,
            seed_prefix=f"eta-sweep-{mult}",
        )
        cell["multiplier"] = mult
        cell["under_estimated"] = mult < 1.0
        sweep.append(cell)
        elapsed = time.time() - t_start
        print(
            f"[{i + 1}/{len(args.multipliers)}] multiplier={mult} eta={mult * _TRUE_BIAS:.4f} "
            f"-> outcomes={cell['outcomes']} error_given_certified="
            f"{cell['error_rate_given_certified']} genuine_abstain="
            f"{cell['genuine_abstention_rate']} round_cap={cell['round_cap_exhausted_rate']} "
            f"elapsed={elapsed:.0f}s"
        )

    # The practical plug-in: estimate eta_hat once per trial (a fresh
    # oracle instance and a fresh pre-fit each time, so the plug-in's own
    # sampling variability is reflected, not just plugged in once).
    plugin_eta_estimates = {"leader": [], "underdog": []}
    plugin_outcomes: Counter[str] = Counter()
    plugin_n_correct_given_certified = 0
    plugin_n_hit_round_cap = 0
    plugin_compute_certified = []
    plugin_compute_all = []
    for run_idx in range(args.n_runs):
        oracle = _TwoArmOracle(trial_salt=run_idx)
        eta_hat = {
            r: estimate_eta_plugin(oracle, r, args.plugin_n_replicates)
            for r in ["leader", "underdog"]
        }
        for r in ["leader", "underdog"]:
            plugin_eta_estimates[r].append(eta_hat[r])
        res = extrapolation_track_and_stop(
            oracle,
            ["leader", "underdog"],
            _FIT_SCALES,
            _TARGET,
            delta=_DELTA,
            eta=eta_hat,
            sigma2=lambda _s: _SIGMA**2,
            model_factory=LogLinear,
            epsilon_0=0.03,
            max_rounds=args.max_rounds,
            solver_n_restarts=1,
            solver_n_iter=args.solver_n_iter,
            min_pulls_per_pair=args.min_pulls_per_pair,
            rng=np.random.default_rng(_stable_seed("eta-plugin", run_idx)),
        )
        plugin_outcomes[res.outcome] += 1
        plugin_compute_all.append(res.compute_spent)
        if res.outcome == "certified":
            plugin_compute_certified.append(res.compute_spent)
            if res.recipe == "underdog":
                plugin_n_correct_given_certified += 1
        if res.certificate.get("reason", "").startswith("max_rounds"):
            plugin_n_hit_round_cap += 1

    plugin_n_certified = plugin_outcomes["certified"]
    plugin_result = {
        "n_runs": args.n_runs,
        "eta_hat_mean": {r: float(np.mean(v)) for r, v in plugin_eta_estimates.items()},
        "eta_hat_std": {r: float(np.std(v)) for r, v in plugin_eta_estimates.items()},
        "true_bias": {"leader": 0.0, "underdog": _TRUE_BIAS},
        "outcomes": dict(plugin_outcomes),
        "error_rate_given_certified": (
            1.0 - plugin_n_correct_given_certified / plugin_n_certified
            if plugin_n_certified > 0
            else None
        ),
        "abstention_rate": plugin_outcomes["abstained"] / args.n_runs,
        "genuine_abstention_rate": (plugin_outcomes["abstained"] - plugin_n_hit_round_cap)
        / args.n_runs,
        "round_cap_exhausted_rate": plugin_n_hit_round_cap / args.n_runs,
        "mean_compute_given_certified": (
            float(np.mean(plugin_compute_certified)) if plugin_compute_certified else None
        ),
    }
    print(
        f"plug-in eta_hat (mean +- std): {plugin_result['eta_hat_mean']} +- "
        f"{plugin_result['eta_hat_std']}"
    )
    print(
        f"plug-in outcomes: {plugin_result['outcomes']}, "
        f"error_given_certified={plugin_result['error_rate_given_certified']}"
    )

    under_cells = [c for c in sweep if c["under_estimated"]]
    # A violation of P[certified AND wrong] <= delta is detected only when the exact
    # interval's LOWER end exceeds delta (PR #32/#34 reviews: a conditional rate from
    # one certified run is not evidence of a violation).
    under_violations = [c for c in under_cells if c["joint_wrong_certified_ci95"][0] > _DELTA]
    well_specified_cells = [c for c in sweep if c["multiplier"] >= 1.0]
    well_specified_violations = [
        c for c in well_specified_cells if c["joint_wrong_certified_ci95"][0] > _DELTA
    ]

    payload = {
        "true_bias": _TRUE_BIAS,
        "true_bias_source": (
            "representative point in P1-06's real measured range (bias magnitude "
            "~0.048-0.17, results/p1_06_decomposition.json) -- see module docstring"
        ),
        "delta": _DELTA,
        "sweep": sweep,
        "plugin": plugin_result,
        # True only for a STATISTICALLY DETECTABLE violation at this n_runs; False means
        # "not detected", not "the guarantee holds" -- at n_runs = 20 the interval on the
        # joint rate is wide.
        "under_estimation_causes_violations": len(under_violations) > 0,
        "under_estimation_max_joint_wrong_certified_rate": max(
            (c["joint_wrong_certified_rate"] for c in under_cells), default=None
        ),
        "under_estimation_violations": under_violations,
        "well_specified_regime_holds": len(well_specified_violations) == 0,
        "well_specified_regime_violations": well_specified_violations,
    }

    provenance.write_result(
        args.out,
        payload=payload,
        config={
            "task": "P3-06",
            "multipliers": args.multipliers,
            "n_runs": args.n_runs,
            "max_rounds": args.max_rounds,
            "solver_n_iter": args.solver_n_iter,
            "min_pulls_per_pair": args.min_pulls_per_pair,
            "plugin_n_replicates": args.plugin_n_replicates,
        },
    )
    print(f"wrote {args.out}")
    print(
        f"under-estimation causes a DETECTABLE delta violation (exact joint interval): "
        f"{payload['under_estimation_causes_violations']}"
    )
    print(f"well-specified regime (multiplier>=1) holds: {payload['well_specified_regime_holds']}")


if __name__ == "__main__":
    main()
