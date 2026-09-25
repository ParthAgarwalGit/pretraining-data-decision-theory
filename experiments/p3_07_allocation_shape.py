"""Task P3-07 (figure F7 data): the shape of the COMPUTED challenger
allocation across scales, under different gap regimes.

**Not a certified-optimal allocation (PR #33's review):** this calls
`solve_allocation`, whose own docstring already documents it as
"validated-reasonable, not certified-optimal" (every restart converges
to a locally max-min-consistent point, not verified globally optimal;
`brute_force_allocation` found a strictly better feasible point on at
least one real instance). It also structurally covers only the
CHALLENGER arms (Theorem 2 Part A's program never perturbs k*'s own
distribution), so k*'s own share never appears in `regimes[...]
["compute_share_by_scale"]` at all. Reported and plotted (F7) as what
`solve_allocation` actually computes for the challengers, not as
evidence of a provably optimal full-instance BAI allocation.

See plan/04-phase3-algorithm.md P3-07. Writes
results/p3_07_allocation_shape.json.

**A real mathematical clarification, not a scope reduction:** the plan
names this "the shape of the optimal allocation... as delta varies."
`solve_allocation` (src/pdt/bai/allocation.py, Theorem 2 Part A's T*(nu)
program) has no dependence on `delta` (the confidence level) at all --
only on the per-challenger gaps `deltas[k]` (`Delta_k`, a different
`delta` than the confidence level, an unfortunate but standard BAI
notational collision) and the cost/noise model. This is not an
oversight: it is the classical Track-and-Stop property (Garivier &
Kaufmann, 2016) that the *optimal allocation shape* `w*(nu)` is
delta-independent -- only the *total* compute scales, via `T*(nu) *
log(1/delta)`. So "as delta varies" is reported here as it actually
holds mathematically: the shape is swept across *gap regimes*
(`Delta_k`, what the T* program genuinely optimizes over), with a note
on the separate, real delta-dependent quantity (total compute, which
scales with `log(1/delta)` at a *fixed* shape).
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
import math  # noqa: E402

import numpy as np  # noqa: E402

from pdt import provenance  # noqa: E402
from pdt.bai.allocation import solve_allocation  # noqa: E402
from pdt.scaling.base import Scale  # noqa: E402
from pdt.scaling.fitters import PowerLawN  # noqa: E402

_FIT_SCALES = [Scale(n=n, d=20 * n) for n in [1e6, 3e6, 1e7, 3e7]]
_TARGET = Scale(n=1e9, d=20e9)
_SIGMA2 = 1e-4


def _model(e: float, a: float, alpha: float) -> PowerLawN:
    m = PowerLawN()
    m._theta = np.array([e, a, alpha])
    return m


# One shared 3-arm instance (kstar, k1, k2), matching
# tests/test_allocation.py's own small_instance fixture in spirit --
# only the *gaps* change across regimes, not the underlying curves.
_MODELS = {
    "kstar": _model(0.8, 2.0, 0.3),
    "k1": _model(0.7, 2.5, 0.25),
    "k2": _model(0.6, 3.0, 0.2),
}

_GAP_REGIMES = {
    "well_separated": {"k1": 0.3, "k2": 0.5},
    "close_top_two": {"k1": 0.03, "k2": 0.5},
    "both_close": {"k1": 0.03, "k2": 0.05},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-restarts", type=int, default=6)
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--out", default="results/p3_07_allocation_shape.json")
    args = parser.parse_args()

    regimes = {}
    for regime_name, deltas in _GAP_REGIMES.items():
        res = solve_allocation(
            _MODELS,
            "kstar",
            _FIT_SCALES,
            _TARGET,
            lambda _s: _SIGMA2,
            deltas,
            n_restarts=args.n_restarts,
            n_iter=args.n_iter,
            rng=np.random.default_rng(0),
        )
        weight_by_scale = {arm: {} for arm in deltas}
        for (arm, i), w in res.weights.items():
            weight_by_scale[arm][_FIT_SCALES[i].n] = w
        compute_share_by_scale = {arm: {} for arm in deltas}
        for arm in deltas:
            total = sum(
                w * s.compute
                for s, w in zip(_FIT_SCALES, weight_by_scale[arm].values(), strict=True)
            )
            for s in _FIT_SCALES:
                w = weight_by_scale[arm][s.n]
                compute_share_by_scale[arm][s.n] = (w * s.compute) / total if total > 0 else 0.0
        regimes[regime_name] = {
            "deltas": deltas,
            "rate": res.rate,
            "t_star": res.t_star,
            "arm_rates": res.arm_rates,
            "weight_by_scale": weight_by_scale,
            "compute_share_by_scale": compute_share_by_scale,
            "message": res.message,
        }
        print(f"{regime_name}: rate={res.rate:.4g} t_star={res.t_star:.4g}")

    # The genuinely delta-dependent quantity, at a *fixed* shape: total
    # required compute scales as T*(nu) * log(1/delta) (Track-and-Stop's
    # classical asymptotic result, cited, not re-derived here).
    example_regime = "well_separated"
    t_star = regimes[example_regime]["t_star"]
    delta_values = [0.2, 0.1, 0.05, 0.01, 0.001]
    compute_vs_delta = [
        {"delta": d, "required_compute": t_star * math.log(1.0 / d)} for d in delta_values
    ]

    payload = {
        "fit_scales": [{"n": s.n, "d": s.d} for s in _FIT_SCALES],
        "target_scale": {"n": _TARGET.n, "d": _TARGET.d},
        "sigma2": _SIGMA2,
        "note": (
            "solve_allocation has no dependence on the confidence level delta -- "
            "only on the per-challenger gaps Delta_k -- per Track-and-Stop's classical "
            "shape/delta-independence property (Garivier & Kaufmann 2016). Regimes here "
            "sweep Delta_k; compute_vs_delta_at_fixed_shape shows the separate, genuinely "
            "delta-dependent quantity (total compute) at one fixed shape. See this "
            "script's own module docstring and docs/decisions.md."
        ),
        "allocation_caveat": (
            "compute_share_by_scale is solve_allocation's own heuristic solution "
            "(validated-reasonable, not certified globally optimal -- see "
            "src/pdt/bai/allocation.py's docstring) and covers only the challenger "
            "arms (k1, k2); k*'s own allocation share is never part of this program "
            "and is not reported here. Not evidence of a provably optimal "
            "full-instance BAI allocation -- see this script's own module docstring."
        ),
        "regimes": regimes,
        "compute_vs_delta_at_fixed_shape": {
            "regime": example_regime,
            "points": compute_vs_delta,
        },
    }

    provenance.write_result(
        args.out,
        payload=payload,
        config={"task": "P3-07-F7", "n_restarts": args.n_restarts, "n_iter": args.n_iter},
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
