"""The common Extrapolator interface and shared fitting machinery.

See plan/02-phase1-datadecide.md task P1-04.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Scale:
    """A (parameter count, token count) pair -- s = (N, D) in the plan's
    own notation. `compute` is the standard C ~= 6*N*D approximation used
    throughout this project (see docs/environment.md / plan notation)."""

    n: float
    d: float

    @property
    def compute(self) -> float:
        return 6.0 * self.n * self.d


class FitFailure(RuntimeError):
    """Raised when a fit is refused (too few scales for the parameter
    count) or when no multi-start restart converges. Never silently
    swallowed -- plan/02-phase1-datadecide.md P1-04 is explicit that "the
    fit failed" is itself evidence about identifiability (Claim 3), and
    must be logged, not dropped."""


class Extrapolator:
    """Common interface every scaling-law fitter in this module implements.

    Subclasses implement `n_params`, `_fit_theta()`, and
    `_predict_from_theta()`. `predict()` and `jacobian()` are provided here
    generically: `jacobian()` is central-difference numerical
    differentiation on the fitted parameter vector, deliberately not
    hand-derived per fitter. Six hand-derived analytic gradients is six
    times the chance of a sign or chain-rule error; one well-tested
    numerical implementation, shared by every fitter, is safer -- and
    plan/02-phase1-datadecide.md's own P1-07 task cross-checks this
    machinery against bootstrap variance rather than assuming perfect
    analytic exactness, so numerical precision here is exactly what that
    later verification is designed to catch if it's ever insufficient.
    """

    n_params: int

    #: True only for a family whose prediction is LINEAR in its parameters and whose fit,
    #: when no parameter bound binds, is exactly ordinary least squares (an exactly
    #: Gaussian / sub-Gaussian linear functional of the noise). `extrapolation_track_and_stop`
    #: certifies under its default `certification="supported_only"` only for such families
    #: (PR #29's third review).
    linear_in_parameters: bool = False

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._theta: np.ndarray | None = None
        self.fit_diagnostics: dict | None = None

    def fit(
        self, scales: list[Scale], values: list[float], weights: list[float] | None = None
    ) -> Extrapolator:
        if len(scales) < self.n_params + 1:
            raise FitFailure(
                f"{type(self).__name__}: need at least {self.n_params + 1} scales to "
                f"identify {self.n_params} parameters, got {len(scales)}"
            )
        self._theta, self.fit_diagnostics = self._fit_theta(scales, values, weights)
        return self

    def bounds_inactive(self, rel_margin: float = 1e-6) -> bool:
        """Whether the fitted parameters lie strictly inside the family's box bounds
        (`_bounds`, if it has any), so the constraint is not binding and a linear family's
        constrained fit coincides with unconstrained least squares. A binding bound makes the
        estimator a clipped -- non-Gaussian, non-centred -- functional of the noise."""
        if self._theta is None:
            return False
        bounds = getattr(self, "_bounds", None)
        if bounds is None:
            return True
        lo, hi = bounds
        margin = rel_margin * (np.asarray(hi) - np.asarray(lo))
        return bool(np.all(self._theta > lo + margin) and np.all(self._theta < hi - margin))

    def _fit_theta(
        self, scales: list[Scale], values: list[float], weights: list[float] | None
    ) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def _predict_from_theta(self, theta: np.ndarray, scale: Scale) -> float:
        raise NotImplementedError

    def predict(self, scale: Scale) -> float:
        if self._theta is None:
            raise FitFailure(f"{type(self).__name__}.predict() called before fit()")
        return self._predict_from_theta(self._theta, scale)

    def jacobian(self, scale: Scale, eps: float = 1e-6) -> np.ndarray:
        if self._theta is None:
            raise FitFailure(f"{type(self).__name__}.jacobian() called before fit()")
        grad = np.zeros(self.n_params)
        for i in range(self.n_params):
            theta_plus = self._theta.copy()
            theta_plus[i] += eps
            theta_minus = self._theta.copy()
            theta_minus[i] -= eps
            grad[i] = (
                self._predict_from_theta(theta_plus, scale)
                - self._predict_from_theta(theta_minus, scale)
            ) / (2 * eps)
        return grad


def multi_start_fit(
    residual_fn,
    n_params: int,
    bounds: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    *,
    n_restarts: int = 8,
    log_uniform_dims: tuple[int, ...] = (),
    extra_starts: tuple[np.ndarray, ...] = (),
) -> tuple[np.ndarray, dict]:
    """Bounded nonlinear least squares from `n_restarts` random starting
    points, keeping the lowest-cost converged result. Scaling-law fits are
    notoriously multi-modal -- a single-start fit is a bug, per the plan.

    `log_uniform_dims` names parameter indices (decay-rate exponents like
    `alpha` in `E + A*N^-alpha`) that are drawn log-uniformly over their
    own `[lower, upper]` bounds instead of linear-uniformly -- **a real
    bug found by external review, not a stylistic choice**. Plain
    `rng.uniform` over a wide exponent range like `[1e-3, 10]` spends
    almost all of its mass on `alpha >~ 1`, where `N^-alpha` and its
    derivatives underflow to numerically zero for the parameter counts
    this project fits over (1e6-1e9) -- a flat region with no gradient
    signal, not a real local optimum. `least_squares` can report
    `success=True` there anyway (it stops because the step size, not the
    residual, went to zero), so **every one of the default 8 restarts can
    land in that flat region and agree with each other**, which passed
    this function's own `objective_spread`-based multi-start sanity check
    while still being badly wrong: reproduced directly with
    `PowerLawN(rng=np.random.default_rng(1))` fit to a noiseless
    `y = 0.9 - 2*N^-0.1` curve on `N` from 1e6 to 1.5e8 -- all 8 restarts
    converged to the identical wrong prediction at the target scale
    (0.504 instead of the true 0.648), `objective_spread` on the order of
    1e-18 (see `tests/test_scaling.py`,
    `docs/decisions.md`). Log-uniform sampling concentrates restarts in
    the small-alpha region where the signal actually lives, without
    narrowing the bounds a legitimately large true alpha would need.

    **Randomized starts alone are not enough (second external review of
    the same fix).** Log-uniform sampling only lowers the probability that
    every start lands in a flat region; it does not remove it. Reproduced
    on the compute-based `PowerLawC` with a noiseless in-family curve
    (`y = 0.9 - 2*C^-0.03`, `default_rng(30)` and `default_rng(54)` of 100
    seeds tried): all 8 restarts converged to the same wrong constant-ish
    fit (prediction 0.246 vs true 0.400), `objective_spread ~ 1e-11`,
    indistinguishable from success by any diagnostic here.
    `extra_starts` takes *deterministic informative starting points* --
    for the power-law families, `fitters._power_law_starts` builds them by
    variable projection (for each candidate exponent on a dense log grid,
    the linear parameters are solved exactly by least squares, so every
    start already fits the data as well as its exponent allows) -- and they
    are refined first, in addition to (not instead of) the random
    restarts. A start is informative by construction, not by luck.
    """
    results = []
    starts = [np.asarray(x0, dtype=float) for x0 in extra_starts]
    for _ in range(n_restarts):
        x0 = rng.uniform(bounds[0], bounds[1])
        for dim in log_uniform_dims:
            x0[dim] = np.exp(rng.uniform(np.log(bounds[0][dim]), np.log(bounds[1][dim])))
        starts.append(x0)
    n_starts = len(starts)
    for x0 in starts:
        x0 = np.clip(x0, bounds[0], bounds[1])
        try:
            res = least_squares(residual_fn, x0, bounds=bounds, max_nfev=2000)
        except Exception:  # noqa: BLE001 -- a single bad restart must not abort the others
            continue
        if res.success:
            results.append(res)

    if not results:
        raise FitFailure(f"no restart converged out of {n_starts} attempts")

    best = min(results, key=lambda r: r.cost)
    costs = [r.cost for r in results]
    diagnostics = {
        "n_restarts": n_starts,
        "n_converged": len(results),
        "best_cost": float(best.cost),
        "objective_spread": float(max(costs) - min(costs)),
    }
    return best.x, diagnostics
