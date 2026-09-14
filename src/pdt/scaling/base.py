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
) -> tuple[np.ndarray, dict]:
    """Bounded nonlinear least squares from `n_restarts` random starting
    points, keeping the lowest-cost converged result. Scaling-law fits are
    notoriously multi-modal -- a single-start fit is a bug, per the plan.

    Returns (best theta, diagnostics dict with n_restarts/n_converged/
    best_cost/objective_spread). Raises FitFailure if zero restarts
    converge -- never silently returns a degenerate or unconverged result.
    """
    results = []
    for _ in range(n_restarts):
        x0 = rng.uniform(bounds[0], bounds[1])
        try:
            res = least_squares(residual_fn, x0, bounds=bounds, max_nfev=2000)
        except Exception:  # noqa: BLE001 -- a single bad restart must not abort the others
            continue
        if res.success:
            results.append(res)

    if not results:
        raise FitFailure(f"no restart converged out of {n_restarts} attempts")

    best = min(results, key=lambda r: r.cost)
    costs = [r.cost for r in results]
    diagnostics = {
        "n_restarts": n_restarts,
        "n_converged": len(results),
        "best_cost": float(best.cost),
        "objective_spread": float(max(costs) - min(costs)),
    }
    return best.x, diagnostics
