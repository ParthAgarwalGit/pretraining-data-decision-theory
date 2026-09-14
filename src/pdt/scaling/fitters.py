"""The six scaling-law extrapolators required by plan/02-phase1-datadecide.md P1-04."""

from __future__ import annotations

import numpy as np

from pdt.scaling.base import Extrapolator, multi_start_fit


def _weights_or_ones(values: list[float], weights: list[float] | None) -> np.ndarray:
    return np.ones(len(values)) if weights is None else np.asarray(weights, dtype=float)


class ConstantExtrapolator(Extrapolator):
    """Predicts the value at the largest fitted scale -- the single-scale
    baseline expressed as an extrapolator. Conceptually important (state
    this in the paper): single-scale ranking is extrapolation under a
    degenerate one-parameter model with large *level* bias (it never
    corrects for scale at all) but potentially small *ordering* bias,
    which is exactly the asymmetry P1-01/P1-03 found DataDecide's puzzle
    turns on.
    """

    n_params = 1

    def _fit_theta(self, scales, values, weights):
        idx = max(range(len(scales)), key=lambda i: scales[i].n)
        theta = np.array([values[idx]])
        diagnostics = {
            "n_restarts": 1,
            "n_converged": 1,
            "best_cost": 0.0,
            "objective_spread": 0.0,
        }
        return theta, diagnostics

    def _predict_from_theta(self, theta, scale):
        return float(theta[0])


class PowerLawN(Extrapolator):
    """y = E + A * N^(-alpha), fit by nonlinear least squares directly on
    the task metric."""

    n_params = 3
    _bounds = (np.array([-2.0, -10.0, 1e-3]), np.array([3.0, 10.0, 10.0]))

    def _fit_theta(self, scales, values, weights):
        ns = np.array([s.n for s in scales], dtype=float)
        ys = np.array(values, dtype=float)
        w = _weights_or_ones(values, weights)

        def residual(theta):
            e, a, alpha = theta
            return w * ((e + a * ns ** (-alpha)) - ys)

        return multi_start_fit(residual, self.n_params, self._bounds, self._rng)

    def _predict_from_theta(self, theta, scale):
        e, a, alpha = theta
        return e + a * scale.n ** (-alpha)


class PowerLawC(Extrapolator):
    """y = E + A * C^(-alpha), C = 6*N*D. Same form as PowerLawN, fit
    against compute instead of parameter count."""

    n_params = 3
    _bounds = (np.array([-2.0, -10.0, 1e-3]), np.array([3.0, 10.0, 10.0]))

    def _fit_theta(self, scales, values, weights):
        cs = np.array([s.compute for s in scales], dtype=float)
        ys = np.array(values, dtype=float)
        w = _weights_or_ones(values, weights)

        def residual(theta):
            e, a, alpha = theta
            return w * ((e + a * cs ** (-alpha)) - ys)

        return multi_start_fit(residual, self.n_params, self._bounds, self._rng)

    def _predict_from_theta(self, theta, scale):
        e, a, alpha = theta
        return e + a * scale.compute ** (-alpha)


class ChinchillaND(Extrapolator):
    """Chinchilla functional form, adapted to an accuracy-like target:
    metric = E - A*N^(-alpha) - B*D^(-beta), with A, B >= 0 so the metric
    rises monotonically toward ceiling E as N and D grow -- the mirror
    image of the original loss form L = E + A*N^-alpha + B*D^-beta, which
    falls toward floor E as loss decreases.

    Fit to the task metric directly, not literal pretraining loss -- this
    project's cached tables carry task accuracy (eval_results/macro_avg),
    not per-recipe validation loss (a different table, ppl_results, not
    used here). See docs/decisions.md for the scoping rationale.
    """

    n_params = 5
    _bounds = (
        np.array([-2.0, 0.0, 1e-3, 0.0, 1e-3]),
        np.array([3.0, 10.0, 10.0, 10.0, 10.0]),
    )

    def _fit_theta(self, scales, values, weights):
        ns = np.array([s.n for s in scales], dtype=float)
        ds = np.array([s.d for s in scales], dtype=float)
        ys = np.array(values, dtype=float)
        w = _weights_or_ones(values, weights)

        def residual(theta):
            e, a, alpha, b, beta = theta
            pred = e - a * ns ** (-alpha) - b * ds ** (-beta)
            return w * (pred - ys)

        return multi_start_fit(residual, self.n_params, self._bounds, self._rng)

    def _predict_from_theta(self, theta, scale):
        e, a, alpha, b, beta = theta
        return e - a * scale.n ** (-alpha) - b * scale.d ** (-beta)


class LogLinear(Extrapolator):
    """y = a + b*log(N). Deliberately misspecified simple model, used in
    the P5-01 misspecification ablation -- not expected to extrapolate
    well, and that is the point of including it."""

    n_params = 2
    _bounds = (np.array([-10.0, -10.0]), np.array([10.0, 10.0]))

    def _fit_theta(self, scales, values, weights):
        log_ns = np.log(np.array([s.n for s in scales], dtype=float))
        ys = np.array(values, dtype=float)
        w = _weights_or_ones(values, weights)

        def residual(theta):
            a, b = theta
            return w * ((a + b * log_ns) - ys)

        return multi_start_fit(residual, self.n_params, self._bounds, self._rng)

    def _predict_from_theta(self, theta, scale):
        a, b = theta
        return a + b * np.log(scale.n)


class TwoStepLadder(Extrapolator):
    """Adapted two-step method following Bhagia et al.'s (arXiv:2412.04403)
    compute-ladder approach, which DataDecide uses as its own scaling-law
    baseline: step 1 fits compute -> an intermediate quantity, step 2 maps
    that quantity to the final task metric through a sigmoid link.

    This is a scoped adaptation, not a literal reproduction: DataDecide's
    step 1 target is actual pretraining validation loss, which lives in a
    *different* cached table (ppl_results) than the task-accuracy data
    (eval_results/macro_avg) this whole module works from, and joining
    those two tables by matching (recipe, scale, seed) is real additional
    work not done here. Instead, step 1 fits a power law in compute
    directly to the task metric as its own intermediate proxy, and step 2
    reshapes that proxy through a 4-parameter sigmoid -- the closest
    same-data-source analogue to "compute -> loss -> metric". See
    docs/decisions.md for the full rationale and what a more faithful
    version would need.

    7 parameters: 3 from step 1 (E1, A1, alpha1), 4 from step 2 (L, U, k,
    x0). Fit sequentially, matching "fit each step separately" in the
    plan's own description -- step 2 regresses the observed metric against
    step 1's in-sample predictions, not a joint 7-parameter optimization.
    """

    n_params = 7
    _step1_bounds = (np.array([-2.0, -10.0, 1e-3]), np.array([3.0, 10.0, 10.0]))
    _step2_bounds = (
        np.array([-2.0, -2.0, -50.0, -10.0]),
        np.array([3.0, 3.0, 50.0, 10.0]),
    )

    def _fit_theta(self, scales, values, weights):
        cs = np.array([s.compute for s in scales], dtype=float)
        ys = np.array(values, dtype=float)
        w = _weights_or_ones(values, weights)

        def step1_residual(theta):
            e1, a1, alpha1 = theta
            return w * ((e1 + a1 * cs ** (-alpha1)) - ys)

        step1_theta, step1_diag = multi_start_fit(step1_residual, 3, self._step1_bounds, self._rng)
        e1, a1, alpha1 = step1_theta
        proxy = e1 + a1 * cs ** (-alpha1)

        def step2_residual(theta):
            lo, hi, k, x0 = theta
            pred = lo + (hi - lo) / (1.0 + np.exp(-k * (proxy - x0)))
            return w * (pred - ys)

        step2_theta, step2_diag = multi_start_fit(step2_residual, 4, self._step2_bounds, self._rng)

        theta = np.concatenate([step1_theta, step2_theta])
        diagnostics = {
            "n_restarts": step1_diag["n_restarts"] + step2_diag["n_restarts"],
            "n_converged": min(step1_diag["n_converged"], step2_diag["n_converged"]),
            "best_cost": step2_diag["best_cost"],
            "objective_spread": step2_diag["objective_spread"],
            "step1": step1_diag,
            "step2": step2_diag,
        }
        return theta, diagnostics

    def _predict_from_theta(self, theta, scale):
        e1, a1, alpha1, lo, hi, k, x0 = theta
        proxy = e1 + a1 * scale.compute ** (-alpha1)
        return lo + (hi - lo) / (1.0 + np.exp(-k * (proxy - x0)))
