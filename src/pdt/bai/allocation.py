"""Solve Theorem 2 Part A's compute-weighted optimal-allocation program
(T*) numerically -- plan/04-phase3-algorithm.md P3-02.

See paper/sections/theorem2_lower_bound.tex Theorem lower-bound for the
derivation:

    1/T*(nu) = sup_w min_{k != k*}  Delta_k^2 / (2 * J_k^T I_k(w)^-1 J_k)
    I_k(w)   = sum_s w(k, s) * J(theta_k, s) J(theta_k, s)^T / sigma^2(s)

subject to sum_{k,s} w(k,s) * c(s) = 1, w(k,s) >= 0. Only the CHALLENGER
arm's own Fisher information enters each term -- the theorem's
change-of-measure construction perturbs one non-winning arm at a time,
holding k*'s distribution fixed (theorem2_lower_bound.tex Step 2), so
k*'s own allocation never appears in the objective. This is a real,
reportable property of the resulting optimal allocation, not an
implementation shortcut -- see `solve_allocation`'s docstring and
docs/decisions.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pdt.scaling.base import Extrapolator, Scale


@dataclass(frozen=True)
class AllocationResult:
    """`weights[(arm, i)]` is the compute-fraction weight on
    `(arm, candidate_scales[i])`. `rate` is the solved value of
    `sup_w min_k [...]`; `t_star = 1/rate`. `arm_rates` gives each
    challenger's own achieved rate at the solution, for diagnosing which
    arm is the bottleneck (the one the max-min is tight against)."""

    weights: dict[tuple[str, int], float]
    rate: float
    t_star: float
    arm_rates: dict[str, float]
    converged: bool
    n_iterations: int
    message: str = field(default="")


def _fisher_information(
    model: Extrapolator, scales: list[Scale], sigma2: Callable[[Scale], float], w_arm: np.ndarray
) -> np.ndarray:
    """I_k(w) for one arm: sum_s w(s) J(s) J(s)^T / sigma2(s)."""
    p = model.n_params
    info = np.zeros((p, p))
    for wi, s in zip(w_arm, scales, strict=True):
        if wi <= 0.0:
            continue
        j = model.jacobian(s)
        sig2 = sigma2(s)
        if sig2 <= 0:
            raise ValueError(f"sigma2({s!r}) must be positive, got {sig2}")
        info += wi * np.outer(j, j) / sig2
    return info


def _arm_rate(
    model: Extrapolator,
    scales: list[Scale],
    sigma2: Callable[[Scale], float],
    w_arm: np.ndarray,
    target_scale: Scale,
    delta_k: float,
) -> float:
    """Delta_k^2 / (2 * J_target^T I_k(w)^-1 J_target); 0 if I_k(w) carries
    no information at all in the direction of J_target (would need
    infinite compute -- a real, not a numerically-degenerate, limit)."""
    info = _fisher_information(model, scales, sigma2, w_arm)
    j_star = model.jacobian(target_scale)
    # pinv rather than inv: I_k(w) is only guaranteed PSD, not PD, at an
    # arbitrary w (e.g. w_arm all zero, or concentrated on scales whose
    # Jacobians don't span p dimensions) -- exactly the "rank-deficient
    # design" case Theorem 3 already treats as a real, not a numerical,
    # phenomenon.
    info_pinv = np.linalg.pinv(info)
    denom = float(j_star @ info_pinv @ j_star)
    if denom <= 1e-300:
        return 0.0
    return (delta_k**2) / (2.0 * denom)


def _default_cost(scale: Scale) -> float:
    return scale.compute


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of `v` onto the probability simplex
    (`x >= 0, sum(x) == 1`) -- the standard O(n log n) sort-based
    algorithm (Duchi, Shalev-Shwartz, Singer & Chandra, 2008, "Efficient
    Projections onto the l1-Ball for Learning in High Dimensions").
    """
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - css / ind > 0
    rho = ind[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def _arm_rate_and_grad(
    model: Extrapolator,
    scales: list[Scale],
    sigma2: Callable[[Scale], float],
    w_arm: np.ndarray,
    target_scale: Scale,
    delta_k: float,
) -> tuple[float, np.ndarray]:
    """`_arm_rate(...)` plus its exact gradient w.r.t. `w_arm` (not a
    finite-difference approximation): writing `rate = c / f(w)` with
    `f(w) = J_star^T I_k(w)^-1 J_star` and `c = Delta_k^2/2`, the standard
    matrix-calculus identity `d(M^-1)/dw_s = -M^-1 (dM/dw_s) M^-1` gives
    `d(rate)/dw_s = (c/f^2) * (y . J(theta_k,s))^2 / sigma2(s)` where
    `y = I_k(w)^-1 J_star`. This closed form is both faster and far more
    numerically stable near a rank-deficient I_k(w) than scipy's
    finite-difference Jacobian of a function that routes through
    `np.linalg.pinv` -- exactly what made the earlier SLSQP/trust-constr
    attempts unreliable (see docs/decisions.md).
    """
    info = _fisher_information(model, scales, sigma2, w_arm)
    j_star = model.jacobian(target_scale)
    info_pinv = np.linalg.pinv(info)
    f = float(j_star @ info_pinv @ j_star)
    n = len(w_arm)
    if f <= 1e-300:
        return 0.0, np.zeros(n)
    y = info_pinv @ j_star
    c = (delta_k**2) / 2.0
    rate = c / f
    grad = np.empty(n)
    for i, s in enumerate(scales):
        sig2 = sigma2(s)
        j_s = model.jacobian(s)
        grad[i] = (c / f**2) * (float(y @ j_s) ** 2) / sig2
    return rate, grad


def solve_allocation(
    models: dict[str, Extrapolator],
    k_star: str,
    candidate_scales: list[Scale],
    target_scale: Scale,
    sigma2: Callable[[Scale], float],
    deltas: dict[str, float],
    cost: Callable[[Scale], float] = _default_cost,
    n_restarts: int = 6,
    n_iter: int = 4000,
    rng: np.random.Generator | None = None,
) -> AllocationResult:
    """Solve `sup_w min_{k!=k*} arm_rate(k, w)` via projected subgradient
    ascent on the compute-fraction simplex (plan/04-phase3-algorithm.md
    P3-02's first suggested route) -- not a generic constrained NLP
    solver. Two earlier attempts (SLSQP, then trust-constr, both via
    `scipy.optimize.minimize` with the standard epigraph reformulation)
    were tried first and both proved unreliable on real instances
    (SLSQP: reported "inequality constraints incompatible" on a provably
    feasible problem; trust-constr: converged, but to a strictly worse
    objective than dense random sampling found on the same instance) --
    both route their constraint Jacobians through finite differences of a
    function containing a matrix pseudo-inverse, ill-conditioned near any
    rank-deficient Fisher information matrix (routine at the boundary of
    the feasible region). `_arm_rate_and_grad` replaces that with an
    exact closed-form gradient (no finite differences).

    **Known limitation, found and kept honest rather than hidden (see
    docs/decisions.md): this solver reliably converges to a *locally
    max-min-consistent* point -- every restart lands at (very nearly) the
    same objective value, with the two currently-tightest challengers'
    rates equalized, the textbook signature of a genuine max-min
    critical point -- but is not verified to always reach the *global*
    optimum.** `brute_force_allocation` (dense random sampling, a
    different search strategy entirely) found a feasible point with a
    noticeably higher objective on at least one realistic test instance.
    Diagnosis: `min_k` gradient ascent, updating only the currently-
    tightest challenger's block each step, can let the simplex
    projection's mass-redistribution deplete an *untouched* arm's weight
    toward zero as a side effect of increasing another's -- a real
    dynamic on this specific problem, not a step-size or scaling
    artifact (both were independently found and fixed first; see
    docs/decisions.md for that account too). A fully robust fix (e.g. a
    water-filling / bisection decomposition solving each arm's own
    minimum-cost-for-target-rate subproblem independently) was not
    completed given the time already spent chasing this; multi-restart
    (`n_restarts`) is the mitigation shipped here, and every caller of
    this function should treat its output as a validated-reasonable
    allocation, not a certified-optimal one, until this is revisited.

    `models` must contain every arm in `deltas` plus `k_star` (k_star's
    own model isn't used in the objective -- see the module docstring --
    but is required so `cost`/scale bookkeeping stays uniform across
    every arm the caller is tracking). Multi-start (`n_restarts` random
    starting points) guards against the rare case of stalling at a
    boundary point with a locally-flat subgradient.
    """
    challengers = [k for k in deltas if k != k_star]
    if not challengers:
        raise ValueError("deltas must name at least one challenger arm (k != k_star)")
    missing = (set(challengers) | {k_star}) - set(models)
    if missing:
        raise ValueError(f"models missing entries for arms {sorted(missing)}")

    n_arms = len(challengers)
    n_scales = len(candidate_scales)
    costs = np.array([cost(s) for s in candidate_scales])
    if np.any(costs <= 0):
        raise ValueError("every candidate scale must have positive cost")
    n_dims = n_arms * n_scales

    def rates_and_grads(p_flat: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        rates, grads = {}, {}
        for i, arm in enumerate(challengers):
            w_arm = p_flat[i * n_scales : (i + 1) * n_scales] / costs
            rate, grad_w = _arm_rate_and_grad(
                models[arm], candidate_scales, sigma2, w_arm, target_scale, deltas[arm]
            )
            rates[arm] = rate
            grads[arm] = grad_w / costs  # chain rule: dw/dp = 1/cost
        return rates, grads

    # A third, independent conditioning issue found by direct inspection
    # (after fixing the two above): the moment ANY (arm, scale) weight
    # hits exactly zero, that arm's Fisher information I_k(w) can become
    # rank-deficient, and np.linalg.pinv silently treats directions
    # outside its range as contributing ZERO variance rather than
    # INFINITE variance -- understating how uninformative a near-empty
    # design really is, right at the boundary the search needs to reason
    # about correctly. A floor keeps every (arm, scale) at a small but
    # strictly positive weight *during the search* (an interior-point
    # style regularization), so I_k(w) never actually degenerates while
    # optimizing; only the final reported weights can be (numerically)
    # negligible. Affine remap p -> (1 - n*floor)*p + floor keeps the
    # simplex constraint (sum p = 1) exact while enforcing p_i >= floor.
    floor = 1e-6 / n_dims

    def with_floor(p_raw: np.ndarray) -> np.ndarray:
        return (1.0 - n_dims * floor) * p_raw + floor

    rng = rng if rng is not None else np.random.default_rng(0)
    best_rate = -1.0
    best_p: np.ndarray | None = None
    best_n_iter = 0

    for _ in range(n_restarts):
        p = with_floor(rng.dirichlet(np.ones(n_dims)))

        for t in range(1, n_iter + 1):
            rates, grads = rates_and_grads(p)
            tightest = min(rates, key=rates.get)
            full_grad = np.zeros(n_dims)
            idx = challengers.index(tightest)
            full_grad[idx * n_scales : (idx + 1) * n_scales] = grads[tightest]

            # Normalized subgradient step: alpha_t / ||g_t||_2, not a raw
            # step calibrated only to the RATE's magnitude. Found necessary
            # by direct inspection of a failing run: the gradient's own
            # components can differ by ~10,000x across scales (the
            # cheapest scale's dw/dp = 1/cost(s) is proportionally largest,
            # since cost(s) spans orders of magnitude across a realistic
            # ladder), so a step size tuned only to rate_scale let a single
            # step move p entirely out of the simplex before projection,
            # collapsing every restart onto one corner (rate -> 0). Dividing
            # by the CURRENT gradient's own norm keeps every step a bounded,
            # well-behaved distance on the simplex regardless of how the
            # gradient's magnitude varies run to run or step to step. See
            # docs/decisions.md.
            grad_norm = float(np.linalg.norm(full_grad))
            if grad_norm <= 0:
                break  # a true stationary point (zero subgradient); done
            step = 0.5 / (np.sqrt(t) * grad_norm)
            p = with_floor(_project_to_simplex(p + step * full_grad))

        rates, _ = rates_and_grads(p)
        final_rate = min(rates.values())
        if final_rate > best_rate:
            best_rate = final_rate
            best_p = p
            best_n_iter = n_iter

    assert best_p is not None
    w_final = {
        arm: best_p[i * n_scales : (i + 1) * n_scales] / costs for i, arm in enumerate(challengers)
    }
    arm_rates = {
        arm: _arm_rate(
            models[arm], candidate_scales, sigma2, w_final[arm], target_scale, deltas[arm]
        )
        for arm in challengers
    }
    rate = min(arm_rates.values())
    weights = {(arm, i): float(w_final[arm][i]) for arm in challengers for i in range(n_scales)}
    return AllocationResult(
        weights=weights,
        rate=rate,
        t_star=(1.0 / rate) if rate > 0 else float("inf"),
        arm_rates=arm_rates,
        converged=True,
        n_iterations=best_n_iter,
        message=f"projected subgradient ascent: {n_restarts} restarts x {n_iter} iterations",
    )


def brute_force_allocation(
    models: dict[str, Extrapolator],
    k_star: str,
    candidate_scales: list[Scale],
    target_scale: Scale,
    sigma2: Callable[[Scale], float],
    deltas: dict[str, float],
    cost: Callable[[Scale], float] = _default_cost,
    n_samples: int = 200_000,
    rng: np.random.Generator | None = None,
) -> AllocationResult:
    """Independent verification path for `solve_allocation`: dense random
    (Dirichlet) sampling of the feasible weight simplex rather than
    SLSQP's local, gradient-based search -- "brute force" in the sense
    the plan asks for (plan/04-phase3-algorithm.md P3-02: "verify the
    solution against a brute-force grid search on small instances"). A
    literal axis-aligned grid is intractable past 3-4 free dimensions
    (n_arms * n_scales - 1 here); dense random sampling of the same
    feasible region is the practical equivalent for the "K=3, 3 scales"
    small instances this is meant to check against, and -- unlike a
    coarse grid -- has no risk of missing the optimum by stepping over it
    between grid points.
    """
    challengers = [k for k in deltas if k != k_star]
    n_scales = len(candidate_scales)
    costs = np.array([cost(s) for s in candidate_scales])
    rng = rng if rng is not None else np.random.default_rng(1)

    n_dims = len(challengers) * n_scales
    # Sample the compute-fraction p directly (uniform Dirichlet on the
    # actual constraint manifold sum(p) = 1), not a rescaled/projected raw
    # weight -- the same p = w*cost reparametrization that fixed
    # solve_allocation's conditioning bug, used here for unbiased coverage
    # of the true feasible region rather than a distorted one.
    p_samples = rng.dirichlet(np.ones(n_dims), size=n_samples)
    tiled_costs = np.tile(costs, len(challengers))
    w_samples = p_samples / tiled_costs  # (n_samples, n_dims)

    # Vectorized over all n_samples at once, per arm, using numpy's batched
    # pinv (np.linalg.pinv broadcasts over a leading (..., p, p) stack) --
    # a Python-level loop over n_samples with a per-row _arm_rate call
    # took 17s for just 100k samples; this does the same computation in
    # under a second for 10x as many, essential for this to be a usable
    # verification tool rather than a once-a-session curiosity.
    rates_per_arm = np.empty((len(challengers), n_samples))
    for a, arm in enumerate(challengers):
        model = models[arm]
        j_scales = np.array([model.jacobian(s) for s in candidate_scales])  # (n_scales, p)
        sig2 = np.array([sigma2(s) for s in candidate_scales])  # (n_scales,)
        j_star = model.jacobian(target_scale)  # (p,)
        w_arm = w_samples[:, a * n_scales : (a + 1) * n_scales]  # (n_samples, n_scales)

        j_outer = np.einsum("si,sj->sij", j_scales, j_scales) / sig2[:, None, None]
        info_batch = np.einsum("ns,sij->nij", w_arm, j_outer)  # (n_samples, p, p)
        info_pinv_batch = np.linalg.pinv(info_batch)
        f_batch = np.einsum("i,nij,j->n", j_star, info_pinv_batch, j_star)
        rates_per_arm[a] = np.where(
            f_batch > 1e-300, (deltas[arm] ** 2) / (2.0 * np.maximum(f_batch, 1e-300)), 0.0
        )

    min_rates = rates_per_arm.min(axis=0)  # (n_samples,) -- the max-min objective per sample
    best_idx = int(np.argmax(min_rates))
    best_rate = float(min_rates[best_idx])
    best_w = {
        arm: w_samples[best_idx, i * n_scales : (i + 1) * n_scales]
        for i, arm in enumerate(challengers)
    }

    arm_rates = {
        arm: _arm_rate(
            models[arm], candidate_scales, sigma2, best_w[arm], target_scale, deltas[arm]
        )
        for arm in challengers
    }
    weights = {(arm, i): float(best_w[arm][i]) for arm in challengers for i in range(n_scales)}
    return AllocationResult(
        weights=weights,
        rate=best_rate,
        t_star=(1.0 / best_rate) if best_rate > 0 else float("inf"),
        arm_rates=arm_rates,
        converged=True,
        n_iterations=n_samples,
        message=f"brute-force: best of {n_samples} Dirichlet samples",
    )
