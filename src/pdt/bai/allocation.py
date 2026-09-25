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
from pdt.theory.identifiability import target_in_row_space


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


def _active_jacobian_rows(
    model: Extrapolator, scales: list[Scale], w_arm: np.ndarray
) -> np.ndarray:
    """Jacobian rows of the scales carrying positive weight -- the *support*
    of the design, which is what decides structural identifiability."""
    rows = [model.jacobian(s) for wi, s in zip(w_arm, scales, strict=True) if wi > 0.0]
    if not rows:
        return np.zeros((0, model.n_params))
    return np.array(rows, dtype=float)


def _weighted_design(
    model: Extrapolator,
    scales: list[Scale],
    sigma2: Callable[[Scale], float],
    w_arm: np.ndarray,
) -> np.ndarray:
    """The weighted design `A` (one row per scale carrying positive weight) with
    `A^T A = I_k(w) = sum_s w(s) J(s) J(s)^T / sigma2(s)`: row `s` is
    `sqrt(w(s) / sigma2(s)) * J(s)`. Everything below works from `A` itself (its
    SVD), never from `I = A^T A`: forming `I` squares the condition number."""
    rows = []
    for wi, s in zip(w_arm, scales, strict=True):
        if wi <= 0.0:
            continue
        sig2 = sigma2(s)
        if sig2 <= 0:
            raise ValueError(f"sigma2({s!r}) must be positive, got {sig2}")
        rows.append(np.sqrt(wi / sig2) * np.asarray(model.jacobian(s), dtype=float))
    if not rows:
        return np.zeros((0, model.n_params))
    return np.array(rows, dtype=float)


#: Relative tolerance on the part of the target Jacobian lying in singular directions of the
#: weighted design that the rank cutoff discards; above it the target's variance is not
#: computable from this design and the arm is reported as having NO information.
_DISCARDED_TARGET_RTOL = 1e-8


def _solve_weighted(a: np.ndarray, j_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`f = J*^T (A^T A)^+ J*` and `y = (A^T A)^+ J*` for a stack of weighted designs
    `a` of shape `(..., n_rows, p)` and one target Jacobian `j_star` of shape `(p,)`,
    returned as `(f, y)` of shapes `(...,)` and `(..., p)`.

    Computed from the SVD of the column-equilibrated design, `f = ||S^-1 V^T j_eq||^2`,
    so a weak but genuinely identified direction (a tiny sampling weight on a scale
    that alone resolves one parameter) contributes its true, LARGE variance.
    Second-round review of PR #28: the earlier version inverted the equilibrated
    information matrix (`pinv(A^T A)`), whose cutoff discarded such a direction as if
    it carried zero variance -- e.g. `LogLinear`, candidates `N = e, e^2`, target
    `N = e^3`, weights `[1, 1e-16]`: rate `.00125` returned against a true `1.25e-19`.

    Fail-closed: if the rank cutoff discards a singular direction in which the target
    has a non-negligible component (relative > `_DISCARDED_TARGET_RTOL`), the result is
    `f = 0`, `y = 0` -- the caller's "no information" contract (`rate = 0`), which is
    the conservative reading -- never a small finite variance from a dropped direction.
    Works identically for one design (add a leading axis) and for the brute-force
    path's stack of designs.
    """
    a = np.asarray(a, dtype=float)
    batch_shape = a.shape[:-2]
    if a.shape[-2] == 0:
        return np.zeros(batch_shape), np.zeros(batch_shape + (a.shape[-1],))
    col = np.linalg.norm(a, axis=-2)  # (..., p)
    col = np.where(col > 0.0, col, 1.0)
    a_eq = a / col[..., None, :]
    j_eq = j_star / col  # (..., p)
    _, sing, vt = np.linalg.svd(a_eq, full_matrices=False)  # sing (..., k), vt (..., k, p)
    tol = max(a.shape[-2:]) * np.finfo(float).eps * sing[..., :1]
    keep = sing > tol
    proj = np.einsum("...kp,...p->...k", vt, j_eq)
    total2 = np.sum(j_eq**2, axis=-1)
    discarded2 = np.sum(np.where(keep, 0.0, proj**2), axis=-1)
    fail = discarded2 > (_DISCARDED_TARGET_RTOL**2) * np.maximum(total2, 1e-300)
    z = np.where(keep, proj / np.where(keep, sing, 1.0), 0.0)  # S^-1 V^T j_eq
    f = np.sum(z**2, axis=-1)
    y_eq = np.einsum("...kp,...k->...p", vt, z / np.where(keep, sing, 1.0))  # V S^-2 V^T j_eq
    y = y_eq / col
    f = np.where(fail, 0.0, f)
    y = np.where(fail[..., None], 0.0, y)
    return f, y


def _target_denom(a: np.ndarray, j_star: np.ndarray, j_rows: np.ndarray) -> float:
    """`J_target^T I^-1 J_target` (with `I = A^T A`), or 0.0 ("no information", the
    contract `_arm_rate` documents) when the target is unidentifiable from the design.

    PR #28's reviews: `np.linalg.pinv` silently treats any direction
    outside `range(I)` as contributing ZERO variance (its minimum-norm
    convention), the OPPOSITE of the truth (INFINITE variance). Reproduced:
    a `LogLinear` model observed only at `N=1` has `J(N=1) = [1, 0]`, so
    `I = diag(1, 0)` regardless of weight; the target at `N=4` has
    `J_target = [1, log(4)]`, and `J_target^T pinv(I) J_target = 1` is a
    small, finite, plausible-looking number when the truth is "cannot be
    estimated at all from this design".

    Three separate questions, three separate checks:
    1. *Structural identifiability* -- is `J_target` in the row space of the
       observed-scale Jacobians (`j_rows`, one per scale with positive weight)?
       Decided on the unweighted, equilibrated Jacobians by
       `pdt.theory.identifiability.target_in_row_space` (a 5%-missing component is
       already rejected; a loose 10% residual test was the first, wrong, fix).
    2. *Numerical resolvability of the weighted design* -- is the target's variance
       computable, or does the rank cutoff of the WEIGHTED design discard a direction
       the target uses? `_solve_weighted` fails closed (returns 0 information).
    3. *Ill-conditioning* -- a weak but resolved direction is a large finite variance,
       reported as such, because the solve works from `A`'s SVD (conditioning of `A`,
       not of `A^T A`).
    """
    if not np.any(j_star):
        return 0.0
    if not target_in_row_space(j_rows, j_star):
        return 0.0
    f, _ = _solve_weighted(a[None, ...], j_star)
    return float(f[0])


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
    j_star = model.jacobian(target_scale)
    # I_k(w) is only guaranteed PSD, not PD, at an arbitrary w (e.g. w_arm all zero, or
    # concentrated on scales whose Jacobians don't span p dimensions) -- the
    # "rank-deficient design" case Theorem 3 treats as a real phenomenon. Work from the
    # weighted design's SVD and confirm the target is actually resolvable (fail closed)
    # before trusting any variance -- see `_target_denom` / `_solve_weighted`.
    denom = _target_denom(
        _weighted_design(model, scales, sigma2, w_arm),
        j_star,
        _active_jacobian_rows(model, scales, w_arm),
    )
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

    Gated by the same structural identifiability check `_arm_rate` uses: if
    `J_target` is not in the row space of the observed-scale Jacobians, the target is structurally
    unidentifiable from every candidate scale regardless of `w` (PR #28's
    review), so `rate=0` for the *entire* feasible simplex here, not just
    this one point -- the true `d(rate)/dw = 0` everywhere, making a zero
    gradient the mathematically correct answer (not a spurious stall),
    and `solve_allocation`'s "zero subgradient -> stationary, done" exit
    is the right call in that case.
    """
    j_star = model.jacobian(target_scale)
    n = len(w_arm)
    a = _weighted_design(model, scales, sigma2, w_arm)
    f = _target_denom(a, j_star, _active_jacobian_rows(model, scales, w_arm))
    if f <= 1e-300:
        return 0.0, np.zeros(n)
    _, y_stack = _solve_weighted(a[None, ...], j_star)
    y = y_stack[0]
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

    # PR #28's review, P2: `converged` must reflect whether THIS restart
    # actually reached a stationarity criterion, not just that the loop
    # ran to completion -- exhausting the `n_iter` budget without the
    # subgradient norm reaching exactly zero is budget exhaustion, not
    # convergence, even though a rate value is still returned either way
    # (the heuristic objective at whatever point the search stopped, not
    # a certified T* optimum). Deliberately an EXACT-zero check, not a
    # small positive tolerance: `grad_norm`'s natural scale depends
    # entirely on the problem's own units (`delta_k`, `sigma2`, `cost`
    # can put rates anywhere from ~1e-20 to ~1 depending on the instance),
    # so any fixed positive tolerance is either too loose for some
    # instances (falsely claiming convergence after essentially zero
    # progress -- tried first, and directly observed to trigger after a
    # single iteration on this module's own realistic-instance tests) or
    # too tight for others. Exact zero is scale-invariant: it means the
    # tightest challenger's own subgradient is a genuine zero vector, a
    # real critical point regardless of units -- the same signal the
    # pre-fix code already used, just now correctly reflected in
    # `converged` instead of being reported as `True` unconditionally
    # regardless of whether that signal was ever seen.
    rng = rng if rng is not None else np.random.default_rng(0)
    best_rate = -1.0
    best_p: np.ndarray | None = None
    best_n_iter = 0
    best_converged = False

    for _ in range(n_restarts):
        p = with_floor(rng.dirichlet(np.ones(n_dims)))
        actual_iters = 0
        restart_converged = False

        for t in range(1, n_iter + 1):
            actual_iters = t
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
                restart_converged = True
                break  # a true stationary point (zero subgradient); done
            step = 0.5 / (np.sqrt(t) * grad_norm)
            p = with_floor(_project_to_simplex(p + step * full_grad))

        rates, _ = rates_and_grads(p)
        final_rate = min(rates.values())
        if final_rate > best_rate:
            best_rate = final_rate
            best_p = p
            best_n_iter = actual_iters
            best_converged = restart_converged

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
    message = (
        f"projected subgradient ascent: {n_restarts} restarts x up to {n_iter} iterations "
        f"(reached a near-stationary point in {best_n_iter} iterations)"
        if best_converged
        else (
            f"projected subgradient ascent: {n_restarts} restarts x {n_iter} iterations -- "
            "budget exhausted without reaching a near-stationary point on the best restart; "
            "treat the returned rate as a heuristic value from wherever the search stopped, "
            "not a certified T* optimum"
        )
    )
    return AllocationResult(
        weights=weights,
        rate=rate,
        t_star=(1.0 / rate) if rate > 0 else float("inf"),
        arm_rates=arm_rates,
        converged=best_converged,
        n_iterations=best_n_iter,
        message=message,
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

        # Weighted-design stack A_n (rows sqrt(w_ns / sigma2_s) J_s): every Dirichlet sample has
        # all weights positive, so the design's support is the full candidate set. The same
        # SVD-based, fail-closed solve as the scalar path -- never a pinv of A^T A.
        if not target_in_row_space(j_scales, j_star):
            rates_per_arm[a] = 0.0
            continue
        a_batch = np.sqrt(w_arm / sig2[None, :])[:, :, None] * j_scales[None, :, :]
        f_batch, _ = _solve_weighted(a_batch, j_star)
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
