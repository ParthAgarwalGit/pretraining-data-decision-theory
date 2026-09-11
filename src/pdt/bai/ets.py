"""Extrapolation-Track-and-Stop and its baselines -- plan/04-phase3-algorithm.md
P3-03.

`extrapolation_track_and_stop` implements exactly the tracking/stopping/
abstention rule specified in paper/sections/theorem4_algorithm.tex's "The
algorithm" section (that document is the spec, cross-checked by
tests/theory/test_theorem4.py; this module builds the *full adaptive*
version -- P3-02's `solve_allocation` plugged in with the current fits,
not a fixed design shape -- which the theory certificate deliberately
does not attempt).

One real gap between the theorem and a runnable algorithm, resolved here
and documented rather than silently patched over: `solve_allocation`'s
T*(nu) program (`src/pdt/bai/allocation.py`) returns weight *only* for
the challenger arms -- by the change-of-measure construction
(theorem2_lower_bound.tex Step 2), k*'s own distribution is never
perturbed, so its allocation genuinely does not appear in that
optimization. A real tracking algorithm still needs to keep pulling the
current leader (to know mu_hat_{k_hat}(s*) at all, and to keep its own
extrapolator identified as the leader potentially changes across
rounds). This module reserves a fixed fraction `kstar_reserve_frac`
(default `1/(n_challengers+1)`, i.e. "treat k* like one more arm") of
the tracking weight for the current leader, spread uniformly across its
own candidate scales -- a simple, explicit heuristic, not something
Theorem 4 specifies, and not tuned here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pdt.bai.allocation import solve_allocation
from pdt.bai.oracle import PullOracle
from pdt.scaling.base import Extrapolator, FitFailure, Scale
from pdt.scaling.fitters import PowerLawN
from pdt.theory.bound import analytic_v_k


@dataclass(frozen=True)
class SelectionResult:
    """Common return type for every method in this module (the four
    baselines and `extrapolation_track_and_stop`), so P3-04/P3-05 can
    compare them uniformly.

    `outcome` is one of `"certified"` (a delta-correct decision, only
    possible for `extrapolation_track_and_stop`), `"abstained"`
    (ETS declined to certify; `recipe` is then the single-scale
    fallback, per theorem4_algorithm.tex), or `"decided"` (a baseline's
    unconditional pick -- no statistical guarantee is claimed).
    `certificate` carries method-specific diagnostics (e.g. ETS's final
    `delta_hat`/`c_t`/`eta` values; a baseline's final predictions).
    """

    method: str
    outcome: str
    recipe: str
    compute_spent: float
    n_pulls: int
    certificate: dict = field(default_factory=dict)


def _default_cost(scale: Scale) -> float:
    return scale.compute


def _beta(t: int, delta: float) -> float:
    """Standard anytime-valid union-bound-over-time threshold, matching
    tests/theory/test_theorem4.py's certificate exactly."""
    return delta / (t * (t + 1))


def _assert_design_identified(n_params: int, model_name: str, distinct_scales: list[Scale]) -> None:
    """Theorem 3's rank condition, asserted directly rather than
    trusted -- but as a property of the *design* (how many distinct
    scales this recipe has been pulled at), not of one particular
    converged fit's numerical Jacobian.

    An earlier version of this check computed the fitted model's own
    Jacobian at each distinct scale and asserted `matrix_rank(J) ==
    n_params`. That was found to be the wrong check by direct testing:
    a real nonlinear fit can converge to a numerically near-degenerate
    point (e.g. `alpha` saturating near its bound, making the
    `d/d(alpha)` and `d/d(a)` directions collinear at every observed
    scale) even when the *design itself* -- the actual set of scales
    pulled -- is perfectly identifiable in the sense Theorem 3 means
    (>= n_params+1 distinct scales, which is what forced exploration
    guarantees). Asserting on the converged Jacobian made this raise on
    legitimate, well-designed runs for a reason having nothing to do
    with the tracking rule's own correctness -- see docs/decisions.md.
    """
    if len(distinct_scales) < n_params + 1:
        raise FitFailure(
            f"{model_name}: only {len(distinct_scales)} distinct scales pulled, need "
            f">= {n_params + 1} to identify {n_params} parameters -- Theorem 3's design "
            f"rank condition is violated."
        )


def _fit_recipe(
    model_factory: Callable[[], Extrapolator],
    scales: list[Scale],
    values: list[float],
    target_scale: Scale,
) -> tuple[Extrapolator, float, float]:
    """Fit one recipe's model on its raw (possibly-duplicated-scale)
    pull history, assert design identifiability, and return `(model,
    mu_hat, v_hat)` at `target_scale` -- `v_hat` via the same analytic
    delta-method machinery as P1-07/Theorem 1 (`analytic_v_k`), not a
    separate ad hoc formula."""
    model = model_factory()
    distinct_scales = sorted(set(scales), key=lambda s: s.n)
    _assert_design_identified(model.n_params, type(model).__name__, distinct_scales)
    model.fit(scales, values)
    mu_hat = model.predict(target_scale)
    v_hat = analytic_v_k(model, scales, values, target_scale)
    return model, mu_hat, v_hat


def _largest_observed_scale(
    scales_by_recipe: dict[str, list[Scale]], candidate_scales: list[Scale]
) -> Scale:
    """The largest candidate scale every recipe has at least one pull
    at -- used for the single-scale fallback recommendation, per
    theorem4_algorithm.tex ("the single-scale recommendation at the
    largest scale actually observed")."""
    observed_by_all = [
        s for s in candidate_scales if all(s in scales_by_recipe[r] for r in scales_by_recipe)
    ]
    if not observed_by_all:
        raise FitFailure("no candidate scale has been pulled for every recipe yet")
    return max(observed_by_all, key=lambda s: s.n)


def single_scale_recommendation(
    oracle: PullOracle,
    recipes: list[str],
    scale: Scale,
    n_replicates: int = 1,
    cost: Callable[[Scale], float] = _default_cost,
) -> SelectionResult:
    """`SingleScale(s_p)`: one (or `n_replicates`) pull per recipe at a
    single scale, rank by the observed mean -- no extrapolation, no
    statistical guarantee. Corollary 1 (theorem1_bound.tex)'s baseline,
    and the fallback ETS recommends on abstention."""
    means: dict[str, float] = {}
    compute_spent = 0.0
    n_pulls = 0
    for recipe in recipes:
        draws = [oracle.pull(recipe, scale, seed=i) for i in range(n_replicates)]
        means[recipe] = float(np.mean(draws))
        compute_spent += n_replicates * cost(scale)
        n_pulls += n_replicates
    best = max(means, key=means.get)
    return SelectionResult(
        method="SingleScale",
        outcome="decided",
        recipe=best,
        compute_spent=compute_spent,
        n_pulls=n_pulls,
        certificate={"scale": scale, "means": means},
    )


def uniform_allocation(
    oracle: PullOracle,
    recipes: list[str],
    candidate_scales: list[Scale],
    target_scale: Scale,
    compute_budget: float,
    model_factory: Callable[[], Extrapolator] = PowerLawN,
    cost: Callable[[Scale], float] = _default_cost,
) -> SelectionResult:
    """`UniformAllocation`: equal compute per recipe, spread evenly over
    the fixed scale ladder -- round-robins (recipe, scale) pairs one
    pull at a time until `compute_budget` is spent, then fits and
    extrapolates each recipe to `target_scale` and ranks by the
    prediction. No statistical guarantee is claimed (`outcome=
    "decided"`, matching the other baselines)."""
    scales_data: dict[str, list[Scale]] = {r: [] for r in recipes}
    values_data: dict[str, list[float]] = {r: [] for r in recipes}
    compute_spent = 0.0
    n_pulls = 0
    seed_counters: dict[tuple[str, float], int] = {}

    pairs = [(r, s) for r in recipes for s in candidate_scales]
    i = 0
    while compute_spent < compute_budget:
        recipe, scale = pairs[i % len(pairs)]
        key = (recipe, scale.n)
        seed = seed_counters.get(key, 0)
        seed_counters[key] = seed + 1
        value = oracle.pull(recipe, scale, seed=seed)
        scales_data[recipe].append(scale)
        values_data[recipe].append(value)
        compute_spent += cost(scale)
        n_pulls += 1
        i += 1

    predictions: dict[str, float] = {}
    for recipe in recipes:
        _model, mu_hat, _v = _fit_recipe(
            model_factory, scales_data[recipe], values_data[recipe], target_scale
        )
        predictions[recipe] = mu_hat
    best = max(predictions, key=predictions.get)
    return SelectionResult(
        method="UniformAllocation",
        outcome="decided",
        recipe=best,
        compute_spent=compute_spent,
        n_pulls=n_pulls,
        certificate={"predictions": predictions},
    )


def fixed_ladder_extrapolation(
    oracle: PullOracle,
    recipes: list[str],
    candidate_scales: list[Scale],
    target_scale: Scale,
    n_replicates: int = 1,
    model_factory: Callable[[], Extrapolator] = PowerLawN,
    cost: Callable[[Scale], float] = _default_cost,
) -> SelectionResult:
    """`FixedLadderExtrapolation`: the DataDecide-style approach -- a
    literal fixed design (`n_replicates` pulls at *every* candidate
    scale for *every* recipe, nothing adaptive), then fit and
    extrapolate each recipe and rank. Distinct from `UniformAllocation`
    in that the design is fixed by scale coverage rather than by a
    total compute budget."""
    compute_spent = 0.0
    n_pulls = 0
    predictions: dict[str, float] = {}
    for recipe in recipes:
        scales_data: list[Scale] = []
        values_data: list[float] = []
        for scale in candidate_scales:
            for seed in range(n_replicates):
                value = oracle.pull(recipe, scale, seed=seed)
                scales_data.append(scale)
                values_data.append(value)
                compute_spent += cost(scale)
                n_pulls += 1
        _model, mu_hat, _v = _fit_recipe(model_factory, scales_data, values_data, target_scale)
        predictions[recipe] = mu_hat
    best = max(predictions, key=predictions.get)
    return SelectionResult(
        method="FixedLadderExtrapolation",
        outcome="decided",
        recipe=best,
        compute_spent=compute_spent,
        n_pulls=n_pulls,
        certificate={"predictions": predictions},
    )


def successive_halving_over_scales(
    oracle: PullOracle,
    recipes: list[str],
    candidate_scales: list[Scale],
    target_scale: Scale,
    n_replicates: int = 1,
    model_factory: Callable[[], Extrapolator] = PowerLawN,
    cost: Callable[[Scale], float] = _default_cost,
) -> SelectionResult:
    """`SuccessiveHalvingOverScales`: at each rung (candidate scale, in
    increasing order), pull every surviving recipe `n_replicates` times,
    then eliminate the bottom half (by that rung's observed mean),
    doubling as a budget-shrinking bandit baseline over the *scale*
    ladder rather than over training iterations. Every rung is visited
    regardless of how many recipes remain (the pool halves, but the
    *scale ladder* does not stop early): the eventual survivor(s) still
    need enough distinct scales to identify `model_factory` (Theorem 3's
    rank condition), so with few recipes and a longer ladder the winner
    keeps getting pulled at every remaining rung even after the field
    has narrowed to one -- what makes its own extrapolation valid.
    Survivors of the last rung are extrapolated (using every point they
    accumulated across all rungs) and ranked -- ties in the elimination
    step are broken by recipe name for determinism."""
    scales_sorted = sorted(candidate_scales, key=lambda s: s.n)
    survivors = list(recipes)
    scales_data: dict[str, list[Scale]] = {r: [] for r in recipes}
    values_data: dict[str, list[float]] = {r: [] for r in recipes}
    compute_spent = 0.0
    n_pulls = 0

    for scale in scales_sorted:
        rung_means: dict[str, float] = {}
        for recipe in survivors:
            draws = [oracle.pull(recipe, scale, seed=i) for i in range(n_replicates)]
            scales_data[recipe].extend([scale] * n_replicates)
            values_data[recipe].extend(draws)
            compute_spent += n_replicates * cost(scale)
            n_pulls += n_replicates
            rung_means[recipe] = float(np.mean(draws))
        n_keep = max(1, len(survivors) // 2)
        survivors = sorted(survivors, key=lambda r: (-rung_means[r], r))[:n_keep]

    predictions: dict[str, float] = {}
    for recipe in survivors:
        _model, mu_hat, _v = _fit_recipe(
            model_factory, scales_data[recipe], values_data[recipe], target_scale
        )
        predictions[recipe] = mu_hat
    best = max(predictions, key=predictions.get)
    return SelectionResult(
        method="SuccessiveHalvingOverScales",
        outcome="decided",
        recipe=best,
        compute_spent=compute_spent,
        n_pulls=n_pulls,
        certificate={"predictions": predictions, "final_survivors": survivors},
    )


def extrapolation_track_and_stop(
    oracle: PullOracle,
    recipes: list[str],
    candidate_scales: list[Scale],
    target_scale: Scale,
    delta: float,
    eta: dict[str, float],
    sigma2: Callable[[Scale], float],
    model_factory: Callable[[], Extrapolator] = PowerLawN,
    cost: Callable[[Scale], float] = _default_cost,
    epsilon_0: float = 1e-4,
    max_rounds: int = 5000,
    solver_n_restarts: int = 1,
    solver_n_iter: int = 200,
    kstar_reserve_frac: float | None = None,
    min_pulls_per_pair: int = 1,
    rng: np.random.Generator | None = None,
) -> SelectionResult:
    """Extrapolation-Track-and-Stop, exactly as specified in
    paper/sections/theorem4_algorithm.tex: at each round, refit every
    recipe's extrapolator on data so far, recompute the plug-in T*(nu)
    weights from `solve_allocation` using the current fits as the
    "instance", track the most-under-sampled (recipe, scale) pair
    relative to that plug-in design (plus the reserved leader share --
    see module docstring), and check Certified/Abstain after every pull.

    `eta` is an *input* (an assumed upper bound on each recipe's bias
    magnitude, `eta_k >= sqrt(sigma2_extrap_k)`), per P2-05's decision
    (route (b): condition on a known/estimated eta rather than inflate
    delta) -- not discovered by this function. `sigma2` is the
    per-scale noise-variance function `solve_allocation` also expects
    (the *planning*-time instance); the stopping rule's own `v_k(t)`
    is estimated from data via the sandwich covariance
    (`analytic_v_k`), not assumed to equal `sigma2`.

    `solver_n_restarts`/`solver_n_iter` default far below
    `solve_allocation`'s own defaults (6 x 4000): T* is re-solved every
    round here, so a cheap solve is a deliberate, documented
    computational trade-off for tractability, not the accuracy-focused
    settings `solve_allocation` ships with for a one-shot call.
    """
    if len(recipes) < 2:
        raise ValueError("need at least 2 recipes to compare")
    if len(candidate_scales) < model_factory().n_params + 1:
        raise ValueError(
            f"need at least {model_factory().n_params + 1} candidate scales to identify "
            f"{type(model_factory()).__name__}, got {len(candidate_scales)}"
        )
    missing_eta = set(recipes) - set(eta)
    if missing_eta:
        raise ValueError(f"eta missing entries for recipes {sorted(missing_eta)}")

    rng = rng if rng is not None else np.random.default_rng(0)
    n_scales = len(candidate_scales)
    scale_idx = {s: i for i, s in enumerate(candidate_scales)}

    scales_data: dict[str, list[Scale]] = {r: [] for r in recipes}
    values_data: dict[str, list[float]] = {r: [] for r in recipes}
    counts: dict[str, np.ndarray] = {r: np.zeros(n_scales, dtype=int) for r in recipes}
    seed_counters: dict[tuple[str, float], int] = {}
    compute_spent = 0.0
    n_pulls = 0

    def pull(recipe: str, scale: Scale) -> None:
        nonlocal compute_spent, n_pulls
        key = (recipe, scale.n)
        seed = seed_counters.get(key, 0)
        seed_counters[key] = seed + 1
        value = oracle.pull(recipe, scale, seed=seed)
        scales_data[recipe].append(scale)
        values_data[recipe].append(value)
        counts[recipe][scale_idx[scale]] += 1
        compute_spent += cost(scale)
        n_pulls += 1

    # Warm-up: every recipe needs >= n_params+1 distinct scales before a
    # first fit is even possible; a deterministic round-robin sweep over
    # every (recipe, scale) pair achieves that for every recipe at once,
    # and doubles as the tracking rule's own natural forced-exploration
    # floor (an unpulled pair always has N=0, the global minimum of
    # N/pi, so it is never starved once the adaptive phase begins).
    # `min_pulls_per_pair` (default 1) lets a caller front-load more than
    # the bare minimum of initial exploration.
    for recipe in recipes:
        for scale in candidate_scales:
            for _ in range(min_pulls_per_pair):
                pull(recipe, scale)

    for t in range(1, max_rounds + 1):
        fits = {
            r: _fit_recipe(model_factory, scales_data[r], values_data[r], target_scale)
            for r in recipes
        }
        mu_hat = {r: fits[r][1] for r in recipes}
        v_hat = {r: fits[r][2] for r in recipes}
        models = {r: fits[r][0] for r in recipes}

        k_hat = max(mu_hat, key=mu_hat.get)
        challengers = [r for r in recipes if r != k_hat]

        certified_all = True
        abstain_any = False
        certificate: dict = {"k_hat": k_hat, "per_challenger": {}}
        for k in challengers:
            delta_hat_k = mu_hat[k_hat] - mu_hat[k]
            c_k = float(
                np.sqrt(max(2 * (v_hat[k_hat] + v_hat[k]) * np.log(1 / _beta(t, delta)), 0.0))
            )
            margin = delta_hat_k - eta[k_hat] - eta[k]
            certificate["per_challenger"][k] = {
                "delta_hat": delta_hat_k,
                "c_t": c_k,
                "margin": margin,
            }
            if not (margin > c_k):
                certified_all = False
            if c_k <= epsilon_0 and margin <= c_k:
                abstain_any = True

        if certified_all:
            return SelectionResult(
                method="ExtrapolationTrackAndStop",
                outcome="certified",
                recipe=k_hat,
                compute_spent=compute_spent,
                n_pulls=n_pulls,
                certificate=certificate,
            )
        if abstain_any:
            s_rec = _largest_observed_scale(scales_data, candidate_scales)
            fallback = single_scale_recommendation(oracle, recipes, s_rec, cost=cost)
            certificate["reason"] = "bias floor"
            certificate["fallback_scale"] = s_rec
            return SelectionResult(
                method="ExtrapolationTrackAndStop",
                outcome="abstained",
                recipe=fallback.recipe,
                compute_spent=compute_spent + fallback.compute_spent,
                n_pulls=n_pulls + fallback.n_pulls,
                certificate=certificate,
            )

        deltas = {k: abs(mu_hat[k_hat] - mu_hat[k]) for k in challengers}
        alloc = solve_allocation(
            models,
            k_hat,
            candidate_scales,
            target_scale,
            sigma2,
            deltas,
            cost=cost,
            n_restarts=solver_n_restarts,
            n_iter=solver_n_iter,
            rng=rng,
        )
        f_kstar = (
            kstar_reserve_frac if kstar_reserve_frac is not None else 1.0 / (len(challengers) + 1)
        )
        # `alloc.weights` is always strictly positive componentwise --
        # `solve_allocation`'s own `with_floor` remap guarantees every
        # weight is >= a positive floor, regardless of how small the
        # plug-in deltas are (see allocation.py) -- so `total_w` is
        # always > 0 here; no degenerate-zero fallback is needed.
        total_w = sum(alloc.weights.values())
        pi = np.zeros((len(recipes), n_scales))
        recipe_idx = {r: i for i, r in enumerate(recipes)}
        for k in challengers:
            for i in range(n_scales):
                pi[recipe_idx[k], i] = (1.0 - f_kstar) * alloc.weights[(k, i)]
        kstar_total = (f_kstar / (1.0 - f_kstar)) * total_w
        pi[recipe_idx[k_hat], :] = kstar_total / n_scales

        counts_matrix = np.array([counts[r] for r in recipes], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(pi > 0, counts_matrix / pi, np.inf)
        target_flat = int(np.argmin(ratio))
        target_r_idx, target_s_idx = np.unravel_index(target_flat, ratio.shape)
        pull(recipes[target_r_idx], candidate_scales[target_s_idx])

    # Round cap exhausted without either resolving -- must still behave
    # exactly like a genuine abstention (same fallback computation, same
    # certificate shape) rather than raising or returning an unusable
    # result: a real caller cannot distinguish "the theorem's asymptotic
    # abstention hasn't kicked in yet within this compute budget" from
    # "abstained for the reason Theorem 4 describes" without also being
    # told which one happened, hence the distinct `reason` string.
    s_rec = _largest_observed_scale(scales_data, candidate_scales)
    fallback = single_scale_recommendation(oracle, recipes, s_rec, cost=cost)
    return SelectionResult(
        method="ExtrapolationTrackAndStop",
        outcome="abstained",
        recipe=fallback.recipe,
        compute_spent=compute_spent + fallback.compute_spent,
        n_pulls=n_pulls + fallback.n_pulls,
        certificate={
            "reason": "max_rounds exhausted without certifying or abstaining",
            "fallback_scale": s_rec,
        },
    )
