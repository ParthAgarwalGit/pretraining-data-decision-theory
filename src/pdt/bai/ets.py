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
from scipy.stats import t as _t_dist

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


def _welch_satterthwaite_df(var_a: float, df_a: float, var_b: float, df_b: float) -> float | None:
    """Satterthwaite's approximation for the degrees of freedom of
    `(mu_a - mu_b) / sqrt(var_a + var_b)` when `var_a`/`var_b` are
    themselves ESTIMATED variances (not known exactly), each with its own
    degrees of freedom `df_a`/`df_b` -- the general form of Welch's
    t-test df formula (`pdt.analysis.rank_reversal.welch_satterthwaite_df`
    is the special case where both variances are `sigma2/n_seeds` with a
    shared `n_seeds`; here `var_a`/`var_b` are already-computed estimator
    variances -- `analytic_v_k`'s sandwich-covariance output -- each with
    its own, generally different, degrees of freedom).
    """
    if df_a <= 0 or df_b <= 0:
        return None
    if var_a == 0 and var_b == 0:
        return None
    numerator = (var_a + var_b) ** 2
    denominator = (var_a**2) / df_a + (var_b**2) / df_b
    if denominator == 0:
        return None
    return numerator / denominator


def _certification_radius(
    var_a: float, df_a: float, var_b: float, df_b: float, beta_t: float
) -> float:
    """The certification radius `c_t` for `mu_hat_a - mu_hat_b`, accounting
    for `var_a`/`var_b` (`analytic_v_k`'s HC0 sandwich-covariance output)
    being ESTIMATED, not known exactly.

    PR #29's review: the original formula, `sqrt(2*(var_a+var_b)*
    log(1/beta_t))`, is a valid sub-Gaussian tail bound only if
    `var_a+var_b` is the TRUE, known variance -- plugging in an estimate
    as if it were exact silently drops the extra uncertainty that
    estimate itself carries, understating the true radius. Reproduced
    exactly as given (LogLinear, scales N=[1,2,3], target N=4, delta=.01,
    single check with only n_params+1=3 pulls per arm, so each arm's own
    HC0 estimate has just 3-2=1 residual degree of freedom): 1,000
    independent trials gave 9.1% actual certification error against a
    1% request, a ~9x violation.

    Fixed with a Student-t radius, `t.ppf(1-beta_t, df) * sqrt(var_a+var_b)`,
    using a Welch-Satterthwaite-combined `df` from each arm's own
    HC0-residual degrees of freedom (`_welch_satterthwaite_df`) -- the
    standard correction for a t-statistic built from independently
    estimated variances with unequal/small degrees of freedom, the same
    remedy this project already applied to P1-09's calibration bug (see
    docs/decisions.md). As `df -> infinity` this converges to the
    original formula's asymptotic regime (`t.ppf -> norm.ppf`, close to
    but not identical to `sqrt(2*log(1/beta))`'s own Chernoff-style
    bound), so the correction is concentrated exactly where the original
    formula was most wrong: small-sample, few-distinct-scale certification
    checks, precisely the reviewer's counterexample.

    **Still not a rigorously PROVEN finite-sample radius** -- a fully
    rigorous fix requires either a proper always-valid confidence sequence
    for unknown variance (the literature the review points to) or
    restricting the guarantee to the known-sigma2 case
    `tests/theory/test_theorem4.py` already certifies; this is a
    substantial, principled improvement over the invalid plug-in Gaussian
    radius, verified empirically against the reviewer's own counterexample
    (see docs/decisions.md), but should be read as a documented, tested
    heuristic correction, not a certified guarantee -- flagged forward
    for the Theorem 4 write-up (paper/sections/theorem4_algorithm.tex,
    PR #26's own review of which independently found the same
    "simultaneous adaptive confidence" gap at the proof level) to resolve
    properly.
    """
    df = _welch_satterthwaite_df(var_a, df_a, var_b, df_b)
    if df is None:
        # Degenerate (a variance genuinely zero, or a residual-df <= 0
        # that should not occur given _assert_design_identified, but
        # guarded rather than assumed): fall back to the original
        # known-variance-style radius rather than raising, since a
        # zero-variance arm is a real (if unusual) input.
        return float(np.sqrt(max(2 * (var_a + var_b) * np.log(1 / beta_t), 0.0)))
    return float(_t_dist.ppf(1 - beta_t, df) * np.sqrt(max(var_a + var_b, 0.0)))


#: Fixed local seed for `_assert_design_identified`'s generic-parameter
#: rank probes, re-seeded FRESH on every call (never a shared, mutating
#: module-level generator) -- deterministic and independent of caller
#: order, since this is a validity check, not a statistical estimate. An
#: earlier version shared one mutating generator across every call, so a
#: call's own probe sequence (and thus whether it happened to hit a
#: bad-luck streak of degenerate draws) depended on how many prior calls
#: had already consumed from it elsewhere in the program -- a real,
#: observed flakiness source (see docs/decisions.md).
_DESIGN_RANK_PROBE_SEED = 0
_N_DESIGN_RANK_PROBES = 8


def _assert_design_identified(
    model_factory: Callable[[], Extrapolator],
    n_params: int,
    model_name: str,
    distinct_scales: list[Scale],
) -> None:
    """Theorem 3's rank condition, asserted directly rather than
    trusted -- but as a property of the *design* (the actual scales this
    recipe has been pulled at), not of one particular converged fit's
    numerical Jacobian.

    An earlier version of this check only counted distinct scales
    (`len(distinct_scales) >= n_params+1`) after finding that asserting
    on the CONVERGED fit's own Jacobian was the wrong check: a real
    nonlinear fit can converge to a numerically near-degenerate point
    (e.g. `alpha` saturating near its bound, making the `d/d(alpha)` and
    `d/d(a)` directions collinear at every observed scale) even when the
    design itself is perfectly identifiable -- asserting on that one
    fit's Jacobian made this raise on legitimate, well-designed runs for
    a reason having nothing to do with the tracking rule's own
    correctness (see docs/decisions.md).

    PR #29's review: the count-only check is necessary but NOT
    sufficient -- `n_params+1` distinct `(N, D)` pairs does not imply the
    model's Jacobian at those pairs has rank `n_params`. `LogLinear`'s
    fit ignores `D` entirely, so `n_params+1` pairs sharing the same `N`
    (different `D`) pass the count check while providing genuinely ZERO
    information about the slope parameter. Fixed by ALSO checking the
    Jacobian's rank at several GENERIC reference parameter values (drawn
    from the model's own declared `_bounds`, from a fixed local seed --
    not this run's own possibly-degenerate converged fit, which avoids
    reintroducing the false-negative failure mode the count-only check
    was adopted to fix): if the model exposes `_bounds` and its Jacobian
    has rank `< n_params` at EVERY one of several generic
    parameterizations, these scales cannot identify this model no matter
    what the true parameters are, and the design is rejected regardless
    of how many nominally-distinct scales it counts. Models without a
    single unified `_bounds` (e.g. `ConstantExtrapolator`, whose n_params=1
    Jacobian is trivially always full rank; `TwoStepLadder`'s two-stage
    `_step1_bounds`/`_step2_bounds`) fall back to the count-only check.
    """
    if len(distinct_scales) < n_params + 1:
        raise FitFailure(
            f"{model_name}: only {len(distinct_scales)} distinct scales pulled, need "
            f">= {n_params + 1} to identify {n_params} parameters -- Theorem 3's design "
            f"rank condition is violated."
        )
    probe_model = model_factory()
    bounds = getattr(probe_model, "_bounds", None)
    if bounds is None:
        return
    lo, hi = bounds
    # Log-uniform, not linear-uniform, for any dimension whose lower
    # bound is strictly positive -- a model-agnostic heuristic that
    # reliably identifies decay-rate/exponent-like parameters (every
    # `alpha`/`beta` bound seen in this project's fitters is strictly
    # positive, e.g. PowerLawN's `[1e-3, 10]`) without needing per-fitter
    # knowledge of which index is which. Necessary, not cosmetic: linear-
    # uniform sampling over such a wide positive range puts the large
    # majority of draws in the numerically-flat region where `N^-alpha`
    # and its derivatives underflow to (numerical, not mathematical)
    # zero for realistic scale magnitudes -- exactly `multi_start_fit`'s
    # own `log_uniform_dims` fix (see docs/decisions.md, PR #12) for the
    # identical reason. Checked directly: linear-uniform probes on a
    # genuinely well-identified 4-scale PowerLawN design gave rank < 3
    # on ~81% of individual draws (an 8-probe run has a ~20% chance of
    # hitting 8-in-a-row and false-rejecting a legitimate design); log-
    # uniform for the positive-bounded dimensions drops that to ~18% per
    # draw (~1e-6 for 8-in-a-row).
    probe_rng = np.random.default_rng(_DESIGN_RANK_PROBE_SEED)
    for _ in range(_N_DESIGN_RANK_PROBES):
        theta = np.empty(n_params)
        for i in range(n_params):
            if lo[i] > 0:
                theta[i] = np.exp(probe_rng.uniform(np.log(lo[i]), np.log(hi[i])))
            else:
                theta[i] = probe_rng.uniform(lo[i], hi[i])
        probe_model._theta = theta
        jac = np.array([probe_model.jacobian(s) for s in distinct_scales])
        if np.linalg.matrix_rank(jac) >= n_params:
            return
    raise FitFailure(
        f"{model_name}: {len(distinct_scales)} distinct scales pulled, but the model's "
        f"Jacobian has rank < {n_params} at every one of {_N_DESIGN_RANK_PROBES} generic "
        "reference parameterizations -- these scales cannot identify this model's "
        "parameters no matter what the true parameters are (e.g. scales that differ only "
        "in a dimension the model ignores)."
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
    _assert_design_identified(model_factory, model.n_params, type(model).__name__, distinct_scales)
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
    "decided"`, matching the other baselines).

    `pairs` cycles every recipe once at the *cheapest* scale before
    moving to the next-cheapest, not every scale for one recipe before
    the next recipe -- found to matter in practice, not just in theory:
    an early (recipe-major) version could exhaust `compute_budget`
    partway through the *first* recipe's own scale ladder on a real,
    many-recipe instance whose few largest scales are individually
    expensive (P3-05's real DataDecide replay, 25 recipes, one scale
    costing a large fraction of a `6x`-scale-cost budget on its own),
    leaving every later recipe with zero pulls and no way to be fit at
    all -- a real correctness bug, not just an inefficiency, since
    `_fit_recipe` then raises rather than silently returning a bad but
    plausible answer. Cycling scale-major, cheapest first, guarantees
    every recipe gets at least the cheap end of the ladder before any
    recipe gets a second pull, for *any* positive budget."""
    scales_data: dict[str, list[Scale]] = {r: [] for r in recipes}
    values_data: dict[str, list[float]] = {r: [] for r in recipes}
    compute_spent = 0.0
    n_pulls = 0
    seed_counters: dict[tuple[str, float], int] = {}

    scales_by_cost = sorted(candidate_scales, key=cost)
    pairs = [(r, s) for s in scales_by_cost for r in recipes]
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

        # Residual degrees of freedom for each recipe's own HC0
        # sandwich-covariance v_hat -- the classic regression-theory
        # analogue (pulls minus fitted parameters) -- fed to
        # `_certification_radius` so the certification check accounts for
        # v_hat being ESTIMATED, not known (PR #29's review).
        resid_df = {r: max(len(scales_data[r]) - models[r].n_params, 1) for r in recipes}

        certified_all = True
        abstain_any = False
        certificate: dict = {"k_hat": k_hat, "per_challenger": {}}
        beta_t = _beta(t, delta)
        for k in challengers:
            delta_hat_k = mu_hat[k_hat] - mu_hat[k]
            c_k = _certification_radius(
                v_hat[k_hat], resid_df[k_hat], v_hat[k], resid_df[k], beta_t
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
