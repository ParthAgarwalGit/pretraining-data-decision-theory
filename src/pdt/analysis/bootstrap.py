"""Bootstrap resampling for the bias/variance decomposition.

See plan/02-phase1-datadecide.md task P1-06. Two resampling schemes, both
producing one perturbed trajectory per (recipe, scale) per replicate:

- **Seed bootstrap**: resample the (up to 3) seed observations at a scale
  with replacement, per the plan's literal "resample seeds with
  replacement at each fitted scale".
- **Parametric bootstrap**: perturb the seed-averaged point by Gaussian
  noise with variance from the P1-05 noise model.

The two schemes treat cross-recipe dependence differently, deliberately:

- **Seed bootstrap** draws ONE shared resample-index pattern per (scale,
  replicate), applied to every recipe's own seed values, because the plan
  requires the pairwise difference statistic `D_k = mu_hat_k*(s*) -
  mu_hat_k(s*)` to be "computed from the same bootstrap replicate (so the
  correlation between the two arms' fits is preserved)". Sharing the
  *index pattern* does not force any correlation: each recipe's resampled
  mean still differs because its own seed values differ. `draw_seed_resample_pattern`
  / `apply_seed_resample`.
- **Parametric bootstrap** draws an INDEPENDENT standard-normal z per
  (recipe, scale, replicate). It used to share one z across recipes, which
  forces an exact +1 correlation between every pair of recipes' noise
  (covariance sigma_a*sigma_b, never estimated) and collapses the pairwise
  bootstrap variance to zero whenever two recipes have equal noise even
  though their observations are independent -- see docs/decisions.md and
  `experiments/p1_06_decomposition.py::_parametric_draws_for_recipe`.
  P1-05 measures only each recipe's own noise variance, no cross-recipe
  covariance, so independence is the honest default.
  `draw_parametric_noise_z` / `apply_parametric_noise`.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from pdt.scaling.base import Scale


def recipe_seed_trajectories(
    long_frame: pl.DataFrame, metric_name: str, proxy_sizes: list[str]
) -> dict[str, dict[str, list[tuple[Scale, list[float]]]]]:
    """{task: {recipe: [(Scale(n, d), [v_seed1, v_seed2, ...]), ...]}}
    across `proxy_sizes`, each recipe's list sorted by scale ascending.

    Unlike `decision_accuracy.recipe_trajectories()`, this keeps every
    seed's raw value instead of averaging them -- seed bootstrap needs the
    individual values to resample from; averaging first would throw away
    exactly what it resamples.
    """
    subset = (
        long_frame.filter(
            (pl.col("metric_name") == metric_name)
            & (pl.col("params_str").is_in(proxy_sizes))
            & (pl.col("is_final"))
        )
        .sort(["recipe", "task", "params_str", "seed"])
        .drop(["seed"])
    )
    if subset.height == 0:
        raise ValueError(
            f"no rows for metric_name={metric_name!r}, proxy_sizes={proxy_sizes!r}, "
            "is_final=True -- check the metric name and size labels are real."
        )

    grouped = subset.group_by(["recipe", "task", "params_str"], maintain_order=True).agg(
        pl.col("params_num").first().alias("n"),
        pl.col("tokens").mean().alias("d"),
        pl.col("metric_value").alias("seed_values"),
    )

    result: dict[str, dict[str, list[tuple[Scale, list[float]]]]] = {}
    for row in grouped.iter_rows(named=True):
        task_dict = result.setdefault(row["task"], {})
        task_dict.setdefault(row["recipe"], []).append(
            (Scale(n=row["n"], d=row["d"]), list(row["seed_values"]))
        )

    for task_dict in result.values():
        for recipe, trajectory in task_dict.items():
            task_dict[recipe] = sorted(trajectory, key=lambda pair: pair[0].n)

    return result


def draw_seed_resample_pattern(rng: np.random.Generator, n_seeds: int) -> np.ndarray:
    """One shared resample pattern for a (scale, replicate): `n_seeds`
    indices into [0, n_seeds), drawn with replacement. Applied identically
    to every recipe's own seed values at that scale via
    `apply_seed_resample`."""
    return rng.integers(0, n_seeds, size=n_seeds)


def apply_seed_resample(pattern: np.ndarray, seed_values: list[float]) -> float:
    """One recipe's seed-bootstrap value at one scale: the mean of its own
    seed values selected by the shared `pattern`."""
    if len(seed_values) != len(pattern):
        raise ValueError(
            f"pattern has {len(pattern)} indices but seed_values has "
            f"{len(seed_values)} -- every recipe must have the same seed count "
            "at a given scale for a shared resample pattern to apply (P1-01 "
            "confirmed this holds for every real cell; a mismatch here means "
            "that no longer holds and the two arms are no longer comparable)."
        )
    return float(np.mean([seed_values[i] for i in pattern]))


def draw_parametric_noise_z(rng: np.random.Generator) -> float:
    """One standard-normal draw for a (recipe, scale, replicate) -- callers
    give each recipe its OWN rng stream (never one shared z across recipes;
    see the module docstring). Applied to that recipe's point estimate and
    noise scale via `apply_parametric_noise`."""
    return float(rng.normal(0.0, 1.0))


def apply_parametric_noise(z: float, mu: float, sigma2_total: float) -> float:
    """One recipe's parametric-bootstrap value at one scale: its own
    seed-averaged point `mu`, perturbed by its own `z` scaled to its own
    noise standard deviation."""
    return mu + z * math.sqrt(max(sigma2_total, 0.0))


def seed_bootstrap_variance_inflation(n_seeds: int) -> float:
    """`n / (n - 1)`: the factor that turns an n-out-of-n seed-bootstrap
    variance into an estimate of the ORIGINAL estimator's sampling variance.

    Second-round external review of `bias_variance_decomposition`: the
    bootstrap replicate variance `v_hat` of a sample mean is the *plug-in*
    variance `s_plug^2 / n = ((n - 1) / n) * s^2 / n` -- a factor
    `(n - 1) / n` below the unbiased estimate of `Var(original mean) =
    sigma^2 / n`. With n = 3 real seeds (every DataDecide cell) that is a
    2/3 shrinkage, so treating `v_hat` as `Var(mu_hat_orig)` left
    `sigma2_extrap_hat` biased UPWARD by `Var(mu_hat_orig) / 3` even for an
    unbiased estimator with zero structural bias (reproduced: independent
    N(0,1) 3-observation datasets, exact target 0, B=200 -> mean
    `v_hat` 0.2225 vs true 1/3, mean unclipped correction +0.106 instead
    of 0).

    Exact for a sample mean. For a smooth (differentiable) fit of the seed
    means -- what every fitter here is -- it is the first-order (delta
    method) version of the same statement, because the bootstrap variance
    of a smooth statistic converges to the plug-in delta-method variance;
    it is an approximation at n = 3 and for non-smooth or boundary-pinned
    fits, not a proof, and is documented as such (docs/decisions.md).
    Only applies to resampling of the n observed seeds; a parametric
    bootstrap draws from an assumed noise model, not from an n-observation
    sample, so it uses no such factor (1.0).
    """
    if n_seeds < 2:
        raise ValueError(f"need at least 2 seeds to estimate a sampling variance, got {n_seeds}")
    return n_seeds / (n_seeds - 1)


def bias_variance_decomposition(
    replicate_predictions: list[float],
    mu_true: float,
    sigma2_target: float,
    *,
    variance_inflation: float = 1.0,
) -> dict:
    """`v_hat`, `bias_hat`, and `sigma2_extrap_hat` from B bootstrap
    replicate predictions of `mu_k(s*)`, per plan/02-phase1-datadecide.md
    P1-06 step 3:

        v_hat = Var_b[mu_hat^(b)]
        bias_hat = mean_b[mu_hat^(b)] - mu_true
        unclipped = bias_hat^2 - variance_inflation * v_hat - v_hat/B - sigma2_target
        sigma2_extrap_hat = max(0, unclipped)

    `bias_hat` is not a clean estimate of the true (structural,
    non-random) extrapolation bias -- it is `mean_pred - mu_true`, and
    `mean_pred` is itself a noisy proxy for the *original* (unresampled)
    fit's prediction `mu_hat_orig`, which is itself a noisy estimate of
    whatever the fitter's functional form converges to. Squaring it
    without correcting picks up extra variance terms on top of the genuine
    squared bias:

    - `v_hat / B`: `mean_pred` averages B bootstrap replicates each with
      variance `v_hat`, so it carries finite-B Monte Carlo noise around
      `mu_hat_orig` of its own -- this shrinks to 0 as B grows.
    - `variance_inflation * v_hat` (not `v_hat / B`): the sampling
      variance of the ORIGINAL estimator `mu_hat_orig`, which does NOT
      vanish as B grows. `v_hat` is a bootstrap estimate of it, and the
      bootstrap generally does not reproduce it exactly: for n-out-of-n
      resampling of n observed seeds it is short by `(n - 1) / n`, so
      callers pass `variance_inflation = n / (n - 1)`
      (`seed_bootstrap_variance_inflation`), 3/2 for three seeds. Default
      1.0 is correct only when `v_hat` is already the sampling variance
      (e.g. a parametric bootstrap drawing from the assumed noise model).
    - `sigma2_target`: noise in the ground-truth `mu_true` itself.

    **Unclipped vs clipped.** `sigma2_extrap_unclipped` is the (approximately)
    unbiased estimator of the squared structural bias `[E mu_hat_orig -
    mu_true]^2`, and is negative about half the time when the true bias is
    small; `sigma2_extrap_hat = max(0, .)` is the reported nonnegative
    *heuristic*, biased upward for small true bias (E[max(0, X)] > E[X]).
    Anything that averages `sigma2_extrap_hat` over many cells inherits that
    upward bias; averages meant to estimate the mean squared bias should use
    the unclipped field. Validated across many independent datasets, not
    just one hand-built series (`tests/test_bootstrap.py`).

    Generic over what "mu_true" and the replicate series mean -- called
    once per recipe for the marginal decomposition, and again on the
    per-replicate difference series `D_k^(b) = mu_hat_k*^(b) - mu_hat_k^(b)`
    (with `mu_true` = the true gap and `sigma2_target` = the *pairwise*
    target noise) for the pairwise decomposition the plan says "the
    decision actually depends on".

    Raises if fewer than 2 replicates are given -- a variance needs at
    least 2 points, and silently returning a degenerate 0 would look like
    a real (if boring) finding instead of "not enough successful
    replicates to say anything".
    """
    b = len(replicate_predictions)
    if b < 2:
        raise ValueError(f"need at least 2 replicate predictions, got {b}")
    if variance_inflation <= 0:
        raise ValueError(f"variance_inflation must be positive, got {variance_inflation}")

    mean_pred = sum(replicate_predictions) / b
    v_hat = sum((p - mean_pred) ** 2 for p in replicate_predictions) / (b - 1)
    bias_hat = mean_pred - mu_true
    sigma2_extrap_unclipped = bias_hat**2 - variance_inflation * v_hat - v_hat / b - sigma2_target
    sigma2_extrap_hat = max(0.0, sigma2_extrap_unclipped)

    return {
        "n_replicates": b,
        "mean_prediction": mean_pred,
        "v_hat": v_hat,
        "bias_hat": bias_hat,
        "sigma2_extrap_hat": sigma2_extrap_hat,
        "sigma2_extrap_unclipped": sigma2_extrap_unclipped,
        "variance_inflation": variance_inflation,
    }
