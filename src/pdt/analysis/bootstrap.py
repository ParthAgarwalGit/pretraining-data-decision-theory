"""Bootstrap resampling for the bias/variance decomposition.

See plan/02-phase1-datadecide.md task P1-06. Two resampling schemes, both
producing one perturbed trajectory per (recipe, scale) per replicate:

- **Seed bootstrap**: resample the (up to 3) seed observations at a scale
  with replacement, per the plan's literal "resample seeds with
  replacement at each fitted scale".
- **Parametric bootstrap**: perturb the seed-averaged point by Gaussian
  noise with variance from the P1-05 noise model.

Both schemes draw ONE shared random pattern per (scale, replicate) --
shared across every recipe, not redrawn per recipe -- because the plan
requires the pairwise difference statistic `D_k = mu_hat_k*(s*) -
mu_hat_k(s*)` to be "computed from the same bootstrap replicate (so the
correlation between the two arms' fits is preserved)". If each recipe drew
independent randomness, there would be no shared "replicate" for a
correlation to survive in the first place -- two recipes evaluated on the
same benchmark instances at the same scale share real correlated noise
(a hard eval instance is hard for every recipe), and this is how that
survives into the resampled data for `D_k` to (partially) cancel it, which
is the effect the plan calls "a real and important effect that the
marginal version misses." `draw_seed_resample_pattern` /
`draw_parametric_noise_z` are the shared per-(scale, replicate) draws;
`apply_seed_resample` / `apply_parametric_noise` apply that same shared
pattern to one recipe's own observed values.
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
    """One shared standard-normal draw for a (scale, replicate). Applied to
    every recipe's own point estimate and noise scale via
    `apply_parametric_noise`."""
    return float(rng.normal(0.0, 1.0))


def apply_parametric_noise(z: float, mu: float, sigma2_total: float) -> float:
    """One recipe's parametric-bootstrap value at one scale: its own
    seed-averaged point `mu`, perturbed by the shared `z` scaled to its own
    noise standard deviation."""
    return mu + z * math.sqrt(max(sigma2_total, 0.0))


def bias_variance_decomposition(
    replicate_predictions: list[float], mu_true: float, sigma2_target: float
) -> dict:
    """`v_hat`, `bias_hat`, and `sigma2_extrap_hat` from B bootstrap
    replicate predictions of `mu_k(s*)`, per plan/02-phase1-datadecide.md
    P1-06 step 3:

        v_hat = Var_b[mu_hat^(b)]
        bias_hat = mean_b[mu_hat^(b)] - mu_true
        sigma2_extrap_hat = max(0, bias_hat^2 - v_hat/B - sigma2_target)

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

    mean_pred = sum(replicate_predictions) / b
    v_hat = sum((p - mean_pred) ** 2 for p in replicate_predictions) / (b - 1)
    bias_hat = mean_pred - mu_true
    sigma2_extrap_hat = max(0.0, bias_hat**2 - v_hat / b - sigma2_target)

    return {
        "n_replicates": b,
        "mean_prediction": mean_pred,
        "v_hat": v_hat,
        "bias_hat": bias_hat,
        "sigma2_extrap_hat": sigma2_extrap_hat,
    }
