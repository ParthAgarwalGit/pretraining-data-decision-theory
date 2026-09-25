"""PullOracle and its three implementations -- plan/04-phase3-algorithm.md
task P3-01.

The algorithm code (P3-02/P3-03) never knows which world it is in: it
only calls `.pull(recipe, scale, seed)`, `.cost(scale)`,
`.available_scales()` on whatever oracle it was handed.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np
import polars as pl

from pdt.data.frame import build_frame
from pdt.scaling.base import Scale


class PullOracle(Protocol):
    def pull(self, recipe: str, scale: Scale, seed: int) -> float: ...
    def cost(self, scale: Scale) -> float: ...
    def available_scales(self) -> list[Scale]: ...


def _stable_seed(*parts: object) -> int:
    """A deterministic, process/platform-independent RNG seed derived from
    `parts` -- Python's built-in `hash()` is salted per-process for str
    (security feature, PYTHONHASHSEED), so it cannot be used here: the
    whole point is that `pull(recipe, scale, seed)` returns the identical
    value on every call, including across separate processes/sessions."""
    key = "|".join(repr(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big")


class SyntheticOracle:
    """Draws from `g(theta_k, s) + h_k(s) + noise`, calibrated to the
    P1-05/P1-06 estimates (not an arbitrary Gaussian toy):

    - Noise variance ~1e-4 (accuracy^2 units), matching P1-05's measured
      sigma2_seed range (results/p1_05_noise.json noise_vs_scale_summary:
      median_sigma2_seed roughly 1e-4 across sizes/tasks) -- and NOT tied
      monotonically to scale (setup.tex Remark 1: P1-05 found no such
      trend), so noise is drawn once per recipe and held constant across
      scale here, not shrunk with N.
    - sqrt(sigma2_extrap) (bias magnitude at the target) drawn from
      [0.03, 0.20], matching P1-06's measured sigma2_extrap range across
      6 fitters x 3 designs (results/p1_06_decomposition.json
      ratio_vs_compute: median_sigma2_extrap_hat ranges ~0.0023 to
      ~0.030, i.e. bias magnitude ~0.048 to ~0.17); the range here is
      widened slightly on both ends to cover fitters/designs P1-06 didn't
      report a ratio_vs_compute cell for exactly, not narrowed to fit.

    The underlying family g(theta,s) = E - A*N^-alpha matches
    `src/pdt/scaling/fitters.py`'s PowerLawN shape (not reusing that
    class directly, since SyntheticOracle needs to draw *many* recipes'
    worth of ground truth cheaply, not fit one).
    """

    _SIGMA2_NOISE_RANGE = (5e-5, 2e-4)
    _BIAS_MAGNITUDE_RANGE = (0.03, 0.20)

    def __init__(
        self,
        recipes: list[str],
        scales: list[Scale],
        target_scale: Scale,
        rng: np.random.Generator,
    ):
        if not recipes:
            raise ValueError("need at least one recipe")
        if not scales:
            raise ValueError("need at least one scale")
        self._scales = list(scales)
        self._target_scale = target_scale
        # An instance-specific salt, drawn once from the constructor's own
        # rng and folded into every pull()'s noise seed below. Without
        # this, pull()'s seed depended only on (recipe, scale, seed) --
        # NOT on which SyntheticOracle instance is calling it -- so two
        # separate instances (e.g. different "independent trials" in an
        # experiment, each with its own constructor rng draw for
        # e/a/alpha/bias/sigma2_noise) would still draw the identical
        # standardized noise innovation for the same (recipe, scale,
        # seed), correlating what are supposed to be independent
        # simulations. Two instances built from the SAME constructor seed
        # still draw the same salt (it's the first thing pulled from that
        # seed's own deterministic stream), so `pull()` stays exactly
        # reproducible given the same construction seed -- only
        # independence ACROSS differently-seeded instances is what this
        # fixes.
        self._instance_salt = int(rng.integers(0, 2**63))
        self._params: dict[str, dict] = {}
        for recipe in recipes:
            e = rng.uniform(0.3, 0.9)
            a = rng.uniform(0.5, 5.0)
            alpha = rng.uniform(0.05, 0.6)
            bias_at_target = rng.uniform(*self._BIAS_MAGNITUDE_RANGE) * rng.choice([-1.0, 1.0])
            sigma2_noise = rng.uniform(*self._SIGMA2_NOISE_RANGE)
            self._params[recipe] = {
                "e": e,
                "a": a,
                "alpha": alpha,
                "bias_at_target": bias_at_target,
                "sigma2_noise": sigma2_noise,
            }

    def _h(self, recipe: str, scale: Scale) -> float:
        """Bounded perturbation, zero at every scale strictly below the
        smallest fitted scale this oracle was built with, rising smoothly
        to `bias_at_target` at (and only at) the target scale -- the same
        `phi(x) = clip(x,0,1)` bump shape as Theorem 2 Part B's
        construction (theorem2_lower_bound.tex), so a SyntheticOracle
        instance can, if desired, be used to build genuinely hard
        (near-impossible-regime) instances by setting bias_at_target
        close to a competing recipe's own true gap -- not just generic
        smooth misspecification.
        """
        params = self._params[recipe]
        s_max_fit = max(s.n for s in self._scales)
        s_star = self._target_scale.n
        if scale.n <= s_max_fit or s_star <= s_max_fit:
            return 0.0
        x = (scale.n - s_max_fit) / (s_star - s_max_fit)
        return params["bias_at_target"] * float(np.clip(x, 0.0, 1.0))

    def _true_mean(self, recipe: str, scale: Scale) -> float:
        p = self._params[recipe]
        return p["e"] - p["a"] * scale.n ** (-p["alpha"]) + self._h(recipe, scale)

    def pull(self, recipe: str, scale: Scale, seed: int) -> float:
        if recipe not in self._params:
            raise KeyError(f"unknown recipe {recipe!r}")
        mean = self._true_mean(recipe, scale)
        sigma = float(np.sqrt(self._params[recipe]["sigma2_noise"]))
        draw_rng = np.random.default_rng(
            _stable_seed(self._instance_salt, recipe, scale.n, scale.d, seed)
        )
        return float(mean + draw_rng.normal(0.0, sigma))

    def true_value_at_target(self, recipe: str) -> float:
        """The noise-free, unobservable ground truth at s* -- for scoring
        a policy's decision in simulation, never for the policy itself to
        read."""
        return self._true_mean(recipe, self._target_scale)

    def cost(self, scale: Scale) -> float:
        return scale.compute

    def available_scales(self) -> list[Scale]:
        return list(self._scales)


class DataDecideOracle:
    """Table lookup into the P1-01/P1-05 tidy frame for one fixed
    (task, metric_name) pair -- the algorithm identifies the best recipe
    for *one* decision-relevant metric at a time, so task/metric are
    constructor arguments, not per-pull ones (matching the plan's literal
    3-arg `pull(recipe, scale, seed)` signature, which has no task slot).

    Seed handling: per P0-06's finding, auxiliary-seed *labels* differ by
    scale ("small aux 2/3" below 1B, "large aux 2/3" at 1B) -- never
    hardcoded here. For each (recipe, scale), the seed labels actually
    present in the data are discovered, sorted for a deterministic order,
    and indexed 0, 1, 2, ... by integer `seed`. If more replicates are
    requested than real seeds exist, falls back to a documented
    pseudo-replicate pool built from each real seed's own last few
    checkpoints (excluding the final one already used as that seed's
    primary value) -- clearly a *pseudo*-replicate (correlated with its
    parent seed's own noise, not independent), used only if the caller
    asks for more replicates than P0-06 confirmed are really there (3,
    everywhere in DataDecide).
    """

    _N_PSEUDO_CHECKPOINTS = 4  # matches pdt.analysis.noise's own window

    def __init__(
        self,
        task: str,
        metric_name: str = "primary_metric",
        source: str = "macro_avg",
    ):
        self.task = task
        self.metric_name = metric_name
        long_frame = build_frame(source=source, metrics=(metric_name,))
        subset = long_frame.filter(
            (pl.col("task") == task) & (pl.col("metric_name") == metric_name)
        )
        if subset.height == 0:
            raise ValueError(
                f"no rows for task={task!r}, metric_name={metric_name!r}, source={source!r} "
                "-- check the task/metric names are real for this source."
            )
        self._final = subset.filter(pl.col("is_final")).sort(["recipe", "params_str", "seed"])
        self._all = subset.sort(["recipe", "params_str", "seed", "step"])

        self._scale_by_params_str: dict[str, Scale] = {}
        for row in (
            self._final.group_by("params_str", maintain_order=True)
            .agg(pl.col("params_num").first(), pl.col("tokens").mean())
            .iter_rows(named=True)
        ):
            self._scale_by_params_str[row["params_str"]] = Scale(
                n=row["params_num"], d=row["tokens"]
            )
        self._params_str_by_n = {s.n: ps for ps, s in self._scale_by_params_str.items()}

        # (recipe, params_str) -> sorted list of (seed_label, metric_value)
        # at the final checkpoint -- the primary, real replicates.
        self._primary: dict[tuple[str, str], list[tuple[str, float]]] = {}
        for (recipe, params_str), rows in self._final.group_by(
            ["recipe", "params_str"], maintain_order=True
        ):
            pairs = sorted(zip(rows["seed"].to_list(), rows["metric_value"].to_list(), strict=True))
            self._primary[(recipe, params_str)] = pairs

    def _params_str_for(self, scale: Scale) -> str:
        if scale.n not in self._params_str_by_n:
            raise KeyError(
                f"scale with n={scale.n:.4g} is not one of this oracle's "
                f"available_scales() -- pull() only accepts scales this "
                f"oracle actually has data for."
            )
        return self._params_str_by_n[scale.n]

    def _pseudo_pool(self, recipe: str, params_str: str) -> list[float]:
        rows = self._all.filter((pl.col("recipe") == recipe) & (pl.col("params_str") == params_str))
        ranked = rows.with_columns(
            pl.col("step").rank(method="ordinal", descending=True).over("seed").alias("_rank")
        )
        tail = ranked.filter(
            (pl.col("_rank") > 1) & (pl.col("_rank") <= self._N_PSEUDO_CHECKPOINTS)
        ).sort(["seed", "step"])
        return tail["metric_value"].to_list()

    def pull(self, recipe: str, scale: Scale, seed: int) -> float:
        params_str = self._params_str_for(scale)
        key = (recipe, params_str)
        if key not in self._primary:
            raise KeyError(f"no data for recipe={recipe!r} at params_str={params_str!r}")
        primary = self._primary[key]
        if seed < len(primary):
            return float(primary[seed][1])

        pseudo = self._pseudo_pool(recipe, params_str)
        pseudo_idx = seed - len(primary)
        if pseudo_idx < len(pseudo):
            return float(pseudo[pseudo_idx])

        raise IndexError(
            f"seed={seed} exceeds available replicates for ({recipe!r}, {params_str!r}): "
            f"{len(primary)} real seeds + {len(pseudo)} pseudo-replicates from nearby "
            f"checkpoints."
        )

    def cost(self, scale: Scale) -> float:
        return scale.compute

    def available_scales(self) -> list[Scale]:
        return list(self._scale_by_params_str.values())


class LiveTrainingOracle:
    """Phase 4 only. Stubbed now so the PullOracle interface is fixed
    before P4 starts -- see plan/05-phase4-training.md."""

    def pull(self, recipe: str, scale: Scale, seed: int) -> float:
        raise NotImplementedError("LiveTrainingOracle is implemented in Phase 4")

    def cost(self, scale: Scale) -> float:
        raise NotImplementedError("LiveTrainingOracle is implemented in Phase 4")

    def available_scales(self) -> list[Scale]:
        raise NotImplementedError("LiveTrainingOracle is implemented in Phase 4")
