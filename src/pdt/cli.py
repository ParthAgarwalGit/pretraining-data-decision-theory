"""Command-line interface: `pdt select --config configs/my_selection.yaml`.

See plan/04-phase3-algorithm.md P3-08. Runs `extrapolation_track_and_stop`
(src/pdt/bai/ets.py) against one of the two built-in `PullOracle`
backends and prints the decision plus certificate as JSON.

**A custom data source (a real training loop, an internal eval harness,
anything that isn't `DataDecideOracle` or `SyntheticOracle`) is not
expressible in YAML** -- plug in your own `pull(recipe, scale, seed)` /
`cost(scale)` / `available_scales()` callback by calling
`extrapolation_track_and_stop` directly from Python instead of through
this CLI; any object implementing that three-method protocol works (see
`pdt.bai.oracle.PullOracle`, and the README quickstart). This CLI covers
the two backends this project ships, not a general plugin system.

**What the printed `outcome` means.** `"certified"` is a delta-level claim, and by
default (`certification: supported_only`, `variance_mode: known_sigma2`) it is printed ONLY
where the argument is proved: a linear model (`LogLinear`) whose fit is unclipped, on the
first check, before any adaptive pull, with the config's `sigma2` the KNOWN noise variance
and `eta` a valid bias bound. Every other time the stopping rule fires (adaptive tracking, a
nonlinear model such as the default `PowerLawN`) it prints `"recommended"`, with
`certificate["unmet_supported_conditions"]` saying why and NO error-probability claim.
`certification: assume_unproved_conditions` is your explicit acceptance of the unproved
conditions (linearization, adaptive-design independence): `"certified"` on any round, with
`certificate["guarantee"]` starting `assumed_unproved`. `variance_mode: hc0_heuristic`
always prints `"recommended"`. `"abstained"` makes no correctness claim
(see `certificate["reason"]`).

**The `datadecide` backend has a FINITE replicate pool** (`DataDecideOracle`'s
own real+pseudo seeds -- see its docstring), and `select`'s default
`max_rounds=5000` can exhaust it well before then (P3-05's own real
replay hit this within 60 rounds on a real task). The oracle is wrapped
in `pdt.bai.oracle.FiniteDataOracle`, so exhaustion is reported as a
distinct `{"outcome": "data_exhausted", ...}` result rather than an
unhandled crash -- it is never silently papered over by recycling an
already-observed value as fresh data. If you see this, lower
`max_rounds` in your config or pick an easier instance (see
`docs/when_to_trust_extrapolation.md`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pdt.bai.ets import SelectionResult, extrapolation_track_and_stop
from pdt.bai.oracle import (
    DataDecideOracle,
    DataExhaustedError,
    FiniteDataOracle,
    PullOracle,
    SyntheticOracle,
)
from pdt.scaling.base import Scale
from pdt.scaling.fitters import (
    ChinchillaND,
    ConstantExtrapolator,
    LogLinear,
    PowerLawC,
    PowerLawN,
    TwoStepLadder,
)

_MODEL_FACTORIES = {
    "PowerLawN": PowerLawN,
    "PowerLawC": PowerLawC,
    "ChinchillaND": ChinchillaND,
    "TwoStepLadder": TwoStepLadder,
    "LogLinear": LogLinear,
    "ConstantExtrapolator": ConstantExtrapolator,
}


def _scale(entry: dict[str, Any]) -> Scale:
    # float(...) explicitly, rather than trusting YAML's own type
    # inference: PyYAML 1.1's resolver does not recognize scientific
    # notation without an explicit exponent sign (`1.0e6` parses as the
    # *string* "1.0e6", not the float 1e6 -- only `1.0e+6` or plain
    # decimal notation like `1000000.0` are recognized) -- a real,
    # easy-to-hit authoring mistake found while testing this CLI against
    # its own example config. Python's float() parses "1.0e6" correctly,
    # so coercing explicitly here is strictly more permissive than
    # requiring users to know YAML's specific float grammar.
    return Scale(n=float(entry["n"]), d=float(entry["d"]))


def _build_oracle(cfg: dict[str, Any]) -> PullOracle:
    oracle_cfg = cfg["oracle"]
    kind = oracle_cfg["type"]
    if kind == "datadecide":
        # Wrapped in FiniteDataOracle: DataDecideOracle's real+pseudo
        # replicate pool is finite, and the default max_rounds=5000 below
        # essentially guarantees an adaptive run exhausts it eventually
        # (P3-05's own real replay hit this within 60 rounds) -- an
        # unwrapped oracle would surface that as a raw, unhandled
        # IndexError instead of the clean "exhausted" result `main()`
        # handles below (PR #34's review).
        return FiniteDataOracle(
            DataDecideOracle(
                task=oracle_cfg["task"],
                metric_name=oracle_cfg.get("metric_name", "primary_metric"),
            )
        )
    if kind == "synthetic":
        return SyntheticOracle(
            recipes=cfg["recipes"],
            scales=[_scale(s) for s in oracle_cfg["fit_scales"]],
            target_scale=_scale(cfg["target_scale"]),
            rng=np.random.default_rng(cfg.get("seed", 0)),
        )
    raise ValueError(f"unknown oracle.type {kind!r}; supported: 'datadecide', 'synthetic'")


def run_selection(cfg: dict[str, Any]) -> SelectionResult:
    oracle = _build_oracle(cfg)
    recipes = cfg["recipes"]
    candidate_scales = [_scale(s) for s in cfg["candidate_scales"]]
    target_scale = _scale(cfg["target_scale"])
    model_factory = _MODEL_FACTORIES[cfg.get("model", "PowerLawN")]

    eta_cfg = cfg["eta"]
    eta = (
        {r: float(v) for r, v in eta_cfg.items()}
        if isinstance(eta_cfg, dict)
        else {r: float(eta_cfg) for r in recipes}
    )
    sigma2_value = float(cfg["sigma2"])

    return extrapolation_track_and_stop(
        oracle,
        recipes,
        candidate_scales,
        target_scale,
        delta=float(cfg["delta"]),
        eta=eta,
        sigma2=lambda _s: sigma2_value,
        model_factory=model_factory,
        max_rounds=int(cfg.get("max_rounds", 5000)),
        solver_n_restarts=int(cfg.get("solver_n_restarts", 1)),
        solver_n_iter=int(cfg.get("solver_n_iter", 200)),
        min_pulls_per_pair=int(cfg.get("min_pulls_per_pair", 1)),
        rng=np.random.default_rng(cfg.get("seed", 0)),
        variance_mode=str(cfg.get("variance_mode", "known_sigma2")),
        certification=str(cfg.get("certification", "supported_only")),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pdt")
    subparsers = parser.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser(
        "select", help="run Extrapolation-Track-and-Stop on a configured instance"
    )
    select_parser.add_argument("--config", required=True, help="path to a YAML config")
    args = parser.parse_args(argv)

    if args.command == "select":
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        try:
            result = run_selection(cfg)
            output = {
                "outcome": result.outcome,
                "recipe": result.recipe,
                "compute_spent": result.compute_spent,
                "n_pulls": result.n_pulls,
                "certificate": result.certificate,
            }
        except DataExhaustedError as exc:
            # A finite real-data source (datadecide) ran out of
            # replicates before the algorithm resolved -- an honest,
            # distinct result, never fabricated as a certification or
            # silently crashed as an unhandled IndexError (PR #34's
            # review). Lower max_rounds or pick an easier instance (see
            # docs/when_to_trust_extrapolation.md) if you see this.
            output = {
                "outcome": "data_exhausted",
                "recipe": None,
                "compute_spent": None,
                "n_pulls": None,
                "certificate": {"reason": str(exc)},
            }
        print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover -- exercised via the `pdt` console script
    main()
