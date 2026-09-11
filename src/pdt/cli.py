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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pdt.bai.ets import SelectionResult, extrapolation_track_and_stop
from pdt.bai.oracle import DataDecideOracle, PullOracle, SyntheticOracle
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
        return DataDecideOracle(
            task=oracle_cfg["task"],
            metric_name=oracle_cfg.get("metric_name", "primary_metric"),
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
        result = run_selection(cfg)
        print(
            json.dumps(
                {
                    "outcome": result.outcome,
                    "recipe": result.recipe,
                    "compute_spent": result.compute_spent,
                    "n_pulls": result.n_pulls,
                    "certificate": result.certificate,
                },
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":  # pragma: no cover -- exercised via the `pdt` console script
    main()
