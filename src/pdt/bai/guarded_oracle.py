"""A `PullOracle` wrapper that refuses to reveal the target scale --
plan/04-phase3-algorithm.md P3-05's leakage guard.

P3-05's offline replay on real DataDecide data must never let an
algorithm (or a baseline, or this script itself) look at `s* = 1B` data
except when explicitly scoring the final decision after the fact. The
plan is explicit that this must be *enforced in code*, not merely
followed by discipline: "Enforce this in code with a guarded oracle
that raises if `s*` is pulled. Do not rely on discipline; rely on the
assertion."
"""

from __future__ import annotations

from pdt.bai.oracle import PullOracle
from pdt.scaling.base import Scale


class TargetScaleLeakageError(RuntimeError):
    """Raised the instant anything tries to pull, cost, or list the
    guarded target scale through a `GuardedOracle` -- never caught and
    silently worked around; a leak here would invalidate the whole
    replay's "free real-data result" claim."""


class GuardedOracle:
    """Wraps any `PullOracle`, forwarding every call except: `pull()`
    and `cost()` raise `TargetScaleLeakageError` if given `target_scale`,
    and `available_scales()` never includes it -- so an algorithm cannot
    even discover the target scale exists as a pullable option, let
    alone observe it."""

    def __init__(self, inner: PullOracle, target_scale: Scale):
        self._inner = inner
        self._target_scale = target_scale

    def pull(self, recipe: str, scale: Scale, seed: int) -> float:
        if scale == self._target_scale:
            raise TargetScaleLeakageError(
                f"attempted to pull the guarded target scale {scale!r} for "
                f"recipe {recipe!r} -- the replay must never observe target-scale "
                f"data except when scoring the final decision, outside this oracle."
            )
        return self._inner.pull(recipe, scale, seed)

    def cost(self, scale: Scale) -> float:
        if scale == self._target_scale:
            raise TargetScaleLeakageError(
                f"attempted to price the guarded target scale {scale!r} -- even "
                f"asking its cost is treated as a leak, since a caller could use "
                f"the mere fact that it priced successfully as information."
            )
        return self._inner.cost(scale)

    def available_scales(self) -> list[Scale]:
        return [s for s in self._inner.available_scales() if s != self._target_scale]
