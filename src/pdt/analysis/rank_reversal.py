"""Rank-reversal classification: does a recipe pair's winner change across scale?

See plan/02-phase1-datadecide.md task P1-09 -- evidence for the
impossibility regime (Claim 2). For a recipe pair (k, k'), the sign of
`mu_k(s) - mu_k'(s)` can differ from one scale to the next just because it
sits within noise near a true near-tie, not because the *true* ranking
actually flips. `classify_pair_trajectory` separates the two using the
same effect-size-vs-threshold logic `ground_truth.py`'s ambiguity flag and
`decision_accuracy.pairwise_decision_accuracy`'s tie exclusion already use
(reusing `AMBIGUOUS_EFFECT_SIZE_THRESHOLD` by default, so "resolved" means
the same thing everywhere in this project).
"""

from __future__ import annotations

import math

from pdt.analysis.ground_truth import AMBIGUOUS_EFFECT_SIZE_THRESHOLD


def pair_effect_size(
    mu_a: float, mu_b: float, sigma2_a: float, sigma2_b: float, n_seeds: int
) -> float | None:
    """Standardized gap for one recipe pair at one scale: `(mu_a - mu_b)`
    divided by the standard error of that gap, `sqrt((sigma2_a + sigma2_b)
    / n_seeds)`. Returns `None` when the pooled variance is zero (both
    recipes have zero seed variance -- degenerate, not expected in real
    data, but handled rather than dividing by zero).
    """
    pooled_variance = (sigma2_a + sigma2_b) / n_seeds
    if pooled_variance == 0:
        return None
    return (mu_a - mu_b) / math.sqrt(pooled_variance)


def resolved_sign(
    effect_size: float | None, *, threshold: float = AMBIGUOUS_EFFECT_SIZE_THRESHOLD
) -> int | None:
    """+1 / -1 if `effect_size` clears `threshold` in magnitude, else
    `None` (the gap at this scale is within noise -- unresolved)."""
    if effect_size is None or abs(effect_size) < threshold:
        return None
    return 1 if effect_size > 0 else -1


def classify_pair_trajectory(sizes_ascending: list[str], resolved_signs: dict[str, int]) -> dict:
    """Classify a (k, k') pair's whole trajectory across the size ladder.

    `resolved_signs`: `{size_label: +1 or -1}` for every scale where the
    pair's gap clears the noise threshold; a scale absent from this dict
    is within noise at that point and doesn't count either way.

    - **within_noise**: no scale resolves the pair at all -- the two
      recipes are statistically indistinguishable everywhere on this
      ladder for this task.
    - **stable**: every resolved scale agrees on the sign.
    - **reversing**: at least two resolved scales disagree -- a genuine
      reversal, not just noise. `crossing_size` is the first resolved
      scale (scanning from the largest size backward) whose sign departs
      from the resolved scale immediately before it -- the boundary of the
      *last* (closest to `s*`) reversal on the ladder.
    """
    resolved = [(s, resolved_signs[s]) for s in sizes_ascending if s in resolved_signs]
    if not resolved:
        return {"classification": "within_noise", "crossing_size": None, "n_resolved_sizes": 0}

    if all(sign == resolved[0][1] for _, sign in resolved):
        return {
            "classification": "stable",
            "crossing_size": None,
            "n_resolved_sizes": len(resolved),
        }

    crossing_size = None
    for i in range(len(resolved) - 1, 0, -1):
        if resolved[i][1] != resolved[i - 1][1]:
            crossing_size = resolved[i][0]
            break

    return {
        "classification": "reversing",
        "crossing_size": crossing_size,
        "n_resolved_sizes": len(resolved),
    }
