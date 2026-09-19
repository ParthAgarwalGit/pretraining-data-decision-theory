"""Exact binomial confidence intervals for simulation error rates."""

from __future__ import annotations

from scipy.stats import beta


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact (Clopper-Pearson) `1 - alpha` interval for a binomial
    proportion after `k` successes in `n` trials.

    Used to judge a delta-correctness guarantee from a finite simulation: the
    guarantee bounds the JOINT probability `P[certified AND wrong] <= delta`, so a
    violation is detected only when the interval's LOWER end exceeds `delta` -- a
    single wrong certification among many runs (or a conditional rate computed
    from one certified run) is not evidence of a violation."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi
