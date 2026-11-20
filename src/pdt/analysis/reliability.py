"""Reliability of small-scale data decisions: target noise vs proxy noise (Phase 7, N7-02).

Every estimator works on ``(R, 3)`` value matrices -- rows are recipes, columns are the
three seeds of one (task, size) cell -- plus a length-``R`` vector of recipe ids. A
*decision* is the sign of the difference between two recipes' means; the decisions are
all unordered pairs ``i < j`` whose recipe ids differ. Comparing ids (not values) keeps
duplicated recipes in a recipe-cluster bootstrap resample from ever being paired with
themselves. Exact ties (sign 0) on either side are excluded from an agreement, and an
agreement over zero scorable pairs is NaN.

Estimands (target = the large scale, e.g. 1B; proxy = a smaller scale):

- ``retest_agreement``: P(sign of one target seed's gap == sign of the mean of the other two
  seeds' gaps), averaged over the held-out seed. A test-retest reliability of the target
  ranking -- *not* a ceiling on proxy accuracy (both sides are noisy).
- ``proxy_agreement``: the same agreement with the 1-seed target replaced by a k-seed
  proxy mean, scored against the *same* held-out 2-seed targets, so it is directly
  comparable with ``retest_agreement``.
- ``stratified_same_target``: proxy-vs-target agreement by bins of the target gap's
  |z|, where binning and scoring use the same 3-seed target. Biased in the middle bins
  (selection on the noisy target), kept only for comparison.
- ``stratified_heldout``: circularity-safe version -- bins come from two target seeds
  and the proxy is scored against the third, held-out seed.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy import stats

DEFAULT_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, float("inf")),
)


# --------------------------------------------------------------------------- pairs


def pair_index(ids: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Positions ``(i, j)``, ``i < j``, of all pairs whose recipe ids differ."""
    ids = np.asarray(ids)
    i, j = np.triu_indices(len(ids), 1)
    keep = ids[i] != ids[j]
    return i[keep], j[keep]


def pair_signs(values: np.ndarray, pairs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Sign of ``values[i] - values[j]`` for every pair."""
    i, j = pairs
    return np.sign(values[i] - values[j])


def _agree_counts(p: np.ndarray, q: np.ndarray, mask: np.ndarray | None = None):
    ok = (p != 0) & (q != 0)
    if mask is not None:
        ok &= mask
    return int(np.sum(p[ok] == q[ok])), int(np.sum(ok))


def sign_agreement(p: np.ndarray, q: np.ndarray) -> float:
    """Fraction of pairs with equal signs, excluding pairs tied on either side."""
    hit, n = _agree_counts(p, q)
    return hit / n if n else float("nan")


def _check(m: np.ndarray, ids: Sequence) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    if m.ndim != 2 or m.shape[1] != 3 or m.shape[0] != len(ids):
        raise ValueError(f"expected an (R, 3) matrix matching ids, got {m.shape}")
    if np.isnan(m).any():
        raise ValueError("value matrix contains NaN")
    return m


def pooled_sd(m: np.ndarray) -> float:
    """sqrt of the mean over recipes of the ddof=1 variance across the given seeds."""
    return float(np.sqrt(np.mean(np.var(m, axis=1, ddof=1))))


# ------------------------------------------------------------- (a), (b) agreement


def retest_agreement(target: np.ndarray, ids: Sequence) -> dict:
    """(a) Target test-retest: one held-out seed vs the mean of the other two."""
    t = _check(target, ids)
    pairs = pair_index(ids)
    per_seed = []
    for c in range(3):
        rest = [s for s in range(3) if s != c]
        per_seed.append(
            sign_agreement(pair_signs(t[:, c], pairs), pair_signs(t[:, rest].mean(1), pairs))
        )
    return {"mean": float(np.nanmean(per_seed)), "per_heldout_seed": per_seed}


def proxy_agreement(proxy: np.ndarray, target: np.ndarray, ids: Sequence, k: int) -> float:
    """(b) k-seed proxy mean vs the held-out 2-seed target means used by ``retest_agreement``.

    Averaged over all ``C(3, k)`` proxy seed combinations and the 3 held-out target seeds.
    Proxy seeds and target seeds are different trained models, so they are independent.
    """
    p, t = _check(proxy, ids), _check(target, ids)
    pairs = pair_index(ids)
    tgt = [pair_signs(t[:, [s for s in range(3) if s != c]].mean(1), pairs) for c in range(3)]
    vals = []
    for combo in itertools.combinations(range(3), k):
        ps = pair_signs(p[:, list(combo)].mean(1), pairs)
        vals.extend(sign_agreement(ps, ts) for ts in tgt)
    return float(np.nanmean(vals))


# ------------------------------------------------------------ (c), (d) stratified


def _bin_masks(z: np.ndarray, bins) -> list[np.ndarray]:
    return [(z >= lo) & (z < hi) for lo, hi in bins]


def stratified_same_target(
    proxy: np.ndarray, target: np.ndarray, ids: Sequence, bins=DEFAULT_BINS
) -> dict:
    """(c) BIASED: bins by the 3-seed target |z| and scores against that same target.

    ``z = |gap of 3-seed target means| / (pooled_sd * sqrt(2/3))``. Selecting pairs on the
    same noisy target that scores them biases the middle bins; use ``stratified_heldout``.
    """
    p, t = _check(proxy, ids), _check(target, ids)
    pairs = pair_index(ids)
    tm = t.mean(1)
    i, j = pairs
    z = np.abs(tm[i] - tm[j]) / (pooled_sd(t) * np.sqrt(2.0 / 3.0))
    ts, ps = pair_signs(tm, pairs), pair_signs(p.mean(1), pairs)
    agree, counts = [], []
    for mask in _bin_masks(z, bins):
        hit, n = _agree_counts(ps, ts, mask)
        agree.append(hit / n if n else float("nan"))
        counts.append(n)
    return {"agreement": agree, "n_pairs": counts, "bins": [list(b) for b in bins]}


def stratified_heldout_single(
    proxy: np.ndarray, target: np.ndarray, ids: Sequence, c: int, bins=DEFAULT_BINS
) -> dict:
    """(d) for one held-out target seed ``c``; returns raw counts per bin.

    Bins use only the other two target seeds ``{a, b}``:
    ``z = |gap of the {a,b} means| / sd_ab`` (the SE of a difference of two 2-seed means is
    ``sd * sqrt(2/2)``). The 3-seed proxy is scored against seed ``c`` alone; the
    target-internal consistency is ``sign(mean over {a,b})`` against seed ``c``.
    """
    p, t = _check(proxy, ids), _check(target, ids)
    pairs = pair_index(ids)
    ab = [s for s in range(3) if s != c]
    m_ab = t[:, ab].mean(1)
    i, j = pairs
    z = np.abs(m_ab[i] - m_ab[j]) / pooled_sd(t[:, ab])
    sc = pair_signs(t[:, c], pairs)
    ps, abs_ = pair_signs(p.mean(1), pairs), pair_signs(m_ab, pairs)
    out = {"proxy_hit": [], "proxy_n": [], "internal_hit": [], "internal_n": []}
    for mask in _bin_masks(z, bins):
        hit, n = _agree_counts(ps, sc, mask)
        out["proxy_hit"].append(hit)
        out["proxy_n"].append(n)
        hit, n = _agree_counts(abs_, sc, mask)
        out["internal_hit"].append(hit)
        out["internal_n"].append(n)
    return out


def stratified_heldout(
    proxy: np.ndarray, target: np.ndarray, ids: Sequence, bins=DEFAULT_BINS
) -> dict:
    """(d) Circularity-safe stratification, pooled over the three held-out seeds.

    Per bin: proxy agreement and target-internal agreement are pooled hit/scored counts
    summed over the held-out seeds (``n_pairs`` counts pair-instances, i.e. up to 3 per
    pair). Empty bins give NaN.
    """
    tot = {"proxy_hit": 0, "proxy_n": 0, "internal_hit": 0, "internal_n": 0}
    tot = {k: np.zeros(len(bins), dtype=int) for k in tot}
    for c in range(3):
        one = stratified_heldout_single(proxy, target, ids, c, bins)
        for key in tot:
            tot[key] += np.asarray(one[key])

    def ratio(h, n):
        return [float(a / b) if b else float("nan") for a, b in zip(h, n, strict=True)]

    return {
        "agreement": ratio(tot["proxy_hit"], tot["proxy_n"]),
        "target_internal_agreement": ratio(tot["internal_hit"], tot["internal_n"]),
        "n_pairs": tot["proxy_n"].tolist(),
        "bins": [list(b) for b in bins],
    }


# ---------------------------------------------------------------- (e) bootstrap


def cluster_bootstrap_indices(n_recipes: int, B: int = 1000, seed: int = 0) -> np.ndarray:
    """One ``(B, n_recipes)`` matrix of recipe-position resamples (with replacement).

    Apply the SAME matrix to every task, size and statistic so that paired differences
    and cross-task means are computed on identical resamples.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_recipes, size=(B, n_recipes))


def bootstrap(stat_fn: Callable[[np.ndarray], object], indices: np.ndarray) -> np.ndarray:
    """Evaluate ``stat_fn(idx)`` for every resample row; returns an array (B, ...)."""
    return np.asarray([np.asarray(stat_fn(idx), dtype=float) for idx in indices])


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """NaN-aware percentile interval over the first axis."""
    a = (1.0 - level) / 2.0
    lo, hi = np.nanpercentile(np.asarray(samples, dtype=float), [100 * a, 100 * (1 - a)], 0)
    return lo.tolist(), hi.tolist()


# --------------------------------------------------------- (f) seeds vs scale


def seeds_vs_scale(
    values_by_size: Mapping[str, np.ndarray],
    compute_by_size: Mapping[str, float],
    target: str,
    ids: Sequence,
    exclude: Sequence[str] = ("750M",),
) -> dict:
    """(f) Accuracy vs cost of k-seed proxies, k in {1, 2, 3}, against the target.

    ``cost = k * compute(size) / compute(target)``; accuracy is ``proxy_agreement``.
    The Pareto frontier keeps points (cost-sorted) that beat every cheaper point; sizes
    in ``exclude`` are reported but left out of the frontier.
    """
    t = values_by_size[target]
    points = []
    for size, mat in values_by_size.items():
        if size == target:
            continue
        for k in (1, 2, 3):
            points.append(
                {
                    "size": size,
                    "k": k,
                    "cost": k * compute_by_size[size] / compute_by_size[target],
                    "accuracy": proxy_agreement(mat, t, ids, k),
                    "excluded": size in exclude,
                }
            )
    points.sort(key=lambda d: (d["cost"], d["size"], d["k"]))
    best, frontier = -np.inf, []
    for pnt in points:
        if not pnt["excluded"] and pnt["accuracy"] > best:
            best = pnt["accuracy"]
            frontier.append({"size": pnt["size"], "k": pnt["k"]})
    return {
        "points": points,
        "frontier": frontier,
        "excluded_sizes": list(exclude),
        "target_test_retest": {"cost": 1.0, "accuracy": retest_agreement(t, ids)["mean"]},
    }


def matched_comparison(
    values_by_size: Mapping[str, np.ndarray],
    compute_by_size: Mapping[str, float],
    target: str,
    ids: Sequence,
    a: tuple[str, int],
    b: tuple[str, int],
) -> dict:
    """Accuracy and cost of two (size, k) arms; ``b`` may be ``(target, 1)``, meaning the
    target's own test-retest. Costs are reported because arms are only approximately
    matched."""
    t = values_by_size[target]

    def arm(size, k):
        acc = (
            retest_agreement(t, ids)["mean"]
            if size == target
            else proxy_agreement(values_by_size[size], t, ids, k)
        )
        return k * compute_by_size[size] / compute_by_size[target], acc

    ca, aa = arm(*a)
    cb, ab = arm(*b)
    return {"cost_a": ca, "acc_a": aa, "cost_b": cb, "acc_b": ab, "diff": aa - ab}


# ------------------------------------------------------------ (g), (h) noise model


def brown_forsythe(m: np.ndarray) -> float:
    """(g) Brown-Forsythe (median-centred Levene) p-value for equal seed variance across
    recipes; each recipe's 3 seeds is one group. Power is very low with n = 3 per group,
    so a non-rejection is weak evidence of homoscedasticity."""
    m = np.asarray(m, dtype=float)
    dev = np.abs(m - np.median(m, axis=1, keepdims=True))
    if np.allclose(dev, dev[0, 0]):
        return float("nan")
    return float(stats.levene(*m, center="median").pvalue)


def log_variance_dispersion(m: np.ndarray, n_sim: int = 2000, seed: int = 0) -> dict:
    """(g') Heteroscedasticity across recipes with real power at n = 3 seeds per recipe.

    Statistic: the across-recipe variance of ``log`` of each recipe's ddof=1 seed
    variance. Under homoscedastic Gaussian seed noise that variance does not depend on
    sigma, so its null distribution is simulated exactly (``n_sim`` draws, fixed seed).
    Returns the statistic, its null median, the ratio, and a one-sided Monte-Carlo
    p-value ``(1 + #null >= obs) / (1 + n_sim)``. Assumes Gaussian seed noise.
    """
    m = np.asarray(m, dtype=float)
    r, c = m.shape
    v = np.var(m, axis=1, ddof=1)
    if np.any(v <= 0):
        return {
            "stat": float("nan"),
            "null_median": float("nan"),
            "ratio": float("nan"),
            "p": float("nan"),
        }
    obs = float(np.var(np.log(v), ddof=1))
    rng = np.random.default_rng(seed)
    null = np.var(np.log(np.var(rng.normal(size=(n_sim, r, c)), axis=2, ddof=1)), axis=1, ddof=1)
    med = float(np.median(null))
    p = float((1 + np.sum(null >= obs)) / (1 + n_sim))
    return {"stat": obs, "null_median": med, "ratio": obs / med, "p": p}


def seed_label_anova(m: np.ndarray) -> dict:
    """(h) Two-way ANOVA without replication (recipe x seed label).

    Tests a seed effect *shared across recipes* (same seed label -> same init/data-order
    perturbation). ``seed_share = var_seed / (var_seed + MS_res)`` with
    ``var_seed = max((MS_seed - MS_res) / R, 0)``.
    """
    m = np.asarray(m, dtype=float)
    r, c = m.shape
    gm = m.mean()
    ss_seed = r * float(((m.mean(0) - gm) ** 2).sum())
    ss_rec = c * float(((m.mean(1) - gm) ** 2).sum())
    ss_res = float(((m - gm) ** 2).sum()) - ss_seed - ss_rec
    ms_seed, ms_res = ss_seed / (c - 1), ss_res / ((r - 1) * (c - 1))
    f = ms_seed / ms_res if ms_res > 0 else float("nan")
    p = float(stats.f.sf(f, c - 1, (r - 1) * (c - 1))) if ms_res > 0 else float("nan")
    var_seed = max((ms_seed - ms_res) / r, 0.0)
    share = var_seed / (var_seed + ms_res) if (var_seed + ms_res) > 0 else float("nan")
    return {"F": f, "p": p, "seed_share": share}
