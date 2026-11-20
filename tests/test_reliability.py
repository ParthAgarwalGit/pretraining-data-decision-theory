"""Tests for pdt.analysis.reliability (N7-02)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pdt.analysis import reliability as rl


def test_pair_index_drops_self_pairs_by_id():
    i, j = rl.pair_index(["a", "b", "a", "c"])
    pairs = {(int(x), int(y)) for x, y in zip(i, j, strict=True)}
    assert (0, 2) not in pairs  # duplicated recipe "a" never pairs with itself
    assert pairs == {(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)}


def test_sign_agreement_excludes_ties_and_handles_empty():
    p = np.array([1, -1, 0, 1, 1])
    q = np.array([1, 1, 1, 0, 1])
    assert rl.sign_agreement(p, q) == pytest.approx(2 / 3)
    assert np.isnan(rl.sign_agreement(np.array([0, 0]), np.array([1, -1])))


def test_retest_agreement_hand_computed():
    # recipes A, B, C; seeds 0..2
    t = np.array([[3.0, 1.0, 2.0], [2.0, 2.0, 2.0], [1.0, 3.0, 0.0]])
    out = rl.retest_agreement(t, ["A", "B", "C"])
    # c=0: seed0 = (3,2,1) -> signs AB+,AC+,BC+; mean(1,2) = (1.5,2,1.5) -> -,0,+ ;
    #      scorable AB (disagree), BC (agree) -> 1/2
    # c=1: seed1 = (1,2,3) -> -,-,- ; mean(0,2) = (2.5,2,0.5) -> +,+,+ -> 0/3
    # c=2: seed2 = (2,2,0) -> 0,+,+ ; mean(0,1) = (2,2,2) -> all ties -> NaN
    assert out["per_heldout_seed"][0] == pytest.approx(0.5)
    assert out["per_heldout_seed"][1] == pytest.approx(0.0)
    assert np.isnan(out["per_heldout_seed"][2])
    assert out["mean"] == pytest.approx(0.25)


def test_retest_agreement_matches_closed_form():
    rng = np.random.default_rng(7)
    r, s, draws = 40, 1.0, 3000
    mu = rng.normal(0.0, 1.0, r)
    ids = np.arange(r)
    i, j = rl.pair_index(ids)
    d = mu[i] - mu[j]
    a, b = norm.cdf(d / (s * np.sqrt(2.0))), norm.cdf(d / s)
    expected = float(np.mean(a * b + (1 - a) * (1 - b)))
    sims = [
        rl.retest_agreement(mu[:, None] + rng.normal(0, s, (r, 3)), ids)["mean"]
        for _ in range(draws)
    ]
    se = np.std(sims) / np.sqrt(draws)
    assert abs(np.mean(sims) - expected) < 4 * se + 1e-3


def test_proxy_agreement_identical_noise_free_proxy_is_perfect():
    mu = np.arange(10, dtype=float)
    target = mu[:, None] + np.array([0.0, 0.01, -0.01])
    proxy = np.repeat(mu[:, None], 3, axis=1)
    for k in (1, 2, 3):
        assert rl.proxy_agreement(proxy, target, np.arange(10), k) == pytest.approx(1.0)


def test_bootstrap_indices_deterministic():
    a = rl.cluster_bootstrap_indices(25, B=50, seed=3)
    b = rl.cluster_bootstrap_indices(25, B=50, seed=3)
    c = rl.cluster_bootstrap_indices(25, B=50, seed=4)
    assert a.shape == (50, 25)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    stat = rl.bootstrap(lambda idx: float(np.mean(idx)), a)
    lo, hi = rl.percentile_ci(stat)
    assert lo <= np.median(stat) <= hi


def _stratified_runs(reps: int = 120):
    """Monte Carlo with real signal: R = 200 recipes, imperfect proxy."""
    rng = np.random.default_rng(1)
    r = 200
    ids = np.arange(r)
    mu_t = rng.normal(0.0, 1.0, r)
    mu_p = mu_t + rng.normal(0.0, 0.5, r)
    pairs = rl.pair_index(ids)
    i, j = pairs
    honest, oracle, same, same_oracle = [], [], [], []
    for _ in range(reps):
        t = mu_t[:, None] + rng.normal(0, 1, (r, 3))
        p = mu_p[:, None] + rng.normal(0, 1, (r, 3))
        fresh_seed = mu_t + rng.normal(0, 1, r)
        fresh_three = mu_t[:, None] + rng.normal(0, 1, (r, 3))
        h = rl.stratified_heldout_single(p, t, ids, 0)
        honest.append(np.divide(h["proxy_hit"], h["proxy_n"]))
        # Oracle, computed independently of the code under test: bins from seeds 1 and 2
        # only, proxy scored against a fresh, independent target seed.
        m12 = t[:, 1:].mean(1)
        sd12 = np.sqrt(np.mean(np.var(t[:, 1:], axis=1, ddof=1)))
        z12 = np.abs(m12[i] - m12[j]) / sd12
        ps3 = np.sign(p.mean(1)[i] - p.mean(1)[j])
        fs = np.sign(fresh_seed[i] - fresh_seed[j])
        orow = []
        for lo, hi in rl.DEFAULT_BINS:
            ok = (z12 >= lo) & (z12 < hi) & (ps3 != 0) & (fs != 0)
            orow.append(np.mean(ps3[ok] == fs[ok]))
        oracle.append(orow)
        same.append(rl.stratified_same_target(p, t, ids)["agreement"])
        tm = t.mean(1)
        z = np.abs(tm[i] - tm[j]) / (rl.pooled_sd(t) * np.sqrt(2.0 / 3.0))
        ps, gs = rl.pair_signs(p.mean(1), pairs), rl.pair_signs(fresh_three.mean(1), pairs)
        row = []
        for mask in rl._bin_masks(z, rl.DEFAULT_BINS):
            hit, n = rl._agree_counts(ps, gs, mask)
            row.append(hit / n)
        same_oracle.append(row)
    return (np.mean(x, axis=0) for x in (honest, oracle, same, same_oracle))


def test_heldout_stratification_is_not_circular():
    # Measured (2026-11-21, 200 reps): honest vs oracle differ by <= 0.002 per bin
    # (MC s.e. of the difference ~0.003 at 120 reps); binning with seeds that include
    # the held-out seed shifts bins by 0.05-0.08. 0.015 sits between the two.
    honest, oracle, same, same_oracle = _stratified_runs()
    assert np.max(np.abs(honest - oracle)) < 0.015
    # The same-target version is visibly biased against its own oracle.
    assert np.max(np.abs(same - same_oracle)) > 0.025


def test_stratified_heldout_pools_counts():
    rng = np.random.default_rng(0)
    mu = rng.normal(0, 1, 30)
    t = mu[:, None] + rng.normal(0, 0.3, (30, 3))
    p = mu[:, None] + rng.normal(0, 0.3, (30, 3))
    out = rl.stratified_heldout(p, t, np.arange(30))
    singles = [rl.stratified_heldout_single(p, t, np.arange(30), c) for c in range(3)]
    assert out["n_pairs"] == np.sum([s["proxy_n"] for s in singles], axis=0).tolist()
    assert sum(out["n_pairs"]) == 3 * 435


def test_seeds_vs_scale_frontier_and_exclusion():
    rng = np.random.default_rng(2)
    mu = rng.normal(0, 1, 15)
    ids = np.arange(15)
    vals = {
        "small": mu[:, None] + rng.normal(0, 2.0, (15, 3)),
        "mid": mu[:, None] + rng.normal(0, 0.5, (15, 3)),
        "odd": mu[:, None] + rng.normal(0, 0.01, (15, 3)),
        "big": mu[:, None] + rng.normal(0, 0.2, (15, 3)),
    }
    comp = {"small": 1.0, "mid": 10.0, "odd": 2.0, "big": 100.0}
    out = rl.seeds_vs_scale(vals, comp, "big", ids, exclude=("odd",))
    assert len(out["points"]) == 9
    assert all(f["size"] != "odd" for f in out["frontier"])
    costs = [p["cost"] for p in out["points"]]
    assert costs == sorted(costs)
    m = rl.matched_comparison(vals, comp, "big", ids, ("mid", 2), ("big", 1))
    assert m["cost_b"] == 1.0 and m["cost_a"] == pytest.approx(0.2)
    assert m["diff"] == pytest.approx(m["acc_a"] - m["acc_b"])


def test_seed_label_anova_detects_shared_seed_effect():
    rng = np.random.default_rng(5)
    base = rng.normal(0, 1, (25, 1)) + rng.normal(0, 0.1, (25, 3))
    shared = base + np.array([0.0, 0.5, -0.5])
    assert rl.seed_label_anova(shared)["p"] < 1e-6
    assert rl.seed_label_anova(shared)["seed_share"] > 0.5
    assert rl.seed_label_anova(base)["seed_share"] < 0.2


def test_heteroscedasticity_diagnostics():
    rng = np.random.default_rng(6)
    homo = rng.normal(0, 1, (25, 3))
    scale = np.r_[np.full(12, 0.2), np.full(13, 5.0)][:, None]
    hetero = rng.normal(0, 1, (25, 3)) * scale
    # The dispersion test has power at n = 3 per recipe; Brown-Forsythe does not, so it
    # is only checked to be a valid p-value (and NaN when every deviation is equal).
    assert rl.log_variance_dispersion(hetero)["p"] < 0.01
    assert rl.log_variance_dispersion(homo)["p"] > 0.01
    assert 0.0 <= rl.brown_forsythe(hetero) <= 1.0
    assert np.isnan(rl.brown_forsythe(np.ones((5, 3))))
    assert rl.log_variance_dispersion(homo) == rl.log_variance_dispersion(homo)
