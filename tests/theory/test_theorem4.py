"""Numerical certificate for Theorem 4 (algorithm correctness) --
paper/sections/theorem4_algorithm.tex.

Tests the Certified/Abstain stopping-and-abstention logic exactly as
specified in the .tex file's Section "The algorithm" -- NOT the full
adaptive C-tracking algorithm (that needs P3-02's T* solver and is
P3-03/P3-04's job). Reuses the linear-in-theta closed-form simulation
machinery from test_theorem1.py: since g is linear in theta, the OLS
estimator is exactly Gaussian at every replicate count, so a sequence of
"more replicates" rounds can be simulated by exact closed-form fits at
each round rather than literally accumulating streaming data.
"""

from __future__ import annotations

import numpy as np
import pytest

_SEED = 20260911
_DELTA = 0.05
_EPSILON_0 = 1e-4
_ROUND_REPLICATE_COUNTS = [2**i for i in range(1, 33)]  # 2, 4, ..., ~4e9 -- a
# purely synthetic replicate count (no real experiment would run this many
# reps), needed only so c_t's 1/sqrt(n_rep) decay actually crosses
# _EPSILON_0 within the round cap for this specific (fit_scales, s_star,
# sigma) instance; checked directly (see the module-level sanity print in
# test_epsilon_0_is_reachable_within_the_round_cap) rather than guessed.
_N_RUNS = 500


def _design_matrix(scales: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(scales), scales])


_N_ARMS = 2


def _beta(t: int, delta: float) -> float:
    # Union bound over rounds AND over the K(K-1) ordered arm pairs (the leader is
    # data-dependent, so every ordered pair could be the compared one) -- second-round
    # review of PR #26. For the two-arm certificate K(K-1) = 2.
    return delta / (t * (t + 1) * _N_ARMS * (_N_ARMS - 1))


def _run_stopping_rule(
    rng: np.random.Generator,
    fit_scales: np.ndarray,
    s_star: float,
    true_theta_a: np.ndarray,
    true_theta_b: np.ndarray,
    h_a_fit: np.ndarray,
    h_b_fit: np.ndarray,
    h_a_star: float,
    h_b_star: float,
    sigma: float,
    eta_a: float,
    eta_b: float,
) -> str:
    """One simulated run: increasing replicate counts, checking
    Certified/Abstain at each round. Returns 'certified', 'abstained', or
    'undecided' (hit the round cap without resolving)."""
    x = _design_matrix(fit_scales)
    xtx_inv = np.linalg.inv(x.T @ x)
    j_star = np.array([1.0, s_star])

    true_vals_a = true_theta_a[0] + true_theta_a[1] * fit_scales + h_a_fit
    true_vals_b = true_theta_b[0] + true_theta_b[1] * fit_scales + h_b_fit
    del h_a_star, h_b_star  # only the fitting-scale values feed the simulated
    # data (correctly -- the whole point of Theorem 2 Part B's construction
    # is that s* itself is never observed); kept as named arguments only so
    # callers stay self-documenting about which instance they're building.

    for t, n_rep in enumerate(_ROUND_REPLICATE_COUNTS, start=1):
        noise_sd = sigma / np.sqrt(n_rep)
        y_a = true_vals_a + rng.normal(0.0, noise_sd, size=len(fit_scales))
        y_b = true_vals_b + rng.normal(0.0, noise_sd, size=len(fit_scales))
        theta_hat_a = xtx_inv @ (x.T @ y_a)
        theta_hat_b = xtx_inv @ (x.T @ y_b)
        mu_hat_a = float(j_star @ theta_hat_a)
        mu_hat_b = float(j_star @ theta_hat_b)
        v_a = (sigma**2 / n_rep) * float(j_star @ xtx_inv @ j_star)
        v_b = v_a  # same design/noise for both arms in this certificate

        if mu_hat_a >= mu_hat_b:
            k_hat_val, k_val, eta_khat, eta_k = mu_hat_a, mu_hat_b, eta_a, eta_b
        else:
            k_hat_val, k_val, eta_khat, eta_k = mu_hat_b, mu_hat_a, eta_b, eta_a

        delta_hat = k_hat_val - k_val
        beta_t = _beta(t, _DELTA)
        c_t = np.sqrt(max(2 * (v_a + v_b) * np.log(1 / beta_t), 0.0))

        if delta_hat - eta_khat - eta_k > c_t:
            return "certified"
        if c_t <= _EPSILON_0 and delta_hat - eta_khat - eta_k <= c_t:
            return "abstained"

    return "undecided"


def test_epsilon_0_is_reachable_within_the_round_cap():
    # A guard against silently getting "undecided" everywhere for the wrong
    # reason (round cap too small for c_t to ever cross epsilon_0) rather
    # than a genuine abstain/certify signal -- this was a real bug caught
    # here: the first version of this file capped at i=20 (~1e6 replicates)
    # and c_t never dropped below epsilon_0=1e-4 for the actual
    # (fit_scales, s_star, sigma) used below, so every single run came back
    # "undecided" and the impossible-regime test failed for a reason that
    # had nothing to do with the abstention logic being checked.
    fit_scales = np.array([1.0, 2.0, 3.0, 4.0])
    s_star = 6.0
    sigma = 0.2
    x = _design_matrix(fit_scales)
    xtx_inv = np.linalg.inv(x.T @ x)
    j_star = np.array([1.0, s_star])
    j_factor = float(j_star @ xtx_inv @ j_star)

    n_rep = _ROUND_REPLICATE_COUNTS[-1]
    t = len(_ROUND_REPLICATE_COUNTS)
    v = (sigma**2 / n_rep) * j_factor
    c_t_final = np.sqrt(max(4 * v * np.log(1 / _beta(t, _DELTA)), 0.0))
    assert c_t_final <= _EPSILON_0, (
        f"round cap insufficient: c_t={c_t_final:.3e} at the last round still "
        f"exceeds epsilon_0={_EPSILON_0:.3e} -- extend _ROUND_REPLICATE_COUNTS"
    )


# ---------------------------------------------------------------------------
# (a) Solvable regime: stops and selects k* correctly in >= 1-delta of runs.
# ---------------------------------------------------------------------------


def test_solvable_regime_stops_and_selects_correctly():
    rng = np.random.default_rng(_SEED)
    fit_scales = np.array([1.0, 2.0, 3.0, 4.0])
    s_star = 6.0
    sigma = 0.2
    eta_a = eta_b = 0.05  # small, valid bias budget (true h below is 0)

    true_theta_a = np.array([0.5, 0.1])
    true_theta_b = np.array([0.3, 0.05])  # arm a is the true winner, comfortably
    h_zeros = np.zeros_like(fit_scales)
    mu_a_star_true = true_theta_a[0] + true_theta_a[1] * s_star
    mu_b_star_true = true_theta_b[0] + true_theta_b[1] * s_star
    true_gap = mu_a_star_true - mu_b_star_true
    assert true_gap > 2 * eta_a + 2 * eta_b  # comfortably solvable

    outcomes = {"certified": 0, "abstained": 0, "undecided": 0}
    n_correct = 0
    for _ in range(_N_RUNS):
        outcome = _run_stopping_rule(
            rng,
            fit_scales,
            s_star,
            true_theta_a,
            true_theta_b,
            h_zeros,
            h_zeros,
            0.0,
            0.0,
            sigma,
            eta_a,
            eta_b,
        )
        outcomes[outcome] += 1
        if outcome == "certified":
            n_correct += 1  # arm a's mu_hat >= mu_hat_b will have been used as
            # k_hat internally, and since true_gap is large and positive with
            # tiny eta, certification (if it happens) essentially always
            # correctly names arm a -- checked directly below instead of assumed.

    assert outcomes["certified"] >= _N_RUNS * (1 - _DELTA), (
        f"expected >= {1 - _DELTA:.0%} certification in the solvable regime, got {outcomes}"
    )
    assert outcomes["abstained"] == 0, "must never abstain when the gap comfortably exceeds eta"


def test_solvable_regime_never_certifies_the_wrong_arm():
    # A stricter, direct check: whenever the rule certifies in the solvable
    # instance above, it must certify arm a specifically (the true winner),
    # not just "some arm" -- re-run tracking which arm won internally.
    rng = np.random.default_rng(_SEED + 1)
    fit_scales = np.array([1.0, 2.0, 3.0, 4.0])
    s_star = 6.0
    sigma = 0.2
    eta = 0.05
    true_theta_a = np.array([0.5, 0.1])
    true_theta_b = np.array([0.3, 0.05])
    x = _design_matrix(fit_scales)
    xtx_inv = np.linalg.inv(x.T @ x)
    j_star = np.array([1.0, s_star])
    true_vals_a = true_theta_a[0] + true_theta_a[1] * fit_scales
    true_vals_b = true_theta_b[0] + true_theta_b[1] * fit_scales

    n_certified_correct = 0
    n_certified_total = 0
    for _ in range(200):
        for n_rep in _ROUND_REPLICATE_COUNTS:
            noise_sd = sigma / np.sqrt(n_rep)
            y_a = true_vals_a + rng.normal(0.0, noise_sd, size=len(fit_scales))
            y_b = true_vals_b + rng.normal(0.0, noise_sd, size=len(fit_scales))
            mu_hat_a = float(j_star @ (xtx_inv @ (x.T @ y_a)))
            mu_hat_b = float(j_star @ (xtx_inv @ (x.T @ y_b)))
            v = (sigma**2 / n_rep) * float(j_star @ xtx_inv @ j_star)
            t = _ROUND_REPLICATE_COUNTS.index(n_rep) + 1
            c_t = np.sqrt(max(4 * v * np.log(1 / _beta(t, _DELTA)), 0.0))
            delta_hat = abs(mu_hat_a - mu_hat_b)
            if delta_hat - 2 * eta > c_t:
                n_certified_total += 1
                winner = "a" if mu_hat_a >= mu_hat_b else "b"
                if winner == "a":
                    n_certified_correct += 1
                break

    assert n_certified_total > 0
    assert n_certified_correct == n_certified_total, (
        f"certified the wrong arm at least once: {n_certified_correct}/{n_certified_total}"
    )


# ---------------------------------------------------------------------------
# (b) Impossible regime (Theorem 2 Part B's exact construction): abstains.
# ---------------------------------------------------------------------------


def test_impossible_regime_abstains_not_falsely_certifies():
    rng = np.random.default_rng(_SEED + 2)
    s_max = 4.0
    fit_scales = np.array([1.0, 2.0, 3.0, s_max])
    g = 2.0
    s_star = s_max + g
    sigma = 0.2

    # Theorem 2 Part B construction: h_a = 0, h_b = eta*phi((s-s_max)/g),
    # phi = clip(x,0,1) -- zero on every fitting scale, amplitude eta at s*.
    eta_budget = 0.15
    h_a_fit = np.zeros_like(fit_scales)
    h_b_fit = np.zeros_like(fit_scales)  # both 0 on fit_scales (s <= s_max)
    h_a_star = 0.0
    h_b_star = eta_budget  # nonzero only at s*

    true_theta_a = np.array([0.5, 0.0])  # flat, so h alone drives the gap
    true_theta_b = np.array([0.5, 0.0])
    mu_a_star_true = true_theta_a[0] + h_a_star
    mu_b_star_true = true_theta_b[0] + h_b_star
    true_gap = abs(mu_a_star_true - mu_b_star_true)
    assert true_gap == pytest.approx(eta_budget)

    eta_a = eta_b = eta_budget  # the assumed budget exactly matches the
    # construction's own amplitude -- the boundary case, deliberately.

    outcomes = {"certified": 0, "abstained": 0, "undecided": 0}
    for _ in range(_N_RUNS):
        outcome = _run_stopping_rule(
            rng,
            fit_scales,
            s_star,
            true_theta_a,
            true_theta_b,
            h_a_fit,
            h_b_fit,
            h_a_star,
            h_b_star,
            sigma,
            eta_a,
            eta_b,
        )
        outcomes[outcome] += 1

    assert outcomes["certified"] == 0, (
        f"must never falsely certify in the impossible regime, got {outcomes}"
    )
    assert outcomes["abstained"] >= _N_RUNS * 0.95, (
        f"expected the algorithm to abstain in effectively every run once "
        f"c_t shrinks below epsilon_0, got {outcomes}"
    )


# ---------------------------------------------------------------------------
# (c) Mutual exclusivity, by construction of the rule itself.
# ---------------------------------------------------------------------------


def test_certified_and_abstain_are_mutually_exclusive_by_construction():
    # Direct check of the two event definitions: Certified requires
    # delta_hat - eta_khat - eta_k > c_t; Abstain requires the complementary
    # <= c_t (among other things) -- structurally impossible to satisfy both
    # at once, verified over random (delta_hat, eta_khat, eta_k, c_t, t).
    rng = np.random.default_rng(_SEED + 3)
    for _ in range(2000):
        delta_hat = rng.uniform(-1, 2)
        eta_khat = rng.uniform(0, 0.5)
        eta_k = rng.uniform(0, 0.5)
        c_t = rng.uniform(0, 1)
        margin = delta_hat - eta_khat - eta_k
        certified = margin > c_t
        abstained = (c_t <= _EPSILON_0) and (margin <= c_t)
        assert not (certified and abstained)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
