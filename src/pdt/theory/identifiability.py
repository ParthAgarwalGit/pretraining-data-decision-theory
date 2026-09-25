"""Structural identifiability of a target prediction from a set of scales.

A model's prediction at the target scale `s*` is *estimable* from
observations at scales `{s_i}` only if the target Jacobian `J(s*)` lies in
the row space of the observed-scale Jacobians `{J(s_i)}`: otherwise some
direction of parameter space that moves the target prediction is invisible
to every observation, and the target's variance is infinite, not finite.

`np.linalg.pinv` does not signal this -- its minimum-norm convention treats
a direction with no information as contributing *zero* variance, the
opposite of the truth. Two earlier attempts to guard against it were wrong
in different ways (PR #17/#28 reviews):

- checking the *weighted* Fisher information's range with a loose 10%
  relative-residual tolerance (`allocation._target_denom`, first version)
  treated a target whose missing component was only ~5% of `||J(s*)||` as
  estimable -- structural non-identifiability can be arbitrarily close to an
  observed scale (LogLinear observed at N=1, target N=exp(0.05)), it does not
  imply an 80-100% residual; and
- a tight tolerance on the weighted information false-triggered on ordinary
  ill-conditioned (condition number ~1e14) but identified designs.

The resolution is to keep the two questions separate:

1. **Structural identifiability** (this module): a property of the *support*
   of the design -- which scales have positive weight -- not of how large
   the weights are, so it is checked on the *unweighted* Jacobian rows, with
   parameter columns equilibrated and a numerical-rank tolerance appropriate
   to that (scaled) matrix.
2. **Numerical ill-conditioning** of the weighted information: left to the
   caller's pseudo-inverse, where it shows up as a large but finite variance,
   which is the honest report for an identified-but-poorly-conditioned design.
"""

from __future__ import annotations

import numpy as np

#: A parameter column whose norm is below this fraction of the largest column
#: norm is numerically zero: no scale in the design moves that parameter.
_ZERO_COLUMN_RTOL = 1e-12


def target_in_row_space(
    jacobian_rows: np.ndarray, j_target: np.ndarray, *, rtol: float = 1e-8
) -> bool:
    """Whether `j_target` lies in the row space of `jacobian_rows`
    (`n_scales x n_params`, one Jacobian row per observed scale), to relative
    tolerance `rtol` on the equilibrated problem.

    Steps: (1) equilibrate parameter columns by their norms so parameters on
    wildly different scales (`E` ~ 1, `alpha`-derivatives ~ 1e-10) are judged
    on equal footing; a column that is numerically zero -- no scale moves
    that parameter -- is dropped, and the target must have (relatively)
    nothing in it; (2) take the numerical rank of the equilibrated matrix
    from its SVD with the standard `max(shape) * eps` cutoff; (3) measure the
    target's residual outside the span of the retained right-singular
    vectors, relative to `||target||`.

    Returns False for an empty design (no observed scale identifies
    anything) unless the target itself is zero.
    """
    rows = np.atleast_2d(np.asarray(jacobian_rows, dtype=float))
    target = np.asarray(j_target, dtype=float)
    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        return True
    if rows.size == 0:
        return False

    col_norms = np.linalg.norm(rows, axis=0)
    max_col = float(col_norms.max())
    if max_col == 0.0:
        return False
    keep = col_norms > _ZERO_COLUMN_RTOL * max_col

    # A parameter no observed scale moves: the target prediction must not
    # depend on it either, up to numerical noise.
    if np.linalg.norm(target[~keep]) > rtol * target_norm:
        return False
    if not keep.any():
        return True  # target depends on nothing that is identified, and nothing else

    a = rows[:, keep] / col_norms[keep]
    t = target[keep] / col_norms[keep]
    t_norm = float(np.linalg.norm(t))
    if t_norm == 0.0:
        return True

    _, singular_values, vt = np.linalg.svd(a, full_matrices=False)
    tol = max(a.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > tol))
    basis = vt[:rank]
    residual = t - basis.T @ (basis @ t)
    return bool(np.linalg.norm(residual) <= rtol * t_norm)


def prediction_influence_weights(
    jacobian_rows: np.ndarray, j_target: np.ndarray, *, rtol: float = 1e-8
) -> np.ndarray | None:
    """Influence weights `g` (one per observed scale) with `prediction = sum_i g_i y_i`,
    i.e. `g = pinv(J)^T j_target`, computed from the SVD of the (column-equilibrated)
    DESIGN `J`, never from `J^T J`. `None` if the target is not identified.

    Second-round review of PR #17: `sandwich_covariance` formed `pinv(J^T J)`, which
    squares the condition number. A full-rank design with `cond(J) ~ 2e8` has
    `cond(J^T J) ~ 6e16`, so `pinv`'s cutoff silently discards a weak but genuinely
    identified direction and reports a variance of 1.5e-4 where the true (SVD-based)
    value is 5e11 -- enormous uncertainty turned into apparent precision. Working from
    the design's own SVD keeps the conditioning at `cond(J)`.

    Fails closed: if any singular direction is discarded by the rank cutoff and the
    target has a non-negligible component in it (relative residual > `rtol`), the
    target's variance is not computable from this design and `None` is returned --
    the caller must treat that as infinite variance / unidentified, never as zero.
    """
    rows = np.atleast_2d(np.asarray(jacobian_rows, dtype=float))
    target = np.asarray(j_target, dtype=float)
    n_scales = rows.shape[0]
    if not np.any(target):
        return np.zeros(n_scales)
    if rows.size == 0:
        return None

    col_norms = np.linalg.norm(rows, axis=0)
    max_col = float(col_norms.max())
    if max_col == 0.0:
        return None
    keep = col_norms > _ZERO_COLUMN_RTOL * max_col
    target_norm = float(np.linalg.norm(target))
    if np.linalg.norm(target[~keep]) > rtol * target_norm:
        return None
    if not keep.any():
        return np.zeros(n_scales)

    a = rows[:, keep] / col_norms[keep]
    t = target[keep] / col_norms[keep]
    u, singular_values, vt = np.linalg.svd(a, full_matrices=False)
    tol = max(a.shape) * np.finfo(float).eps * singular_values[0]
    retained = singular_values > tol
    basis = vt[retained]
    residual = t - basis.T @ (basis @ t)
    t_norm = float(np.linalg.norm(t))
    if t_norm > 0.0 and np.linalg.norm(residual) > rtol * t_norm:
        return None
    coeffs = (basis @ t) / singular_values[retained]
    return u[:, retained] @ coeffs
