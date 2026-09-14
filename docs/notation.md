# Notation

Every symbol in `paper/sections/setup.tex` (and the theorem statements that build on
it), mapped to the code identifier that computes or estimates it. Written for
`plan/03-phase2-theory.md` task P2-01's definition of done: "a setup section a
stranger can read without the source PDF, plus `docs/notation.md` mapping every
symbol to the code identifier that computes it."

Kept current as later Phase 2 tasks introduce new symbols (Theorems 1-4); each row
names which task introduced it.

## Arms, scales, target

| Symbol | Meaning | Code identifier | Task |
|---|---|---|---|
| `K` | number of arms (candidate recipes) | `len(RECIPES)` conceptually; DataDecide's real value is 25 (`src/pdt/data/datadecide.py`) | P2-01 |
| `k` | an arm index, `k in {1..K}` | a `recipe` string everywhere in `src/pdt/` (e.g. `frame["recipe"]`) | P2-01 |
| `s = (N, D)` | a scale: parameter count and token count | `pdt.scaling.base.Scale(n, d)` | P2-01 |
| `s*` | the target scale | `Scale` for DataDecide's 1B row; `_TARGET`-style constants in `experiments/p1_*.py` | P2-01 |
| `S` | the set of candidate scales | the 14-size DataDecide ladder, or Pythia's 8-size ladder (P1-10) | P2-01 |
| `S_fit` | the design: scales actually observed to fit `theta_k` | the `scales` argument to `Extrapolator.fit()`; a "design" name like `S_fit_le_530M` in `experiments/p1_04_extrapolation_baselines.py` | P2-01 |
| `mu_k(s)` | arm `k`'s true decision-relevant value at scale `s` | not directly observable; estimated by `recipe_means()` / `recipe_trajectories()` (`src/pdt/analysis/decision_accuracy.py`) at scales that are actually run | P2-01 |
| `k*` | the true winner, `argmax_k mu_k(s*)` | `compute_ground_truth()` (`src/pdt/analysis/ground_truth.py`) | P2-01 |
| `Delta_k` | the gap `mu_{k*}(s*) - mu_k(s*)` | `gap` field in `results/p1_02_target.json` / P1-09's per-pair `gap` | P2-01 |

## The scaling family (Assumption 1, `setup.tex` sec. 2)

| Symbol | Meaning | Code identifier | Task |
|---|---|---|---|
| `g(theta, s)` | the known parametric family | `Extrapolator._predict_from_theta(theta, scale)` (subclass-specific; abstract in `pdt.scaling.base.Extrapolator`) | P2-01 |
| `theta_k` | arm `k`'s true (unknown) parameter vector | `Extrapolator._theta` after `.fit()` (the *estimate*; the true value is never observed) | P2-01 |
| `Theta` | the parameter space | implicit in each fitter's `bounds` passed to `multi_start_fit()` | P2-01 |
| `p` | number of parameters | `Extrapolator.n_params` (1 for `ConstantExtrapolator`, 2 `LogLinear`, 3 `PowerLawN`/`PowerLawC`, 5 `ChinchillaND`, 7 `TwoStepLadder`) | P2-01 |
| `J(theta, s)` | Jacobian `d g / d theta` | `Extrapolator.jacobian(scale)` (central-difference numerical differentiation, shared across all six fitters) | P2-01 |

## The observation model (Assumption 2, `setup.tex` sec. 3)

| Symbol | Meaning | Code identifier | Task |
|---|---|---|---|
| `y_{k,i}(s)` | one noisy observation | one row's `metric_value` in the long frame (`pdt.data.frame.build_frame`) | P2-01 |
| `eps_{k,i}(s)` | observation noise | not materialized directly; its variance is what `pdt.analysis.noise` estimates | P2-01 |
| `sigma^2(s)` | proxy noise variance at scale `s` | three components, combined per estimator: `noise.seed_variance()` (`sigma2_seed`), `noise.checkpoint_jitter()` (`sigma2_ckpt`), `noise.eval_sampling_noise()` / `eval_sampling_noise_of_mean()` | P2-01 |
| `sigma^2_target,k` | noise in the *ground-truth* measurement at `s*` | `sigma2_target` in `results/p1_05_noise.json`, defined as `sigma2_seed / n_seeds` at the target scale (matches `compute_ground_truth`'s own seed-averaging estimator) | P2-01 |

## The cost model (Assumption 3, `setup.tex` sec. 4)

| Symbol | Meaning | Code identifier | Task |
|---|---|---|---|
| `c(s)` | cost of one observation at scale `s` | `Scale.compute` property, `= 6.0 * n * d` | P2-01 |
| `C` | total compute spent | `compute_cost` field throughout `results/p1_06_decomposition.json` / `p1_07_bound_coverage.json` | P2-01 |

## The misspecification model (`setup.tex` sec. 5 -- the paper's central object)

| Symbol | Meaning | Code identifier | Task |
|---|---|---|---|
| `h_k(s)` | arm `k`'s true deviation from the family | never directly observed; its *effect* at `s*` is what `sigma^2_extrap,k` measures | P2-01 |
| `H`, `eta` | the perturbation class and its bound | not yet estimated in code; P2-06 measures empirical `h_k(s)` residuals against DataDecide to check whether an assumed `H` actually contains them | P2-01 / P2-06 |
| `theta_k^dagger(S_fit)` | population-level in-family projection | the *infinite-replicate limit* of `Extrapolator._theta` for a given design; never computed directly (only its noisy estimate is) | P2-01 |
| `sigma^2_extrap,k(S_fit)` | extrapolation bias at `s*`, population level | the quantity `bias_variance_decomposition()`'s `sigma2_extrap_hat` estimates | P2-01 |
| `sigma2_extrap_hat` | the empirical estimator of the above | `pdt.analysis.bootstrap.bias_variance_decomposition()`'s return field; `max(0, bias_hat**2 - v_hat/B - sigma2_target)` | P2-01, estimator built in P1-06 |
| `v_k(C)` | estimation variance of `mu_hat_k(s*)` | bootstrap: `bias_variance_decomposition()`'s `v_hat` field. Analytic (delta-method): `pdt.theory.bound.analytic_v_k()`, `= J_target^T Sigma_theta J_target` | P2-01 / P2-02, both built in P1-06/P1-07 |
| `Sigma_theta` | sandwich covariance of the fitted `theta` | `pdt.theory.bound.sandwich_covariance()` | P1-07, reused by P2-02 |
| `D_k` | pairwise difference statistic `mu_hat_{k*}(s*) - mu_hat_k(s*)` | the per-replicate difference series bootstrapped in P1-06 step 4; `bias(D_k)`, `v(D_k)` are `bias_variance_decomposition()` called on that series | P1-06, used in P2-02's Corollary 1 |

## The policy class (`setup.tex` sec. 6)

| Symbol | Meaning | Code identifier | Task |
|---|---|---|---|
| `tau` | the stopping time | P3's algorithm loop (`experiments/p3_*` / `src/pdt/bai/`, not yet built) | P2-01, implemented P3-03 |
| `k_hat` | the recommended arm at stopping | same | P2-01, implemented P3-03 |
| `delta` | the correctness parameter | a config value, never hardcoded (per `plan/00-agent-protocol.md` sec. 6) | P2-01 |
| `P[k_hat != k*]` | selection-error probability | empirically estimated by Monte-Carlo in `experiments/p1_07_bound_coverage.py`'s `empirical_error_rate`, and (theoretically) bounded by Theorem 1 | P1-07 empirical, P2-02 theoretical |

## Bound forms already implemented (P1-07, reused by Theorem 1)

| Symbol | Meaning | Code identifier |
|---|---|---|
| marginal bound term | `exp(-Delta_k^2 / (2*(sigma2_extrap_k + v_k)))` | `pdt.theory.bound.marginal_bound_term()` |
| pairwise bound term | `exp(-Delta_k^2 / (2*(bias(D_k)^2 + v(D_k))))` | `pdt.theory.bound.pairwise_bound_term()` |
| union bound | `sum over k != k*` of either term | `pdt.theory.bound.marginal_bound()` / `.pairwise_bound()` |

Symbols introduced by Theorems 2-4 (P2-03 through P2-05) will be appended here as
those tasks land, not in a separate file -- one notation reference, kept current.
