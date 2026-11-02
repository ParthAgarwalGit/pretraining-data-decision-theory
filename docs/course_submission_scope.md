# Course self-study submission scope

This repository contains an exploratory implementation and analysis of selecting
pretraining data using small-model results. The course submission should describe
what was studied, how the implementation works, and what the evidence supports.
It is not a completed research paper or a validated claim of adaptive optimality.

## Corrections for the final report

- The corrected Welch/Bonferroni rank-reversal artifact reports 32 reversing pairs
  among 3,300 (about 0.97%), 1,424 stable pairs, and 1,844 unresolved within noise.
  The unresolved pairs must not be described as evidence of stable rankings.
  Source: `results/p1_09_rank_reversals.json`,
  `data.thresholds.bonferroni.summary`. The earlier 15.2% figure is superseded.
- Uncertainty calculations now scale each nonzero parameter column before deciding
  numerical rank. Parameter units must not turn a weakly observed direction into
  apparent certainty. Regression tests exercise extreme unit changes.
- In supported-only mode, a bounded linear model returns a recommendation when the
  stopping criterion fires. An interior observed fit does not make the estimator
  unconstrained across possible samples. In particular, built-in `LogLinear` is
  bounded. Adaptive or nonlinear guarantees also remain unproved.
- The constant-rank identifiability converse requires an interior parameter;
  exponent identification in the power-law example requires nonzero amplitude.

## Historical results and limitations

The committed experimental JSON files record historical runs and their generating
commits. They have not been regenerated after this correction. Do not describe
their variance values, selection labels, or dependent figures as freshly validated
results of this branch. The rank-reversal census is separate from the changed
uncertainty/selection code; its corrected counts above are read from that artifact.

For the course report, prefer an explanation of the algorithm, the corrected
rank-reversal census, and regression examples of what was learned. Any historical
pilot figure should be explicitly identified as exploratory, with its limitations.
No claim of demonstrated compute savings, generally valid certification, or new
large-model training is justified by this cleanup.

## What is and is not needed for this course submission

Include the problem, background concepts, methods studied, implementation details,
selected observations, limitations, references, and a reflection on learning.
Separate teammate contributions, existing public data/code, personal work, and
AI-assisted implementation or writing. This correction was AI-assisted.

New GPU training, expanded ablations, and completion of the research publication
plan are outside this course cleanup. The outstanding research tasks remain in
`STATUS.md`; they are not prerequisites for an honest self-study report.
