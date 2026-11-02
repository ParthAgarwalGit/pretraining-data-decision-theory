# Stage 7 Integrated findings

## Three primary findings
1. Across the ten individual tasks, mean Spearman correlation rises from 0.130 at 4M to 0.830 at 530M, then falls to 0.787 at 750M. At those sizes, winner selection succeeds on 2/10, 4/10 and 5/10 tasks. Average regret is 4.60, 0.71 and 0.93 accuracy percentage points. Evidence: scale_summary.csv and cross_scale_predictability.csv.
2. Mean ranking correlation across source sizes is 0.954 for ARC-Easy, 0.060 for BoolQ and 0.040 for Winogrande. The aggregate winner is never exactly selected at a smaller size (0/13), which does not imply all selections are materially worse. Evidence: task_heterogeneity.csv and task_ranking_heatmap.png.
3. Across 100 individual task/cutoff settings, the largest-observed baseline has top-1 success 27%, versus 23% linear and 26% for both log-linear and power-law. Power-law has the lowest average RMSE (0.127) and average regret (1.83 percentage points), but not the strongest ranking correlation or winner rate. At the 750M cutoff, linear has lower RMSE and regret than power-law. Evidence: extrapolation_metrics.csv. These are descriptive comparisons, not proof of superiority.

## Secondary findings
Single- and two-seed subsets match the full-seed winner 46.9% and 67.9% of the time. The nested reference makes this optimistic relative to an independent run. Absolute Welch statistics below 1 versus at least 4 are associated with different cross-scale agreement, but do not identify the causal source of instability. Evidence: seed_subsampling_summary.csv and signal_to_noise_summary.csv.

All 3300 historical calibrated pair classifications reproduce without discrepancy. At alpha=0.05 there are 32 reversing, 1424 stable and 1844 within-noise trajectories. Across alpha 0.01 to 0.10, reversals range from 7 to 72. The count is sensitive, although resolved reversals remain a small fraction of this panel. This is not evidence that the true reversal rate is low.

## Important limitations and contrary evidence
2344/2750 power-law fits reach an exponent bound. 1460 linear predictions fall outside [0,1]. These diagnostics are retained, not hidden; power-law parameters should not be interpreted as established scaling exponents. The fitting range and model assumptions materially affect results. Repeated sizes, shared recipes and correlated tasks are not independent replications. Target averages remain noisy. No new model training, measured compute savings or general algorithm guarantee is established.

## Narrative
Small-scale experiments contain useful but task-dependent ranking information. Better average ranking does not ensure correct winner selection. Seed uncertainty, target measurement noise and extrapolation choices all limit confidence. A defensible conclusion is to evaluate rankings, regret and method diagnostics together, with abstention or additional evidence where decisions are uncertain.

The original DataDecide observations belong to the released dataset authors. This study's contribution is the additional offline analysis and validation of that dataset. Existing numerical debugging evidence is separate from these empirical observations. The report is a course self-study, not a claim of new large-scale training or a new general theorem.
