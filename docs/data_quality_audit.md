# Stage 1 Data quality audit

The pinned source contains 235,125 rows. Excluded 10,725 untrained rows and 212,850 intermediate rows by the maximum-step rule, leaving 11,550 final observations. No duplicate keys, missing final cells, nonfinite scores or out-of-range scores were found. There are 25 recipes, 14 sizes, 10 individual task families and one aggregate; every cell has three seeds. Budgets are consistent across recipes and seeds within each size. Tokens change with size: this is a joint size/training-budget ladder, not an isolated causal effect of parameter count.

The aggregate is the arithmetic mean of the ten component scores (maximum discrepancy 2.22e-16). It is excluded from task-level summaries. The primary metric is a task-specific accuracy score supplied by DataDecide; metric definitions can differ between tasks, so all rankings are within task, and raw regret is summarized explicitly as accuracy percentage points.

Small auxiliary labels occur below 1B and large auxiliary labels at 1B. Label matching does not prove paired runs. Welch comparisons use an independent-run assumption, three seeds per recipe, unbiased sample variance and a per-trajectory Bonferroni correction over sizes. With three observations, normality is not testable and the inference remains assumption-dependent. This does not control errors across every recipe pair and task.

Historical pair classifications were independently recounted and checked for duplicates in historical_census_recount.json. Recomputing their statistics from the newly loaded data occurs in Stage 6; recounting alone is not statistical reproduction. The inventory preserves older artifact provenance.

All input hashes, filtering counts, seed labels, budgets and coverage assertions are saved in results/empirical_study. All four proposed empirical analyses can proceed using this complete panel. No missing values were imputed.
