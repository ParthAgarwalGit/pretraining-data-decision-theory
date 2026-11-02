"""Stage 0: preserve an inventory before acquiring data or changing analyses."""
from pathlib import Path
import hashlib, json, subprocess

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'results/empirical_study'
out.mkdir(parents=True, exist_ok=True)
inventory = []
for folder in ['results', 'data']:
    for p in sorted((ROOT / folder).rglob('*')):
        if p.is_file() and 'empirical_study' not in p.parts:
            record = {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size,
                      'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            if p.suffix == '.json':
                d = json.loads(p.read_text(encoding='utf8'))
                record['provenance'] = d.get('provenance')
            inventory.append(record)
(out / 'artifact_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf8')
sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
(ROOT / 'docs/research_audit.md').write_text(f'''# Stage 0 Research audit

Baseline commit: {sha}. Work isolated on course/empirical-study. Historical artifacts remain unchanged.

## Architecture and available evidence
src/pdt/data/datadecide.py downloads pinned Hugging Face tables and parses metrics; frame.py creates tidy observations. src/pdt/analysis contains seed variance, recipe ranking and reversal analysis. src/pdt/scaling contains constant, power-law and other extrapolators. src/pdt/theory implements uncertainty and identifiability calculations. src/pdt/bai contains Extrapolation Track and Stop, allocation and guarded oracles. experiments and tests provide reusable implementations and checks.

The checkout has historical JSON results but no raw or cached observations. Inventory and hashes are in results/empirical_study/artifact_inventory.json. Historical outputs contain their own older commit and timestamp; they are not regenerated results from the November corrections. The rank census stores individual pair classifications, permitting independent count verification offline, but not a fresh statistical census without the observations.

## Dataset and acquisition
Use allenai/DataDecide-eval-results at revision 9919b5a0e61e57a85021263918fa82d6ceaee038, also recorded by the historical census. The upstream API currently resolves to this same revision. Retrieve only data/macro_avg-00000-of-00001.parquet and README.md, not model weights, token corpora or individual predictions.

Expected raw columns from the loader are data, params, task, seed, step, tokens, compute, chinchilla and metrics. These remain expectations until Stage 1 inspects the actual file. The 11-task table includes an aggregate of 10 components. The 66-task training table has MMLU subject splits and is not interchangeable with it. Final checkpoint selection must use each model run's maximum step, independently of its scores.

## Statistical and implementation concerns
Seed labels differ at 1B. A common label alone does not establish statistical pairing; use independent-run comparisons, document the assumption and avoid cross-scale paired differences. Three repetitions cannot establish population uncertainty. Single-seed variance is undefined. Pair trajectories share recipes; the aggregate overlaps its components. The stored Welch calculation assumes equal replicate counts and must only be reused after checking that condition. Its correction is per trajectory across scales, not across all 3300 trajectories. A stable label means resolved signs agree, not proof of equal ranking at every scale.

The existing PowerLawN has form E+A*N**(-alpha), with bounds E in [-2,3], A in [-10,10], alpha in [0.001,10]. Those amplitude bounds depend on the units of N. The new comparison will explicitly normalize N to millions and document its own fixed fitting specification; it will not claim a numerical reproduction of that bounded fitter. ConstantExtrapolator averages the largest observed scale. Preserve the existing selection code and its distinction between recommendations and certification.

## Sequential execution plan
1. Acquire the single pinned table and audit actual coverage, duplicates, metric validity, checkpoints, seed labels and provenance. Stop on unresolved invalidity.
2. Cross-scale ranking with a held-out target, identical recipe sets, explicit ties and descriptive regret.
3. Task heterogeneity, treating the aggregate separately.
4. Seed variation and all subsets; full-seed estimates are references, not truth.
5. Fixed linear, log-linear, power-law and last-observation methods, without target-driven tuning; retain failures.
6. Sensitivity to thresholds, ties, training cutoffs, targets and task exclusion.
7. Trace findings to saved results, including contrary evidence.
8. Re-run offline, compare numeric outputs, produce report and reproducibility package.

Without downloading data, only artifact inventory, historical count verification, synthetic numerical checks and software tests are feasible. The four new empirical investigations require the pinned macro-average table. No GPU work is required.
''', encoding='utf8')
print('Stage 0 audit and inventory saved')
