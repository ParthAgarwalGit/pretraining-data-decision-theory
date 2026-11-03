# Reproducing Beyond Rank Reversals

This course self-study uses AI2's released DataDecide evaluation scores. It does not train models or require a GPU. Start with `submission/Beyond_Rank_Reversals_Final_Report.pdf`. The Word and Markdown versions are editable. The score table, checksums, scripts, stage records and actual outputs are included in the full archive.

## Run on a new machine

Use Python 3.12 and install `python -m pip install -r requirements-study.txt`. Package installation may require internet access. From the extracted repository root run:

```
python tools/study_reproduce.py --output reproduced-study
```

Choose a directory that does not exist yet. After dependencies are installed, the command runs offline using the bundled pinned Parquet table. It recomputes analytical stages 1 through 7 in order, runs the six focused tests and compares CSV rows/columns and numerical values with the supplied reference artifacts. It also checks the result JSON files. The program exits nonzero if a check fails. Individual stage stdout/stderr are retained as logs in the fresh output directory. The comparison record is written to `results/empirical_study/reproducibility_verification.json`.

The original Stage 0 audit in `docs/research_audit.md` and its inventory describe the initial checkout **before** the download. Preserve these records rather than rerunning the inventory over the post-download tree. Stage 8 is packaging/report composition. `python tools/study_notebooks.py` rebuilds the five result-inspection notebooks; `python tools/build_empirical_report.py` regenerates the editable report from the supplied outputs. Export that DOCX to PDF in Word or LibreOffice. Do not overwrite the historical `results/p*_*.json` files.

## Layout

| Location | Purpose |
|---|---|
| `data/study_raw/` | Pinned AI2 macro-average Parquet file, dataset card and source manifest |
| `results/empirical_study/analysis_observations.csv` | Final-checkpoint observations keyed by recipe, size, task and seed |
| `results/empirical_study/` | Audits, per-setting measurements, sensitivity analyses, findings, verification and provenance |
| `figures/empirical_study/` | Plots generated from the numerical results |
| `notebooks/01_...` to `05_...` | Executable result-inspection notebooks with actual captured output and figures |
| `tools/empirical_study.py` | Statistical calculations, stages 1–7 |
| `tests/test_empirical_study.py` | Independent hand-calculated and SciPy reference checks |
| `docs/` | Per-stage methods and integrated findings |
| `submission/` | Final PDF, editable Word and Markdown |

All new empirical outputs are separate from historical results. Per-stage Git SHA records identify the working parent commit, while script SHA-256 identifies the exact executing script. `study_revision.json` identifies the packed source commit when running without a `.git` folder. Most analyses are deterministic enumerations and least-squares fits, not random resampling; the 20261103 seed recorded in provenance is reserved, not a claim of additional stochastic replications.

## Data attribution and limitations

Magnusson et al., *DataDecide: How to Predict Best Pretraining Data with Small Experiments* (2025). Dataset: https://huggingface.co/datasets/allenai/DataDecide-eval-results, revision `9919b5a0e61e57a85021263918fa82d6ceaee038`. The upstream dataset card and attribution accompany the downloaded input. The original repository code and new analysis code are separate from the upstream dataset license. Input Parquet SHA-256: `b5012e963952ba2b5814bc07c260f8389c288db0ad7558e312db49fa1ab57fdf`.

The aggregate task is not independent of the ten component tasks. Three seeds provide limited uncertainty information. Subset agreement is not independent replication. Rank correlations, top-1 success, numeric prediction error and regret answer different questions. Neither reduced real-world compute use nor a universal algorithmic guarantee is demonstrated.
