# Reproducing Beyond Rank Reversals

This course self-study uses AI2's released DataDecide evaluation scores. It does not train models or require a GPU. Start with `submission/Beyond_Rank_Reversals_Final_Report.pdf`. The Word and Markdown versions are editable. The score table, checksums, scripts, stage records and actual outputs are included in the full archive.

## Live demonstration for the course meeting

Use Python 3.12. From the repository root, run `python tools/study_demo.py`. The demo uses Python's standard library and the included observations. It independently recomputes all 143 task/size correlations, winner selections and regrets, checking against the reference results. The default 530M-to-1B display should show correlation 0.830, winners 4/10 and regret 0.713 percentage points.

Try changing the source size or task during the meeting.

```
python tools/study_demo.py --source 4M
python tools/study_demo.py --source 750M --task boolq
python tools/study_demo.py --output demo-output
```

The last command creates a new directory containing `demo.html`, `summary.json` and `comparisons.csv`. Open `demo.html` in a browser. Choose a new output name for each run. The target remains 1B. The selected recipe and target winner are printed for each task. The full command below also runs the seed, curve-fitting and robustness analyses.

## Full study on a new machine

Use Python 3.12. Create a virtual environment with `python -m venv .venv` and activate it. On Windows PowerShell use `.venv\Scripts\Activate.ps1`. On macOS/Linux use `source .venv/bin/activate`. Install `python -m pip install -r requirements-study.txt`.

The full submission ZIP includes the pinned raw input. A GitHub clone or source ZIP needs an initial download of the 139 MB evaluation table. Run `python tools/study_download.py` to retrieve it and its dataset card. Dependency installation and the initial download need internet access.

From the repository root run:

```
python tools/study_reproduce.py --output reproduced-study
```

Choose a directory that does not exist yet. After dependencies are installed, the command runs offline using the pinned Parquet table. It recomputes stages 1 through 7 in order, runs six focused tests and compares CSV rows, columns and numerical values with the reference artifacts. It also checks the result JSON files. The program exits nonzero if a check fails. Logs, generated methods notes and the comparison record `reproducibility_verification.json` are written to the fresh output directory.

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

Per-stage Git SHA records identify the working parent commit, while script SHA-256 identifies the exact executing script. For source archives without Git metadata, the recorded Git commit is null and the script hash still identifies the executing code. The full submission archive also includes `study_revision.json` naming its source commit. Most analyses are deterministic enumerations and least-squares fits. The 20261103 seed recorded in provenance is reserved for stochastic extensions.

## Data attribution and limitations

Magnusson et al., *DataDecide: How to Predict Best Pretraining Data with Small Experiments* (2025). Dataset: https://huggingface.co/datasets/allenai/DataDecide-eval-results, revision `9919b5a0e61e57a85021263918fa82d6ceaee038`. The upstream dataset card and attribution accompany the downloaded input. The original repository code and new analysis code are separate from the upstream dataset license. Input Parquet SHA-256: `b5012e963952ba2b5814bc07c260f8389c288db0ad7558e312db49fa1ab57fdf`.

The aggregate task is not independent of the ten component tasks. Three seeds provide limited uncertainty information. Subset agreement is not independent replication. Rank correlations, top-1 success, numeric prediction error and regret answer different questions. Neither reduced real-world compute use nor a universal algorithmic guarantee is demonstrated.
