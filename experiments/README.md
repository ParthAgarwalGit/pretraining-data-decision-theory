# experiments/

Thin runnable scripts, one per task ID (e.g. `p1_06_decomposition.py`).
Each script imports from `src/pdt/`, reads a config from `configs/`, and
writes its output to `results/<task_id>_<name>.json` via
`pdt.provenance.write_result` (see `plan/01-phase0-setup.md` task P0-04).

No analysis logic lives here — only orchestration. Logic belongs in
`src/pdt/` where it can be unit tested.
