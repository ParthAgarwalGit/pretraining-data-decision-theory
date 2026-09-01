# Environment

Verified by the executing agent as task P0-01, folded into the repo as part
of P0-02 (repo creation).

**Checked:** 2026-09-01 (UTC offset not queried; local machine clock)
**Machine:** Windows-11-10.0.26200-SP0, drive C: 925 GB total / 546 GB available

## Toolchain

| Tool | Version | Notes |
|---|---|---|
| git | 2.51.0.windows.2 | |
| GitHub CLI (`gh`) | 2.96.0 | Logged in as `ParthAgarwalGit`, scopes `gist, read:org, repo, workflow` — sufficient for repo creation, branches, PRs |
| Python | 3.12.10 | Matches `pyproject.toml` constraint planned for P0-03 (`>=3.11,<3.13`) <!-- NUMBER-OK --> |
| `uv` | 0.11.3 | |
| `huggingface_hub` | 0.35.3 | |
| `hf` CLI | present at `/c/Users/Parth/AppData/Local/Programs/Python/Python312/Scripts/hf` | `hf auth whoami` → user `Parth4105`, orgs `Algoverse-AYJP` — matches the expected identity from `plan/00-agent-protocol.md` §4 |

## GPU

NVIDIA GeForce RTX 4060 Laptop, 8188 MiB VRAM, driver 566.07, CUDA 12.7. <!-- NUMBER-OK: hardware/driver specs, not a research result -->

**Sufficient for Phase 1 (analysis, no training) and Phase 3 (algorithm
development, simulation, offline replay against DataDecide — all table
lookups, no training).**

**Not sufficient for Phase 4 (confirmatory pretraining runs).** Even the
smallest ladder rungs in `plan/08-compute-requests.md` are sized for A100
40GB-class hardware and a multi-day job; Phase 4 is explicitly gated behind
GATE-C and a provisioned cluster, not this machine.

## Credentials status

- **GitHub:** authenticated, write-capable (`repo` + `workflow` scopes present).
  Sufficient to create the repository and open PRs once GATE-0 naming is
  approved (P0-02).
- **Hugging Face:** authenticated via `hf auth whoami` as `Parth4105`, member
  of org `Algoverse-AYJP`. Token scope was not independently re-verified
  against "write" access in this session (the MCP identity check in P0-08 tool
  discovery reported `scopes: openid, profile, read-mcp, read-repos, jobs,
  contribute-repos, inference-api` for the OAuth-backed connector) — **before
  any HF push (P0-08 onward), confirm the local `hf` CLI token specifically
  has write access**, since CLI login and the MCP connector may be backed by
  different credentials. If `hf upload` fails with a permission error, stop
  and ask the PI for a write token per protocol §4; do not attempt to work
  around it.

## Build tooling (added during P0-03)

- **`make`** was not present on this machine (`/usr/bin/bash: make: command not
  found`) — Windows has no default `make`, unlike the `ubuntu-latest` CI
  runner planned for P0-05. Installed GNU Make 3.81 <!-- NUMBER-OK: tool version --> via
  `winget install -e --id GnuWin32.Make` and added
  `C:\Program Files (x86)\GnuWin32\bin` to the **user** PATH permanently.
  **New terminal sessions pick this up automatically; a shell that was
  already open when this ran will not** — export the directory onto `PATH`
  manually for the remainder of an already-running session.
- **`uv lock` / `uv sync`** both work without any extra configuration; PyPI
  is reachable from this machine. `uv sync --all-extras` resolves 77
  packages and installs 74 (`pdt` itself is the 74th, editable).
- **Note on `huggingface_hub`:** the version floor pinned in
  `pyproject.toml` (`>=0.35`) <!-- NUMBER-OK: version constraint, not a result --> resolved to **1.29.0** — a major-version jump
  from the 0.35.3 seen system-wide in P0-01. Not a blocker (the API used so
  far, `hf auth whoami`, is stable across the bump), but flagging it because
  `src/pdt/data/datadecide.py` (P0-06) and `src/pdt/hub.py` (P0-08) should be
  written and tested against the **locked** version (1.29.0), not assumed
  to match older `huggingface_hub` 0.x documentation or examples.

## Outcome

All P0-01 checks pass. `make setup && make check` verified passing on a
genuine fresh clone (not just the working tree) as of P0-03. No blockers.
