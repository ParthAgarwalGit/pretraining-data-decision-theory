.PHONY: setup lint format test check figures paper clean

# Installs all optional-dependency groups (core, hub, fitting, plotting, dev)
# and activates the pre-commit hooks. Does NOT install the `train` extra --
# that arrives in task P4-01 and is opt-in via `uv sync --all-extras --extra train`.
setup:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest -q

# Every JSON file under results/ must carry valid provenance (non-dirty git
# SHA) -- see src/pdt/provenance.py and plan/00-agent-protocol.md Rule 3.
check: lint test
	uv run python -m pdt.provenance --validate results/

# Regenerates every figure in figures/ from results/*.json. The build_all
# entry point lands in task P1-11.
figures:
	uv run python -m pdt.viz.build_all

# Builds paper/main.tex. Lands in task P6-01.
paper:
	latexmk -pdf -cd paper/main.tex

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
