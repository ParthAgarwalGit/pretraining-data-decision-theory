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

# Provenance validation (src/pdt/provenance.py) lands in task P0-04. Until
# then this target degrades gracefully instead of failing setup for anyone
# who clones the repo between P0-03 and P0-04 landing.
check: lint test
	@if [ -f src/pdt/provenance.py ]; then \
		uv run python -m pdt.provenance --validate results/ ; \
	else \
		echo "pdt.provenance not yet implemented (lands in task P0-04) -- skipping provenance validation" ; \
	fi

# Regenerates every figure in figures/ from results/*.json. The build_all
# entry point lands in task P1-11.
figures:
	uv run python -m pdt.viz.build_all

# Builds paper/main.tex. Lands in task P6-01.
paper:
	latexmk -pdf -cd paper/main.tex

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
