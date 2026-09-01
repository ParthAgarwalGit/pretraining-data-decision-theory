"""Smoke test: the package installs and imports correctly.

Exists so `pytest` has at least one test to collect (an empty test suite
exits non-zero) and so a broken editable install fails loudly at the very
first `make check` rather than silently in some later, harder-to-diagnose
task.
"""

import pdt


def test_package_imports():
    assert pdt.__version__ == "0.0.0"
