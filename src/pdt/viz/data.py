"""Tiny shared results-loading helper for every F1-F5 module."""

from __future__ import annotations

import json
from pathlib import Path


def load(path: str) -> dict:
    with open(Path(path), encoding="utf-8") as f:
        return json.load(f)["data"]
