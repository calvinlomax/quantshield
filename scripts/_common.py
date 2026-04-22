"""Shared runtime helpers for top-level QuantShield scripts."""

from __future__ import annotations

from pathlib import Path
import sys


def bootstrap_project_root(caller_file: str) -> Path:
    """Return the repo root and ensure repo imports resolve consistently.

    This supports both direct script execution, e.g. ``python scripts/foo.py``,
    and package-style imports from tests, e.g. ``import scripts.foo``.
    """
    root = Path(caller_file).resolve().parents[1]
    root_str = str(root)
    src_str = str(root / "src")
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return root
