"""Launch the QuantShield desktop replay app."""

from __future__ import annotations

import sys

try:
    from scripts._common import bootstrap_project_root
except ImportError:  # pragma: no cover - direct script execution
    from _common import bootstrap_project_root

bootstrap_project_root(__file__)

from quantshield_app.main import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
