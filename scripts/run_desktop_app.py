"""Launch the QuantShield desktop replay app."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield_app.main import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
