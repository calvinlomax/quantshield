"""Desktop application entrypoint for QuantShield."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the QuantShield PySide6 desktop app for policy inference and replay."
    )
    parser.add_argument(
        "--checkpoint-root",
        action="append",
        default=[],
        help="Optional root directory to search for actor_critic_policy.pt checkpoints. Can be passed multiple times.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - depends on optional desktop deps
        raise SystemExit(
            "The desktop app requires PySide6. Install it with `pip install -r requirements-app.txt` or "
            "`pip install -e .[app]`."
        ) from exc

    from quantshield_app.services import CheckpointService
    from quantshield_app.ui import QuantShieldMainWindow

    search_roots = args.checkpoint_root or None
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    window = QuantShieldMainWindow(
        checkpoint_service=CheckpointService(search_roots=search_roots) if search_roots else CheckpointService()
    )
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - manual launcher
    raise SystemExit(main())
