"""Build duration-specific replay checkpoint suites for the desktop app."""

from __future__ import annotations

import argparse
import subprocess
import sys

try:
    from scripts._common import bootstrap_project_root
except ImportError:  # pragma: no cover - direct script execution
    from _common import bootstrap_project_root

ROOT = bootstrap_project_root(__file__)

from quantshield.replay_durations import REPLAY_DURATION_PROFILES, checkpoint_root_for_duration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all duration-specific replay checkpoint suites.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Base QuantShield config.")
    parser.add_argument("--start-date", default="2018-01-01", help="Historical sample start date.")
    parser.add_argument("--candidate-pool-size", type=int, default=80, help="Random S&P 500 candidate pool size.")
    parser.add_argument("--random-universes", type=int, default=256, help="Random universes per duration suite.")
    parser.add_argument("--device", help="Optional torch device override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_executable = sys.executable

    for profile in REPLAY_DURATION_PROFILES:
        output_dir = checkpoint_root_for_duration(profile.key)
        command = [
            python_executable,
            "scripts/train_random_sp500_policy.py",
            "--config",
            args.config,
            "--duration-key",
            profile.key,
            "--output-dir",
            str(output_dir),
            "--start-date",
            args.start_date,
            "--candidate-pool-size",
            str(args.candidate_pool_size),
            "--random-universes",
            str(args.random_universes),
        ]
        if args.device:
            command.extend(["--device", args.device])

        print(f"Training duration suite {profile.key} -> {output_dir}", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
