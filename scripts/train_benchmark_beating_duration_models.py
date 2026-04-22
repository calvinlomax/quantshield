"""Train new duration-specific ETF-universe models until each horizon has two qualified candidates."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import subprocess
import sys

import pandas as pd

try:
    from scripts._common import bootstrap_project_root
except ImportError:  # pragma: no cover - direct script execution
    from _common import bootstrap_project_root

ROOT = bootstrap_project_root(__file__)

from quantshield.replay_durations import REPLAY_DURATION_PROFILES
from quantshield.universe import CANONICAL_TOP_ETF_UNIVERSE


DEFAULT_DURATION_FREQUENCIES = {
    "1mo": "B",
    "3mo": "3B",
    "6mo": "W-FRI",
    "1y": "2W-FRI",
    "3y": "ME",
    "5y": "ME",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ETF-benchmark-beating models for every replay duration.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Base QuantShield config.")
    parser.add_argument("--start-date", default="2018-01-01", help="Historical sample start date.")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="Historical sample end date.")
    parser.add_argument("--target-per-duration", type=int, default=2, help="Qualified models required per duration.")
    parser.add_argument("--max-rounds", type=int, default=3, help="Maximum fit rounds per duration.")
    parser.add_argument(
        "--candidate-mode",
        choices=["standard", "experimental", "comprehensive"],
        default="comprehensive",
        help="Candidate sweep size per fit round.",
    )
    parser.add_argument("--epochs", type=int, default=64, help="Base epoch budget for each fit round.")
    parser.add_argument("--batch-size", type=int, default=64, help="Base batch size for each fit round.")
    parser.add_argument("--seed", type=int, default=42, help="Initial seed. Each round increments it.")
    parser.add_argument("--device", help="Optional torch device override.")
    return parser.parse_args()


def _qualified_candidates(qualification_path: Path) -> list[str]:
    if not qualification_path.exists():
        return []
    frame = pd.read_csv(qualification_path)
    if frame.empty:
        return []
    required_splits = {"validation", "all"}
    qualified: list[str] = []
    for candidate in sorted(frame["candidate"].astype(str).unique()):
        candidate_rows = frame.loc[frame["candidate"] == candidate]
        if set(candidate_rows["split"]) != required_splits:
            continue
        if bool(candidate_rows["beats_all_tickers"].all()):
            qualified.append(candidate)
    return qualified


def main() -> None:
    args = parse_args()
    python_executable = sys.executable
    output_root = ROOT / "outputs" / "model_experiments"
    run_stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    summary_rows: list[dict[str, object]] = []

    for profile in REPLAY_DURATION_PROFILES:
        qualified_paths: list[str] = []
        duration_root = output_root / profile.key
        duration_root.mkdir(parents=True, exist_ok=True)

        for round_index in range(1, args.max_rounds + 1):
            if len(qualified_paths) >= args.target_per_duration:
                break

            round_name = f"{run_stamp}_benchmark_beater_round{round_index:02d}"
            output_dir = duration_root / round_name
            command = [
                python_executable,
                "scripts/fit_portfolio_model.py",
                "--config",
                args.config,
                "--name",
                f"{profile.key}_{round_name}",
                "--duration-key",
                profile.key,
                "--start-date",
                args.start_date,
                "--end-date",
                args.end_date,
                "--benchmark",
                "SPY",
                "--rebalance-frequency",
                DEFAULT_DURATION_FREQUENCIES.get(profile.key, "ME"),
                "--candidate-mode",
                args.candidate_mode,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(args.seed + round_index - 1),
                "--output-dir",
                str(output_dir),
                "--tickers",
                *CANONICAL_TOP_ETF_UNIVERSE,
            ]
            if args.device:
                command.extend(["--device", args.device])

            print(f"Training {profile.key} round {round_index} -> {output_dir}", flush=True)
            subprocess.run(command, check=True, cwd=ROOT)

            sweep_path = output_dir / "model_sweep.csv"
            if not sweep_path.exists():
                summary_rows.append(
                    {
                        "duration": profile.key,
                        "round": round_index,
                        "output_dir": output_dir.as_posix(),
                        "qualified_candidates": 0,
                        "status": "missing_model_sweep",
                    }
                )
                continue

            sweep = pd.read_csv(sweep_path)
            qualification_rows: list[dict[str, object]] = []
            for candidate_row in sweep.itertuples(index=False):
                candidate_dir = Path(candidate_row.candidate_dir)
                qualification_path = candidate_dir / "baseline_ticker_qualification.csv"
                candidate_qualification = pd.read_csv(qualification_path)
                for split_row in candidate_qualification.itertuples(index=False):
                    qualification_rows.append(
                        {
                            "candidate": candidate_row.candidate,
                            "split": split_row.split,
                            "beats_all_tickers": bool(split_row.beats_all_tickers),
                            "significant_vs_all_tickers": bool(split_row.significant_vs_all_tickers),
                            "min_mean_excess_return": float(split_row.min_mean_excess_return),
                            "candidate_dir": candidate_dir.as_posix(),
                        }
                    )
            qualification_frame = pd.DataFrame(qualification_rows)
            if not qualification_frame.empty:
                qualification_output = output_dir / "benchmark_etf_qualification.csv"
                qualification_frame.to_csv(qualification_output, index=False)
                for candidate in _qualified_candidates(qualification_output):
                    candidate_dir = sweep.loc[sweep["candidate"] == candidate, "candidate_dir"].iloc[0]
                    if candidate_dir not in qualified_paths:
                        qualified_paths.append(str(candidate_dir))

            summary_rows.append(
                {
                    "duration": profile.key,
                    "round": round_index,
                    "output_dir": output_dir.as_posix(),
                    "qualified_candidates": len(qualified_paths),
                    "status": "qualified" if len(qualified_paths) >= args.target_per_duration else "needs_more",
                }
            )

        summary_rows.append(
            {
                "duration": profile.key,
                "round": "final",
                "output_dir": duration_root.as_posix(),
                "qualified_candidates": len(qualified_paths),
                "status": "complete" if len(qualified_paths) >= args.target_per_duration else "incomplete",
            }
        )
        print(
            f"{profile.key}: found {len(qualified_paths)} qualified models "
            f"(target={args.target_per_duration}).",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_root / f"benchmark_beating_duration_summary_{run_stamp}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved duration summary to {summary_path}")


if __name__ == "__main__":
    main()
