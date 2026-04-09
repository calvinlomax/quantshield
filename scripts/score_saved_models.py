"""Score saved QuantShield policy checkpoints across all replay duration suites."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield.model_scoring import build_model_score_summary
from quantshield.utils import ensure_directory, save_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score saved QuantShield policy checkpoints.")
    parser.add_argument(
        "--suite-root",
        default="outputs/replay_checkpoint_suites",
        help="Directory containing replay-duration checkpoint suites.",
    )
    parser.add_argument(
        "--output",
        default="outputs/replay_checkpoint_suites/model_scoreboard.csv",
        help="CSV path for the aggregated scoreboard.",
    )
    parser.add_argument(
        "--write-per-model",
        action="store_true",
        help="Write model_score_summary.csv into checkpoint directories when missing or stale.",
    )
    return parser.parse_args()


def _find_model_directories(root: Path) -> list[Path]:
    directories: list[Path] = []
    for benchmark_path in root.rglob("benchmark_summary.csv"):
        directory = benchmark_path.parent
        if (directory / "evaluation_summary.csv").exists():
            directories.append(directory)
    return sorted(set(directories))


def _duration_and_model_name(root: Path, directory: Path) -> tuple[str, str]:
    relative = directory.relative_to(root)
    parts = relative.parts
    duration_key = parts[0] if parts else "unknown"
    if len(parts) >= 3 and parts[1] == "candidate_models":
        return duration_key, parts[2]
    return duration_key, "promoted"


def _flatten_row(prefix: str, row: pd.Series) -> dict[str, float]:
    return {f"{prefix}_{column}": float(row[column]) for column in row.index}


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root)
    output_path = Path(args.output)
    ensure_directory(output_path.parent)

    rows: list[dict[str, object]] = []
    for directory in _find_model_directories(suite_root):
        benchmark_summary = pd.read_csv(directory / "benchmark_summary.csv", index_col=0)
        evaluation_summary = pd.read_csv(directory / "evaluation_summary.csv", index_col=0)
        model_score_summary = build_model_score_summary(benchmark_summary, evaluation_summary)
        if args.write_per_model:
            save_frame(model_score_summary, directory / "model_score_summary.csv")

        duration_key, model_name = _duration_and_model_name(suite_root, directory)
        all_score = model_score_summary.loc["all"]
        validation_score = model_score_summary.loc["validation"]
        all_benchmark = benchmark_summary.loc["all"]
        validation_benchmark = benchmark_summary.loc["validation"]
        all_evaluation = evaluation_summary.loc["all"]
        validation_evaluation = evaluation_summary.loc["validation"]
        rows.append(
            {
                "duration_key": duration_key,
                "model_name": model_name,
                "directory": str(directory),
                **_flatten_row("all_score", all_score),
                **_flatten_row("validation_score", validation_score),
                **_flatten_row("all_benchmark", all_benchmark),
                **_flatten_row("validation_benchmark", validation_benchmark),
                **_flatten_row("all_evaluation", all_evaluation),
                **_flatten_row("validation_evaluation", validation_evaluation),
            }
        )

    if not rows:
        raise SystemExit(f"No checkpoint directories with benchmark/evaluation summaries found under {suite_root}.")

    scoreboard = pd.DataFrame(rows).sort_values(
        ["all_score_composite_score", "validation_score_composite_score", "all_benchmark_policy_mean_excess_return"],
        ascending=[False, False, False],
    )
    save_frame(scoreboard, output_path)
    print(scoreboard.head(20).to_string(index=False, float_format=lambda value: f"{value:0.6f}"))
    print("")
    print(f"Saved scoreboard to {output_path}")


if __name__ == "__main__":
    main()
