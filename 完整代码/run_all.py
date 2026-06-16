"""Unified project runner for the AQI hourly pipeline.

This script stitches together the existing repo entry points into one
top-level command so the project has a clear "run the main workflow"
interface. By default it uses the runtime-safe settings already adopted in
the handoff scripts for the deep models.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    """One executable pipeline step."""

    title: str
    command: list[str]


def build_steps(args: argparse.Namespace) -> list[Step]:
    """Build the ordered command list for the selected workflow."""
    py = args.python
    steps: list[Step] = []

    if not args.skip_preprocess:
        steps.append(
            Step(
                title="Preprocess data and build features",
                command=[
                    py,
                    "main.py",
                    "--steps",
                    "all",
                    "--granularity",
                    args.granularity,
                ],
            )
        )

    if not args.skip_training:
        steps.extend(
            [
                Step(
                    title="Train XGBoost (h1 + seq6)",
                    command=[py, "scripts/train_xgboost.py", "--task", "both"],
                ),
                Step(
                    title="Train ARIMA (h1 + seq6)",
                    command=[py, "scripts/train_arima.py"],
                ),
            ]
        )

        prophet_cmd = [py, "scripts/train_prophet.py", "--task", "both"]
        if args.run_prophet:
            prophet_cmd.append("--run")
        steps.append(
            Step(
                title="Train Prophet (h1 + seq6)"
                if args.run_prophet
                else "Write Prophet blocked outputs",
                command=prophet_cmd,
            )
        )

        steps.extend(
            [
                Step(
                    title=f"Train LSTM (h1 + seq6, profile={args.lstm_profile})",
                    command=[
                        py,
                        "scripts/train_lstm.py",
                        "--task",
                        "both",
                        "--profile",
                        args.lstm_profile,
                    ],
                ),
                Step(
                    title="Train Transformer (h1 + seq6)",
                    command=[
                        py,
                        "scripts/train_transformer.py",
                        "--task",
                        "both",
                        "--epochs",
                        str(args.transformer_epochs),
                        "--patience",
                        str(args.transformer_patience),
                        "--batch-size",
                        str(args.transformer_batch_size),
                        "--lr",
                        str(args.transformer_lr),
                    ],
                ),
            ]
        )

    if not args.skip_analysis:
        steps.append(
            Step(
                title="Run hourly analysis",
                command=[py, "scripts/run_hourly_analysis.py"],
            )
        )

    if not args.skip_cross_city:
        steps.append(
            Step(
                title="Run cross-city XGBoost evaluation",
                command=[py, "scripts/run_cross_city_xgboost.py"],
            )
        )

    if not args.skip_assemble:
        steps.append(
            Step(
                title="Assemble summary outputs",
                command=[py, "scripts/assemble_hourly_outputs.py"],
            )
        )

    if not args.skip_figures:
        steps.append(
            Step(
                title="Generate report figures",
                command=[py, "scripts/generate_hourly_figures.py"],
            )
        )

    if not args.skip_shap:
        steps.append(
            Step(
                title="Generate SHAP figures",
                command=[py, "scripts/generate_shap_figures.py"],
            )
        )

    return steps


def run_step(step: Step, dry_run: bool) -> None:
    """Print and execute one pipeline step."""
    printable = " ".join(step.command)
    print(f"\n=== {step.title} ===")
    print(printable)
    if dry_run:
        return
    subprocess.run(step.command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the AQI project main workflow")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for child scripts (default: current interpreter).",
    )
    parser.add_argument(
        "--granularity",
        default="hourly",
        choices=["daily", "hourly", "both"],
        help="Feature granularity passed to main.py.",
    )
    parser.add_argument(
        "--lstm-profile",
        default="handoff",
        choices=["full", "handoff"],
        help="LSTM profile. 'handoff' is the safer default for a full repo run.",
    )
    parser.add_argument(
        "--transformer-epochs",
        type=int,
        default=60,
        help="Base Transformer epochs for h1; seq6 uses +20 inside the training script.",
    )
    parser.add_argument("--transformer-patience", type=int, default=8)
    parser.add_argument("--transformer-batch-size", type=int, default=64)
    parser.add_argument("--transformer-lr", type=float, default=5e-4)
    parser.add_argument(
        "--run-prophet",
        action="store_true",
        help="Actually train Prophet. If omitted, the existing script writes blocked outputs.",
    )
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip main.py preprocessing/features step.")
    parser.add_argument("--skip-training", action="store_true", help="Skip all model training steps.")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip scripts/run_hourly_analysis.py.")
    parser.add_argument("--skip-cross-city", action="store_true", help="Skip scripts/run_cross_city_xgboost.py.")
    parser.add_argument("--skip-assemble", action="store_true", help="Skip scripts/assemble_hourly_outputs.py.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip scripts/generate_hourly_figures.py.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip scripts/generate_shap_figures.py.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the command sequence without executing it.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the selected pipeline steps in order."""
    args = parse_args()
    steps = build_steps(args)
    if not steps:
        print("No steps selected. Nothing to run.")
        return

    print(f"Project root: {ROOT}")
    if not args.run_prophet:
        print("Prophet note: --run-prophet not set, so Prophet will emit blocked outputs.")

    for step in steps:
        run_step(step, dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete.")
    else:
        print("\nAll selected steps completed.")


if __name__ == "__main__":
    main()
