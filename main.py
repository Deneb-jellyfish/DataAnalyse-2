"""One-click pipeline entry point (Member A preprocessing + feature steps)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_preprocess() -> None:
    """Run data cleaning and city alignment."""
    from utils.preprocess import run_full_preprocess

    result = run_full_preprocess()
    print(
        f"[preprocess] Beijing daily: {result['beijing_rows']}, "
        f"Shanghai daily: {result['shanghai_rows']}, "
        f"Aligned: {result['aligned_beijing_rows']}"
    )


def run_features() -> None:
    """Build feature matrices for tree-based models."""
    from utils.feature_engineering import build_feature_matrices

    metadata = build_feature_matrices()
    print(
        f"[features] X_train {metadata['n_train']}x{metadata['n_features']}, "
        f"X_test {metadata['n_test']}x{metadata['n_features']}"
    )


def main() -> None:
    """Execute pipeline steps selected via CLI flags."""
    parser = argparse.ArgumentParser(description="AQI project pipeline")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["preprocess", "features"],
        choices=["preprocess", "features", "all"],
        help="Pipeline steps to run",
    )
    args = parser.parse_args()
    steps = args.steps
    if "all" in steps:
        steps = ["preprocess", "features"]

    if "preprocess" in steps:
        run_preprocess()
    if "features" in steps:
        run_features()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
