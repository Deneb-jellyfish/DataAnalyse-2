"""CLI entry point for feature matrix generation."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.feature_engineering import build_feature_matrices


def main() -> None:
    """Build feature matrices and print summary."""
    metadata = build_feature_matrices()
    print("Feature engineering complete.")
    print(f"Valid rows: {metadata['valid_rows']}")
    print(f"X_train shape: ({metadata['n_train']}, {metadata['n_features']})")
    print(f"X_test shape: ({metadata['n_test']}, {metadata['n_features']})")
    print(f"Train period: {metadata['train_start']} to {metadata['train_end']}")
    print(f"Test period: {metadata['test_start']} to {metadata['test_end']}")
    print("Features:")
    for name in metadata["feature_columns"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
