"""Move legacy flat hourly artifacts into semantic subdirectories."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hourly_output_paths import ensure_hourly_output_dirs, organize_existing_hourly_outputs


def main() -> None:
    ensure_hourly_output_dirs()
    moves = organize_existing_hourly_outputs()
    if not moves:
        print("No root-level hourly artifacts needed moving.")
        return

    print(f"Moved {len(moves)} hourly artifacts:")
    for source, target in moves:
        print(f"- {source.relative_to(ROOT).as_posix()} -> {target.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
