"""CLI entry point for the air quality cleaning pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.preprocess import run_full_preprocess


def main() -> None:
    """Run full preprocessing and print a short summary."""
    result = run_full_preprocess()
    beijing_summary = result["beijing_summary"]
    city_summary = result["city_summary"]

    print("Cleaning complete.")
    print(f"PRSA files cleaned: {beijing_summary['files']}")
    print(f"PRSA rows written: {beijing_summary['rows']}")
    print(f"Beijing daily rows (unified): {result['beijing_rows']}")
    print(f"Beijing hourly rows (unified): {result['beijing_hourly_rows']}")
    print(f"Shanghai daily rows (unified): {result['shanghai_rows']}")
    print(f"Shanghai hourly rows (unified): {result['shanghai_hourly_rows']}")
    print(f"Aligned Beijing rows: {result['aligned_beijing_rows']}")
    print(f"Aligned Shanghai rows: {result['aligned_shanghai_rows']}")
    print("PRSA missing values before interpolation:")
    for column, count in sorted(
        beijing_summary["missing_before"].items(), key=lambda item: item[1], reverse=True
    ):
        if count:
            print(f"  {column}: {count}")
    print("PRSA missing values after interpolation:")
    for column, count in sorted(
        beijing_summary["missing_after"].items(), key=lambda item: item[1], reverse=True
    ):
        if count:
            print(f"  {column}: {count}")
    print("3-sigma spike removals:")
    for column, count in sorted(
        beijing_summary.get("spike_counts", {}).items(), key=lambda item: item[1], reverse=True
    ):
        if count:
            print(f"  {column}: {count}")
    print("Five-city files cleaned:")
    for city_name, details in city_summary.items():
        pm_fields = ", ".join(details["pm_columns"])
        print(
            f"  {city_name}: {details['rows']} rows, "
            f"{details['overlap_rows']} overlap rows, PM fields: {pm_fields}"
        )


if __name__ == "__main__":
    main()
