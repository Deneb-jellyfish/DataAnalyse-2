"""Generate data/processed/beijing_hourly.csv from beijing_hourly_city.csv.

This is a lightweight conversion that maps PRSA column names to unified schema,
without re-running the full 12-station preprocessing pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import UNIFIED_HOURLY_FIELDS, estimate_humidity, write_csv, PROCESSED_DIR


def main() -> None:
    src = PROCESSED_DIR / "beijing" / "beijing_hourly_city.csv"
    dst = PROCESSED_DIR / "beijing_hourly.csv"

    print(f"Reading: {src}")
    df = pd.read_csv(src)
    print(f"  rows: {len(df)}, cols: {list(df.columns)}")

    rows = []
    skipped = 0
    for _, row in df.iterrows():
        temp = row.get("TEMP")
        dewp = row.get("DEWP")
        pm25 = row.get("PM2.5")

        # Skip rows where PM2.5 is missing (can't be used for modeling)
        if pd.isna(pm25):
            skipped += 1
            continue

        humidity = estimate_humidity(temp, dewp)

        rows.append({
            "datetime": str(row["datetime"]),
            "city": "Beijing",
            "pm25": pm25,
            "pm10": row.get("PM10"),
            "so2": row.get("SO2"),
            "no2": row.get("NO2"),
            "co": row.get("CO"),
            "o3": row.get("O3"),
            "temp": temp,
            "pres": row.get("PRES"),
            "dewp": dewp,
            "humidity": humidity,
            "wind_dir": None,
            "wind_speed": row.get("WSPM"),
            "precipitation": row.get("RAIN"),
        })

    print(f"  valid rows: {len(rows)}, skipped (missing PM2.5): {skipped}")
    write_csv(dst, UNIFIED_HOURLY_FIELDS, rows)
    print(f"Wrote: {dst}")


if __name__ == "__main__":
    main()
