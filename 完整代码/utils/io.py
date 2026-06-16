"""CSV I/O helpers and project path constants."""

from __future__ import annotations

import csv
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BEIJING_DIR = PROCESSED_DIR / "beijing"
FIVE_CITY_DIR = PROCESSED_DIR / "five_city"
ALIGNED_DIR = PROCESSED_DIR / "aligned"
FEATURES_DIR = PROCESSED_DIR / "features"
FEATURES_HOURLY_DIR = PROCESSED_DIR / "features_hourly"
ALIGNED_HOURLY_DIR = PROCESSED_DIR / "aligned_hourly"
DOCS_DIR = ROOT / "docs"

MISSING_TOKENS: set[Any] = {"", "NA", "NaN", "nan", None}
SHORT_GAP_LIMIT = 6
OVERLAP_START = datetime(2013, 3, 1, 0, 0, 0)
OVERLAP_END = datetime(2015, 12, 31, 23, 0, 0)

UNIFIED_FIELDS = [
    "date",
    "city",
    "pm25",
    "pm10",
    "so2",
    "no2",
    "co",
    "o3",
    "temp",
    "pres",
    "dewp",
    "humidity",
    "wind_dir",
    "wind_speed",
    "precipitation",
]

UNIFIED_HOURLY_FIELDS = [
    "datetime",
    "city",
    "pm25",
    "pm10",
    "so2",
    "no2",
    "co",
    "o3",
    "temp",
    "pres",
    "dewp",
    "humidity",
    "wind_dir",
    "wind_speed",
    "precipitation",
]

SPIKE_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "WSPM"]
NUMERIC_COLUMNS = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]
POLLUTANT_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]


def parse_float(value: Any) -> float | None:
    """Parse a CSV cell into a float or None for missing values."""
    if value in MISSING_TOKENS:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def parse_int(value: Any) -> int | None:
    """Parse a CSV cell into an int or None for missing values."""
    if value in MISSING_TOKENS:
        return None
    return int(value)


def format_number(value: Any) -> str:
    """Format numeric values for CSV output."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def mean_ignore_none(values: Sequence[float | None]) -> float | None:
    """Return the arithmetic mean of non-null values."""
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def interpolate_short_gaps(
    values: Sequence[float | None], max_gap: int = SHORT_GAP_LIMIT
) -> list[float | None]:
    """Linearly interpolate missing runs up to max_gap hours."""
    filled = list(values)
    index = 0
    length = len(filled)
    while index < length:
        if filled[index] is not None:
            index += 1
            continue
        start = index
        while index < length and filled[index] is None:
            index += 1
        end = index
        gap_size = end - start
        left_index = start - 1
        right_index = end
        left_value = filled[left_index] if left_index >= 0 else None
        right_value = filled[right_index] if right_index < length else None
        if gap_size <= max_gap and left_value is not None and right_value is not None:
            step = (right_value - left_value) / (gap_size + 1)
            for offset in range(gap_size):
                filled[start + offset] = left_value + step * (offset + 1)
    return filled


def fill_categorical(values: Sequence[Any]) -> list[Any]:
    """Forward-fill categorical values and fall back to the mode."""
    filled = list(values)
    counter = Counter(value for value in filled if value not in MISSING_TOKENS)
    fallback = counter.most_common(1)[0][0] if counter else ""
    previous = None
    for index, value in enumerate(filled):
        if value in MISSING_TOKENS:
            if previous is not None:
                filled[index] = previous
            else:
                filled[index] = None
        else:
            previous = value
    for index, value in enumerate(filled):
        if value in MISSING_TOKENS:
            filled[index] = fallback
    return filled


def estimate_humidity(temp: float | None, dewp: float | None) -> float | None:
    """Estimate relative humidity (%) from temperature and dew point."""
    if temp is None or dewp is None:
        return None
    try:
        es_t = 6.112 * math.exp((17.67 * temp) / (temp + 243.5))
        es_d = 6.112 * math.exp((17.67 * dewp) / (dewp + 243.5))
        if es_t == 0:
            return None
        return min(100.0, max(0.0, 100.0 * es_d / es_t))
    except (OverflowError, ZeroDivisionError):
        return None


def ensure_output_dirs() -> None:
    """Create processed output directories if needed."""
    for directory in (
        BEIJING_DIR,
        FIVE_CITY_DIR,
        ALIGNED_DIR,
        ALIGNED_HOURLY_DIR,
        FEATURES_DIR,
        FEATURES_HOURLY_DIR,
        DOCS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    """Write rows to a UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted: dict[str, str] = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, float):
                    formatted[field] = format_number(value)
                else:
                    formatted[field] = "" if value is None else str(value)
            writer.writerow(formatted)
