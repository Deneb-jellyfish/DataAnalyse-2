"""Air quality data cleaning, aggregation, and city alignment pipeline."""

from __future__ import annotations

import glob
import math
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from utils.io import (
    ALIGNED_DIR,
    BEIJING_DIR,
    FIVE_CITY_DIR,
    MISSING_TOKENS,
    NUMERIC_COLUMNS,
    OVERLAP_END,
    OVERLAP_START,
    POLLUTANT_COLUMNS,
    PROCESSED_DIR,
    RAW_DIR,
    SHORT_GAP_LIMIT,
    SPIKE_COLUMNS,
    UNIFIED_FIELDS,
    ensure_output_dirs,
    estimate_humidity,
    fill_categorical,
    interpolate_short_gaps,
    mean_ignore_none,
    parse_float,
    parse_int,
    write_csv,
)


def remove_spikes_3sigma(
    values: list[float | None], sigma: float = 3.0
) -> tuple[list[float | None], int]:
    """Replace values outside global mean +/- sigma*std with None."""
    valid = [value for value in values if value is not None]
    if len(valid) < 2:
        return values, 0

    mean_value = sum(valid) / len(valid)
    variance = sum((value - mean_value) ** 2 for value in valid) / len(valid)
    std_value = math.sqrt(variance)
    if std_value == 0:
        return values, 0

    lower = mean_value - sigma * std_value
    upper = mean_value + sigma * std_value
    cleaned = list(values)
    removed = 0
    for index, value in enumerate(cleaned):
        if value is not None and (value < lower or value > upper):
            cleaned[index] = None
            removed += 1
    return cleaned, removed


def mark_long_gap_dates(
    datetimes: list[datetime], pm25_values: list[float | None], max_gap: int = SHORT_GAP_LIMIT
) -> set[date]:
    """Return dates that contain PM2.5 gaps longer than max_gap hours."""
    bad_dates: set[date] = set()
    index = 0
    length = len(pm25_values)
    while index < length:
        if pm25_values[index] is not None:
            index += 1
            continue
        start = index
        while index < length and pm25_values[index] is None:
            index += 1
        gap_size = index - start
        if gap_size > max_gap:
            for gap_index in range(start, index):
                bad_dates.add(datetimes[gap_index].date())
    return bad_dates


def drop_long_gap_rows(
    daily_rows: list[dict[str, Any]], bad_dates: set[str]
) -> tuple[list[dict[str, Any]], int]:
    """Remove daily rows whose date falls in bad_dates."""
    filtered = [row for row in daily_rows if row["date"] not in bad_dates]
    return filtered, len(daily_rows) - len(filtered)


def beijing_daily_to_unified(daily_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map Beijing PRSA daily rows to the unified handoff schema."""
    unified: list[dict[str, Any]] = []
    for row in daily_rows:
        temp = row.get("TEMP")
        dewp = row.get("DEWP")
        unified.append(
            {
                "date": row["date"],
                "city": "Beijing",
                "pm25": row.get("PM2.5"),
                "pm10": row.get("PM10"),
                "so2": row.get("SO2"),
                "no2": row.get("NO2"),
                "co": row.get("CO"),
                "o3": row.get("O3"),
                "temp": temp,
                "pres": row.get("PRES"),
                "dewp": dewp,
                "humidity": estimate_humidity(temp, dewp),
                "wind_dir": None,
                "wind_speed": row.get("WSPM"),
                "precipitation": row.get("RAIN"),
            }
        )
    return unified


def shanghai_hourly_to_daily(hourly_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate Shanghai hourly cleaned rows to daily means."""
    daily_bucket: dict[str, dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    wind_dirs: dict[str, list[str]] = defaultdict(list)

    for row in hourly_rows:
        day_key = row["datetime"][:10]
        for field, source in [
            ("pm25", "pm25_city"),
            ("temp", "TEMP"),
            ("pres", "PRES"),
            ("dewp", "DEWP"),
            ("humidity", "HUMI"),
            ("wind_speed", "wind_speed"),
            ("precipitation", "precipitation"),
        ]:
            value = row.get(source)
            if isinstance(value, (int, float)):
                daily_bucket[day_key][field].append(float(value))
        wind_value = row.get("wind_dir")
        if wind_value not in MISSING_TOKENS and wind_value is not None:
            wind_dirs[day_key].append(str(wind_value))

    daily_rows: list[dict[str, Any]] = []
    for day_key in sorted(daily_bucket):
        bucket = daily_bucket[day_key]
        mode_wind = Counter(wind_dirs[day_key]).most_common(1)
        daily_rows.append(
            {
                "date": day_key,
                "city": "Shanghai",
                "pm25": mean_ignore_none(bucket["pm25"]),
                "pm10": None,
                "so2": None,
                "no2": None,
                "co": None,
                "o3": None,
                "temp": mean_ignore_none(bucket["temp"]),
                "pres": mean_ignore_none(bucket["pres"]),
                "dewp": mean_ignore_none(bucket["dewp"]),
                "humidity": mean_ignore_none(bucket["humidity"]),
                "wind_dir": mode_wind[0][0] if mode_wind else None,
                "wind_speed": mean_ignore_none(bucket["wind_speed"]),
                "precipitation": mean_ignore_none(bucket["precipitation"]),
            }
        )
    return daily_rows


def align_cities(
    beijing_rows: list[dict[str, Any]], shanghai_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inner-join Beijing and Shanghai daily rows on date."""
    shanghai_by_date = {row["date"]: row for row in shanghai_rows}
    aligned_beijing: list[dict[str, Any]] = []
    aligned_shanghai: list[dict[str, Any]] = []
    for row in beijing_rows:
        sh_row = shanghai_by_date.get(row["date"])
        if sh_row is not None:
            aligned_beijing.append(row)
            aligned_shanghai.append(sh_row)
    return aligned_beijing, aligned_shanghai


def clean_prsa_files() -> dict[str, Any]:
    """Clean Beijing PRSA station files and write intermediate outputs."""
    prsa_files = sorted(glob.glob(str(RAW_DIR / "PRSA_Data_*.csv")))
    beijing_rows: list[dict[str, Any]] = []
    city_hourly: dict[datetime, dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    total_missing_before: Counter[str] = Counter()
    total_missing_after: Counter[str] = Counter()
    spike_counts: Counter[str] = Counter()
    station_bad_dates: set[date] = set()

    for file_path in prsa_files:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as handle:
            import csv

            reader = csv.DictReader(handle)
            station_rows: list[dict[str, Any]] = []
            for raw_row in reader:
                row: dict[str, Any] = {
                    "datetime": datetime(
                        parse_int(raw_row["year"]),
                        parse_int(raw_row["month"]),
                        parse_int(raw_row["day"]),
                        parse_int(raw_row["hour"]),
                    ),
                    "station": raw_row["station"],
                    "wd": None if raw_row["wd"] in MISSING_TOKENS else raw_row["wd"],
                }
                for column in NUMERIC_COLUMNS:
                    value = parse_float(raw_row[column])
                    if column in POLLUTANT_COLUMNS and value is not None and value < 0:
                        value = None
                    if column in {"RAIN", "WSPM"} and value is not None and value < 0:
                        value = None
                    row[column] = value
                    if value is None:
                        total_missing_before[column] += 1
                if row["wd"] is None:
                    total_missing_before["wd"] += 1
                station_rows.append(row)

        station_rows.sort(key=lambda item: item["datetime"])

        for column in SPIKE_COLUMNS:
            values = [row[column] for row in station_rows]
            cleaned, removed = remove_spikes_3sigma(values)
            spike_counts[column] += removed
            for row, cleaned_value in zip(station_rows, cleaned):
                row[column] = cleaned_value

        for column in NUMERIC_COLUMNS:
            values = [row[column] for row in station_rows]
            for row, filled_value in zip(station_rows, interpolate_short_gaps(values)):
                row[column] = filled_value

        wind_values = [row["wd"] for row in station_rows]
        for row, filled_value in zip(station_rows, fill_categorical(wind_values)):
            row["wd"] = filled_value

        datetimes = [row["datetime"] for row in station_rows]
        pm25_values = [row["PM2.5"] for row in station_rows]
        station_bad_dates.update(mark_long_gap_dates(datetimes, pm25_values))

        for row in station_rows:
            for column in NUMERIC_COLUMNS:
                if row[column] is None:
                    total_missing_after[column] += 1
            if row["wd"] in MISSING_TOKENS:
                total_missing_after["wd"] += 1

            beijing_rows.append(row)
            for column in NUMERIC_COLUMNS:
                city_hourly[row["datetime"]][column].append(row[column])

    beijing_rows.sort(key=lambda item: (item["station"], item["datetime"]))
    write_csv(
        BEIJING_DIR / "beijing_hourly_all_stations.csv",
        [
            "datetime",
            "station",
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
            "wd",
            "WSPM",
        ],
        [
            {
                "datetime": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                "station": row["station"],
                "PM2.5": row["PM2.5"],
                "PM10": row["PM10"],
                "SO2": row["SO2"],
                "NO2": row["NO2"],
                "CO": row["CO"],
                "O3": row["O3"],
                "TEMP": row["TEMP"],
                "PRES": row["PRES"],
                "DEWP": row["DEWP"],
                "RAIN": row["RAIN"],
                "wd": row["wd"],
                "WSPM": row["WSPM"],
            }
            for row in beijing_rows
        ],
    )

    city_rows: list[dict[str, Any]] = []
    daily_bucket: dict[date, dict[str, list[float | None]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for dt in sorted(city_hourly):
        row: dict[str, Any] = {"datetime": dt.strftime("%Y-%m-%d %H:%M:%S")}
        station_count = 0
        for column in NUMERIC_COLUMNS:
            averaged = mean_ignore_none(city_hourly[dt][column])
            row[column] = averaged
            if column == "PM2.5":
                station_count = len(
                    [value for value in city_hourly[dt][column] if value is not None]
                )
            if averaged is not None:
                daily_bucket[dt.date()][column].append(averaged)
        row["pm25_station_count"] = station_count
        city_rows.append(row)

    write_csv(
        BEIJING_DIR / "beijing_hourly_city.csv",
        [
            "datetime",
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
            "pm25_station_count",
        ],
        city_rows,
    )

    daily_rows: list[dict[str, Any]] = [
        {
            "date": date_value.isoformat(),
            **{
                column: mean_ignore_none(daily_bucket[date_value][column])
                for column in NUMERIC_COLUMNS
            },
        }
        for date_value in sorted(daily_bucket)
    ]
    daily_rows, dropped_days = drop_long_gap_rows(
        daily_rows, {d.isoformat() for d in station_bad_dates}
    )

    write_csv(
        BEIJING_DIR / "beijing_daily_city.csv",
        ["date"] + NUMERIC_COLUMNS,
        daily_rows,
    )

    return {
        "files": len(prsa_files),
        "rows": len(beijing_rows),
        "missing_before": dict(total_missing_before),
        "missing_after": dict(total_missing_after),
        "spike_counts": dict(spike_counts),
        "dropped_daily_rows": dropped_days,
        "daily_rows": daily_rows,
    }


def clean_five_city_files() -> dict[str, Any]:
    """Clean five-city PM files and write hourly outputs."""
    city_files = sorted(glob.glob(str(RAW_DIR / "othercity" / "*PM20100101_20151231.csv")))
    combined_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    shanghai_hourly: list[dict[str, Any]] = []

    common_numeric = ["DEWP", "HUMI", "PRES", "TEMP", "Iws", "precipitation", "Iprec"]

    for file_path in city_files:
        city_name = Path(file_path).name.replace("PM20100101_20151231.csv", "")
        with open(file_path, "r", encoding="utf-8-sig", newline="") as handle:
            import csv

            reader = csv.DictReader(handle)
            rows: list[dict[str, Any]] = []
            fieldnames = reader.fieldnames or []
            pm_columns = [name for name in fieldnames if name.startswith("PM_")]
            for raw_row in reader:
                dt = datetime(
                    parse_int(raw_row["year"]),
                    parse_int(raw_row["month"]),
                    parse_int(raw_row["day"]),
                    parse_int(raw_row["hour"]),
                )
                row: dict[str, Any] = {
                    "datetime": dt,
                    "city": city_name,
                    "season": parse_int(raw_row["season"]),
                    "cbwd": None if raw_row["cbwd"] in MISSING_TOKENS else raw_row["cbwd"],
                }
                for column in common_numeric + pm_columns:
                    value = parse_float(raw_row[column])
                    if column.startswith("PM_") and value is not None and value < 0:
                        value = None
                    if column in {"Iws", "precipitation", "Iprec"} and value is not None and value < 0:
                        value = None
                    row[column] = value
                rows.append(row)

        rows.sort(key=lambda item: item["datetime"])

        for column in common_numeric + pm_columns:
            values = [row[column] for row in rows]
            filled = interpolate_short_gaps(values)
            for row, filled_value in zip(rows, filled):
                row[column] = filled_value

        for categorical_column in ["cbwd"]:
            values = [row[categorical_column] for row in rows]
            filled = fill_categorical(values)
            for row, filled_value in zip(rows, filled):
                row[categorical_column] = filled_value

        cleaned_rows: list[dict[str, Any]] = []
        overlap_rows: list[dict[str, Any]] = []
        for row in rows:
            pm_values = [row[column] for column in pm_columns]
            row["pm25_city"] = mean_ignore_none(pm_values)
            row["pm_site_count"] = len([value for value in pm_values if value is not None])
            normalized = {
                "datetime": row["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                "city": row["city"],
                "season": row["season"],
                "pm25_city": row["pm25_city"],
                "pm_site_count": row["pm_site_count"],
                "DEWP": row["DEWP"],
                "HUMI": row["HUMI"],
                "PRES": row["PRES"],
                "TEMP": row["TEMP"],
                "wind_dir": row["cbwd"],
                "wind_speed": row["Iws"],
                "precipitation": row["precipitation"],
                "cum_precipitation": row["Iprec"],
            }
            cleaned_rows.append(normalized)
            if OVERLAP_START <= row["datetime"] <= OVERLAP_END:
                overlap_rows.append(normalized)
                combined_rows.append(normalized)
            if city_name == "Shanghai":
                shanghai_hourly.append(normalized)

        write_csv(
            FIVE_CITY_DIR / f"{city_name.lower()}_hourly_clean.csv",
            [
                "datetime",
                "city",
                "season",
                "pm25_city",
                "pm_site_count",
                "DEWP",
                "HUMI",
                "PRES",
                "TEMP",
                "wind_dir",
                "wind_speed",
                "precipitation",
                "cum_precipitation",
            ],
            cleaned_rows,
        )
        write_csv(
            FIVE_CITY_DIR / f"{city_name.lower()}_hourly_overlap_20130301_20151231.csv",
            [
                "datetime",
                "city",
                "season",
                "pm25_city",
                "pm_site_count",
                "DEWP",
                "HUMI",
                "PRES",
                "TEMP",
                "wind_dir",
                "wind_speed",
                "precipitation",
                "cum_precipitation",
            ],
            overlap_rows,
        )

        summary[city_name] = {
            "rows": len(cleaned_rows),
            "overlap_rows": len(overlap_rows),
            "pm_columns": pm_columns,
        }

    combined_rows.sort(key=lambda item: (item["city"], item["datetime"]))
    write_csv(
        FIVE_CITY_DIR / "five_city_hourly_overlap_combined.csv",
        [
            "datetime",
            "city",
            "season",
            "pm25_city",
            "pm_site_count",
            "DEWP",
            "HUMI",
            "PRES",
            "TEMP",
            "wind_dir",
            "wind_speed",
            "precipitation",
            "cum_precipitation",
        ],
        combined_rows,
    )

    summary["shanghai_hourly"] = shanghai_hourly
    return summary


def write_preprocess_log(result: dict[str, Any]) -> None:
    """Write preprocessing statistics to docs/preprocess_log.md."""
    from utils.io import DOCS_DIR

    beijing = result["beijing_summary"]
    log_path = DOCS_DIR / "preprocess_log.md"
    lines = [
        "# 数据预处理日志",
        "",
        "## 北京 PRSA 数据",
        "",
        f"- 处理文件数：{beijing['files']}",
        f"- 站点小时级行数：{beijing['rows']}",
        f"- 长缺口剔除后日级行数：{result['beijing_rows']}",
        f"- 剔除日级行数（PM2.5 长缺口）：{beijing.get('dropped_daily_rows', 0)}",
        "",
        "### 插值前缺失值",
        "",
    ]
    for column, count in sorted(
        beijing["missing_before"].items(), key=lambda item: item[1], reverse=True
    ):
        if count:
            lines.append(f"- {column}: {count}")
    lines.extend(["", "### 插值后缺失值", ""])
    for column, count in sorted(
        beijing["missing_after"].items(), key=lambda item: item[1], reverse=True
    ):
        if count:
            lines.append(f"- {column}: {count}")
    lines.extend(["", "### 3σ 尖刺剔除数量", ""])
    for column, count in sorted(
        beijing.get("spike_counts", {}).items(), key=lambda item: item[1], reverse=True
    ):
        if count:
            lines.append(f"- {column}: {count}")

    lines.extend(
        [
            "",
            "## 统一交接文件",
            "",
            f"- `beijing.csv` 行数：{result['beijing_rows']}",
            f"- `shanghai.csv` 行数：{result['shanghai_rows']}",
            f"- `aligned/beijing.csv` 行数：{result['aligned_beijing_rows']}",
            f"- `aligned/shanghai.csv` 行数：{result['aligned_shanghai_rows']}",
            f"- 对齐日期范围：{OVERLAP_START.date().isoformat()} 至 {OVERLAP_END.date().isoformat()}",
            "",
            "## 五城市小时级输出",
            "",
        ]
    )
    for city_name, details in result["city_summary"].items():
        if city_name == "shanghai_hourly":
            continue
        lines.append(
            f"- {city_name}: {details['rows']} 行，"
            f"重叠时段 {details['overlap_rows']} 行"
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full_preprocess() -> dict[str, Any]:
    """Run the full preprocessing pipeline and write all outputs."""
    ensure_output_dirs()

    beijing_summary = clean_prsa_files()
    city_summary = clean_five_city_files()

    beijing_unified = beijing_daily_to_unified(beijing_summary["daily_rows"])
    shanghai_unified = shanghai_hourly_to_daily(city_summary["shanghai_hourly"])
    aligned_beijing, aligned_shanghai = align_cities(beijing_unified, shanghai_unified)

    write_csv(PROCESSED_DIR / "beijing.csv", UNIFIED_FIELDS, beijing_unified)
    write_csv(PROCESSED_DIR / "shanghai.csv", UNIFIED_FIELDS, shanghai_unified)
    write_csv(ALIGNED_DIR / "beijing.csv", UNIFIED_FIELDS, aligned_beijing)
    write_csv(ALIGNED_DIR / "shanghai.csv", UNIFIED_FIELDS, aligned_shanghai)

    result = {
        "beijing_summary": beijing_summary,
        "city_summary": {k: v for k, v in city_summary.items() if k != "shanghai_hourly"},
        "beijing_rows": len(beijing_unified),
        "shanghai_rows": len(shanghai_unified),
        "aligned_beijing_rows": len(aligned_beijing),
        "aligned_shanghai_rows": len(aligned_shanghai),
    }
    write_preprocess_log(result)
    return result
