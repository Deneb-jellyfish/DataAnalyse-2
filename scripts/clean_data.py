import csv
import glob
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
BEIJING_DIR = PROCESSED_DIR / "beijing"
FIVE_CITY_DIR = PROCESSED_DIR / "five_city"

MISSING_TOKENS = {"", "NA", "NaN", "nan", None}
SHORT_GAP_LIMIT = 6
OVERLAP_START = datetime(2013, 3, 1, 0, 0, 0)
OVERLAP_END = datetime(2015, 12, 31, 23, 0, 0)


def parse_float(value):
    if value in MISSING_TOKENS:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def parse_int(value):
    if value in MISSING_TOKENS:
        return None
    return int(value)


def format_number(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def mean_ignore_none(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def interpolate_short_gaps(values, max_gap=SHORT_GAP_LIMIT):
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
        if (
            gap_size <= max_gap
            and left_value is not None
            and right_value is not None
        ):
            step = (right_value - left_value) / (gap_size + 1)
            for offset in range(gap_size):
                filled[start + offset] = left_value + step * (offset + 1)
    return filled


def fill_categorical(values):
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


def ensure_output_dirs():
    BEIJING_DIR.mkdir(parents=True, exist_ok=True)
    FIVE_CITY_DIR.mkdir(parents=True, exist_ok=True)


def clean_prsa_files():
    prsa_files = sorted(glob.glob(str(RAW_DIR / "PRSA_Data_*.csv")))
    beijing_rows = []
    city_hourly = defaultdict(lambda: defaultdict(list))
    total_missing_before = Counter()
    total_missing_after = Counter()

    numeric_columns = [
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
    pollutant_columns = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

    for file_path in prsa_files:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            station_rows = []
            for raw_row in reader:
                row = {
                    "datetime": datetime(
                        parse_int(raw_row["year"]),
                        parse_int(raw_row["month"]),
                        parse_int(raw_row["day"]),
                        parse_int(raw_row["hour"]),
                    ),
                    "station": raw_row["station"],
                    "wd": None if raw_row["wd"] in MISSING_TOKENS else raw_row["wd"],
                }
                for column in numeric_columns:
                    value = parse_float(raw_row[column])
                    if column in pollutant_columns and value is not None and value < 0:
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

        for column in numeric_columns:
            values = [row[column] for row in station_rows]
            for row, filled_value in zip(
                station_rows, interpolate_short_gaps(values)
            ):
                row[column] = filled_value
        wind_values = [row["wd"] for row in station_rows]
        for row, filled_value in zip(station_rows, fill_categorical(wind_values)):
            row["wd"] = filled_value

        for row in station_rows:
            for column in numeric_columns:
                if row[column] is None:
                    total_missing_after[column] += 1
            if row["wd"] in MISSING_TOKENS:
                total_missing_after["wd"] += 1

            beijing_rows.append(row)
            for column in numeric_columns:
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

    city_rows = []
    daily_bucket = defaultdict(lambda: defaultdict(list))
    for dt in sorted(city_hourly):
        row = {"datetime": dt.strftime("%Y-%m-%d %H:%M:%S")}
        station_count = 0
        for column in numeric_columns:
            averaged = mean_ignore_none(city_hourly[dt][column])
            row[column] = averaged
            if column == "PM2.5":
                station_count = len([x for x in city_hourly[dt][column] if x is not None])
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

    daily_rows = []
    for date_value in sorted(daily_bucket):
        row = {"date": date_value.isoformat()}
        for column in numeric_columns:
            row[column] = mean_ignore_none(daily_bucket[date_value][column])
        daily_rows.append(row)

    write_csv(
        BEIJING_DIR / "beijing_daily_city.csv",
        ["date"] + numeric_columns,
        daily_rows,
    )

    return {
        "files": len(prsa_files),
        "rows": len(beijing_rows),
        "missing_before": dict(total_missing_before),
        "missing_after": dict(total_missing_after),
    }


def clean_five_city_files():
    city_files = sorted(glob.glob(str(RAW_DIR / "othercity" / "*PM20100101_20151231.csv")))
    combined_rows = []
    summary = {}

    common_numeric = ["DEWP", "HUMI", "PRES", "TEMP", "Iws", "precipitation", "Iprec"]

    for file_path in city_files:
        city_name = Path(file_path).name.replace("PM20100101_20151231.csv", "")
        with open(file_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = []
            fieldnames = reader.fieldnames or []
            pm_columns = [name for name in fieldnames if name.startswith("PM_")]
            for raw_row in reader:
                dt = datetime(
                    parse_int(raw_row["year"]),
                    parse_int(raw_row["month"]),
                    parse_int(raw_row["day"]),
                    parse_int(raw_row["hour"]),
                )
                row = {
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

        cleaned_rows = []
        overlap_rows = []
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
    return summary


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for field in fieldnames:
                value = row.get(field)
                if isinstance(value, float):
                    formatted[field] = format_number(value)
                else:
                    formatted[field] = "" if value is None else str(value)
            writer.writerow(formatted)


def print_summary(beijing_summary, city_summary):
    print("Cleaning complete.")
    print(f"PRSA files cleaned: {beijing_summary['files']}")
    print(f"PRSA rows written: {beijing_summary['rows']}")
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
    print("Five-city files cleaned:")
    for city_name, details in city_summary.items():
        pm_fields = ", ".join(details["pm_columns"])
        print(
            f"  {city_name}: {details['rows']} rows, "
            f"{details['overlap_rows']} overlap rows, PM fields: {pm_fields}"
        )


def main():
    ensure_output_dirs()
    beijing_summary = clean_prsa_files()
    city_summary = clean_five_city_files()
    print_summary(beijing_summary, city_summary)


if __name__ == "__main__":
    main()
