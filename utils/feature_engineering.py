"""Feature engineering for Beijing PM2.5 prediction tasks."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.io import FEATURES_DIR, FEATURES_HOURLY_DIR, PROCESSED_DIR


class StandardScaler:
    """Minimal feature scaler matching sklearn StandardScaler interface."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        """Compute per-column mean and standard deviation."""
        self.mean_ = np.mean(x, axis=0)
        self.scale_ = np.std(x, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Standardize features with fitted statistics."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler has not been fit.")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit scaler and transform input."""
        return self.fit(x).transform(x)


WIND_DIR_ENCODING = {
    "N": 0.0,
    "NNE": math.pi / 8,
    "NE": math.pi / 4,
    "ENE": 3 * math.pi / 8,
    "E": math.pi / 2,
    "ESE": 5 * math.pi / 8,
    "SE": 3 * math.pi / 4,
    "SSE": 7 * math.pi / 8,
    "S": math.pi,
    "SSW": -7 * math.pi / 8,
    "SW": -3 * math.pi / 4,
    "WSW": -5 * math.pi / 8,
    "W": -math.pi / 2,
    "WNW": -3 * math.pi / 8,
    "NW": -math.pi / 4,
    "NNW": -math.pi / 8,
    "cv": 0.0,
}

DAILY_FEATURE_DESCRIPTIONS = {
    "pm25_lag1": "PM2.5 one day ago.",
    "pm25_lag3": "PM2.5 three days ago.",
    "pm25_lag7": "PM2.5 one week ago.",
    "pm25_roll_mean_7": "Trailing 7-day PM2.5 mean using only past values.",
    "pm25_roll_mean_14": "Trailing 14-day PM2.5 mean using only past values.",
    "pm25_roll_std_7": "Trailing 7-day PM2.5 standard deviation.",
    "pm25_roll_std_14": "Trailing 14-day PM2.5 standard deviation.",
    "month_sin": "Month-of-year cyclic encoding (sine).",
    "month_cos": "Month-of-year cyclic encoding (cosine).",
    "weekday": "Weekday index, Monday=0.",
    "is_holiday": "Chinese public holiday indicator.",
    "temp": "Average temperature.",
    "pres": "Average pressure.",
    "dewp": "Average dew point.",
    "wind_speed": "Average wind speed.",
    "precipitation": "Daily precipitation.",
    "temp_x_humidity": "Temperature multiplied by humidity.",
    "wind_speed_x_wind_dir_sin": "Wind speed scaled by wind-direction sine.",
}

HOURLY_FEATURE_DESCRIPTIONS = {
    "pm25_lag1": "PM2.5 one hour ago.",
    "pm25_lag3": "PM2.5 three hours ago.",
    "pm25_lag6": "PM2.5 six hours ago.",
    "pm25_lag12": "PM2.5 twelve hours ago.",
    "pm25_lag18": "PM2.5 eighteen hours ago.",
    "pm25_lag24": "PM2.5 one day ago at the same hour.",
    "pm25_lag36": "PM2.5 thirty-six hours ago.",
    "pm25_lag48": "PM2.5 two days ago at the same hour.",
    "pm25_roll_mean_6": "Trailing 6-hour PM2.5 mean using only past values.",
    "pm25_roll_mean_12": "Trailing 12-hour PM2.5 mean using only past values.",
    "pm25_roll_mean_24": "Trailing 24-hour PM2.5 mean using only past values.",
    "pm25_roll_mean_168": "Trailing 7-day PM2.5 mean using only past values.",
    "pm25_roll_std_6": "Trailing 6-hour PM2.5 standard deviation.",
    "pm25_roll_std_12": "Trailing 12-hour PM2.5 standard deviation.",
    "pm25_roll_std_24": "Trailing 24-hour PM2.5 standard deviation.",
    "pm25_roll_std_168": "Trailing 7-day PM2.5 standard deviation.",
    "pm25_diff_1": "Current PM2.5 minus PM2.5 one hour ago.",
    "pm25_diff_3": "Current PM2.5 minus PM2.5 three hours ago.",
    "pm25_diff_6": "Current PM2.5 minus PM2.5 six hours ago.",
    "pm25_diff_12": "Current PM2.5 minus PM2.5 twelve hours ago.",
    "hour_sin": "Hour-of-day cyclic encoding (sine).",
    "hour_cos": "Hour-of-day cyclic encoding (cosine).",
    "month_sin": "Month-of-year cyclic encoding (sine).",
    "month_cos": "Month-of-year cyclic encoding (cosine).",
    "weekday": "Weekday index, Monday=0.",
    "is_weekend": "Weekend indicator.",
    "is_holiday": "Chinese public holiday indicator.",
    "is_daytime": "Indicator for 06:00-17:59 local time.",
    "is_rush_hour": "Indicator for commute-heavy hours (7-9, 17-19).",
    "temp": "Current temperature.",
    "pres": "Current pressure.",
    "dewp": "Current dew point.",
    "humidity": "Current humidity.",
    "wind_speed": "Current wind speed.",
    "precipitation": "Current precipitation.",
    "wind_dir_sin": "Wind direction encoded as sine.",
    "wind_dir_cos": "Wind direction encoded as cosine.",
    "precipitation_flag": "Indicator for any precipitation in the current hour.",
    "dewp_temp_gap": "Temperature minus dew point.",
    "temp_diff_1": "Current temperature minus temperature one hour ago.",
    "temp_diff_3": "Current temperature minus temperature three hours ago.",
    "pres_diff_1": "Current pressure minus pressure one hour ago.",
    "pres_diff_3": "Current pressure minus pressure three hours ago.",
    "wind_speed_diff_1": "Current wind speed minus wind speed one hour ago.",
    "wind_speed_diff_3": "Current wind speed minus wind speed three hours ago.",
    "temp_x_humidity": "Temperature multiplied by humidity.",
    "wind_speed_x_wind_dir_sin": "Wind speed scaled by wind-direction sine.",
    "wind_speed_x_pm25_lag1": "Wind speed multiplied by PM2.5 one hour ago.",
}


def _is_holiday(date_value: pd.Timestamp) -> int:
    """Return 1 if date is a Chinese public holiday, else 0."""
    try:
        import chinesecalendar

        return int(chinesecalendar.is_holiday(pd.Timestamp(date_value).date()))
    except Exception:
        return 0


def encode_wind_dir(value: Any) -> float:
    """Encode wind direction label to radians."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return WIND_DIR_ENCODING.get(str(value).strip(), 0.0)


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construct model features from unified Beijing daily data."""
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    frame["pm25_lag1"] = frame["pm25"].shift(1)
    frame["pm25_lag3"] = frame["pm25"].shift(3)
    frame["pm25_lag7"] = frame["pm25"].shift(7)
    frame["pm25_roll_mean_7"] = frame["pm25"].shift(1).rolling(7).mean()
    frame["pm25_roll_mean_14"] = frame["pm25"].shift(1).rolling(14).mean()
    frame["pm25_roll_std_7"] = frame["pm25"].shift(1).rolling(7).std()
    frame["pm25_roll_std_14"] = frame["pm25"].shift(1).rolling(14).std()

    frame["month_sin"] = np.sin(2 * np.pi * frame["date"].dt.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["date"].dt.month / 12)
    frame["weekday"] = frame["date"].dt.weekday
    frame["is_holiday"] = frame["date"].map(_is_holiday)

    humidity = frame["humidity"].fillna(0.0)
    temp = frame["temp"].fillna(0.0)
    wind_speed = frame["wind_speed"].fillna(0.0)
    wind_dir_rad = frame["wind_dir"].map(encode_wind_dir)

    frame["temp_x_humidity"] = temp * humidity
    frame["wind_speed_x_wind_dir_sin"] = wind_speed * np.sin(wind_dir_rad)

    feature_columns = [
        "pm25_lag1",
        "pm25_lag3",
        "pm25_lag7",
        "pm25_roll_mean_7",
        "pm25_roll_mean_14",
        "pm25_roll_std_7",
        "pm25_roll_std_14",
        "month_sin",
        "month_cos",
        "weekday",
        "is_holiday",
        "temp",
        "pres",
        "dewp",
        "wind_speed",
        "precipitation",
        "temp_x_humidity",
        "wind_speed_x_wind_dir_sin",
    ]
    return frame, feature_columns


def build_hourly_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construct richer hourly features aimed at multi-hour PM2.5 forecasting."""
    frame = df.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values("datetime").reset_index(drop=True)

    # Mid-range lags help +12h prediction because they preserve half-day memory
    # without relying only on the previous hour or the previous day.
    frame["pm25_lag1"] = frame["pm25"].shift(1)
    frame["pm25_lag3"] = frame["pm25"].shift(3)
    frame["pm25_lag6"] = frame["pm25"].shift(6)
    frame["pm25_lag12"] = frame["pm25"].shift(12)
    frame["pm25_lag18"] = frame["pm25"].shift(18)
    frame["pm25_lag24"] = frame["pm25"].shift(24)
    frame["pm25_lag36"] = frame["pm25"].shift(36)
    frame["pm25_lag48"] = frame["pm25"].shift(48)

    frame["pm25_roll_mean_6"] = frame["pm25"].shift(1).rolling(6).mean()
    frame["pm25_roll_mean_12"] = frame["pm25"].shift(1).rolling(12).mean()
    frame["pm25_roll_mean_24"] = frame["pm25"].shift(1).rolling(24).mean()
    frame["pm25_roll_mean_168"] = frame["pm25"].shift(1).rolling(168).mean()
    frame["pm25_roll_std_6"] = frame["pm25"].shift(1).rolling(6).std()
    frame["pm25_roll_std_12"] = frame["pm25"].shift(1).rolling(12).std()
    frame["pm25_roll_std_24"] = frame["pm25"].shift(1).rolling(24).std()
    frame["pm25_roll_std_168"] = frame["pm25"].shift(1).rolling(168).std()

    frame["pm25_diff_1"] = frame["pm25"] - frame["pm25_lag1"]
    frame["pm25_diff_3"] = frame["pm25"] - frame["pm25_lag3"]
    frame["pm25_diff_6"] = frame["pm25"] - frame["pm25_lag6"]
    frame["pm25_diff_12"] = frame["pm25"] - frame["pm25_lag12"]

    frame["hour_sin"] = np.sin(2 * np.pi * frame["datetime"].dt.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["datetime"].dt.hour / 24)
    frame["month_sin"] = np.sin(2 * np.pi * frame["datetime"].dt.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["datetime"].dt.month / 12)
    frame["weekday"] = frame["datetime"].dt.weekday
    frame["is_weekend"] = (frame["weekday"] >= 5).astype(int)
    frame["is_holiday"] = frame["datetime"].dt.date.map(_is_holiday)

    hour = frame["datetime"].dt.hour
    frame["is_daytime"] = hour.between(6, 17, inclusive="both").astype(int)
    frame["is_rush_hour"] = hour.isin([7, 8, 9, 17, 18, 19]).astype(int)

    humidity = frame["humidity"].fillna(0.0)
    temp = frame["temp"].fillna(0.0)
    pressure = frame["pres"].fillna(0.0)
    dew_point = frame["dewp"].fillna(0.0)
    wind_speed = frame["wind_speed"].fillna(0.0)
    wind_dir_rad = frame["wind_dir"].map(encode_wind_dir)
    precipitation = frame["precipitation"].fillna(0.0)

    frame["humidity"] = humidity
    frame["wind_dir_sin"] = np.sin(wind_dir_rad)
    frame["wind_dir_cos"] = np.cos(wind_dir_rad)
    frame["precipitation_flag"] = (precipitation > 0).astype(int)
    frame["dewp_temp_gap"] = temp - dew_point
    frame["temp_diff_1"] = temp - temp.shift(1)
    frame["temp_diff_3"] = temp - temp.shift(3)
    frame["pres_diff_1"] = pressure - pressure.shift(1)
    frame["pres_diff_3"] = pressure - pressure.shift(3)
    frame["wind_speed_diff_1"] = wind_speed - wind_speed.shift(1)
    frame["wind_speed_diff_3"] = wind_speed - wind_speed.shift(3)
    frame["temp_x_humidity"] = temp * humidity
    frame["wind_speed_x_wind_dir_sin"] = wind_speed * np.sin(wind_dir_rad)
    frame["wind_speed_x_pm25_lag1"] = wind_speed * frame["pm25_lag1"]

    feature_columns = [
        "pm25_lag1",
        "pm25_lag3",
        "pm25_lag6",
        "pm25_lag12",
        "pm25_lag18",
        "pm25_lag24",
        "pm25_lag36",
        "pm25_lag48",
        "pm25_roll_mean_6",
        "pm25_roll_mean_12",
        "pm25_roll_mean_24",
        "pm25_roll_mean_168",
        "pm25_roll_std_6",
        "pm25_roll_std_12",
        "pm25_roll_std_24",
        "pm25_roll_std_168",
        "pm25_diff_1",
        "pm25_diff_3",
        "pm25_diff_6",
        "pm25_diff_12",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "weekday",
        "is_weekend",
        "is_holiday",
        "is_daytime",
        "is_rush_hour",
        "temp",
        "pres",
        "dewp",
        "humidity",
        "wind_speed",
        "precipitation",
        "wind_dir_sin",
        "wind_dir_cos",
        "precipitation_flag",
        "dewp_temp_gap",
        "temp_diff_1",
        "temp_diff_3",
        "pres_diff_1",
        "pres_diff_3",
        "wind_speed_diff_1",
        "wind_speed_diff_3",
        "temp_x_humidity",
        "wind_speed_x_wind_dir_sin",
        "wind_speed_x_pm25_lag1",
    ]
    return frame, feature_columns


def build_feature_matrices(
    input_path: Path | None = None,
    output_dir: Path | None = None,
    test_ratio: float = 0.2,
    granularity: str = "daily",
) -> dict[str, Any]:
    """Build scaled train/test feature matrices and persist numpy artifacts."""
    if granularity not in {"daily", "hourly"}:
        raise ValueError("granularity must be 'daily' or 'hourly'")

    if input_path is None:
        input_path = (
            PROCESSED_DIR / "beijing_hourly.csv"
            if granularity == "hourly"
            else PROCESSED_DIR / "beijing.csv"
        )
    if output_dir is None:
        output_dir = FEATURES_HOURLY_DIR if granularity == "hourly" else FEATURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    if granularity == "hourly":
        frame, feature_columns = build_hourly_feature_frame(raw)
        time_column = "datetime"
        time_format = "%Y-%m-%d %H:%M:%S"
    else:
        frame, feature_columns = build_feature_frame(raw)
        time_column = "date"
        time_format = "%Y-%m-%d"

    model_frame = frame.dropna(subset=feature_columns + ["pm25"]).reset_index(drop=True)
    if model_frame.empty:
        raise ValueError("No valid rows remain after feature construction.")

    split_index = int(len(model_frame) * (1 - test_ratio))
    train_frame = model_frame.iloc[:split_index]
    test_frame = model_frame.iloc[split_index:]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_frame[feature_columns].to_numpy(dtype=np.float64))
    x_test = scaler.transform(test_frame[feature_columns].to_numpy(dtype=np.float64))
    y_train = train_frame["pm25"].to_numpy(dtype=np.float64)
    y_test = test_frame["pm25"].to_numpy(dtype=np.float64)
    dates_train = train_frame[time_column].dt.strftime(time_format).to_numpy()
    dates_test = test_frame[time_column].dt.strftime(time_format).to_numpy()

    np.save(output_dir / "X_train.npy", x_train)
    np.save(output_dir / "X_test.npy", x_test)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "y_test.npy", y_test)
    np.save(output_dir / "dates_train.npy", dates_train)
    np.save(output_dir / "dates_test.npy", dates_test)

    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as handle:
        json.dump(feature_columns, handle, indent=2)

    with open(output_dir / "scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)

    metadata = {
        "granularity": granularity,
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "feature_columns": feature_columns,
        "n_train": int(x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
        "n_features": int(x_train.shape[1]),
        "train_start": str(dates_train[0]),
        "train_end": str(dates_train[-1]),
        "test_start": str(dates_test[0]),
        "test_end": str(dates_test[-1]),
        "valid_rows": int(len(model_frame)),
        "dropped_rows": int(len(raw) - len(model_frame)),
    }
    write_feature_doc(metadata)
    return metadata


def write_feature_doc(metadata: dict[str, Any]) -> None:
    """Write a readable feature reference for the generated matrices."""
    from utils.io import DOCS_DIR

    granularity = metadata.get("granularity", "daily")
    if granularity == "hourly":
        title = "Hourly Feature Reference"
        input_path = "data/processed/beijing_hourly.csv"
        output_dir = "data/processed/features_hourly/"
        load_dir = "data/processed/features_hourly"
        descriptions = HOURLY_FEATURE_DESCRIPTIONS
        doc_path = DOCS_DIR / "feature_doc_hourly.md"
    else:
        title = "Daily Feature Reference"
        input_path = "data/processed/beijing.csv"
        output_dir = "data/processed/features/"
        load_dir = "data/processed/features"
        descriptions = DAILY_FEATURE_DESCRIPTIONS
        doc_path = DOCS_DIR / "feature_doc.md"

    lines = [
        f"# {title}",
        "",
        "## Overview",
        "",
        "- Target: `pm25`",
        f"- Input dataset: `{input_path}`",
        f"- Output directory: `{output_dir}`",
        "",
        "## Train/Test Split",
        "",
        f"- Valid rows after feature construction: {metadata['valid_rows']}",
        f"- Dropped rows caused by lag/rolling windows: {metadata['dropped_rows']}",
        f"- Train rows: {metadata['n_train']} ({metadata['train_start']} -> {metadata['train_end']})",
        f"- Test rows: {metadata['n_test']} ({metadata['test_start']} -> {metadata['test_end']})",
        "- Split rule: chronological split, last 20% used as the test set",
        "",
        "## Feature List",
        "",
        "| Feature | Description |",
        "|---|---|",
    ]

    for feature_name in metadata["feature_columns"]:
        description = descriptions.get(feature_name, "Derived feature.")
        lines.append(f"| {feature_name} | {description} |")

    lines.extend(
        [
            "",
            "## Matrix Shapes",
            "",
            f"- X_train: ({metadata['n_train']}, {metadata['n_features']})",
            f"- X_test: ({metadata['n_test']}, {metadata['n_features']})",
            f"- y_train: ({metadata['n_train']},)",
            f"- y_test: ({metadata['n_test']},)",
            "",
            "## Loading Example",
            "",
            "```python",
            "import json",
            "import numpy as np",
            "",
            f"X_train = np.load('{load_dir}/X_train.npy')",
            f"y_train = np.load('{load_dir}/y_train.npy')",
            f"with open('{load_dir}/feature_names.json', encoding='utf-8') as f:",
            "    feature_names = json.load(f)",
            "```",
            "",
            "The scaler is fit on the training split only and stored as `scaler.pkl`.",
        ]
    )

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")
