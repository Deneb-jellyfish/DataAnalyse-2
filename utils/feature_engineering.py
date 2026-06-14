"""Feature engineering for Beijing PM2.5 daily prediction."""

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


def _is_holiday(date_value: pd.Timestamp) -> int:
    """Return 1 if date is a Chinese public holiday, else 0."""
    try:
        import chinese_calendar

        return int(chinese_calendar.is_holiday(date_value.date()))
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
    """Construct 48-dim hourly features tuned for h1/h6 PM2.5 prediction."""
    frame = df.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.sort_values("datetime").reset_index(drop=True)

    # Ensure numeric columns are available and stable
    for col in ["pm25", "temp", "pres", "dewp", "humidity", "wind_speed", "precipitation"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Core lag features
    frame["pm25_lag1"] = frame["pm25"].shift(1)
    frame["pm25_lag2"] = frame["pm25"].shift(2)
    frame["pm25_lag3"] = frame["pm25"].shift(3)
    frame["pm25_lag4"] = frame["pm25"].shift(4)
    frame["pm25_lag6"] = frame["pm25"].shift(6)
    frame["pm25_lag8"] = frame["pm25"].shift(8)
    frame["pm25_lag12"] = frame["pm25"].shift(12)
    frame["pm25_lag24"] = frame["pm25"].shift(24)

    # Rolling statistics (past-only)
    frame["pm25_roll_mean_3"] = frame["pm25"].shift(1).rolling(3).mean()
    frame["pm25_roll_mean_6"] = frame["pm25"].shift(1).rolling(6).mean()
    frame["pm25_roll_mean_12"] = frame["pm25"].shift(1).rolling(12).mean()
    frame["pm25_roll_mean_24"] = frame["pm25"].shift(1).rolling(24).mean()
    frame["pm25_roll_std_3"] = frame["pm25"].shift(1).rolling(3).std()
    frame["pm25_roll_std_6"] = frame["pm25"].shift(1).rolling(6).std()
    frame["pm25_roll_std_12"] = frame["pm25"].shift(1).rolling(12).std()
    frame["pm25_roll_std_24"] = frame["pm25"].shift(1).rolling(24).std()

    # Trend / difference features
    frame["pm25_diff_1"] = frame["pm25"] - frame["pm25_lag1"]
    frame["pm25_diff_2"] = frame["pm25"] - frame["pm25_lag2"]
    frame["pm25_diff_3"] = frame["pm25"] - frame["pm25_lag3"]
    frame["pm25_diff_6"] = frame["pm25"] - frame["pm25_lag6"]

    # Time / calendar features
    hour = frame["datetime"].dt.hour
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    frame["month_sin"] = np.sin(2 * np.pi * frame["datetime"].dt.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["datetime"].dt.month / 12)
    frame["weekday"] = frame["datetime"].dt.weekday
    frame["is_weekend"] = frame["weekday"].isin([5, 6]).astype(int)
    frame["is_holiday"] = frame["datetime"].dt.date.map(_is_holiday)
    frame["is_daytime"] = hour.between(6, 17).astype(int)
    frame["is_rush_hour"] = hour.isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Meteorological raw + directional transform
    wind_dir_rad = frame["wind_dir"].map(encode_wind_dir)
    frame["wind_dir_sin"] = np.sin(wind_dir_rad)
    frame["wind_dir_cos"] = np.cos(wind_dir_rad)
    frame["precipitation_flag"] = (frame["precipitation"].fillna(0.0) > 0).astype(int)
    frame["dewp_temp_gap"] = frame["temp"] - frame["dewp"]

    # Meteorological trend features
    frame["temp_diff_1"] = frame["temp"] - frame["temp"].shift(1)
    frame["temp_diff_2"] = frame["temp"] - frame["temp"].shift(2)
    frame["temp_diff_6"] = frame["temp"] - frame["temp"].shift(6)
    frame["pres_diff_1"] = frame["pres"] - frame["pres"].shift(1)
    frame["pres_diff_2"] = frame["pres"] - frame["pres"].shift(2)
    frame["pres_diff_6"] = frame["pres"] - frame["pres"].shift(6)
    frame["wind_speed_diff_1"] = frame["wind_speed"] - frame["wind_speed"].shift(1)
    frame["wind_speed_diff_2"] = frame["wind_speed"] - frame["wind_speed"].shift(2)
    frame["wind_speed_diff_6"] = frame["wind_speed"] - frame["wind_speed"].shift(6)

    # Interaction features
    frame["temp_x_humidity"] = frame["temp"].fillna(0.0) * frame["humidity"].fillna(0.0)
    frame["wind_speed_x_wind_dir_sin"] = frame["wind_speed"].fillna(0.0) * frame["wind_dir_sin"]
    frame["wind_speed_x_pm25_lag1"] = frame["wind_speed"].fillna(0.0) * frame["pm25_lag1"].fillna(0.0)

    feature_columns = [
        "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag4", "pm25_lag6", "pm25_lag8", "pm25_lag12", "pm25_lag24",
        "pm25_roll_mean_3", "pm25_roll_mean_6", "pm25_roll_mean_12", "pm25_roll_mean_24",
        "pm25_roll_std_3", "pm25_roll_std_6", "pm25_roll_std_12", "pm25_roll_std_24",
        "pm25_diff_1", "pm25_diff_2", "pm25_diff_3", "pm25_diff_6",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "weekday", "is_weekend", "is_holiday", "is_daytime", "is_rush_hour",
        "temp", "pres", "dewp", "humidity", "wind_speed", "precipitation", "wind_dir_sin", "wind_dir_cos", "precipitation_flag", "dewp_temp_gap",
        "temp_diff_1", "temp_diff_2", "pres_diff_1", "pres_diff_2", "wind_speed_diff_1", "wind_speed_diff_2",
        "temp_x_humidity", "wind_speed_x_wind_dir_sin", "wind_speed_x_pm25_lag1",
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
    x_train = scaler.fit_transform(train_frame[feature_columns])
    x_test = scaler.transform(test_frame[feature_columns])
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
    """将特征说明写入 docs/feature_doc.md（中文）。"""
    from utils.io import DOCS_DIR

    granularity = metadata.get("granularity", "daily")
    if granularity == "hourly":
        lines = [
            "# 小时级特征说明文档",
            "",
            "## 概述",
            "",
            "- 预测目标：`pm25`（PM2.5 小时浓度，单位 μg/m³）",
            "- 当前特征集更偏向 `h1 / h6` 这类短中期点预测任务",
            "- 输入数据：`data/processed/beijing_hourly.csv`",
            "- 输出目录：`data/processed/features_hourly/`",
            "",
            "## 训练/测试划分",
            "",
            f"- 滞后/滚动特征构造后的有效行数：{metadata['valid_rows']}",
            f"- 因滞后产生的丢弃行数：{metadata['dropped_rows']}",
            f"- 训练样本：{metadata['n_train']}（{metadata['train_start']} 至 {metadata['train_end']}）",
            f"- 测试样本：{metadata['n_test']}（{metadata['test_start']} 至 {metadata['test_end']}）",
            f"- 划分方式：按时间顺序，后 20% 为测试集",
            "",
            "## 特征列表",
            "",
            "| 特征名 | 说明 |",
            "|--------|------|",
            "| pm25_lag1 | 前 1 小时 PM2.5 |",
            "| pm25_lag2 | 前 2 小时 PM2.5 |",
            "| pm25_lag3 | 前 3 小时 PM2.5 |",
            "| pm25_lag4 | 前 4 小时 PM2.5 |",
            "| pm25_lag6 | 前 6 小时 PM2.5 |",
            "| pm25_lag8 | 前 8 小时 PM2.5 |",
            "| pm25_lag12 | 前 12 小时 PM2.5 |",
            "| pm25_lag24 | 前 24 小时 PM2.5 |",
            "| pm25_roll_mean_3 | 3 小时滚动均值（滞后 1 小时） |",
            "| pm25_roll_mean_6 | 6 小时滚动均值（滞后 1 小时） |",
            "| pm25_roll_mean_12 | 12 小时滚动均值（滞后 1 小时） |",
            "| pm25_roll_mean_24 | 24 小时滚动均值（滞后 1 小时） |",
            "| pm25_roll_std_3 | 3 小时滚动标准差（滞后 1 小时） |",
            "| pm25_roll_std_6 | 6 小时滚动标准差（滞后 1 小时） |",
            "| pm25_roll_std_12 | 12 小时滚动标准差（滞后 1 小时） |",
            "| pm25_roll_std_24 | 24 小时滚动标准差（滞后 1 小时） |",
            "| pm25_diff_1 | 当前 PM2.5 相对 1 小时前变化 |",
            "| pm25_diff_2 | 当前 PM2.5 相对 2 小时前变化 |",
            "| pm25_diff_3 | 当前 PM2.5 相对 3 小时前变化 |",
            "| pm25_diff_6 | 当前 PM2.5 相对 6 小时前变化 |",
            "| hour_sin | 小时正弦周期编码 |",
            "| hour_cos | 小时余弦周期编码 |",
            "| month_sin | 月份正弦周期编码 |",
            "| month_cos | 月份余弦周期编码 |",
            "| weekday | 星期几（0=周一） |",
            "| is_weekend | 是否周末（0/1） |",
            "| is_holiday | 中国法定节假日标记（0/1） |",
            "| is_daytime | 是否白天（06:00-17:59） |",
            "| is_rush_hour | 是否通勤高峰（7-9,17-19） |",
            "| temp | 小时温度（℃） |",
            "| pres | 小时气压（hPa） |",
            "| dewp | 小时露点（℃） |",
            "| humidity | 小时湿度（%） |",
            "| wind_speed | 小时风速（m/s） |",
            "| precipitation | 小时降水量（mm） |",
            "| wind_dir_sin | 风向正弦编码 |",
            "| wind_dir_cos | 风向余弦编码 |",
            "| precipitation_flag | 是否有降水（0/1） |",
            "| dewp_temp_gap | 温度与露点差（temp-dewp） |",
            "| temp_diff_1 | 温度相对 1 小时前变化 |",
            "| temp_diff_2 | 温度相对 2 小时前变化 |",
            "| pres_diff_1 | 气压相对 1 小时前变化 |",
            "| pres_diff_2 | 气压相对 2 小时前变化 |",
            "| wind_speed_diff_1 | 风速相对 1 小时前变化 |",
            "| wind_speed_diff_2 | 风速相对 2 小时前变化 |",
            "| temp_x_humidity | 温度 × 湿度交互项 |",
            "| wind_speed_x_wind_dir_sin | 风速 × sin(风向角) 交互项 |",
            "| wind_speed_x_pm25_lag1 | 风速 × PM2.5(t-1) 交互项 |",
            "",
            "## 矩阵维度",
            "",
            f"- X_train: ({metadata['n_train']}, {metadata['n_features']})",
            f"- X_test: ({metadata['n_test']}, {metadata['n_features']})",
            f"- y_train: ({metadata['n_train']},)",
            f"- y_test: ({metadata['n_test']},)",
            "",
            "## 加载示例",
            "",
            "```python",
            "import json",
            "import numpy as np",
            "",
            "X_train = np.load('data/processed/features_hourly/X_train.npy')",
            "y_train = np.load('data/processed/features_hourly/y_train.npy')",
            "with open('data/processed/features_hourly/feature_names.json') as f:",
            "    feature_names = json.load(f)",
            "```",
            "",
            "特征已使用 `utils/feature_engineering.py` 中的 StandardScaler 在训练集上拟合标准化，",
            "测试集使用同一 scaler 变换。推理新样本时请加载 `scaler.pkl`。",
            "",
        ]
        doc_path = DOCS_DIR / "feature_doc_hourly.md"
    else:
        lines = [
            "# 特征说明文档",
            "",
            "## 概述",
            "",
            "- 预测目标：`pm25`（PM2.5 日均浓度，单位 μg/m³）",
            "- 输入数据：`data/processed/beijing.csv`",
            "- 输出目录：`data/processed/features/`",
            "",
            "## 训练/测试划分",
            "",
            f"- 滞后/滚动特征构造后的有效行数：{metadata['valid_rows']}",
            f"- 因滞后产生的丢弃行数：{metadata['dropped_rows']}",
            f"- 训练样本：{metadata['n_train']}（{metadata['train_start']} 至 {metadata['train_end']}）",
            f"- 测试样本：{metadata['n_test']}（{metadata['test_start']} 至 {metadata['test_end']}）",
            f"- 划分方式：按时间顺序，后 20% 为测试集",
            "",
            "## 特征列表",
            "",
            "| 特征名 | 说明 |",
            "|--------|------|",
            "| pm25_lag1 | 前 1 日 PM2.5 |",
            "| pm25_lag3 | 前 3 日 PM2.5 |",
            "| pm25_lag7 | 前 7 日 PM2.5 |",
            "| pm25_roll_mean_7 | 7 日滚动均值（滞后 1 日） |",
            "| pm25_roll_mean_14 | 14 日滚动均值（滞后 1 日） |",
            "| pm25_roll_std_7 | 7 日滚动标准差（滞后 1 日） |",
            "| pm25_roll_std_14 | 14 日滚动标准差（滞后 1 日） |",
            "| month_sin | 月份正弦周期编码 |",
            "| month_cos | 月份余弦周期编码 |",
            "| weekday | 星期几（0=周一） |",
            "| is_holiday | 中国法定节假日标记（0/1） |",
            "| temp | 日均温度（℃） |",
            "| pres | 日均气压（hPa） |",
            "| dewp | 日均露点（℃） |",
            "| wind_speed | 日均风速（m/s） |",
            "| precipitation | 日降水量（mm） |",
            "| temp_x_humidity | 温度 × 湿度交互项 |",
            "| wind_speed_x_wind_dir_sin | 风速 × sin(风向角) 交互项 |",
            "",
            "## 矩阵维度",
            "",
            f"- X_train: ({metadata['n_train']}, {metadata['n_features']})",
            f"- X_test: ({metadata['n_test']}, {metadata['n_features']})",
            f"- y_train: ({metadata['n_train']},)",
            f"- y_test: ({metadata['n_test']},)",
            "",
            "## 加载示例",
            "",
            "```python",
            "import json",
            "import numpy as np",
            "",
            "X_train = np.load('data/processed/features/X_train.npy')",
            "y_train = np.load('data/processed/features/y_train.npy')",
            "with open('data/processed/features/feature_names.json') as f:",
            "    feature_names = json.load(f)",
            "```",
            "",
            "特征已使用 `utils/feature_engineering.py` 中的 StandardScaler 在训练集上拟合标准化，",
            "测试集使用同一 scaler 变换。推理新样本时请加载 `scaler.pkl`。",
            "",
        ]
        doc_path = DOCS_DIR / "feature_doc.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines), encoding="utf-8")
