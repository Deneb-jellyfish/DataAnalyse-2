"""小时级 PM2.5 项目 Streamlit 演示台。

当前版本以中文界面为主，先用 mock 数据把展示框架与交互结构搭起来。
后续实验结果产出后，可按页面中标注的接口位置替换为真实 CSV / 模型输出。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_HOURLY_DIR = PROCESSED_DIR / "features_hourly"
OUTPUTS_HOURLY_DIR = ROOT / "outputs" / "hourly"
DOCS_DIR = ROOT / "docs"

MODEL_ORDER = ["ARIMA", "Prophet", "XGBoost", "LSTM", "Transformer"]
MODEL_COLORS = {
    "真实值": "#1f2937",
    "ARIMA": "#5f6caf",
    "Prophet": "#f59e0b",
    "XGBoost": "#ef4444",
    "LSTM": "#10b981",
    "Transformer": "#6366f1",
}
MODEL_DESCRIPTIONS = {
    "ARIMA": "传统统计时间序列基线，重点看可解释性与稳定性。",
    "Prophet": "趋势与季节项分解模型，适合展示可解释时间规律。",
    "XGBoost": "当前最有希望的主力模型，适合承接特征工程成果。",
    "LSTM": "循环神经网络模型，关注时序记忆能力。",
    "Transformer": "注意力模型，关注较长序列中的依赖关系。",
}
SHORT_TERM_FILES = [
    "outputs/hourly/xgboost_predictions_h1.csv",
    "outputs/hourly/arima_predictions_h1.csv",
    "outputs/hourly/prophet_predictions_h1.csv",
    "outputs/hourly/lstm_predictions_h1.csv",
    "outputs/hourly/transformer_predictions_h1.csv",
    "outputs/hourly/overall_metrics_summary.csv",
    "outputs/hourly/aqi_bucket_metrics.csv",
    "outputs/hourly/cross_city_metrics.csv",
]
MID_TERM_FILES = [
    "outputs/hourly/xgboost_predictions_h12.csv",
    "outputs/hourly/arima_predictions_h12.csv",
    "outputs/hourly/prophet_predictions_h12.csv",
    "outputs/hourly/lstm_predictions_h12.csv",
    "outputs/hourly/transformer_predictions_h12.csv",
    "outputs/hourly/feature_ablation.csv",
    "outputs/hourly/hourly_error_by_hour.csv",
    "outputs/hourly/high_pollution_error_analysis.csv",
]


@dataclass
class DatasetSummary:
    name: str
    rows: int
    start: str
    end: str
    columns: list[str]


def set_matplotlib_style() -> None:
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 158, 66, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(67, 146, 124, 0.15), transparent 28%),
                linear-gradient(180deg, #f8f5ef 0%, #efe8dd 100%);
            color: #1f2937;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }
        .hero {
            background: linear-gradient(135deg, rgba(255,255,255,0.88), rgba(247,240,230,0.92));
            border: 1px solid rgba(112, 90, 52, 0.14);
            box-shadow: 0 18px 48px rgba(84, 62, 35, 0.08);
            border-radius: 26px;
            padding: 1.5rem 1.6rem;
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.3rem;
            letter-spacing: 0.01em;
        }
        .hero p {
            margin: 0.18rem 0;
            line-height: 1.7;
            font-size: 1rem;
        }
        .soft-card {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(112, 90, 52, 0.12);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 30px rgba(84, 62, 35, 0.05);
            margin-bottom: 0.85rem;
        }
        .soft-card h3, .soft-card h4 {
            margin-top: 0;
        }
        .tiny-label {
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            background: rgba(239, 68, 68, 0.10);
            color: #b91c1c;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }
        .note-box {
            background: rgba(16, 185, 129, 0.08);
            border-left: 4px solid #10b981;
            padding: 0.85rem 0.95rem;
            border-radius: 14px;
            margin: 0.55rem 0 0.8rem 0;
        }
        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(112, 90, 52, 0.12);
            border-radius: 18px;
            padding: 0.95rem 1rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ece7dc 0%, #e2ddd2 100%);
            border-right: 1px solid rgba(112, 90, 52, 0.10);
        }
        section[data-testid="stSidebar"] .stRadio label {
            padding: 0.45rem 0.8rem !important;
            border-radius: 12px !important;
            margin: 0.1rem 0 !important;
            color: #243242 !important;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(53, 97, 85, 0.10) !important;
        }
        section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background: rgba(53, 97, 85, 0.16) !important;
            color: #1f4d3f !important;
            font-weight: 700 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 2px solid rgba(112, 90, 52, 0.14);
            margin-bottom: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding: 0 22px;
            color: #5b4b35;
            border-bottom: 3px solid transparent !important;
            margin-bottom: -2px;
        }
        .stTabs [aria-selected="true"] {
            color: #d9485f !important;
            border-bottom: 3px solid #d9485f !important;
            font-weight: 700 !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="soft-card">
            <h4>{title}</h4>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notice(text: str) -> None:
    st.markdown(f'<div class="note-box">{text}</div>', unsafe_allow_html=True)


def safe_read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, parse_dates=parse_dates)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_dataset_summary(file_name: str, time_col: str, display_name: str) -> DatasetSummary:
    path = PROCESSED_DIR / file_name
    df = safe_read_csv(path, parse_dates=[time_col])
    if df is None or df.empty:
        return DatasetSummary(display_name, 0, "-", "-", [])
    ordered = df.sort_values(time_col)
    return DatasetSummary(
        name=display_name,
        rows=int(len(ordered)),
        start=str(pd.to_datetime(ordered[time_col].iloc[0])),
        end=str(pd.to_datetime(ordered[time_col].iloc[-1])),
        columns=ordered.columns.tolist(),
    )


@st.cache_data(show_spinner=False)
def load_hourly_preview() -> pd.DataFrame:
    df = safe_read_csv(PROCESSED_DIR / "beijing_hourly.csv", parse_dates=["datetime"])
    if df is None or df.empty:
        now = pd.date_range("2016-05-14 00:00:00", periods=12, freq="h")
        return pd.DataFrame(
            {
                "datetime": now,
                "city": ["Beijing"] * len(now),
                "pm25": np.linspace(82, 118, len(now)).round(2),
                "temp": np.linspace(23, 29, len(now)).round(2),
                "humidity": np.linspace(45, 72, len(now)).round(2),
                "wind_speed": np.linspace(0.6, 2.4, len(now)).round(2),
            }
        )
    return df.head(12)


@st.cache_data(show_spinner=False)
def load_feature_names() -> list[str]:
    path = FEATURES_HOURLY_DIR / "feature_names.json"
    if path.exists():
        try:
            return pd.read_json(path, typ="series").tolist()
        except Exception:
            try:
                import json

                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return []
    return []


def categorize_features(feature_names: list[str]) -> dict[str, list[str]]:
    groups = {
        "滞后特征": [],
        "滚动统计": [],
        "变化趋势": [],
        "时间周期": [],
        "气象原值": [],
        "交互与增强": [],
    }
    for name in feature_names:
        if "lag" in name:
            groups["滞后特征"].append(name)
        elif "roll" in name:
            groups["滚动统计"].append(name)
        elif "diff" in name:
            groups["变化趋势"].append(name)
        elif any(tag in name for tag in ["hour_", "month_", "weekday", "weekend", "daytime", "rush", "holiday"]):
            groups["时间周期"].append(name)
        elif name in {"temp", "pres", "dewp", "humidity", "wind_speed", "precipitation", "wind_dir_sin", "wind_dir_cos"}:
            groups["气象原值"].append(name)
        else:
            groups["交互与增强"].append(name)
    return groups


def build_mock_signal(horizon_hours: int) -> pd.DataFrame:
    rng = np.random.default_rng(42 + horizon_hours)
    periods = 14 * 24 if horizon_hours == 1 else 10 * 24
    ts = pd.date_range("2016-11-01 00:00:00", periods=periods, freq="h")
    x = np.arange(periods)
    y_true = (
        92
        + 22 * np.sin(2 * np.pi * x / 24)
        + 14 * np.sin(2 * np.pi * x / (24 * 5))
        + 7 * np.cos(2 * np.pi * x / (24 * 3))
        + rng.normal(0, 4.2 if horizon_hours == 1 else 5.6, periods)
    )
    base = pd.DataFrame({"timestamp": ts, "y_true": np.clip(y_true, 8, None)})
    base["ARIMA"] = np.clip(base["y_true"].rolling(3, min_periods=1).mean().shift(1).bfill() + rng.normal(0, 7 if horizon_hours == 1 else 12, periods), 5, None)
    base["Prophet"] = np.clip(base["y_true"].rolling(8, min_periods=1).mean().shift(2).bfill() + rng.normal(0, 10 if horizon_hours == 1 else 16, periods), 5, None)
    base["XGBoost"] = np.clip(base["y_true"] * (0.97 if horizon_hours == 1 else 0.93) + rng.normal(0, 4.8 if horizon_hours == 1 else 8.8, periods), 5, None)
    base["LSTM"] = np.clip(base["y_true"].rolling(4, min_periods=1).mean() + rng.normal(0, 6.6 if horizon_hours == 1 else 11.5, periods), 5, None)
    base["Transformer"] = np.clip(base["y_true"].rolling(5, min_periods=1).mean() + rng.normal(0, 6.1 if horizon_hours == 1 else 10.5, periods), 5, None)
    numeric_cols = [col for col in base.columns if col != "timestamp"]
    base[numeric_cols] = base[numeric_cols].round(2)
    return base


def compute_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    true = pred_df["y_true"].to_numpy(dtype=float)
    for model in MODEL_ORDER:
        pred = pred_df[model].to_numpy(dtype=float)
        err = pred - true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        mape = float(np.mean(np.abs(err) / np.clip(np.abs(true), 1.0, None)) * 100)
        rows.append(
            {
                "model": model,
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "mape": round(mape, 2),
                "n_samples": int(len(pred_df)),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def build_bucket_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    bucket_edges = [0, 35, 75, 115, 150, np.inf]
    bucket_labels = ["0-35", "35-75", "75-115", "115-150", ">150"]
    frame = pred_df.copy()
    frame["aqi_bucket"] = pd.cut(frame["y_true"], bins=bucket_edges, labels=bucket_labels, right=False)
    rows: list[dict[str, str | float]] = []
    for model in MODEL_ORDER:
        for bucket in bucket_labels:
            subset = frame[frame["aqi_bucket"] == bucket]
            if subset.empty:
                value = np.nan
            else:
                value = float(np.mean(np.abs(subset[model] - subset["y_true"])))
            rows.append({"model": model, "aqi_bucket": bucket, "mae": value})
    return pd.DataFrame(rows)


def build_hourly_error_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    frame = pred_df.copy()
    frame["hour"] = frame["timestamp"].dt.hour
    rows: list[dict[str, str | float | int]] = []
    for model in MODEL_ORDER:
        for hour in range(24):
            subset = frame[frame["hour"] == hour]
            mae = float(np.mean(np.abs(subset[model] - subset["y_true"])))
            rows.append({"model": model, "hour": hour, "mae": mae})
    return pd.DataFrame(rows)


def build_dayperiod_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    frame = pred_df.copy()
    hour = frame["timestamp"].dt.hour
    frame["day_period"] = np.select(
        [
            hour < 6,
            hour < 10,
            hour < 17,
            hour < 21,
        ],
        ["凌晨", "早高峰", "白天平峰", "晚高峰"],
        default="夜间",
    )
    rows: list[dict[str, str | float | int]] = []
    for model in MODEL_ORDER:
        for period in ["凌晨", "早高峰", "白天平峰", "晚高峰", "夜间"]:
            subset = frame[frame["day_period"] == period]
            mae = float(np.mean(np.abs(subset[model] - subset["y_true"])))
            rows.append({"model": model, "day_period": period, "mae": mae})
    return pd.DataFrame(rows)


def build_city_transfer_metrics(horizon_hours: int) -> pd.DataFrame:
    base = compute_metrics(build_mock_signal(horizon_hours))
    rows: list[dict[str, str | float]] = []
    for _, item in base.iterrows():
        rows.append({"city": "北京测试", "model": item["model"], "rmse": item["rmse"], "mae": item["mae"]})
        rows.append(
            {
                "city": "上海迁移测试",
                "model": item["model"],
                "rmse": round(float(item["rmse"]) * (1.12 if horizon_hours == 1 else 1.18), 2),
                "mae": round(float(item["mae"]) * (1.10 if horizon_hours == 1 else 1.16), 2),
            }
        )
    return pd.DataFrame(rows)


def plot_prediction_lines(pred_df: pd.DataFrame, models: list[str], recent_hours: int, title: str) -> plt.Figure:
    set_matplotlib_style()
    frame = pred_df.tail(recent_hours).copy()
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.plot(frame["timestamp"], frame["y_true"], label="真实值", linewidth=2.7, color=MODEL_COLORS["真实值"])
    for model in models:
        ax.plot(frame["timestamp"], frame[model], label=model, linewidth=1.9, color=MODEL_COLORS[model], alpha=0.95)
    ax.set_title(title)
    ax.set_xlabel("时间")
    ax.set_ylabel("PM2.5")
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    return fig


def plot_metric_bars(metric_df: pd.DataFrame, metric_name: str, title: str) -> plt.Figure:
    set_matplotlib_style()
    ordered = metric_df.sort_values(metric_name)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    colors = [MODEL_COLORS[m] for m in ordered["model"]]
    ax.bar(ordered["model"], ordered[metric_name], color=colors, alpha=0.88)
    for idx, value in enumerate(ordered[metric_name]):
        ax.text(idx, value + max(ordered[metric_name]) * 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(metric_name.upper())
    ax.grid(axis="y", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_heatmap(bucket_df: pd.DataFrame, title: str) -> plt.Figure:
    set_matplotlib_style()
    pivot = bucket_df.pivot(index="aqi_bucket", columns="model", values="mae").reindex(columns=MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    matrix = pivot.to_numpy(dtype=float)
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                label = "-"
            else:
                label = f"{matrix[i, j]:.1f}"
            ax.text(j, i, label, ha="center", va="center", color="#3f2a14", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    return fig


def plot_hour_error_curve(hour_df: pd.DataFrame, models: list[str], title: str) -> plt.Figure:
    set_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for model in models:
        subset = hour_df[hour_df["model"] == model]
        ax.plot(subset["hour"], subset["mae"], marker="o", linewidth=1.9, label=model, color=MODEL_COLORS[model])
    ax.set_title(title)
    ax.set_xlabel("小时")
    ax.set_ylabel("MAE")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_city_transfer(city_df: pd.DataFrame, metric_name: str, title: str) -> plt.Figure:
    set_matplotlib_style()
    pivot = city_df.pivot(index="model", columns="city", values=metric_name).reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    positions = np.arange(len(pivot.index))
    width = 0.34
    ax.bar(positions - width / 2, pivot["北京测试"], width=width, color="#ef4444", label="北京测试")
    ax.bar(positions + width / 2, pivot["上海迁移测试"], width=width, color="#0f766e", label="上海迁移测试")
    ax.set_xticks(positions)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel(metric_name.upper())
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_horizon_degradation(h1_metrics: pd.DataFrame, h12_metrics: pd.DataFrame) -> plt.Figure:
    set_matplotlib_style()
    merged = h1_metrics[["model", "rmse"]].merge(
        h12_metrics[["model", "rmse"]],
        on="model",
        suffixes=("_h1", "_h12"),
    ).set_index("model").reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for model in merged.index:
        ax.plot(["+1h", "+12h"], [merged.loc[model, "rmse_h1"], merged.loc[model, "rmse_h12"]], marker="o", linewidth=2, label=model, color=MODEL_COLORS[model])
    ax.set_title("预测步长拉长后的 RMSE 退化对比")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def build_live_input(hours: int) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    ts = pd.date_range("2017-02-27 00:00:00", periods=hours, freq="h")
    x = np.arange(hours)
    return pd.DataFrame(
        {
            "时间": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "PM2.5": np.clip(88 + 18 * np.sin(2 * np.pi * x / 24) + rng.normal(0, 3.8, hours), 8, None).round(2),
            "温度": np.clip(18 + 8 * np.sin(2 * np.pi * x / 24 + 0.4) + rng.normal(0, 1.2, hours), -10, None).round(2),
            "湿度": np.clip(55 + 18 * np.cos(2 * np.pi * x / 24) + rng.normal(0, 2.0, hours), 10, 100).round(2),
            "风速": np.clip(1.6 + 0.8 * np.sin(2 * np.pi * x / 12) + rng.normal(0, 0.15, hours), 0.1, None).round(2),
            "降水": np.clip(rng.choice([0.0, 0.0, 0.0, 0.3, 0.8], size=hours), 0, None).round(2),
        }
    )


def run_mock_live_prediction(history_df: pd.DataFrame, selected_model: str) -> tuple[dict[str, float | str], pd.DataFrame]:
    frame = history_df.copy()
    frame["时间"] = pd.to_datetime(frame["时间"])
    pm = pd.to_numeric(frame["PM2.5"], errors="coerce")
    temp = pd.to_numeric(frame["温度"], errors="coerce")
    humidity = pd.to_numeric(frame["湿度"], errors="coerce")
    wind_speed = pd.to_numeric(frame["风速"], errors="coerce")
    rain = pd.to_numeric(frame["降水"], errors="coerce")

    last_pm = float(pm.iloc[-1])
    mean_6 = float(pm.tail(6).mean())
    mean_12 = float(pm.tail(12).mean())
    trend_6 = float(pm.iloc[-1] - pm.iloc[-6]) / 6 if len(pm) >= 6 else 0.0
    humidity_boost = float(humidity.iloc[-1] - 60) * 0.12
    wind_suppress = float(wind_speed.iloc[-1]) * 2.4
    rain_suppress = float(rain.tail(6).sum()) * 4.0

    model_offsets = {
        "ARIMA": (0.6, 3.0),
        "Prophet": (2.4, 5.2),
        "XGBoost": (-1.2, -2.0),
        "LSTM": (0.8, 1.4),
        "Transformer": (0.3, 0.9),
    }
    short_bias, long_bias = model_offsets[selected_model]
    next_1h = max(5.0, last_pm + trend_6 * 1.6 + humidity_boost - wind_suppress - rain_suppress + short_bias)
    next_12h = max(5.0, mean_12 + trend_6 * 7.5 + humidity_boost * 1.3 - wind_suppress * 0.8 - rain_suppress * 0.4 + long_bias)

    future_times = pd.date_range(frame["时间"].iloc[-1] + pd.Timedelta(hours=1), periods=12, freq="h")
    path = np.linspace(next_1h, next_12h, 12)
    wave = 4.0 * np.sin(np.linspace(0, np.pi, 12))
    future_values = np.clip(path + wave, 5, None)
    forecast = pd.DataFrame({"时间": future_times, "预测PM2.5": future_values.round(2)})

    level = "优"
    if next_12h > 150:
        level = "重污染风险高"
    elif next_12h > 115:
        level = "较高污染风险"
    elif next_12h > 75:
        level = "中度污染风险"

    summary = {
        "next_1h": round(next_1h, 2),
        "next_12h": round(next_12h, 2),
        "risk_level": level,
        "driver": "近 12 小时均值 + 最近 6 小时趋势 + 湿度/风速/降水修正（mock 占位）",
    }
    return summary, forecast


def plot_live_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame, model_name: str) -> plt.Figure:
    set_matplotlib_style()
    history = history_df.copy()
    history["时间"] = pd.to_datetime(history["时间"])
    fig, ax = plt.subplots(figsize=(12.2, 5.0))
    ax.plot(history["时间"], history["PM2.5"], color="#1f2937", linewidth=2.4, label="输入历史")
    ax.plot(forecast_df["时间"], forecast_df["预测PM2.5"], color=MODEL_COLORS[model_name], linewidth=2.2, marker="o", label=f"{model_name} 预测路径")
    ax.axvline(history["时间"].iloc[-1], linestyle="--", color="#7c6b52", linewidth=1.2)
    ax.set_title("实时预测结果（当前为 mock 引擎占位）")
    ax.set_xlabel("时间")
    ax.set_ylabel("PM2.5")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    return fig


def render_file_targets(file_list: list[str]) -> None:
    st.markdown("**后续真实数据接入文件**")
    table = pd.DataFrame({"预期文件": file_list, "当前状态": ["待接入"] * len(file_list)})
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_overview_page() -> None:
    beijing_hourly = load_dataset_summary("beijing_hourly.csv", "datetime", "北京小时级主数据")
    shanghai_hourly = load_dataset_summary("shanghai_hourly.csv", "datetime", "上海小时级数据")
    feature_names = load_feature_names()
    feature_groups = categorize_features(feature_names)
    preview = load_hourly_preview()

    st.subheader("总览与数据")
    render_notice(
        "当前页面优先承接项目总览、数据规模、特征工程与后续实验接口。"
        " 这里展示的数据概况尽量读真实文件；模型结果部分则先用 mock 占位。"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("北京小时级样本", f"{beijing_hourly.rows}")
    c2.metric("上海小时级样本", f"{shanghai_hourly.rows}")
    c3.metric("当前特征维度", f"{len(feature_names)}")
    c4.metric("正式对比模型数", f"{len(MODEL_ORDER)}")

    left, right = st.columns([1.15, 0.85])
    with left:
        render_card(
            "项目总览",
            """
            本项目当前已经从“日级下一天预测”切换到“小时级 PM2.5 预测”主线。
            正式实验采用双任务设计：
            <br/>1. 短期预测：未来 <strong>+1h</strong>
            <br/>2. 中短期预测：未来 <strong>+12h</strong>
            <br/><br/>目标是在统一口径下完成 ARIMA、Prophet、XGBoost、LSTM、Transformer 五模型系统对比。
            """,
        )
        render_card(
            "数据处理与特征工程",
            """
            当前小时级特征工程已经完成升级，重点围绕 <strong>+12h</strong> 做增强。
            <br/>包含中尺度滞后、滚动统计、趋势变化、昼夜时段、气象变化与交互特征。
            <br/>后续建模同学只需要围绕 `beijing_hourly.csv` 与 `features_hourly/` 接模型即可。
            """,
        )
        render_card(
            "当前实验主线",
            """
            这版 Streamlit 先把展示框架搭好：
            <br/>- 总览和数据
            <br/>- 短期预测（+1h）
            <br/>- 中长期预测（+12h）
            <br/>- 真实预测
            <br/><br/>后续真实实验结果出来后，只需要把 CSV 接到对应页面即可。
            """,
        )
    with right:
        st.markdown("**数据集概况**")
        data_table = pd.DataFrame(
            [
                {"数据集": beijing_hourly.name, "样本数": beijing_hourly.rows, "起始时间": beijing_hourly.start, "结束时间": beijing_hourly.end},
                {"数据集": shanghai_hourly.name, "样本数": shanghai_hourly.rows, "起始时间": shanghai_hourly.start, "结束时间": shanghai_hourly.end},
            ]
        )
        st.dataframe(data_table, use_container_width=True, hide_index=True)
        st.markdown("**主数据预览**")
        st.dataframe(preview, use_container_width=True, hide_index=True)

    st.markdown("### 特征工程结构")
    group_cols = st.columns(3)
    for col, (group_name, items) in zip(group_cols * 2, feature_groups.items()):
        with col:
            render_card(
                group_name,
                f"共 <strong>{len(items)}</strong> 个特征。<br/>示例：<code>{', '.join(items[:6]) if items else '待接入'}</code>",
            )

    with st.expander("查看完整特征列表", expanded=False):
        st.code("\n".join(feature_names) if feature_names else "当前未检测到 feature_names.json", language="text")

    st.markdown("### 后续接入位置")
    interface_table = pd.DataFrame(
        {
            "模块": ["短期预测页", "中长期预测页", "真实预测页", "误差分析页"],
            "后续接入内容": [
                "各模型 `*_predictions_h1.csv` 与 `overall_metrics_summary.csv`",
                "各模型 `*_predictions_h12.csv`、`feature_ablation.csv`、高污染分析结果",
                "训练后保存的统一推理接口或模型权重",
                "`aqi_bucket_metrics.csv`、`hourly_error_by_hour.csv`、`cross_city_metrics.csv`",
            ],
        }
    )
    st.dataframe(interface_table, use_container_width=True, hide_index=True)


def render_short_term_page() -> None:
    pred_df = build_mock_signal(1)
    metric_df = compute_metrics(pred_df)
    bucket_df = build_bucket_metrics(pred_df)
    hour_df = build_hourly_error_metrics(pred_df)
    dayperiod_df = build_dayperiod_metrics(pred_df)
    city_df = build_city_transfer_metrics(1)

    st.subheader("短期预测（+1h）")
    render_notice(
        "本页当前使用 mock 结果占位，目的是先把展示结构、调节控件与后续真实结果接口留好。"
        " 等 `outputs/hourly/*_h1.csv` 出来后，可直接替换数据源。"
    )

    tab1, tab2, tab3, tab4 = st.tabs(["预测结果", "模型对比", "误差分析", "城市迁移"])

    with tab1:
        chart_col, ctrl_col = st.columns([2.35, 0.85])
        with ctrl_col:
            st.markdown("**图表控制**")
            selected_models = st.multiselect("选择展示模型", MODEL_ORDER, default=["XGBoost", "ARIMA", "LSTM"])
            recent_hours = st.slider("显示最近多少小时", min_value=48, max_value=240, value=120, step=24)
            st.caption("后续接入真实结果后，这里的模型列表与时间范围会自动跟随真实 CSV 更新。")
            render_file_targets(SHORT_TERM_FILES[:5])
        with chart_col:
            fig = plot_prediction_lines(
                pred_df,
                selected_models or ["XGBoost"],
                recent_hours=recent_hours,
                title="短期预测结果图：真实值与多模型 +1h 预测对比（mock）",
            )
            st.pyplot(fig, use_container_width=True)
            preview_cols = ["timestamp", "y_true"] + (selected_models[:3] if selected_models else ["XGBoost"])
            st.dataframe(pred_df.tail(18)[preview_cols], use_container_width=True, hide_index=True)

    with tab2:
        left, right = st.columns([1.15, 0.85])
        with right:
            ranking_metric = st.radio("主排序指标", ["rmse", "mae", "mape"], horizontal=True)
            best_model = str(metric_df.sort_values(ranking_metric).iloc[0]["model"])
            st.metric("当前 mock 最优模型", best_model)
            st.caption(MODEL_DESCRIPTIONS[best_model])
            render_file_targets(["outputs/hourly/overall_metrics_summary.csv"])
        with left:
            fig = plot_metric_bars(metric_df, ranking_metric, f"+1h 模型对比：按 {ranking_metric.upper()} 排序")
            st.pyplot(fig, use_container_width=True)
        compare_table = metric_df.copy()
        compare_table["排名"] = range(1, len(compare_table) + 1)
        compare_table = compare_table[["排名", "model", "rmse", "mae", "mape", "n_samples"]]
        compare_table.columns = ["排名", "模型", "RMSE", "MAE", "MAPE", "样本数"]
        st.dataframe(compare_table, use_container_width=True, hide_index=True)

    with tab3:
        left, right = st.columns(2)
        with left:
            fig = plot_heatmap(bucket_df, "AQI 分层误差热力图（MAE，mock）")
            st.pyplot(fig, use_container_width=True)
        with right:
            focus_models = st.multiselect("误差曲线模型", MODEL_ORDER, default=["XGBoost", "LSTM", "Transformer"], key="short_error_models")
            fig = plot_hour_error_curve(hour_df, focus_models or ["XGBoost"], "按小时误差变化（MAE，mock）")
            st.pyplot(fig, use_container_width=True)
        st.markdown("**时段误差预留区域**")
        st.dataframe(dayperiod_df, use_container_width=True, hide_index=True)
        render_file_targets(["outputs/hourly/aqi_bucket_metrics.csv", "outputs/hourly/hourly_error_by_hour.csv", "outputs/hourly/hourly_error_by_dayperiod.csv"])

    with tab4:
        left, right = st.columns([1.2, 0.8])
        with right:
            transfer_metric = st.radio("迁移观察指标", ["rmse", "mae"], horizontal=True, key="short_transfer_metric")
            render_card(
                "迁移解释",
                """
                本页后续用于展示“北京训练、上海测试”的泛化结果。
                当前先用 mock 结果说明版面：
                <br/>- 北京测试：原域表现
                <br/>- 上海迁移测试：跨城市落地后的误差变化
                """,
            )
        with left:
            fig = plot_city_transfer(city_df, transfer_metric, f"城市迁移对比：{transfer_metric.upper()}（mock）")
            st.pyplot(fig, use_container_width=True)
        st.dataframe(city_df, use_container_width=True, hide_index=True)
        render_file_targets(["outputs/hourly/cross_city_metrics.csv", "outputs/hourly/cross_city_predictions.csv"])


def render_mid_term_page() -> None:
    pred_h1 = build_mock_signal(1)
    pred_h12 = build_mock_signal(12)
    metric_h1 = compute_metrics(pred_h1)
    metric_h12 = compute_metrics(pred_h12)
    bucket_h12 = build_bucket_metrics(pred_h12)
    hour_h12 = build_hourly_error_metrics(pred_h12)

    st.subheader("中长期预测（+12h）")
    render_notice(
        "这一页对应新的正式扩展任务 `+12h`。"
        " 这里不仅放预测结果，还会放步长退化分析、特征消融预留位与高污染场景分析入口。"
    )

    tab1, tab2, tab3, tab4 = st.tabs(["任务说明", "预测结果", "步长退化", "深入分析预留"])

    with tab1:
        left, right = st.columns([1.2, 0.8])
        with left:
            render_card(
                "为什么是 +12h",
                """
                `+12h` 比 `+24h` 更适合作为当前正式扩展任务：
                <br/>- 仍然有难度，不是简单预测
                <br/>- 更能发挥新做的中尺度特征工程
                <br/>- 更容易把五个模型全部跑齐并做公平比较
                <br/>- 更适合在报告中写“步长拉长后的性能退化规律”
                """,
            )
            render_card(
                "后续这一页会承接什么",
                """
                真实实验完成后，本页将放入：
                <br/>1. 各模型 `+12h` 预测曲线
                <br/>2. `+1h` 与 `+12h` 退化对比
                <br/>3. `+12h` 特征消融结果
                <br/>4. 高污染与昼夜时段误差分析
                """,
            )
        with right:
            render_file_targets(MID_TERM_FILES)

    with tab2:
        chart_col, ctrl_col = st.columns([2.35, 0.85])
        with ctrl_col:
            selected_models = st.multiselect("选择展示模型", MODEL_ORDER, default=["XGBoost", "Transformer", "LSTM"], key="mid_models")
            recent_hours = st.slider("显示最近多少小时", min_value=72, max_value=240, value=144, step=24, key="mid_hours")
            st.caption("这里的图位已经预留好，后续直接接 `*_predictions_h12.csv`。")
        with chart_col:
            fig = plot_prediction_lines(
                pred_h12,
                selected_models or ["XGBoost"],
                recent_hours=recent_hours,
                title="中长期预测结果图：真实值与多模型 +12h 预测对比（mock）",
            )
            st.pyplot(fig, use_container_width=True)
            preview_cols = ["timestamp", "y_true"] + (selected_models[:3] if selected_models else ["XGBoost"])
            st.dataframe(pred_h12.tail(18)[preview_cols], use_container_width=True, hide_index=True)

    with tab3:
        top_left, top_right = st.columns([1.15, 0.85])
        with top_left:
            fig = plot_horizon_degradation(metric_h1, metric_h12)
            st.pyplot(fig, use_container_width=True)
        with top_right:
            degrade = metric_h1[["model", "rmse"]].merge(metric_h12[["model", "rmse"]], on="model", suffixes=("_h1", "_h12"))
            degrade["退化幅度"] = (degrade["rmse_h12"] - degrade["rmse_h1"]).round(2)
            degrade["退化倍率"] = (degrade["rmse_h12"] / degrade["rmse_h1"]).round(2)
            st.dataframe(degrade.rename(columns={"model": "模型", "rmse_h1": "RMSE(+1h)", "rmse_h12": "RMSE(+12h)"}), use_container_width=True, hide_index=True)
            render_card(
                "这个模块后续的作用",
                """
                这里会成为答辩里最关键的一页之一：
                <br/>- 说明预测步长拉长后的性能下降
                <br/>- 说明树模型与深度模型谁退化更快
                <br/>- 说明新特征工程对 `+12h` 是否真的有效
                """,
            )

    with tab4:
        left, right = st.columns(2)
        with left:
            fig = plot_heatmap(bucket_h12, "+12h AQI 分层误差热力图（mock）")
            st.pyplot(fig, use_container_width=True)
            render_card(
                "特征消融预留位",
                """
                后续建议直接接一张 5 组特征集的对比表：
                <br/>- base_short_lag
                <br/>- base_plus_daily_cycle
                <br/>- mid_range_memory
                <br/>- trend_and_roll
                <br/>- full_48
                """,
            )
        with right:
            fig = plot_hour_error_curve(hour_h12, ["XGBoost", "LSTM", "Transformer"], "+12h 按小时误差变化（mock）")
            st.pyplot(fig, use_container_width=True)
            render_card(
                "高污染场景预留位",
                """
                这里后续建议补：
                <br/>- `pm25 > 150` 样本数
                <br/>- 各模型高污染 MAE / RMSE
                <br/>- 极端污染时段预测曲线截图
                """,
            )


def render_live_prediction_page() -> None:
    st.subheader("真实预测")
    render_notice(
        "当前这一页先使用 mock 推理逻辑占位，重点是把输入区、输出区、未来真实模型接口全部留出来。"
        " 后续训练完成后，可直接把按钮逻辑替换成真实模型推理函数。"
    )

    left, right = st.columns([1.25, 0.75])
    with right:
        city = st.selectbox("预测城市", ["北京", "上海（后续接入）"])
        selected_model = st.selectbox("推理模型", MODEL_ORDER, index=2)
        history_hours = st.slider("输入最近多少小时数据", min_value=24, max_value=72, value=36, step=12)
        render_card(
            "未来真实接口位置",
            """
            后续建议替换为：
            <br/>- `predict_next_1h(input_frame, model_name)`
            <br/>- `predict_next_12h(input_frame, model_name)`
            <br/>- 或统一的 `run_inference(input_frame, horizon, model_name)`
            """,
        )
    with left:
        default_input = build_live_input(history_hours)
        edited = st.data_editor(
            default_input,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key=f"live_input_{history_hours}",
        )

    if st.button("开始预测", type="primary"):
        try:
            summary, forecast = run_mock_live_prediction(edited, selected_model)
            m1, m2, m3 = st.columns(3)
            m1.metric("下一小时预测", f"{summary['next_1h']:.2f}")
            m2.metric("未来 12 小时预测终点", f"{summary['next_12h']:.2f}")
            m3.metric("风险提示", str(summary["risk_level"]))

            render_card("当前预测解释", f"城市：{city}<br/>模型：{selected_model}<br/>当前说明：{summary['driver']}")

            chart_col, table_col = st.columns([1.7, 1.0])
            with chart_col:
                fig = plot_live_forecast(edited, forecast, selected_model)
                st.pyplot(fig, use_container_width=True)
            with table_col:
                st.markdown("**未来 12 小时预测明细**")
                show_table = forecast.copy()
                show_table["时间"] = show_table["时间"].dt.strftime("%Y-%m-%d %H:%M:%S")
                st.dataframe(show_table, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"预测失败：{exc}")


def main() -> None:
    st.set_page_config(page_title="小时级 PM2.5 项目工作台", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <div class="tiny-label">第一版中文演示台 · mock 占位版</div>
            <h1>小时级 PM2.5 项目工作台</h1>
            <p>这一版先把 Streamlit 的页面架构完整搭起来：左侧做目录导航，主区域按“总览与数据 / 短期预测 / 中长期预测 / 真实预测”组织。</p>
            <p>当前模型图表先用 mock 数据占位，但真实数据接口、文件接入位置与后续分析模块都已经预留好，后面实验结果出来后可直接落位。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## 页面目录")
        page = st.radio(
            "页面导航",
            ["1 总览与数据", "2 短期预测", "3 中长期预测", "4 真实预测"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("当前说明")
        st.caption("1 总览与数据：看项目背景、数据集与特征工程。")
        st.caption("2 短期预测：主看 +1h 结果结构。")
        st.caption("3 中长期预测：主看 +12h 结果结构。")
        st.caption("4 真实预测：先保留交互输入与推理结果位置。")

    if page == "1 总览与数据":
        render_overview_page()
    elif page == "2 短期预测":
        render_short_term_page()
    elif page == "3 中长期预测":
        render_mid_term_page()
    else:
        render_live_prediction_page()


if __name__ == "__main__":
    main()
