"""Streamlit dashboard for the hourly PM2.5 experiment."""

from __future__ import annotations

from html import escape
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from models.lstm_model import LSTMModel, LSTMTrainer
from models.transformer_model import TimeSeriesTransformer
from utils.feature_engineering import StandardScaler, build_hourly_feature_frame

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "hourly"
FIGURE_DIR = OUTPUT_DIR / "figures"
FEATURE_DIR = ROOT / "data" / "processed" / "features_hourly"
PROCESSED_DIR = ROOT / "data" / "processed"

MODEL_COLORS = {
    "ARIMA": "#6d645a",
    "Prophet": "#d66a5d",
    "XGBoost": "#2f8078",
    "LSTM": "#d49d2d",
    "Transformer": "#5e73b9",
}

MODEL_DESCRIPTIONS = {
    "ARIMA": "统计时序基线，短期稳定，但多步误差累积明显。",
    "Prophet": "趋势与季节分解模型，可解释性强，但对突发峰值不够敏感。",
    "XGBoost": "特征工程驱动的树模型，是当前 6 小时序列预测的综合最优模型。",
    "LSTM": "循环神经网络模型，当前 h1 单步预测表现最好。",
    "Transformer": "注意力时序模型，在多步预测上优于 LSTM，但均值仍落后于 XGBoost。",
}

REALTIME_COLUMNS = [
    "datetime",
    "pm25",
    "temp",
    "pres",
    "dewp",
    "humidity",
    "wind_dir",
    "wind_speed",
    "precipitation",
]


st.set_page_config(
    page_title="小时级 PM2.5 实验展示系统",
    page_icon="AQI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def set_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.facecolor"] = "#fffdf8"
    plt.rcParams["figure.facecolor"] = "#fffdf8"
    plt.rcParams["axes.edgecolor"] = "#d7c9b3"
    plt.rcParams["axes.labelcolor"] = "#352d23"
    plt.rcParams["xtick.color"] = "#4a4032"
    plt.rcParams["ytick.color"] = "#4a4032"
    plt.rcParams["grid.color"] = "#e8dcc9"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(223, 178, 90, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(58, 124, 114, 0.10), transparent 24%),
                linear-gradient(180deg, #f7f2e7 0%, #efe4d2 100%);
            color: #241d17;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.8rem;
            max-width: 1520px;
        }
        .hero {
            padding: 1.7rem 1.9rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,253,248,0.96), rgba(248,238,220,0.92));
            border: 1px solid rgba(104, 83, 53, 0.14);
            box-shadow: 0 18px 56px rgba(68, 52, 25, 0.07);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.55rem;
            font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
        }
        .hero p {
            margin: 0.25rem 0;
            line-height: 1.75;
            font-size: 1rem;
            max-width: 1040px;
        }
        .panel {
            padding: 1.15rem 1.25rem;
            border-radius: 22px;
            background: rgba(255, 252, 246, 0.94);
            border: 1px solid rgba(104, 83, 53, 0.14);
            margin-bottom: 1rem;
            box-shadow: 0 10px 26px rgba(80, 60, 30, 0.04);
        }
        .panel strong {
            color: #7a5722;
            display: inline-block;
            margin-bottom: 0.45rem;
            font-size: 1.03rem;
        }
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 18px;
            margin: 0.95rem 0 1.25rem 0;
        }
        .small-card {
            padding: 1.1rem 1.15rem;
            border-radius: 18px;
            background: rgba(255, 252, 246, 0.95);
            border: 1px solid rgba(104, 83, 53, 0.14);
            min-height: 170px;
            box-shadow: 0 12px 30px rgba(80, 60, 30, 0.05);
        }
        .small-card h4 {
            margin: 0 0 0.35rem 0;
            color: #7a5722;
            font-size: 1.6rem;
        }
        .small-card p {
            margin: 0;
            line-height: 1.78;
            font-size: 1.02rem;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin: 0.6rem 0 1.3rem 0;
        }
        .stat-card {
            padding: 1rem 1.15rem 1.1rem 1.15rem;
            border-radius: 22px;
            background: rgba(255, 252, 246, 0.96);
            border: 1px solid rgba(104, 83, 53, 0.14);
            box-shadow: 0 14px 34px rgba(80, 60, 30, 0.05);
            min-height: 146px;
        }
        .stat-card .label {
            color: #5b5141;
            font-size: 0.98rem;
            margin-bottom: 0.6rem;
        }
        .stat-card .value {
            color: #241d17;
            font-size: clamp(2rem, 2.8vw, 3.35rem);
            line-height: 1.08;
            font-weight: 700;
            letter-spacing: -0.02em;
            overflow-wrap: anywhere;
        }
        .stat-card .value.long {
            font-size: clamp(1.2rem, 1.9vw, 1.9rem);
            line-height: 1.28;
        }
        .stat-card .note {
            margin-top: 0.55rem;
            color: #8b7a60;
            font-size: 0.84rem;
            line-height: 1.55;
        }
        .section-head {
            margin: 1.25rem 0 0.85rem 0;
        }
        .section-kicker {
            display: inline-block;
            padding: 0.24rem 0.72rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(193, 90, 72, 0.16), rgba(82, 123, 114, 0.13));
            color: #8b4c3d;
            font-size: 0.84rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 0.45rem;
        }
        .section-head h3 {
            margin: 0;
            color: #2f3448;
            font-size: 2rem;
            line-height: 1.12;
        }
        .section-head p {
            margin: 0.32rem 0 0 0;
            color: #66594a;
            line-height: 1.7;
            max-width: 920px;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 18px;
            margin: 0.75rem 0 1.2rem 0;
        }
        .feature-block {
            padding: 1rem 1.05rem;
            border-radius: 20px;
            background: rgba(255, 252, 246, 0.95);
            border: 1px solid rgba(104, 83, 53, 0.14);
            box-shadow: 0 10px 28px rgba(80, 60, 30, 0.04);
        }
        .feature-block h4 {
            margin: 0 0 0.8rem 0;
            color: #7a5722;
            font-size: 1.1rem;
        }
        .chip-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .feature-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.66rem;
            border-radius: 999px;
            background: rgba(61, 111, 101, 0.09);
            border: 1px solid rgba(61, 111, 101, 0.11);
            color: #294a43;
            font-size: 0.9rem;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        .feature-empty {
            color: #8b7a60;
            font-size: 0.92rem;
        }
        .stDataFrame, div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(104, 83, 53, 0.12);
        }
        div[data-testid="metric-container"] {
            background: rgba(255, 252, 246, 0.95);
            border: 1px solid rgba(104, 83, 53, 0.14);
            border-radius: 18px;
            padding: 0.85rem 1rem;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #efe8d9 0%, #e7decb 100%);
            border-right: 1px solid rgba(104, 83, 53, 0.10);
        }
        section[data-testid="stSidebar"] .stRadio label {
            padding: 0.46rem 0.78rem !important;
            border-radius: 10px !important;
            margin: 2px 0 !important;
        }
        section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background: rgba(61, 111, 101, 0.16) !important;
            color: #294a43 !important;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 2px solid rgba(104, 83, 53, 0.14);
            margin-bottom: 1rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 42px;
            font-size: 0.98rem;
            color: #5d513f;
            border-bottom: 3px solid transparent !important;
        }
        .stTabs [aria-selected="true"] {
            color: #c15a48 !important;
            border-bottom: 3px solid #c15a48 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, body: str) -> None:
    st.markdown(
        f'<div class="panel"><strong>{escape(title)}</strong><br/>{body}</div>',
        unsafe_allow_html=True,
    )


def render_cards(cards: list[tuple[str, str]]) -> None:
    blocks = []
    for title, text in cards:
        blocks.append(
            f'<div class="small-card"><h4>{escape(title)}</h4><p>{text}</p></div>'
        )
    st.markdown(f'<div class="card-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)


def render_stat_cards(cards: list[tuple[str, str, str | None]]) -> None:
    blocks = []
    for label, value, note in cards:
        value_text = str(value)
        value_class = "value long" if len(value_text) > 12 else "value"
        note_html = f'<div class="note">{escape(note)}</div>' if note else ""
        blocks.append(
            f'<div class="stat-card"><div class="label">{escape(label)}</div><div class="{value_class}">{escape(value_text)}</div>{note_html}</div>'
        )
    st.markdown(f'<div class="stat-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)


def render_section_header(kicker: str, title: str, desc: str | None = None) -> None:
    desc_html = f"<p>{escape(desc)}</p>" if desc else ""
    st.markdown(
        f'<div class="section-head"><div class="section-kicker">{escape(kicker)}</div><h3>{escape(title)}</h3>{desc_html}</div>',
        unsafe_allow_html=True,
    )


def render_feature_groups(groups: dict[str, list[str]]) -> None:
    blocks = []
    for title, items in groups.items():
        chip_html = "".join(f'<span class="feature-chip">{escape(item)}</span>' for item in items)
        if not chip_html:
            chip_html = '<span class="feature-empty">暂无对应特征</span>'
        blocks.append(
            f'<div class="feature-block"><h4>{escape(title)}</h4><div class="chip-wrap">{chip_html}</div></div>'
        )
    st.markdown(f'<div class="feature-grid">{"".join(blocks)}</div>', unsafe_allow_html=True)


def show_saved_figure(filename: str, caption: str) -> None:
    path = FIGURE_DIR / filename
    if not path.exists():
        render_panel("图片缺失", f"未找到 {escape(filename)}，请先运行 `python scripts/generate_hourly_figures.py`。")
        return
    st.image(str(path), caption=caption, use_column_width=True)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_overall_metrics() -> pd.DataFrame:
    df = load_csv(OUTPUT_DIR / "overall_metrics_summary.csv")
    if df.empty:
        return df
    df["horizon_hours"] = pd.to_numeric(df["horizon_hours"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    files = [
        "arima_predictions_h1.csv",
        "arima_predictions_seq6.csv",
        "prophet_predictions_h1.csv",
        "prophet_predictions_seq6.csv",
        "xgboost_predictions_h1.csv",
        "xgboost_predictions_seq6.csv",
        "lstm_predictions_h1.csv",
        "lstm_predictions_seq6.csv",
        "transformer_predictions_h1.csv",
        "transformer_predictions_seq6.csv",
    ]
    frames: list[pd.DataFrame] = []
    for filename in files:
        df = load_csv(OUTPUT_DIR / filename)
        if df.empty:
            continue
        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        if "task" not in df.columns:
            df["task"] = "seq6" if "seq6" in filename else "h1"
        df["forecast_origin_time"] = pd.to_datetime(df["forecast_origin_time"])
        df["target_time"] = pd.to_datetime(df["target_time"])
        df["horizon_hours"] = pd.to_numeric(df["horizon_hours"], errors="coerce")
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def load_hourly_data() -> pd.DataFrame:
    df = load_csv(PROCESSED_DIR / "beijing_hourly.csv")
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_feature_names() -> list[str]:
    path = FEATURE_DIR / "feature_names.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_feature_matrix_meta() -> dict[str, int]:
    X_train = np.load(FEATURE_DIR / "X_train.npy")
    X_test = np.load(FEATURE_DIR / "X_test.npy")
    y_train = np.load(FEATURE_DIR / "y_train.npy")
    y_test = np.load(FEATURE_DIR / "y_test.npy")
    dates_train = np.load(FEATURE_DIR / "dates_train.npy", allow_pickle=True)
    dates_test = np.load(FEATURE_DIR / "dates_test.npy", allow_pickle=True)
    return {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "train_start": str(dates_train[0]),
        "train_end": str(dates_train[-1]),
        "test_start": str(dates_test[0]),
        "test_end": str(dates_test[-1]),
        "y_train_mean": float(np.mean(y_train)),
        "y_test_mean": float(np.mean(y_test)),
    }


@st.cache_resource(show_spinner=False)
def rebuild_hourly_scaler(feature_names: tuple[str, ...]) -> StandardScaler:
    raw = load_hourly_data()
    if raw.empty:
        raise FileNotFoundError("缺少 data/processed/beijing_hourly.csv，无法重建实时预测 scaler。")

    frame, _ = build_hourly_feature_frame(raw)
    model_frame = frame.dropna(subset=list(feature_names) + ["pm25"]).reset_index(drop=True)
    if model_frame.empty:
        raise ValueError("重建实时预测 scaler 失败：特征工程后没有可用样本。")

    split_index = int(len(model_frame) * 0.8)
    train_frame = model_frame.iloc[:split_index]
    scaler = StandardScaler()
    scaler.fit(train_frame[list(feature_names)].to_numpy(dtype=np.float64))
    return scaler


@st.cache_resource(show_spinner=False)
def load_realtime_assets():
    with open(FEATURE_DIR / "feature_names.json", encoding="utf-8") as handle:
        feature_names = json.load(handle)
    try:
        with open(FEATURE_DIR / "scaler.pkl", "rb") as handle:
            scaler = pickle.load(handle)
    except Exception:
        scaler = rebuild_hourly_scaler(tuple(feature_names))
    with open(OUTPUT_DIR / "lstm_h1_config.json", encoding="utf-8") as handle:
        lstm_config = json.load(handle)
    lstm_model = LSTMModel(
        input_dim=len(feature_names),
        hidden_dim=int(lstm_config["hidden_dim"]),
        num_layers=int(lstm_config["num_layers"]),
        dropout=float(lstm_config["dropout"]),
        output_dim=int(lstm_config["output_len"]),
    )
    lstm_trainer = LSTMTrainer.load(OUTPUT_DIR / "lstm_h1.pkl", model=lstm_model)
    lstm_trainer.model.eval()

    transformer_checkpoint = torch.load(
        OUTPUT_DIR / "transformer_seq6.pt",
        map_location="cpu",
        weights_only=False,
    )
    transformer_model = TimeSeriesTransformer(**transformer_checkpoint["config"])
    transformer_model.load_state_dict(transformer_checkpoint["model_state_dict"])
    transformer_model.eval()
    return scaler, feature_names, lstm_config, lstm_trainer.model, transformer_checkpoint["config"], transformer_model


def dedupe_metrics(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if df.empty:
        return df
    if task == "h1":
        sub = df[(df["task"] == "h1") & (df["horizon_hours"] == 1)].copy()
    elif task == "seq6":
        sub = df[df["task"] == "seq6"].copy()
    else:
        sub = df[df["task"] == "seq6_mean"].copy()
    return sub.sort_values(["model", "task", "horizon_hours", "rmse"]).drop_duplicates(
        subset=["model", "task", "horizon_hours"],
        keep="first",
    )


def format_metric_table(df: pd.DataFrame, include_horizon: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    table = df.sort_values("rmse").copy()
    table["排名"] = range(1, len(table) + 1)
    table["RMSE"] = table["rmse"].round(2)
    table["MAE"] = table["mae"].round(2)
    table["MAPE"] = table["mape"].round(2)
    table["样本数"] = table["n_samples"].astype(int)
    keep = ["排名", "model", "RMSE", "MAE", "MAPE", "样本数"]
    rename = {"model": "模型"}
    if include_horizon:
        table["步长"] = table["horizon_hours"].astype(int)
        keep.insert(2, "步长")
    return table[keep].rename(columns=rename)


def get_best_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    return df.sort_values("rmse").iloc[0]


def to_summary_markdown(best_h1: pd.Series | None, best_seq6: pd.Series | None, seq6_df: pd.DataFrame) -> str:
    lines = []
    if best_h1 is not None:
        lines.append(f"- h1 最优模型：`{best_h1['model']}`，RMSE `{best_h1['rmse']:.3f}`，MAE `{best_h1['mae']:.3f}`。")
    if best_seq6 is not None:
        lines.append(f"- 未来 6 小时平均表现最优模型：`{best_seq6['model']}`，RMSE `{best_seq6['rmse']:.3f}`，MAE `{best_seq6['mae']:.3f}`。")
    xgb = seq6_df[seq6_df["model"] == "XGBoost"].sort_values("horizon_hours")
    if not xgb.empty:
        trend = "基本单调上升" if xgb["rmse"].is_monotonic_increasing else "并非严格单调"
        values = "，".join(f"h{int(row.horizon_hours)}={row.rmse:.2f}" for row in xgb.itertuples())
        lines.append(f"- XGBoost 在 6 小时序列任务上的 RMSE 随步长 {trend}：{values}。")
    return "\n".join(lines)


def draw_metric_bars(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "暂无指标", ha="center", va="center")
            ax.axis("off")
        return fig

    plot_df = df.copy()
    plot_df["标签"] = plot_df["model"] + " · " + plot_df["task"]
    for ax, metric in zip(axes, ["rmse", "mae", "mape"]):
        ordered = plot_df.sort_values(metric)
        colors = [MODEL_COLORS.get(model, "#888888") for model in ordered["model"]]
        ax.barh(ordered["标签"], ordered[metric], color=colors, alpha=0.92)
        ax.set_title(metric.upper(), fontsize=13, weight="bold")
        ax.grid(axis="x", alpha=0.22, linestyle="--")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def draw_prediction_curve(df: pd.DataFrame, models: list[str], horizon: int, max_points: int = 168) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    sub = df[(df["horizon_hours"] == horizon) & (df["model"].isin(models))].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "当前筛选条件下没有可展示的预测数据", ha="center", va="center")
        ax.axis("off")
        return fig

    truth = sub[["target_time", "y_true"]].drop_duplicates().sort_values("target_time").tail(max_points)
    ax.plot(
        truth["target_time"],
        truth["y_true"],
        color="#1f1f1f",
        linewidth=2.8,
        label="真实值",
        zorder=5,
    )

    for model in models:
        model_df = sub[sub["model"] == model][["target_time", "y_pred"]]
        model_df = model_df[model_df["target_time"].isin(truth["target_time"])].sort_values("target_time")
        if model_df.empty:
            continue
        ax.plot(
            model_df["target_time"],
            model_df["y_pred"],
            linewidth=2.1,
            color=MODEL_COLORS.get(model, "#888888"),
            alpha=0.95,
            label=model,
        )

    ax.set_title(f"最近 {len(truth)} 个点的预测曲线对比（h{horizon}）", fontsize=14, weight="bold")
    ax.set_xlabel("目标时刻")
    ax.set_ylabel("PM2.5")
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(ncol=6, frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def draw_seq6_curve(seq6_df: pd.DataFrame, models: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    sub = seq6_df[seq6_df["model"].isin(models)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "暂无 6 小时序列分步结果", ha="center", va="center")
        ax.axis("off")
        return fig

    for model in models:
        model_df = sub[sub["model"] == model].sort_values("horizon_hours")
        if model_df.empty:
            continue
        ax.plot(
            model_df["horizon_hours"],
            model_df["rmse"],
            marker="o",
            markersize=6,
            linewidth=2.4,
            color=MODEL_COLORS.get(model, "#888888"),
            label=model,
        )
    ax.set_title("未来 6 小时逐步 RMSE 变化", fontsize=14, weight="bold")
    ax.set_xlabel("预测步长（小时）")
    ax.set_ylabel("RMSE")
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, ncol=5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def draw_aqi_bucket(df: pd.DataFrame, task: str, horizon: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.2, 4.9))
    sub = df[(df["task"] == task) & (df["horizon_hours"] == horizon)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "当前条件下没有 AQI 分层结果", ha="center", va="center")
        ax.axis("off")
        return fig

    bucket_order = ["0-35", "35-75", "75-115", "115-150", ">150"]
    for model in sub["model"].drop_duplicates():
        model_df = sub[sub["model"] == model].copy()
        model_df["aqi_bucket"] = pd.Categorical(model_df["aqi_bucket"], bucket_order, ordered=True)
        model_df = model_df.sort_values("aqi_bucket")
        ax.plot(
            model_df["aqi_bucket"].astype(str),
            model_df["rmse"],
            marker="o",
            linewidth=2.2,
            color=MODEL_COLORS.get(model, "#888888"),
            label=model,
        )

    ax.set_title(f"AQI 分层误差对比（{task}，h{horizon}）", fontsize=14, weight="bold")
    ax.set_xlabel("AQI 分层")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.18, linestyle="--")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def draw_feature_importance(path: Path, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 6))
    df = load_csv(path)
    if df.empty:
        ax.text(0.5, 0.5, "暂无特征重要性结果", ha="center", va="center")
        ax.axis("off")
        return fig

    top = df.head(18).iloc[::-1]
    values = top["gain_norm"] if "gain_norm" in top.columns else top["gain"]
    ax.barh(top["feature"], values, color="#2f8078", alpha=0.92)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("相对重要性")
    ax.grid(axis="x", alpha=0.18, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def draw_feature_ablation(df: pd.DataFrame, task: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    if df.empty:
        ax.text(0.5, 0.5, "暂无特征消融结果", ha="center", va="center")
        ax.axis("off")
        return fig

    order = ["base_short_lag", "base_plus_daily_cycle", "mid_range_memory", "trend_and_roll", "weather_enhanced", "full_48"]
    if task == "h1":
        sub = df[(df["task"] == "h1") & (df["horizon_hours"] == 1)].copy()
    else:
        sub = df[df["task"] == "seq6"].groupby("feature_group", as_index=False)["rmse"].mean()
    sub["feature_group"] = pd.Categorical(sub["feature_group"], order, ordered=True)
    sub = sub.sort_values("feature_group")

    ax.plot(
        sub["feature_group"].astype(str),
        sub["rmse"],
        marker="o",
        markersize=6,
        linewidth=2.4,
        color="#d49d2d",
    )
    ax.set_title("XGBoost 特征消融结果", fontsize=14, weight="bold")
    ax.set_xlabel("特征组")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.18, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def draw_high_pollution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    if df.empty:
        ax.text(0.5, 0.5, "暂无高污染结果", ha="center", va="center")
        ax.axis("off")
        return fig

    sub = df.copy()
    sub["标签"] = sub["model"] + " · " + sub["task"] + " · h" + sub["horizon_hours"].astype(int).astype(str)
    ordered = sub.sort_values("rmse")
    colors = [MODEL_COLORS.get(model, "#888888") for model in ordered["model"]]
    ax.barh(ordered["标签"], ordered["rmse"], color=colors, alpha=0.92)
    ax.set_title("高污染样本误差对比（PM2.5 > 150）", fontsize=14, weight="bold")
    ax.set_xlabel("RMSE")
    ax.grid(axis="x", alpha=0.18, linestyle="--")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def draw_hourly_error(df: pd.DataFrame, task: str, horizon: int, models: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    sub = df[(df["task"] == task) & (df["horizon_hours"] == horizon) & (df["model"].isin(models))].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "暂无分小时误差结果", ha="center", va="center")
        ax.axis("off")
        return fig

    for model in models:
        model_df = sub[sub["model"] == model].sort_values("hour")
        if model_df.empty:
            continue
        ax.plot(
            model_df["hour"],
            model_df["mae"],
            linewidth=2.1,
            color=MODEL_COLORS.get(model, "#888888"),
            label=model,
        )

    ax.set_title(f"按小时 MAE 变化（{task}，h{horizon}）", fontsize=14, weight="bold")
    ax.set_xlabel("小时")
    ax.set_ylabel("MAE")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.18, linestyle="--")
    ax.legend(frameon=False, ncol=5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def draw_cross_city(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if df.empty:
        ax.text(0.5, 0.5, "暂无跨城市泛化结果", ha="center", va="center")
        ax.axis("off")
        return fig

    sub = df.copy()
    sub["标签"] = sub["task"] + "-h" + sub["horizon_hours"].astype(int).astype(str)
    ax.bar(sub["标签"], sub["rmse"], color="#5e73b9", alpha=0.9)
    ax.set_title("跨城市泛化：北京训练，上海测试", fontsize=14, weight="bold")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.18, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def hourly_data_summary(df: pd.DataFrame) -> dict[str, str | int | float]:
    return {
        "rows": int(len(df)),
        "start": df["datetime"].min().strftime("%Y-%m-%d %H:%M"),
        "end": df["datetime"].max().strftime("%Y-%m-%d %H:%M"),
        "pm25_mean": float(df["pm25"].mean()),
        "pm25_max": float(df["pm25"].max()),
    }


def get_realtime_default_table(df: pd.DataFrame, n_rows: int = 120) -> pd.DataFrame:
    default = df[REALTIME_COLUMNS].tail(n_rows).copy().reset_index(drop=True)
    default["datetime"] = default["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return default


def run_realtime_prediction(input_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = input_df.copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    for col in ["pm25", "temp", "pres", "dewp", "humidity", "wind_speed", "precipitation"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.sort_values("datetime").reset_index(drop=True)

    scaler, feature_names, lstm_config, lstm_h1_model, transformer_config, transformer_seq6_model = load_realtime_assets()
    frame, _ = build_hourly_feature_frame(work)
    valid = frame.dropna(subset=feature_names).reset_index(drop=True)

    seq_len_h1 = int(lstm_config["seq_len"])
    seq_len_seq6 = int(transformer_config["seq_len"])
    min_feature_rows = max(seq_len_h1, seq_len_seq6)
    if len(valid) < min_feature_rows:
        raise ValueError(
            f"输入数据不足：至少需要 {min_feature_rows + 24} 小时原始观测，"
            f"当前仅能构造 {len(valid)} 条有效特征行。"
        )

    scaled = scaler.transform(valid[feature_names].to_numpy(dtype=np.float64)).astype(np.float32)
    base_time = pd.to_datetime(valid["datetime"].iloc[-1])

    h1_window = torch.from_numpy(scaled[-seq_len_h1:]).unsqueeze(0)
    seq6_window = torch.from_numpy(scaled[-seq_len_seq6:]).unsqueeze(0)

    with torch.no_grad():
        h1_value = max(0.0, float(lstm_h1_model(h1_window).cpu().numpy().reshape(-1)[0]))
        seq6_raw = transformer_seq6_model(seq6_window).cpu().numpy().reshape(-1)
    seq6_values = [max(0.0, float(value)) for value in seq6_raw.tolist()]

    summary = pd.DataFrame(
        [
            {"任务": "h1 单步预测（LSTM）", "预测值": round(h1_value, 2)},
            {"任务": "未来 6 小时均值（Transformer）", "预测值": round(float(np.mean(seq6_values)), 2)},
            {"任务": "未来 6 小时峰值（Transformer）", "预测值": round(float(np.max(seq6_values)), 2)},
        ]
    )

    detail = pd.DataFrame(
        {
            "目标时刻": [(base_time + pd.Timedelta(hours=h)).strftime("%Y-%m-%d %H:%M:%S") for h in range(1, 7)],
            "预测步长": [f"h{h}" for h in range(1, 7)],
            "预测 PM2.5": [round(value, 2) for value in seq6_values],
        }
    )
    return summary, detail


def draw_realtime_forecast(detail: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(
        detail["预测步长"],
        detail["预测 PM2.5"],
        marker="o",
        markersize=7,
        linewidth=2.5,
        color="#2f8078",
    )
    ax.fill_between(detail["预测步长"], detail["预测 PM2.5"], color="#2f8078", alpha=0.12)
    ax.set_title("未来 6 小时实时预测结果", fontsize=14, weight="bold")
    ax.set_xlabel("预测步长")
    ax.set_ylabel("预测 PM2.5")
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


set_plot_style()
inject_css()

overall_metrics = load_overall_metrics()
predictions = load_predictions()
hourly_df = load_hourly_data()
feature_names = load_feature_names()
feature_meta = load_feature_matrix_meta()
aqi_bucket = load_csv(OUTPUT_DIR / "aqi_bucket_metrics.csv")
feature_ablation = load_csv(OUTPUT_DIR / "feature_ablation.csv")
high_pollution = load_csv(OUTPUT_DIR / "high_pollution_error_analysis.csv")
hourly_error = load_csv(OUTPUT_DIR / "hourly_error_by_hour.csv")
dayperiod_error = load_csv(OUTPUT_DIR / "hourly_error_by_dayperiod.csv")
cross_city = load_csv(OUTPUT_DIR / "cross_city_metrics.csv")
registry = load_csv(OUTPUT_DIR / "experiment_registry.csv")

h1_metrics = dedupe_metrics(overall_metrics, "h1")
seq6_metrics = dedupe_metrics(overall_metrics, "seq6")
seq6_mean = dedupe_metrics(overall_metrics, "seq6_mean")
summary_metrics = pd.concat([h1_metrics.assign(task="h1"), seq6_mean.assign(task="seq6_mean")], ignore_index=True)

best_h1 = get_best_row(h1_metrics)
best_seq6 = get_best_row(seq6_mean)
summary_text = to_summary_markdown(best_h1, best_seq6, seq6_metrics)
all_models = sorted(predictions["model"].dropna().unique().tolist()) if not predictions.empty else []
hourly_info = hourly_data_summary(hourly_df)

st.markdown(
    """
    <div class="hero">
      <h1>小时级 PM2.5 实验展示系统</h1>
      <p>本页面围绕当前已经完成的真实实验产物构建，展示数据处理与特征工程、h1 单步预测、未来 6 小时逐小时预测、误差分析与实时预测。</p>
      <p>所有指标和图表都直接读取 <code>outputs/hourly</code> 与 <code>data/processed/features_hourly</code> 中的结果，不使用占位数据或过期页面文案。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 导航")
    current_page = st.radio(
        "页面导航",
        ["实验总览", "数据处理与特征工程", "h1 单步预测", "6 小时序列预测", "深入分析", "真实预测"],
        label_visibility="collapsed",
    )

if current_page == "实验总览":
    st.subheader("实验总览")

    render_stat_cards(
        [
            ("小时级样本数", f"{hourly_info['rows']}", "统一按小时排序后进入建模流程"),
            ("特征维度", f"{feature_meta['n_features']}", "当前特征工程共输出 48 维输入"),
            ("h1 最优模型", best_h1["model"] if best_h1 is not None else "-", "按 RMSE 排名"),
            ("6 小时均值最优模型", best_seq6["model"] if best_seq6 is not None else "-", "按 seq6 平均 RMSE 排名"),
        ]
    )

    render_cards(
        [
            (
                "h1 结论",
                f"当前 h1 单步预测中，{best_h1['model']} 表现最好，RMSE 为 {best_h1['rmse']:.2f}，MAE 为 {best_h1['mae']:.2f}。"
                if best_h1 is not None
                else "当前暂无 h1 指标。",
            ),
            (
                "6 小时序列结论",
                f"未来 6 小时逐小时预测中，{best_seq6['model']} 的平均表现最优，RMSE 为 {best_seq6['rmse']:.2f}。"
                if best_seq6 is not None
                else "当前暂无 6 小时序列指标。",
            ),
            (
                "模型结构",
                "本轮对比包含 ARIMA、Prophet、XGBoost、LSTM 和 Transformer 五类模型，覆盖统计模型、树模型和深度学习模型。",
            ),
        ]
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        render_section_header("图 3", "模型整体指标比较", "当前展示的是最新汇总结果生成的整体指标图，已包含增强版 Prophet。")
        show_saved_figure("fig07_overall_rmse.png", "图 3a  Overall RMSE comparison")
        show_saved_figure("fig08_overall_mae.png", "图 3b  Overall MAE comparison")
        show_saved_figure("fig09_overall_mape.png", "图 3c  Overall MAPE comparison")
        st.markdown("### 指标明细表")
        st.dataframe(format_metric_table(summary_metrics), use_container_width=True, hide_index=True)
    with right:
        render_panel("自动结论摘要", summary_text.replace("\n", "<br/>"))
        render_panel(
            "数据范围",
            f"小时级主数据覆盖 {hourly_info['start']} 到 {hourly_info['end']}，"
            f"平均 PM2.5 为 {hourly_info['pm25_mean']:.2f}，最大值为 {hourly_info['pm25_max']:.2f}。",
        )

elif current_page == "数据处理与特征工程":
    st.subheader("数据处理与特征工程")

    groups = {
        "污染物滞后特征": [x for x in feature_names if x.startswith("pm25_lag")],
        "滚动统计特征": [x for x in feature_names if x.startswith("pm25_roll")],
        "趋势变化特征": [x for x in feature_names if x.startswith("pm25_diff")],
        "时间与日历特征": [x for x in feature_names if x in {"hour_sin", "hour_cos", "month_sin", "month_cos", "weekday", "is_weekend", "is_holiday", "is_daytime", "is_rush_hour"}],
        "气象与交互特征": [x for x in feature_names if x not in {
            *[n for n in feature_names if n.startswith("pm25_lag")],
            *[n for n in feature_names if n.startswith("pm25_roll")],
            *[n for n in feature_names if n.startswith("pm25_diff")],
            "hour_sin", "hour_cos", "month_sin", "month_cos", "weekday", "is_weekend", "is_holiday", "is_daytime", "is_rush_hour",
        }],
    }

    render_stat_cards(
        [
            ("数据起点", hourly_info["start"], "北京小时级主数据起始时间"),
            ("数据终点", hourly_info["end"], "北京小时级主数据结束时间"),
            ("训练样本数", f"{feature_meta['n_train']}", "按时间顺序切分得到"),
            ("测试样本数", f"{feature_meta['n_test']}", "保留后 20% 作为测试集"),
            ("特征总数", f"{feature_meta['n_features']}", "同一套输入同时支撑 h1 与 seq6"),
        ]
    )

    render_cards(
        [
            ("原始粒度", "实验使用北京城市级小时数据作为主数据源，并在跨城市实验中引入上海对齐小时数据。"),
            ("清洗策略", "统一字段、数值化污染物与气象字段、保留时间顺序，并在特征构造前完成缺失值处理和格式清洗。"),
            ("划分方式", "按时间顺序切分，前 80% 为训练集，后 20% 为测试集，避免时间穿越。"),
        ]
    )

    upper_left, upper_right = st.columns([0.9, 1.1])
    with upper_left:
        render_section_header("图 2", "数据处理流程", "先清洗、再构造特征、最后按时间切分，保证训练和测试严格顺序一致。")
        render_panel(
            "处理流程",
            "1. 统一小时级污染物与气象字段。<br/>"
            "2. 保证 datetime 升序排列。<br/>"
            "3. 在特征工程前完成数值型字段清洗。<br/>"
            "4. 构造滞后、滚动统计、时间与气象交互特征。<br/>"
            "5. 使用固定时间切分生成训练集和测试集。",
        )
        render_panel(
            "训练 / 测试时间范围",
            f"训练集：{feature_meta['train_start']} 到 {feature_meta['train_end']}。<br/>"
            f"测试集：{feature_meta['test_start']} 到 {feature_meta['test_end']}。",
        )
    with upper_right:
        render_section_header("图 3", "小时级样本预览", "保留最近一段原始样本，方便展示字段结构、时间戳和气象变量。")
        preview_rows = st.slider("预览行数", min_value=8, max_value=30, value=12, step=2, key="raw_preview")
        preview_cols = ["datetime", "pm25", "pm10", "temp", "pres", "dewp", "humidity", "wind_speed", "precipitation"]
        preview_df = hourly_df[preview_cols].tail(preview_rows).copy()
        preview_df["datetime"] = preview_df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(preview_df, use_container_width=True, hide_index=True, height=440)

    render_panel(
        "设计思路",
        "这套 48 维特征围绕“短期惯性 + 日内周期 + 气象扰动”构建，目的是让同一套输入同时支撑 h1 单步预测和未来 6 小时序列预测。",
    )

    render_section_header("图 4", "特征组与重要性", "把长串特征拆成分组标签展示，再用重要性与消融实验验证哪些输入最关键。")

    col_a, col_b = st.columns([1.0, 1.0])
    with col_a:
        render_feature_groups(groups)
    with col_b:
        tabs = st.tabs(["h1 重要性", "6 小时重要性", "特征消融"])
        with tabs[0]:
            show_saved_figure("fig10_feature_importance_h1.png", "图 4  XGBoost feature importance for h1")
        with tabs[1]:
            show_saved_figure("fig11_feature_importance_seq6.png", "图 5  XGBoost feature importance for seq6")
        with tabs[2]:
            task_choice = st.radio("查看任务", ["h1", "seq6"], horizontal=True, key="ablation_static_task")
            show_saved_figure(
                "fig14_feature_ablation_h1.png" if task_choice == "h1" else "fig15_feature_ablation_seq6.png",
                f"图 7  Feature ablation for {task_choice}",
            )

elif current_page == "h1 单步预测":
    st.subheader("h1 单步预测")

    render_section_header("图 1", "h1 综合表现", "使用正式成图展示预测曲线、模型排序与残差分布，不再混用页面临时绘图。")
    show_saved_figure("fig01_h1_prediction_curve.png", "图 1a  h1 prediction curve")
    show_saved_figure("fig02_h1_model_rmse.png", "图 1b  h1 model RMSE comparison")
    show_saved_figure("fig03_h1_residual_kde.png", "图 1c  h1 residual KDE")

    st.markdown("### h1 指标表")
    metric_view = h1_metrics.copy()
    st.dataframe(format_metric_table(metric_view), use_container_width=True, hide_index=True)

    best_local = get_best_row(h1_metrics)
    if best_local is not None:
        render_panel(
            "结果解读",
            f"当前 h1 任务下，{best_local['model']} 的 RMSE 最低，为 {best_local['rmse']:.2f}。"
            f" LSTM 在短期 1 小时预测上略优于 XGBoost 和 Transformer，说明它对局部时间依赖建模更有优势。",
        )

elif current_page == "6 小时序列预测":
    st.subheader("6 小时序列预测")

    render_section_header("图 2", "未来 6 小时逐小时预测综合表现", "左侧保留 h6 预测曲线，右侧同时展示随步长变化的 RMSE 趋势和均值排名。")
    show_saved_figure("fig04_h6_prediction_curve.png", "图 2a  h6 prediction curve")
    show_saved_figure("fig05_rmse_vs_horizon.png", "图 2b  RMSE vs horizon")
    show_saved_figure("fig06_seq6_avg_rmse.png", "图 2c  seq6 average RMSE")

    st.markdown("### 分步指标表")
    step_table = seq6_metrics.copy()
    st.dataframe(format_metric_table(step_table, include_horizon=True), use_container_width=True, hide_index=True)

    st.markdown("### 6 小时均值指标表")
    st.dataframe(format_metric_table(seq6_mean), use_container_width=True, hide_index=True)

    render_panel(
        "结果解读",
        "当前真实实验结果表明：XGBoost 在未来 6 小时平均指标上最好；Transformer 在 h1~h6 的多步走势上明显优于 LSTM；"
        "随着步长从 h1 增加到 h6，几乎所有模型的误差都会持续变大。",
    )

elif current_page == "深入分析":
    st.subheader("深入分析")

    tab1, tab2, tab3, tab4 = st.tabs(["AQI 分层", "高污染与时段", "特征消融", "跨城市泛化"])

    with tab1:
        task_choice = st.radio("任务", ["h1", "seq6"], horizontal=True, key="aqi_task")
        show_saved_figure(
            "fig12_aqi_heatmap_h1.png" if task_choice == "h1" else "fig13_aqi_heatmap_seq6.png",
            f"图 6  AQI bucket heatmap for {task_choice}",
        )
        horizon_choice = 1 if task_choice == "h1" else st.select_slider("查看步长", options=[1, 2, 3, 4, 5, 6], value=6)
        table = aqi_bucket[(aqi_bucket["task"] == task_choice) & (aqi_bucket["horizon_hours"] == horizon_choice)].copy()
        st.dataframe(table.round(2), use_container_width=True, hide_index=True)

    with tab2:
        left, right = st.columns([1.0, 1.0])
        with left:
            hp_task = st.radio("高污染任务", ["h1", "seq6"], horizontal=True, key="hp_task")
            show_saved_figure(
                "fig16_high_pollution_h1.png" if hp_task == "h1" else "fig17_high_pollution_seq6.png",
                f"图 8  High-pollution error for {hp_task}",
            )
            hp_view = high_pollution[high_pollution["task"] == hp_task].copy()
            st.dataframe(hp_view.round(2), use_container_width=True, hide_index=True)
        with right:
            task_choice = st.radio("时段分析任务", ["h1", "seq6"], horizontal=True, key="hourly_task")
            show_saved_figure(
                "fig18_error_by_hour_h1.png" if task_choice == "h1" else "fig19_error_by_hour_seq6.png",
                f"图 9  Hourly error curve for {task_choice}",
            )
            show_saved_figure(
                "fig20_dayperiod_heatmap_h1.png" if task_choice == "h1" else "fig21_dayperiod_heatmap_seq6.png",
                f"图 10  Day-period heatmap for {task_choice}",
            )
            horizon_choice = 1 if task_choice == "h1" else 6
            day_view = dayperiod_error[(dayperiod_error["task"] == task_choice) & (dayperiod_error["horizon_hours"] == horizon_choice)].copy()
            st.dataframe(day_view.round(2), use_container_width=True, hide_index=True)

    with tab3:
        task_choice = st.radio("查看任务", ["h1", "seq6"], horizontal=True, key="ablation_task_2")
        show_saved_figure(
            "fig14_feature_ablation_h1.png" if task_choice == "h1" else "fig15_feature_ablation_seq6.png",
            f"图 7  Feature ablation for {task_choice}",
        )
        ablation_view = feature_ablation[feature_ablation["task"] == task_choice].copy()
        st.dataframe(ablation_view.round(2), use_container_width=True, hide_index=True)

    with tab4:
        show_saved_figure("fig22_cross_city_rmse.png", "图 11a  Cross-city RMSE comparison")
        show_saved_figure("fig23_generalization_gap.png", "图 11b  Generalization gap")
        st.dataframe(cross_city.round(2), use_container_width=True, hide_index=True)
        render_panel(
            "分析结论",
            "跨城市泛化目前基于 XGBoost 完成。北京训练、上海测试时，h1 还能保持可接受水平，但步长增加后误差持续扩大，"
            "说明 6 小时序列任务对城市分布差异更敏感。",
        )

elif current_page == "真实预测":
    st.subheader("真实预测")
    st.write("在这页中，你可以输入最近一段小时级观测数据，系统会直接调用仓库中已经训练完成的深度模型，给出 h1 单步预测和未来 6 小时逐小时预测结果。")

    render_panel(
        "输入要求",
        "建议直接保留最近 120 小时的数据。<br/>"
        "若要完整运行 h1 + 未来 6 小时序列预测，至少需要 96 小时原始观测。<br/>"
        "必填列：datetime、pm25、temp、pres、dewp、humidity、wind_dir、wind_speed、precipitation。<br/>"
        "如果不上传 CSV，可以直接在下方表格里修改默认示例数据。",
    )

    upload = st.file_uploader("上传小时级 CSV（可选）", type=["csv"])
    default_table = get_realtime_default_table(hourly_df)
    if upload is not None:
        uploaded_df = pd.read_csv(upload)
        input_table = uploaded_df.copy()
    else:
        input_table = default_table.copy()

    input_table = input_table[REALTIME_COLUMNS].copy()
    edited = st.data_editor(
        input_table,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="realtime_editor",
    )

    if st.button("开始预测", type="primary"):
        try:
            summary, detail = run_realtime_prediction(edited)
            c1, c2, c3 = st.columns(3)
            c1.metric("h1 预测值", f"{summary.iloc[0]['预测值']:.2f}")
            c2.metric("未来 6 小时均值", f"{summary.iloc[1]['预测值']:.2f}")
            c3.metric("未来 6 小时峰值", f"{summary.iloc[2]['预测值']:.2f}")

            left, right = st.columns([0.95, 1.05])
            with left:
                st.markdown("### 分步预测结果")
                st.dataframe(detail, use_container_width=True, hide_index=True)
            with right:
                st.pyplot(draw_realtime_forecast(detail), use_container_width=True)

            render_panel(
                "推理说明",
                "当前实时预测采用与现有实验权重直接对应的深度模型组合：h1 使用 LSTM 单步模型，"
                "未来 6 小时使用 Transformer 序列模型。这样既能保持真实模型推理，也能直接输出逐小时结果。",
            )
        except Exception as exc:
            st.error(f"预测失败：{exc}")

if registry.empty:
    st.caption("提示：experiment_registry.csv 暂未加载。")
