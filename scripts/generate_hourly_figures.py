"""Generate required figures for hourly experiment deliverables."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "hourly"
FIG_DIR = OUT / "figures"


def _safe_read(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_prediction_curve(horizon: int, filename: str) -> None:
    model_files = {
        "ARIMA": f"arima_predictions_h{horizon}.csv",
        "XGBoost": f"xgboost_predictions_h{horizon}.csv",
        "LSTM": f"lstm_predictions_h{horizon}.csv",
        "Transformer": f"transformer_predictions_h{horizon}.csv",
    }

    merged = None
    for model, fn in model_files.items():
        df = _safe_read(fn)
        if df.empty:
            continue
        df = df[df.get("split", "test") == "test"].copy() if "split" in df.columns else df
        df["target_time"] = pd.to_datetime(df["target_time"])
        df = df[["target_time", "y_true", "y_pred"]].rename(columns={"y_pred": f"y_pred_{model}"})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["target_time", f"y_pred_{model}"]], on="target_time", how="inner")

    if merged is None or merged.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"No prediction data for h{horizon}", ha="center", va="center")
        ax.axis("off")
        _save(fig, filename)
        return

    show_n = min(168, len(merged))
    show = merged.sort_values("target_time").iloc[:show_n]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(show["target_time"], show["y_true"], label="y_true", linewidth=2.0, color="black")

    for model in model_files:
        col = f"y_pred_{model}"
        if col in show.columns:
            ax.plot(show["target_time"], show[col], label=model, alpha=0.9)

    ax.set_title(f"Prediction Curve Comparison (h{horizon}, first {show_n} points)")
    ax.set_xlabel("target_time")
    ax.set_ylabel("PM2.5")
    ax.legend(ncol=5, fontsize=8)
    _save(fig, filename)


def fig_overall_metrics_bar() -> None:
    df = _safe_read("overall_metrics_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No overall_metrics_summary.csv", ha="center", va="center")
            ax.axis("off")
        _save(fig, "03_overall_metrics_bar.png")
        return

    df = df.dropna(subset=["rmse", "mae", "mape"]).copy()
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No valid metrics", ha="center", va="center")
            ax.axis("off")
        _save(fig, "03_overall_metrics_bar.png")
        return

    df["model_h"] = df["model"] + "-h" + df["horizon_hours"].astype(int).astype(str)
    for ax, metric in zip(axes, ["rmse", "mae", "mape"]):
        d = df.sort_values(metric)
        ax.bar(d["model_h"], d[metric])
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=45)
    _save(fig, "03_overall_metrics_bar.png")


def fig_feature_importance(task: str, filename: str) -> None:
    df = _safe_read(f"xgboost_feature_importance_{task}.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    if df.empty:
        ax.text(0.5, 0.5, f"No xgboost_feature_importance_{task}.csv", ha="center", va="center")
        ax.axis("off")
        _save(fig, filename)
        return
    top = df.head(20).iloc[::-1]
    vals = top["gain_norm"] if "gain_norm" in top.columns else top["gain"]
    ax.barh(top["feature"], vals)
    ax.set_title(f"XGBoost Feature Importance ({task})")
    _save(fig, filename)


def fig_aqi_bucket_heatmap() -> None:
    df = _safe_read("aqi_bucket_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No aqi_bucket_metrics.csv", ha="center", va="center")
            ax.axis("off")
        _save(fig, "06_aqi_bucket_error_heatmap.png")
        return

    for ax, h in zip(axes, [1, 12]):
        sub = df[df["horizon_hours"] == h].copy()
        piv = sub.pivot(index="model", columns="aqi_bucket", values="rmse")
        if piv.empty:
            ax.text(0.5, 0.5, f"No h{h} data", ha="center", va="center")
            ax.axis("off")
            continue
        mat = piv.values
        im = ax.imshow(mat, aspect="auto")
        ax.set_title(f"RMSE by AQI bucket (h{h})")
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=45)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "06_aqi_bucket_error_heatmap.png")


def fig_feature_ablation() -> None:
    df = _safe_read("feature_ablation.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "No feature_ablation.csv", ha="center", va="center")
        ax.axis("off")
        _save(fig, "07_feature_ablation_comparison.png")
        return

    order = ["base_short_lag", "base_plus_daily_cycle", "mid_range_memory", "trend_and_roll", "full_48"]
    for h in [1, 12]:
        sub = df[df["horizon_hours"] == h].copy()
        sub["feature_group"] = pd.Categorical(sub["feature_group"], categories=order, ordered=True)
        sub = sub.sort_values("feature_group")
        ax.plot(sub["feature_group"].astype(str), sub["rmse"], marker="o", label=f"h{h}")
    ax.set_title("XGBoost Feature Ablation (RMSE)")
    ax.set_ylabel("RMSE")
    ax.legend()
    _save(fig, "07_feature_ablation_comparison.png")


def fig_high_pollution() -> None:
    df = _safe_read("high_pollution_error_analysis.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "No high_pollution_error_analysis.csv", ha="center", va="center")
        ax.axis("off")
        _save(fig, "08_high_pollution_error.png")
        return

    df["model_h"] = df["model"] + "-h" + df["horizon_hours"].astype(int).astype(str)
    d = df.sort_values("rmse")
    ax.bar(d["model_h"], d["rmse"])
    ax.set_title("High-pollution RMSE (y_true > 150)")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "08_high_pollution_error.png")


def fig_error_by_hour() -> None:
    df = _safe_read("hourly_error_by_hour.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No hourly_error_by_hour.csv", ha="center", va="center")
            ax.axis("off")
        _save(fig, "09_error_by_hour_curve.png")
        return

    for ax, h in zip(axes, [1, 12]):
        sub = df[df["horizon_hours"] == h]
        for model in sorted(sub["model"].unique()):
            m = sub[sub["model"] == model].sort_values("hour")
            ax.plot(m["hour"], m["mae"], label=model)
        ax.set_title(f"MAE by hour (h{h})")
        ax.set_xlabel("Hour")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("MAE")
    axes[1].legend(fontsize=8)
    _save(fig, "09_error_by_hour_curve.png")


def fig_dayperiod_error() -> None:
    df = _safe_read("hourly_error_by_dayperiod.csv")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "No hourly_error_by_dayperiod.csv", ha="center", va="center")
        ax.axis("off")
        _save(fig, "10_dayperiod_error_bar.png")
        return

    order = ["凌晨", "早高峰", "白天平峰", "晚高峰", "夜间"]
    df["label"] = df["model"] + "-h" + df["horizon_hours"].astype(int).astype(str)
    grp = df.pivot_table(index="dayperiod", columns="label", values="mae")
    grp = grp.reindex(order)
    grp.plot(kind="bar", ax=ax)
    ax.set_title("MAE by day period")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, "10_dayperiod_error_bar.png")


def fig_cross_city() -> None:
    df = _safe_read("cross_city_metrics.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "No cross_city_metrics.csv", ha="center", va="center")
        ax.axis("off")
        _save(fig, "11_cross_city_generalization.png")
        return

    labels = [f"h{int(h)}" for h in df["horizon_hours"]]
    ax.bar(labels, df["rmse"])
    ax.set_title("Cross-city generalization (Beijing -> Shanghai)")
    ax.set_ylabel("RMSE")
    _save(fig, "11_cross_city_generalization.png")


def main() -> None:
    fig_prediction_curve(1, "01_prediction_curve_h1.png")
    fig_prediction_curve(12, "02_prediction_curve_h12.png")
    fig_overall_metrics_bar()
    fig_feature_importance("h1", "04_xgboost_feature_importance_h1.png")
    fig_feature_importance("h12", "05_xgboost_feature_importance_h12.png")
    fig_aqi_bucket_heatmap()
    fig_feature_ablation()
    fig_high_pollution()
    fig_error_by_hour()
    fig_dayperiod_error()
    fig_cross_city()
    print(f"Generated figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
