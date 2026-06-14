"""Generate figures for the hourly h1 + seq6 experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "hourly"
FIG_DIR = OUT / "figures"


def safe_read(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def prediction_curve(task: str, filename: str, horizon_filter: int) -> None:
    model_files = {
        "ARIMA": f"arima_predictions_{task}.csv" if task == "seq6" else "arima_predictions_h1.csv",
        "Prophet": f"prophet_predictions_{task}.csv" if task == "seq6" else "prophet_predictions_h1.csv",
        "XGBoost": f"xgboost_predictions_{task}.csv" if task == "seq6" else "xgboost_predictions_h1.csv",
        "LSTM": f"lstm_predictions_{task}.csv" if task == "seq6" else "lstm_predictions_h1.csv",
        "Transformer": f"transformer_predictions_{task}.csv" if task == "seq6" else "transformer_predictions_h1.csv",
    }

    merged = None
    for model, filename_in in model_files.items():
        df = safe_read(filename_in)
        if df.empty:
            continue
        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        df = df[df["horizon_hours"] == horizon_filter].copy()
        if df.empty:
            continue
        df["target_time"] = pd.to_datetime(df["target_time"])
        df = df[["target_time", "y_true", "y_pred"]].rename(columns={"y_pred": f"y_pred_{model}"})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["target_time", f"y_pred_{model}"]], on="target_time", how="inner")

    if merged is None or merged.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"No prediction data for {task}", ha="center", va="center")
        ax.axis("off")
        save(fig, filename)
        return

    show = merged.sort_values("target_time").iloc[: min(168, len(merged))]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(show["target_time"], show["y_true"], label="y_true", linewidth=2.0, color="black")
    for model in model_files:
        column = f"y_pred_{model}"
        if column in show.columns:
            ax.plot(show["target_time"], show[column], label=model, alpha=0.9)

    title = "Prediction Curve Comparison (h1)" if task == "h1" else "Prediction Curve Comparison (seq6, +6h step)"
    ax.set_title(title)
    ax.set_xlabel("target_time")
    ax.set_ylabel("PM2.5")
    ax.legend(ncol=5, fontsize=8)
    save(fig, filename)


def overall_metrics_bar() -> None:
    df = safe_read("overall_metrics_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No overall_metrics_summary.csv", ha="center", va="center")
            ax.axis("off")
        save(fig, "03_overall_metrics_bar.png")
        return

    df = df[df["task"].isin(["h1", "seq6_mean"])].dropna(subset=["rmse", "mae", "mape"]).copy()
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No valid metrics", ha="center", va="center")
            ax.axis("off")
        save(fig, "03_overall_metrics_bar.png")
        return

    df["label"] = df["model"] + "-" + df["task"]
    for ax, metric in zip(axes, ["rmse", "mae", "mape"]):
        plot_df = df.sort_values(metric)
        ax.bar(plot_df["label"], plot_df[metric])
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=45)
    save(fig, "03_overall_metrics_bar.png")


def feature_importance(task: str, filename: str) -> None:
    df = safe_read(f"xgboost_feature_importance_{task}.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    if df.empty:
        ax.text(0.5, 0.5, f"No xgboost_feature_importance_{task}.csv", ha="center", va="center")
        ax.axis("off")
        save(fig, filename)
        return

    top = df.head(20).iloc[::-1]
    values = top["gain_norm"] if "gain_norm" in top.columns else top["gain"]
    ax.barh(top["feature"], values)
    ax.set_title(f"XGBoost Feature Importance ({task})")
    save(fig, filename)


def aqi_bucket_heatmap() -> None:
    df = safe_read("aqi_bucket_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No aqi_bucket_metrics.csv", ha="center", va="center")
            ax.axis("off")
        save(fig, "06_aqi_bucket_error_heatmap.png")
        return

    panels = [("h1", 1, "h1"), ("seq6", 6, "seq6 +6h step")]
    for ax, (task, horizon, title) in zip(axes, panels):
        sub = df[(df["task"] == task) & (df["horizon_hours"] == horizon)].copy()
        pivot = sub.pivot(index="model", columns="aqi_bucket", values="rmse")
        if pivot.empty:
            ax.text(0.5, 0.5, f"No {title} data", ha="center", va="center")
            ax.axis("off")
            continue
        image = ax.imshow(pivot.values, aspect="auto")
        ax.set_title(f"RMSE by AQI bucket ({title})")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save(fig, "06_aqi_bucket_error_heatmap.png")


def feature_ablation() -> None:
    df = safe_read("feature_ablation.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "No feature_ablation.csv", ha="center", va="center")
        ax.axis("off")
        save(fig, "07_feature_ablation_comparison.png")
        return

    order = ["base_short_lag", "base_plus_daily_cycle", "mid_range_memory", "trend_and_roll", "weather_enhanced", "full_48"]
    h1_df = df[(df["task"] == "h1") & (df["horizon_hours"] == 1)].copy()
    seq6_df = df[df["task"] == "seq6"].groupby("feature_group", as_index=False)["rmse"].mean()
    seq6_df["task"] = "seq6_mean"
    merged = [("h1", h1_df), ("seq6_mean", seq6_df)]
    for label, sub in merged:
        sub["feature_group"] = pd.Categorical(sub["feature_group"], categories=order, ordered=True)
        sub = sub.sort_values("feature_group")
        ax.plot(sub["feature_group"].astype(str), sub["rmse"], marker="o", label=label)
    ax.set_title("XGBoost Feature Ablation (RMSE)")
    ax.set_ylabel("RMSE")
    ax.legend()
    save(fig, "07_feature_ablation_comparison.png")


def high_pollution() -> None:
    df = safe_read("high_pollution_error_analysis.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "No high_pollution_error_analysis.csv", ha="center", va="center")
        ax.axis("off")
        save(fig, "08_high_pollution_error.png")
        return

    df["label"] = df["model"] + "-" + df["task"] + "-h" + df["horizon_hours"].astype(int).astype(str)
    plot_df = df.sort_values("rmse")
    ax.bar(plot_df["label"], plot_df["rmse"])
    ax.set_title("High-pollution RMSE (y_true > 150)")
    ax.tick_params(axis="x", rotation=45)
    save(fig, "08_high_pollution_error.png")


def error_by_hour() -> None:
    df = safe_read("hourly_error_by_hour.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No hourly_error_by_hour.csv", ha="center", va="center")
            ax.axis("off")
        save(fig, "09_error_by_hour_curve.png")
        return

    panels = [("h1", 1, "h1"), ("seq6", 6, "seq6 +6h step")]
    for ax, (task, horizon, title) in zip(axes, panels):
        sub = df[(df["task"] == task) & (df["horizon_hours"] == horizon)].copy()
        for model in sorted(sub["model"].unique()):
            model_df = sub[sub["model"] == model].sort_values("hour")
            ax.plot(model_df["hour"], model_df["mae"], label=model)
        ax.set_title(f"MAE by hour ({title})")
        ax.set_xlabel("Hour")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("MAE")
    axes[1].legend(fontsize=8)
    save(fig, "09_error_by_hour_curve.png")


def dayperiod_error() -> None:
    df = safe_read("hourly_error_by_dayperiod.csv")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "No hourly_error_by_dayperiod.csv", ha="center", va="center")
        ax.axis("off")
        save(fig, "10_dayperiod_error_bar.png")
        return

    subset = df[((df["task"] == "h1") & (df["horizon_hours"] == 1)) | ((df["task"] == "seq6") & (df["horizon_hours"] == 6))].copy()
    subset["label"] = subset["model"] + "-" + subset["task"] + "-h" + subset["horizon_hours"].astype(int).astype(str)
    pivot = subset.pivot_table(index="dayperiod", columns="label", values="mae")
    order = ["overnight", "morning_peak", "daytime", "evening_peak", "night"]
    pivot = pivot.reindex(order)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("MAE by day period")
    ax.set_ylabel("MAE")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "10_dayperiod_error_bar.png")


def cross_city() -> None:
    df = safe_read("cross_city_metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if df.empty:
        ax.text(0.5, 0.5, "No cross_city_metrics.csv", ha="center", va="center")
        ax.axis("off")
        save(fig, "11_cross_city_generalization.png")
        return

    df["label"] = df["task"] + "-h" + df["horizon_hours"].astype(int).astype(str)
    ax.bar(df["label"], df["rmse"])
    ax.set_title("Cross-city generalization (Beijing -> Shanghai)")
    ax.set_ylabel("RMSE")
    save(fig, "11_cross_city_generalization.png")


def main() -> None:
    prediction_curve("h1", "01_prediction_curve_h1.png", horizon_filter=1)
    prediction_curve("seq6", "02_prediction_curve_seq6.png", horizon_filter=6)
    overall_metrics_bar()
    feature_importance("h1", "04_xgboost_feature_importance_h1.png")
    feature_importance("seq6", "05_xgboost_feature_importance_seq6.png")
    aqi_bucket_heatmap()
    feature_ablation()
    high_pollution()
    error_by_hour()
    dayperiod_error()
    cross_city()
    print(f"Generated figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
