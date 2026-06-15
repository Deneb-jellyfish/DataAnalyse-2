"""Generate SHAP beeswarm summary plots for XGBoost h1 and seq6 models.

Run with:   <traffic_env_python> scripts/generate_shap_figures.py

Root cause of previous bug: aligned_hourly/beijing.csv only covers 2013-2015,
but test predictions span 2016-2017 → zero merge overlap → all-NaN features
→ constant SHAP values → beeswarm collapses to vertical lines.

Fix: use data/processed/beijing/beijing_hourly_city.csv (2013-2017).
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

ROOT    = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT     = ROOT / "outputs" / "hourly"
FIG_DIR = OUT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Correct data source: covers full 2013-03-01 → 2017-02-28
DATA_PATH = ROOT / "data" / "processed" / "beijing" / "beijing_hourly_city.csv"

FEAT_COLS = [
    "pm25_lag1","pm25_lag2","pm25_lag3","pm25_lag4","pm25_lag6","pm25_lag8","pm25_lag12","pm25_lag24",
    "pm25_roll_mean_3","pm25_roll_mean_6","pm25_roll_mean_12","pm25_roll_mean_24",
    "pm25_roll_std_3","pm25_roll_std_6","pm25_roll_std_12","pm25_roll_std_24",
    "pm25_diff_1","pm25_diff_2","pm25_diff_3","pm25_diff_6",
    "hour_sin","hour_cos","month_sin","month_cos","weekday","is_weekend","is_holiday",
    "is_daytime","is_rush_hour",
    "temp","pres","dewp","humidity","wind_speed","precipitation",
    "wind_dir_sin","wind_dir_cos","precipitation_flag","dewp_temp_gap",
    "temp_diff_1","temp_diff_2","pres_diff_1","pres_diff_2","wind_speed_diff_1","wind_speed_diff_2",
    "temp_x_humidity","wind_speed_x_wind_dir_sin","wind_speed_x_pm25_lag1",
]


# ── Feature engineering ───────────────────────────────────────────────────────

def build_feature_frame() -> pd.DataFrame:
    """Load beijing_hourly_city.csv and compute all model features."""
    raw = pd.read_csv(str(DATA_PATH))
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw = raw.rename(columns={
        "PM2.5": "pm25", "TEMP": "temp", "PRES": "pres",
        "DEWP": "dewp", "RAIN": "precipitation", "WSPM": "wind_speed",
    })
    # humidity and wd not present in this file; wind_dir_* features have
    # near-zero importance in the trained model so neutral fill is acceptable
    raw["humidity"] = 60.0
    raw["wind_dir"] = 0.0  # neutral → sin=0, cos=1

    f = raw.sort_values("datetime").reset_index(drop=True)
    for col in ["pm25","temp","pres","dewp","humidity","wind_speed","precipitation"]:
        f[col] = pd.to_numeric(f[col], errors="coerce")

    # Lag features
    for lag in [1, 2, 3, 4, 6, 8, 12, 24]:
        f[f"pm25_lag{lag}"] = f["pm25"].shift(lag)
    # Rolling features (shift(1) prevents leakage)
    for w in [3, 6, 12, 24]:
        f[f"pm25_roll_mean_{w}"] = f["pm25"].shift(1).rolling(w).mean()
        f[f"pm25_roll_std_{w}"]  = f["pm25"].shift(1).rolling(w).std()
    # Diff features
    for d in [1, 2, 3, 6]:
        f[f"pm25_diff_{d}"] = f["pm25"] - f[f"pm25_lag{d}"]

    # Temporal
    hour = f["datetime"].dt.hour
    f["hour_sin"]    = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"]    = np.cos(2 * np.pi * hour / 24)
    f["month_sin"]   = np.sin(2 * np.pi * f["datetime"].dt.month / 12)
    f["month_cos"]   = np.cos(2 * np.pi * f["datetime"].dt.month / 12)
    f["weekday"]     = f["datetime"].dt.weekday
    f["is_weekend"]  = f["weekday"].isin([5, 6]).astype(int)
    f["is_holiday"]  = 0
    f["is_daytime"]  = hour.between(6, 17).astype(int)
    f["is_rush_hour"]= hour.isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Wind direction (neutral since feature has 0 importance)
    wd = pd.to_numeric(f["wind_dir"], errors="coerce").fillna(0.0)
    f["wind_dir_sin"] = np.sin(np.deg2rad(wd))
    f["wind_dir_cos"] = np.cos(np.deg2rad(wd))

    # Weather derived
    f["precipitation_flag"] = (f["precipitation"].fillna(0) > 0).astype(int)
    f["dewp_temp_gap"]      = f["temp"] - f["dewp"]
    f["temp_diff_1"]        = f["temp"]        - f["temp"].shift(1)
    f["temp_diff_2"]        = f["temp"]        - f["temp"].shift(2)
    f["pres_diff_1"]        = f["pres"]        - f["pres"].shift(1)
    f["pres_diff_2"]        = f["pres"]        - f["pres"].shift(2)
    f["wind_speed_diff_1"]  = f["wind_speed"]  - f["wind_speed"].shift(1)
    f["wind_speed_diff_2"]  = f["wind_speed"]  - f["wind_speed"].shift(2)
    f["temp_x_humidity"]           = f["temp"].fillna(0) * f["humidity"].fillna(0)
    f["wind_speed_x_wind_dir_sin"] = f["wind_speed"].fillna(0) * f["wind_dir_sin"]
    f["wind_speed_x_pm25_lag1"]    = f["wind_speed"].fillna(0) * f["pm25_lag1"].fillna(0)

    return f[["datetime"] + FEAT_COLS]


def build_X_test(pred_csv: str, full_frame: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_csv(pred_csv)
    pred = pred[pred["split"] == "test"].copy()
    pred["forecast_origin_time"] = pd.to_datetime(pred["forecast_origin_time"])

    merged = pred[["forecast_origin_time"]].merge(
        full_frame.rename(columns={"datetime": "forecast_origin_time"}),
        on="forecast_origin_time", how="left",
    )
    X = merged[FEAT_COLS].reset_index(drop=True)
    med = X.median(numeric_only=True)
    X   = X.fillna(med)
    return X


# ── Main pipeline ─────────────────────────────────────────────────────────────

def make_shap_fig(
    model_path: str,
    pred_csv:   str,
    fig_path:   str,
    fig_label:  str,
    full_frame: pd.DataFrame,
    n_sample:   int = 500,
    n_top:      int = 20,
) -> None:
    import shap

    print(f"\n── {fig_label} ──")

    with open(model_path, "rb") as fh:
        obj = pickle.load(fh)
    model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    X = build_X_test(pred_csv, full_frame)
    print(f"  X shape: {X.shape}  non-null rows: {X.dropna().shape[0]}")

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(n_sample, len(X)), replace=False)
    X_s = X.iloc[idx].reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    sv        = explainer.shap_values(X_s.values)
    print(f"  shap_values shape: {sv.shape}  range [{sv.min():.2f}, {sv.max():.2f}]")

    # Rank features by mean |SHAP|, keep top-n (most important first)
    mean_abs  = np.abs(sv).mean(axis=0)
    top_idx   = np.argsort(mean_abs)[::-1][:n_top]
    top_names = [X_s.columns[i] for i in top_idx]   # index 0 = most important
    mean_top  = mean_abs[top_idx]                    # index 0 = most important
    sv_top    = sv[:, top_idx]
    X_top     = X_s.iloc[:, top_idx]

    # ── 1. render beeswarm via shap.summary_plot (viridis colormap) ──────────
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    })
    fig, ax0 = plt.subplots(figsize=(9.5, 7.5))
    plt.sca(ax0)

    shap.summary_plot(
        sv_top, X_top,
        cmap=plt.get_cmap("viridis"),
        show=False,
        max_display=n_top,
        plot_size=None,
        color_bar=True,
    )

    # ax0 is always fig.axes[0]; colorbar (if added) is fig.axes[1]
    ax = fig.axes[0]

    # ── 2. y-axis labels: "feature_name  (mean_abs_value)" ───────────────────
    # shap places features at integer y-positions 0…n_top-1, then calls
    # invert_yaxis → position 0 (bottom) = least important = top_names[n_top-1]
    #               position n_top-1 (top) = most important = top_names[0]
    n = n_top
    new_labels = [
        f"{top_names[n - 1 - j]}  ({mean_top[n - 1 - j]:.4f})"
        for j in range(n)
    ]
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(new_labels, fontsize=8.0)

    # ── 3. background bars on twin top x-axis (Mean|SHAP| scale) ─────────────
    ax_bar = ax.twiny()
    # bar_widths[j] = Mean|SHAP| for feature at y-position j
    bar_widths = mean_top[::-1]   # reversed: least→most important for y=0→n-1
    y_pos = np.arange(n)

    # Color each bar with rank-based viridis tint (avoids compression from outliers)
    # j=0 is least important (bottom, purple), j=n-1 is most important (top, yellow)
    cmap_v = plt.get_cmap("viridis")
    bar_colors = []
    for j in range(n):
        rank_norm = j / max(n - 1, 1)   # 0 (least important) → 1 (most important)
        rgba = np.array(cmap_v(rank_norm))
        # blend ~38% viridis + 62% white → soft pastel
        light = 0.38 * rgba[:3] + 0.62 * np.ones(3)
        bar_colors.append((*light, 0.90))

    ax_bar.barh(y_pos, bar_widths, color=bar_colors,
                height=0.72, left=0, zorder=0)
    ax_bar.set_xlim(0, mean_top.max() * 1.2)
    ax_bar.set_xlabel("Mean(|SHAP value|)", fontsize=9, color="#444444", labelpad=5)
    ax_bar.tick_params(axis="x", labelsize=8, colors="#555555", length=3)
    for sp in ["right", "left", "bottom"]:
        ax_bar.spines[sp].set_visible(False)
    ax_bar.spines["top"].set_color("#BBBBBB")
    ax_bar.spines["top"].set_linewidth(0.6)

    # put scatter (ax) visually in front of bars (ax_bar)
    ax.set_zorder(ax_bar.get_zorder() + 1)
    ax.patch.set_alpha(0.0)

    # ── 4. styling ────────────────────────────────────────────────────────────
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="-",
               alpha=0.6, zorder=5)
    ax.grid(axis="x", linestyle=":", alpha=0.25, color="gray", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("SHAP value  (impact on model output)", fontsize=10, labelpad=6)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8.0, length=0)

    fig.patch.set_facecolor("white")
    fig.suptitle(fig_label, fontsize=11, fontweight="bold",
                 x=0.02, ha="left", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  saved  {Path(fig_path).name}")


def main() -> None:
    print("Building feature frame from", DATA_PATH)
    full_frame = build_feature_frame()
    print(f"Feature frame: {full_frame.shape}  "
          f"{full_frame['datetime'].min()} → {full_frame['datetime'].max()}")

    make_shap_fig(
        model_path=str(OUT / "xgboost_h1.pkl"),
        pred_csv  =str(OUT / "xgboost_predictions_h1.csv"),
        fig_path  =str(FIG_DIR / "fig10_shap_h1.png"),
        fig_label ="(a)  SHAP feature importance — XGBoost h1",
        full_frame=full_frame,
    )

    make_shap_fig(
        model_path=str(OUT / "xgboost_seq6_h6.pkl"),
        pred_csv  =str(OUT / "xgboost_predictions_seq6.csv"),
        fig_path  =str(FIG_DIR / "fig11_shap_seq6.png"),
        fig_label ="(b)  SHAP feature importance — XGBoost seq6 (h6 model)",
        full_frame=full_frame,
    )

    print("\nAll SHAP figures saved.")


if __name__ == "__main__":
    main()
