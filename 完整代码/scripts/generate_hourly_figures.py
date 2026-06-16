"""Generate publication-quality figures for the hourly h1 + seq6 experiment.

Each sub-panel is saved as an independent file with white background and English labels.
Figure naming: fig01_… through fig23_…
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hourly_output_paths import resolve_artifact_path

OUT = ROOT / "outputs" / "hourly"
FIG_DIR = OUT / "figures"

# ── Color palette (Wong, colorblind-safe) — used for line/scatter plots ───────
MODEL_COLORS = {
    "XGBoost":     "#0072B2",
    "Transformer": "#56B4E9",
    "LSTM":        "#009E73",
    "ARIMA":       "#D55E00",
    "Prophet":     "#CC79A7",
}

# ── Bar-chart palette (Morandi / muted academic) ──────────────────────────────
BAR_COLORS = {
    "LSTM":        "#9999D4",
    "Transformer": "#F0CFA0",
    "ARIMA":       "#F0A8A0",
    "XGBoost":     "#A8C4E0",
    "Prophet":     "#A8D4C8",
}

ABLATION_LABELS = {
    "base_short_lag":        "Base (short lag)",
    "base_plus_daily_cycle": "Base + Daily Cycle",
    "mid_range_memory":      "Mid-Range Memory",
    "trend_and_roll":        "Trend & Rolling",
    "weather_enhanced":      "Weather Enhanced",
    "full_48":               "Full (48 h)",
}

DAYPERIOD_LABELS = {
    "overnight":    "Overnight",
    "morning_peak": "Morning Peak",
    "daytime":      "Daytime",
    "evening_peak": "Evening Peak",
    "night":        "Night",
}

HOUR_SPANS = [
    (0,  5,  "#EEF3FA", "Overnight"),
    (6,  9,  "#EEF7EE", "Morning\nPeak"),
    (10, 16, "#FDFAE8", "Daytime"),
    (17, 20, "#FFF0EE", "Evening\nPeak"),
    (21, 23, "#F4EEF8", "Night"),
]


# ── Theme ─────────────────────────────────────────────────────────────────────

def apply_theme() -> None:
    mpl.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.titlesize":    10.5,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "legend.frameon":    True,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#CCCCCC",
        "axes.linewidth":    0.8,
        "axes.edgecolor":    "#555555",
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
        "axes.grid":         True,
        "grid.color":        "#E5E5E5",
        "grid.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
    })


def save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  saved  {name}")


def safe_read(name: str) -> pd.DataFrame:
    path = resolve_artifact_path(name)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    mape  = float(np.nanmean(np.abs(err) / denom) * 100)
    return {"rmse": rmse, "mae": mae, "mape": mape}


def load_prediction_bundle(task: str, horizon_filter: int):
    model_files = {
        "ARIMA":       "arima_predictions_seq6.csv"       if task == "seq6" else "arima_predictions_h1.csv",
        "Prophet":     "prophet_predictions_seq6.csv"     if task == "seq6" else "prophet_predictions_h1.csv",
        "XGBoost":     "xgboost_predictions_seq6.csv"    if task == "seq6" else "xgboost_predictions_h1.csv",
        "LSTM":        "lstm_predictions_seq6.csv"        if task == "seq6" else "lstm_predictions_h1.csv",
        "Transformer": "transformer_predictions_seq6.csv" if task == "seq6" else "transformer_predictions_h1.csv",
    }
    merged = None
    raw_map: dict[str, pd.DataFrame] = {}
    metrics_rows: list[dict] = []

    for model, fname in model_files.items():
        df = safe_read(fname)
        if df.empty:
            continue
        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        df = df[df["horizon_hours"] == horizon_filter].copy()
        if df.empty:
            continue
        df["target_time"] = pd.to_datetime(df["target_time"])
        df = df.sort_values("target_time").reset_index(drop=True)
        raw_map[model] = df
        m = metric_summary(df["y_true"].to_numpy(float), df["y_pred"].to_numpy(float))
        metrics_rows.append({"model": model, **m})
        compact = df[["target_time", "y_true", "y_pred"]].rename(
            columns={"y_pred": f"y_pred_{model}"})
        merged = (compact if merged is None
                  else merged.merge(compact[["target_time", f"y_pred_{model}"]],
                                    on="target_time", how="inner"))

    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse").reset_index(drop=True)
    return (merged if merged is not None else pd.DataFrame()), raw_map, metrics_df


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
        subset=["model", "task", "horizon_hours"], keep="first")


def darken(hex_color: str, factor: float = 0.72) -> str:
    """Return a darkened version of a hex color by multiplying RGB by factor."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02X}{:02X}{:02X}".format(
        int(r * factor), int(g * factor), int(b * factor))


def hbar(ax: plt.Axes, values: pd.Series, labels: pd.Series,
         color_map: dict[str, str], unit: str = "") -> None:
    y      = np.arange(len(labels))
    colors = [color_map.get(lbl, "#BBBBCC") for lbl in labels]
    edges  = [darken(c) for c in colors]
    for i, (v, c, ec) in enumerate(zip(values, colors, edges)):
        ax.barh(i, v, color=c, alpha=1.0, height=0.55, edgecolor=ec, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    xmax = float(values.max()) if len(values) else 1.0
    for i, v in enumerate(values):
        ax.text(v + xmax * 0.02, i, f"{v:.2f}{unit}", va="center", fontsize=9)


# ── Fig 01: h1 prediction time series ────────────────────────────────────────

def fig01_h1_prediction_curve() -> None:
    _colors = {
        "LSTM":        "#1B7F79",
        "Transformer": "#2E5C8A",
        "ARIMA":       "#C76B2C",
        "XGBoost":     "#1F6FA8",
        "Prophet":     "#A8407C",
    }
    model_files = {
        "ARIMA":       "arima_predictions_h1.csv",
        "Prophet":     "prophet_predictions_h1.csv",
        "XGBoost":     "xgboost_predictions_h1.csv",
        "LSTM":        "lstm_predictions_h1.csv",
        "Transformer": "transformer_predictions_h1.csv",
    }

    # Use XGBoost (densest hourly data) to find best contiguous 48h window
    xdf = safe_read("xgboost_predictions_h1.csv")
    if xdf.empty:
        fig, ax = plt.subplots(figsize=(3.54, 2.8))
        ax.text(0.5, 0.5, "No h1 prediction data available", ha="center", va="center",
                transform=ax.transAxes)
        save(fig, "fig01_h1_prediction_curve.png")
        return
    if "split" in xdf.columns:
        xdf = xdf[xdf["split"] == "test"].copy()
    xdf = xdf[xdf["horizon_hours"] == 1].sort_values("target_time").reset_index(drop=True)
    xdf["target_time"] = pd.to_datetime(xdf["target_time"])

    # Find highest-variability 48-row window that is truly contiguous (no gaps)
    roll = xdf["y_true"].rolling(48).std()
    t_start, t_end = None, None
    for ei in roll.nlargest(50).index:
        si = max(0, ei - 47)
        span_h = (xdf.loc[ei, "target_time"] - xdf.loc[si, "target_time"]).total_seconds() / 3600
        if abs(span_h - 47) < 0.1:
            t_start = xdf.loc[si, "target_time"]
            t_end   = xdf.loc[ei, "target_time"]
            break
    if t_start is None:
        t_end   = xdf["target_time"].iloc[-1]
        t_start = t_end - pd.Timedelta(hours=47)

    fig, ax = plt.subplots(figsize=(3.54, 2.8))

    # Plot each model independently in the time window
    y_true_ref = None
    for model in ["XGBoost", "Transformer", "LSTM", "ARIMA", "Prophet"]:
        df = safe_read(model_files[model])
        if df.empty:
            continue
        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        df = df[df["horizon_hours"] == 1].copy()
        df["target_time"] = pd.to_datetime(df["target_time"])
        w = df[(df["target_time"] >= t_start) & (df["target_time"] <= t_end)].sort_values("target_time")
        if w.empty:
            continue
        if y_true_ref is None:
            y_true_ref = w
        ax.plot(w["target_time"], w["y_pred"], linewidth=1.1,
                color=_colors[model], alpha=0.8, label=model, zorder=3)

    if y_true_ref is not None:
        ax.plot(y_true_ref["target_time"], y_true_ref["y_true"],
                color="black", linewidth=1.5, label="Observed", zorder=5)

    ax.grid(color="gray", alpha=0.3, linestyle="--", linewidth=0.5, zorder=0)
    ax.set_ylabel(r"PM$_{2.5}$ (μg m$^{-3}$)", fontsize=8)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:00"))
    ax.tick_params(axis="x", labelsize=6.5, rotation=0)
    ax.tick_params(axis="y", labelsize=8)

    # Legend: models first then Observed — reorder so Observed is first
    handles, labels = ax.get_legend_handles_labels()
    obs_idx = labels.index("Observed") if "Observed" in labels else -1
    if obs_idx > 0:
        handles = [handles[obs_idx]] + [h for i, h in enumerate(handles) if i != obs_idx]
        labels  = [labels[obs_idx]]  + [l for i, l in enumerate(labels)  if i != obs_idx]
    ax.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.38),
              fontsize=7, handlelength=1.4, handletextpad=0.5,
              columnspacing=0.8, frameon=True, framealpha=0.9, edgecolor="#cccccc")

    ax.set_title("(a)  One-step-ahead prediction vs. observed",
                 loc="left", fontweight="bold", pad=6, fontsize=8.5)
    save(fig, "fig01_h1_prediction_curve.png")


# ── Fig 02: h1 model RMSE bar chart ──────────────────────────────────────────

def fig02_h1_model_rmse() -> None:
    _, _, metrics_df = load_prediction_bundle("h1", horizon_filter=1)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    if metrics_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, "fig02_h1_model_rmse.png")
        return

    hbar(ax, metrics_df["rmse"], metrics_df["model"], BAR_COLORS)
    ax.set_xlabel(r"RMSE (μg m$^{-3}$)")
    ax.set_title("(b)  Model comparison — h1 RMSE",
                 loc="left", fontweight="bold", pad=8)
    fig.tight_layout()
    save(fig, "fig02_h1_model_rmse.png")


# ── Fig 03: h1 residual KDE ───────────────────────────────────────────────────

def fig03_h1_residual_kde() -> None:
    _, raw_map, metrics_df = load_prediction_bundle("h1", horizon_filter=1)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    top_models = metrics_df["model"].tolist()
    all_resid: list[np.ndarray] = []
    for model in top_models:
        if model not in raw_map:
            continue
        resid = (raw_map[model]["y_pred"] - raw_map[model]["y_true"]).to_numpy(float)
        all_resid.append(resid)
        sns.kdeplot(x=resid, ax=ax, fill=True, alpha=0.15, linewidth=1.6,
                    color=MODEL_COLORS[model], label=model)

    ax.axvline(0, color="#333333", linewidth=0.9, linestyle="--", alpha=0.7)
    if all_resid:
        combined = np.concatenate(all_resid)
        lo, hi   = np.nanquantile(combined, [0.01, 0.99])
        margin   = max((hi - lo) * 0.15, 5.0)
        ax.set_xlim(lo - margin, hi + margin)

    ax.set_xlabel(r"Prediction residual ($\hat{y} - y$,  μg m$^{-3}$)")
    ax.set_ylabel("Density")
    ax.set_title("(c)  h1 residual distribution",
                 loc="left", fontweight="bold", pad=8)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig03_h1_residual_kde.png")


# ── Fig 04: h6 prediction time series ────────────────────────────────────────

def fig04_h6_prediction_curve() -> None:
    merged, _, metrics_df = load_prediction_bundle("seq6", horizon_filter=6)
    fig, ax = plt.subplots(figsize=(9, 4.2))

    if merged.empty:
        ax.text(0.5, 0.5, "No seq6 prediction data available", ha="center", va="center",
                transform=ax.transAxes, color="#888888")
        save(fig, "fig04_h6_prediction_curve.png")
        return

    top_models = metrics_df["model"].tolist()
    show = merged.sort_values("target_time").tail(min(168, len(merged)))

    ax.fill_between(show["target_time"], show["y_true"],
                    color="#BBDDEE", alpha=0.25, zorder=1)
    ax.plot(show["target_time"], show["y_true"],
            color="black", linewidth=1.8, label="Observed", zorder=5)
    for model in top_models:
        col = f"y_pred_{model}"
        if col in show.columns:
            ax.plot(show["target_time"], show[col], linewidth=1.4,
                    color=MODEL_COLORS[model], alpha=0.92, label=model)

    ax.set_ylabel(r"PM$_{2.5}$ (μg m$^{-3}$)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", labelsize=8.5, rotation=0)
    ax.legend(ncol=3, loc="upper left", handlelength=1.5)
    ax.set_title("(a)  6-step-ahead (h6) prediction vs. observed",
                 loc="left", fontweight="bold", pad=8)
    fig.tight_layout()
    save(fig, "fig04_h6_prediction_curve.png")


# ── Fig 05: RMSE vs forecast horizon ─────────────────────────────────────────

def fig05_rmse_vs_horizon() -> None:
    overall = safe_read("overall_metrics_summary.csv")
    fig, ax  = plt.subplots(figsize=(5.5, 4.0))

    if overall.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, "fig05_rmse_vs_horizon.png")
        return

    seq6_steps = dedupe_metrics(overall, "seq6")
    for model in ["XGBoost", "Transformer", "LSTM", "ARIMA", "Prophet"]:
        sub = seq6_steps[seq6_steps["model"] == model].sort_values("horizon_hours")
        if sub.empty:
            continue
        ax.plot(sub["horizon_hours"], sub["rmse"],
                marker="o", markersize=5, linewidth=1.8,
                color=MODEL_COLORS[model], label=model)

    ax.set_xticks(range(1, 7))
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel(r"RMSE (μg m$^{-3}$)")
    ax.set_title("(b)  RMSE vs. forecast horizon (h1–h6)",
                 loc="left", fontweight="bold", pad=8)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig05_rmse_vs_horizon.png")


# ── Fig 06: seq6 average RMSE bar chart ──────────────────────────────────────

def fig06_seq6_avg_rmse() -> None:
    overall = safe_read("overall_metrics_summary.csv")
    fig, ax  = plt.subplots(figsize=(5.5, 3.8))

    if overall.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, "fig06_seq6_avg_rmse.png")
        return

    seq6_mean = dedupe_metrics(overall, "seq6_mean")
    if seq6_mean.empty:
        ax.text(0.5, 0.5, "No seq6_mean data", ha="center", va="center", transform=ax.transAxes)
        save(fig, "fig06_seq6_avg_rmse.png")
        return

    seq6_mean = seq6_mean.sort_values("rmse").reset_index(drop=True)
    hbar(ax, seq6_mean["rmse"], seq6_mean["model"], BAR_COLORS)
    ax.set_xlabel(r"Mean RMSE (μg m$^{-3}$)")
    ax.set_title("(c)  seq6 average model ranking",
                 loc="left", fontweight="bold", pad=8)
    fig.tight_layout()
    save(fig, "fig06_seq6_avg_rmse.png")


# ── Fig 07–09: overall metric comparisons ────────────────────────────────────

def fig_overall_metric(metric: str, panel_label: str, fig_num: int) -> None:
    df = safe_read("overall_metrics_summary.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, f"fig{fig_num:02d}_overall_{metric}.png")
        return

    df = (df[df["task"].isin(["h1", "seq6_mean"])]
          .dropna(subset=["rmse", "mae", "mape"])
          .copy())
    df = df.sort_values(["task", metric]).drop_duplicates(
        subset=["model", "task"], keep="first")
    df["label"] = (df["model"] + "  ("
                   + df["task"].map({"h1": "h1", "seq6_mean": "seq6 mean"}) + ")")

    plot_df = df.sort_values(metric).reset_index(drop=True)
    y       = np.arange(len(plot_df))
    colors  = [BAR_COLORS.get(m, "#BBBBCC") for m in plot_df["model"]]
    # h1 bars: full color; seq6 bars: slightly lighter (mix with white)
    alphas  = [1.0 if t == "h1" else 0.65 for t in plot_df["task"]]

    for i, (v, c, a) in enumerate(zip(plot_df[metric], colors, alphas)):
        ax.barh(i, v, color=c, alpha=a, height=0.55,
                edgecolor=darken(c), linewidth=0.8)
        ax.text(v + plot_df[metric].max() * 0.02, i,
                f"{v:.2f}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    unit = " (%)" if metric == "mape" else r" (μg m$^{-3}$)"
    ax.set_xlabel(metric.upper() + unit)
    ax.set_title(f"({panel_label})  {metric.upper()} comparison — h1 vs. seq6 mean",
                 loc="left", fontweight="bold", pad=8)
    fig.tight_layout()
    save(fig, f"fig{fig_num:02d}_overall_{metric}.png")


# ── Fig 10–11: XGBoost feature importance ────────────────────────────────────

def fig_feature_importance(task: str, panel_label: str, fig_num: int) -> None:
    df = safe_read(f"xgboost_feature_importance_{task}.csv")
    fig, ax = plt.subplots(figsize=(7, 5.5))

    if df.empty:
        ax.text(0.5, 0.5, "No feature importance data", ha="center", va="center",
                transform=ax.transAxes)
        save(fig, f"fig{fig_num:02d}_feature_importance_{task}.png")
        return

    top    = df.head(14).iloc[::-1].copy()
    values = (top["gain_norm"] if "gain_norm" in top.columns else top["gain"]).to_numpy(float)
    labels = top["feature"].tolist()
    y      = np.arange(len(top))

    gradient = plt.cm.Blues(np.linspace(0.30, 0.80, len(top)))
    ax.barh(y, values, color=gradient, edgecolor="none", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Normalized gain (relative importance)")
    ax.set_title(f"({panel_label})  XGBoost feature importance — {task}",
                 loc="left", fontweight="bold", pad=8)
    xmax = float(values.max())
    for i, v in enumerate(values):
        ax.text(v + xmax * 0.025, i, f"{v * 100:.1f}%", va="center", fontsize=8.5)
    ax.grid(True, axis="x", alpha=0.5)
    ax.grid(False, axis="y")
    fig.tight_layout()
    save(fig, f"fig{fig_num:02d}_feature_importance_{task}.png")


# ── Heatmap helpers ───────────────────────────────────────────────────────────

_HEATMAP_BG   = "white"
# Muted blue → white → muted rose  (matches reference style)
_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "muted_bwr", ["#6BA3C8", "#FFFFFF", "#C47878"], N=256
)


def _fix_heatmap_text_contrast(ax: plt.Axes) -> None:
    """Switch annotation text between white / dark-gray based on cell luminance."""
    coll = ax.collections[0]
    fc   = coll.get_facecolors()
    ncol = int(round(ax.get_xlim()[1]))
    for text in ax.texts:
        x, y = text.get_position()
        idx  = int(y) * ncol + int(x)
        if 0 <= idx < len(fc):
            r, g, b, _ = fc[idx]
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            text.set_color("white" if lum < 0.50 else "#444444")


def _style_heatmap(fig: plt.Figure, ax: plt.Axes) -> None:
    """Apply shared styling to a seaborn heatmap axes."""
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Remove colorbar outline
    try:
        cbar = ax.collections[0].colorbar
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(size=0)
    except Exception:
        pass
    _fix_heatmap_text_contrast(ax)


# ── Fig 12–13: AQI bucket heatmaps ───────────────────────────────────────────

def fig_aqi_heatmap(task: str, horizon: int, panel_label: str, fig_num: int) -> None:
    df = safe_read("aqi_bucket_metrics.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor(_HEATMAP_BG)
    ax.set_facecolor(_HEATMAP_BG)

    if df.empty:
        ax.text(0.5, 0.5, "No AQI data", ha="center", va="center", transform=ax.transAxes)
        save(fig, f"fig{fig_num:02d}_aqi_heatmap_{task}.png")
        return

    sub   = df[(df["task"] == task) & (df["horizon_hours"] == horizon)].copy()
    pivot = sub.pivot(index="model", columns="aqi_bucket", values="rmse")

    if pivot.empty:
        ax.text(0.5, 0.5, "No data for this task/horizon",
                ha="center", va="center", transform=ax.transAxes)
        save(fig, f"fig{fig_num:02d}_aqi_heatmap_{task}.png")
        return

    sns.heatmap(pivot, annot=True, fmt=".1f", cmap=_HEATMAP_CMAP,
                linewidths=1.0, linecolor="white",
                cbar_kws={"label": r"RMSE (μg m$^{-3}$)", "shrink": 0.85},
                ax=ax, annot_kws={"fontsize": 10.5})
    _style_heatmap(fig, ax)
    ax.set_xlabel(r"PM$_{2.5}$ concentration bucket (μg m$^{-3}$)")
    ax.set_ylabel("")
    task_name = "h1 (one-step)" if task == "h1" else "h6 (6-step)"
    ax.set_title(f"({panel_label})  RMSE by AQI bucket — {task_name}",
                 loc="left", fontweight="bold", pad=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    save(fig, f"fig{fig_num:02d}_aqi_heatmap_{task}.png")


# ── Fig 14–15: Feature ablation ───────────────────────────────────────────────

def fig_feature_ablation(task: str, panel_label: str, fig_num: int) -> None:
    df = safe_read("feature_ablation.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))

    if df.empty:
        ax.text(0.5, 0.5, "No ablation data", ha="center", va="center",
                transform=ax.transAxes)
        save(fig, f"fig{fig_num:02d}_feature_ablation_{task}.png")
        return

    order = ["base_short_lag", "base_plus_daily_cycle", "mid_range_memory",
             "trend_and_roll", "weather_enhanced", "full_48"]

    if task == "h1":
        sub = df[(df["task"] == "h1") & (df["horizon_hours"] == 1)].copy()
    else:
        sub = (df[df["task"] == "seq6"]
               .groupby(["feature_group", "n_features"], as_index=False)[["rmse", "mae"]]
               .mean())

    sub["feature_group"] = pd.Categorical(
        sub["feature_group"], categories=order, ordered=True)
    sub = sub.sort_values("feature_group").dropna(subset=["rmse"])
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, f"fig{fig_num:02d}_feature_ablation_{task}.png")
        return

    x_labels = [ABLATION_LABELS.get(g, g) for g in sub["feature_group"].astype(str)]
    x_pos    = np.arange(len(sub))
    color    = MODEL_COLORS["XGBoost"] if task == "h1" else MODEL_COLORS["LSTM"]

    ax.plot(x_pos, sub["rmse"].to_numpy(float),
            color=color, linewidth=2.0, marker="o", markersize=6, zorder=3)
    ax.fill_between(x_pos, sub["rmse"].to_numpy(float), color=color, alpha=0.10)
    rmse_max = sub["rmse"].max()
    for i, (_, row) in enumerate(sub.iterrows()):
        ax.text(i, row["rmse"] + rmse_max * 0.018,
                f"{row['rmse']:.2f}", ha="center", fontsize=8.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=22, ha="right")
    ax.set_ylabel(r"RMSE (μg m$^{-3}$)")
    task_name = "h1 (one-step)" if task == "h1" else "seq6 (6-step mean)"
    ax.set_title(f"({panel_label})  Feature ablation study — {task_name}",
                 loc="left", fontweight="bold", pad=8)
    fig.tight_layout()
    save(fig, f"fig{fig_num:02d}_feature_ablation_{task}.png")


# ── Fig 16–17: High-pollution error ──────────────────────────────────────────

def fig_high_pollution(task: str, horizon: int, panel_label: str, fig_num: int) -> None:
    df = safe_read("high_pollution_error_analysis.csv")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, f"fig{fig_num:02d}_high_pollution_{task}.png")
        return

    sub = (df[(df["task"] == task) & (df["horizon_hours"] == horizon)]
           .sort_values("rmse").copy())
    if sub.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, f"fig{fig_num:02d}_high_pollution_{task}.png")
        return

    hbar(ax, sub["rmse"], sub["model"], BAR_COLORS)
    ax.set_xlabel(r"RMSE (μg m$^{-3}$)")
    task_name = "h1 (one-step)" if task == "h1" else "h6 (6-step)"
    ax.set_title(f"({panel_label})  High-pollution RMSE — {task_name}",
                 loc="left", fontweight="bold", pad=8)
    ax.text(0.98, 0.02, r"PM$_{2.5}$ > 150 μg m$^{-3}$",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#666666",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.85))
    fig.tight_layout()
    save(fig, f"fig{fig_num:02d}_high_pollution_{task}.png")


# ── Fig 18–19: Diurnal error pattern ─────────────────────────────────────────

def fig_error_by_hour(task: str, horizon: int, panel_label: str, fig_num: int) -> None:
    df = safe_read("hourly_error_by_hour.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, f"fig{fig_num:02d}_error_by_hour_{task}.png")
        return

    sub = df[(df["task"] == task) & (df["horizon_hours"] == horizon)].copy()

    for left, right, color, _ in HOUR_SPANS:
        ax.axvspan(left - 0.5, right + 0.5, color=color, alpha=0.55, zorder=0)

    for model in ["XGBoost", "Transformer", "LSTM", "ARIMA"]:
        model_df = sub[sub["model"] == model].sort_values("hour")
        if model_df.empty:
            continue
        y = model_df["mae"].rolling(3, center=True, min_periods=1).mean()
        ax.plot(model_df["hour"], y,
                color=MODEL_COLORS[model], linewidth=1.8, label=model)

    # Period labels just above plot area
    for left, right, _, label in HOUR_SPANS:
        mid = (left + right) / 2
        ax.text(mid, 1.02, label, ha="center", va="bottom",
                fontsize=7.5, color="#777777",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(r"MAE (μg m$^{-3}$)")
    task_name = "h1 (one-step)" if task == "h1" else "h6 (6-step)"
    ax.set_title(f"({panel_label})  Diurnal error pattern — {task_name}",
                 loc="left", fontweight="bold", pad=28)  # extra pad for period labels
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, f"fig{fig_num:02d}_error_by_hour_{task}.png")


# ── Fig 20–21: Time-period heatmaps ──────────────────────────────────────────

def fig_dayperiod_heatmap(task: str, horizon: int, panel_label: str, fig_num: int) -> None:
    df = safe_read("hourly_error_by_dayperiod.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor(_HEATMAP_BG)
    ax.set_facecolor(_HEATMAP_BG)

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, f"fig{fig_num:02d}_dayperiod_heatmap_{task}.png")
        return

    order = ["overnight", "morning_peak", "daytime", "evening_peak", "night"]
    sub   = df[(df["task"] == task) & (df["horizon_hours"] == horizon)].copy()
    pivot = sub.pivot(index="model", columns="dayperiod", values="mae")

    present = [c for c in order if c in pivot.columns]
    pivot   = pivot[present]
    pivot.columns = [DAYPERIOD_LABELS.get(c, c) for c in pivot.columns]

    if pivot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, f"fig{fig_num:02d}_dayperiod_heatmap_{task}.png")
        return

    sns.heatmap(pivot, annot=True, fmt=".1f", cmap=_HEATMAP_CMAP,
                linewidths=1.0, linecolor="white",
                cbar_kws={"label": r"MAE (μg m$^{-3}$)", "shrink": 0.85},
                ax=ax, annot_kws={"fontsize": 10.5})
    _style_heatmap(fig, ax)
    ax.set_xlabel("Time-of-day period")
    ax.set_ylabel("")
    task_name = "h1 (one-step)" if task == "h1" else "h6 (6-step)"
    ax.set_title(f"({panel_label})  MAE by time-of-day period — {task_name}",
                 loc="left", fontweight="bold", pad=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    save(fig, f"fig{fig_num:02d}_dayperiod_heatmap_{task}.png")


# ── Fig 22–23: Cross-city generalization ─────────────────────────────────────

def _build_compare() -> pd.DataFrame | None:
    cross_df = safe_read("cross_city_metrics.csv")
    overall  = safe_read("overall_metrics_summary.csv")
    if cross_df.empty or overall.empty:
        return None

    local_h1   = dedupe_metrics(overall, "h1")
    local_seq6 = dedupe_metrics(overall, "seq6")
    local_rows: list[dict] = []

    xgb_h1 = local_h1[local_h1["model"] == "XGBoost"]
    if not xgb_h1.empty:
        local_rows.append({"horizon_hours": 1, "rmse": float(xgb_h1.iloc[0]["rmse"])})
    for _, row in local_seq6[local_seq6["model"] == "XGBoost"].iterrows():
        local_rows.append({"horizon_hours": int(row["horizon_hours"]),
                           "rmse": float(row["rmse"])})

    local_df  = (pd.DataFrame(local_rows)
                 .drop_duplicates("horizon_hours")
                 .sort_values("horizon_hours"))
    cross_use = (cross_df[["horizon_hours", "rmse"]]
                 .drop_duplicates()
                 .sort_values("horizon_hours"))
    compare   = local_df.merge(cross_use, on="horizon_hours",
                               suffixes=("_local", "_cross"))
    compare["gap"] = compare["rmse_cross"] - compare["rmse_local"]
    return compare


def fig22_cross_city_rmse() -> None:
    compare = _build_compare()
    fig, ax = plt.subplots(figsize=(6, 4.2))

    if compare is None or compare.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, "fig22_cross_city_rmse.png")
        return

    ax.plot(compare["horizon_hours"], compare["rmse_local"],
            color=MODEL_COLORS["XGBoost"], marker="o", linewidth=2.0,
            label="Beijing (local test)")
    ax.plot(compare["horizon_hours"], compare["rmse_cross"],
            color=MODEL_COLORS["ARIMA"], marker="o", linewidth=2.0,
            label="Shanghai (cross-city test)")
    ax.fill_between(compare["horizon_hours"],
                    compare["rmse_local"], compare["rmse_cross"],
                    color=MODEL_COLORS["ARIMA"], alpha=0.10)
    ax.set_xticks(range(1, int(compare["horizon_hours"].max()) + 1))
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel(r"RMSE (μg m$^{-3}$)")
    ax.set_title("(a)  Local vs. cross-city RMSE — XGBoost",
                 loc="left", fontweight="bold", pad=8)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig22_cross_city_rmse.png")


def fig23_generalization_gap() -> None:
    compare = _build_compare()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    if compare is None or compare.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        save(fig, "fig23_generalization_gap.png")
        return

    colors = [BAR_COLORS["ARIMA"] if g >= 8 else BAR_COLORS["XGBoost"]
              for g in compare["gap"]]
    edges = [darken(c) for c in colors]
    bars  = ax.bar(compare["horizon_hours"].astype(str), compare["gap"],
                   color=colors, alpha=1.0, width=0.6)
    for bar, ec in zip(bars, edges):
        bar.set_edgecolor(ec)
        bar.set_linewidth(0.8)
    for x, v in zip(compare["horizon_hours"].astype(str), compare["gap"]):
        ax.text(x, v + compare["gap"].max() * 0.03, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel(r"$\Delta$RMSE (cross-city $-$ local,  μg m$^{-3}$)")
    ax.set_title("(b)  Generalization gap by forecast horizon",
                 loc="left", fontweight="bold", pad=8)
    fig.tight_layout()
    save(fig, "fig23_generalization_gap.png")


# ── Fig 03b: Actual vs Predicted scatter grid — density scatter + marginal KDE ─

def fig03_h1_scatter_grid() -> None:
    """Actual vs. Predicted scatter grid for all h1 models.

    Layout  : 2 rows × 3 cols (5 panels), each with main scatter + top/right marginals.
    Scatter : KDE density colouring, single-hue gradient per model, alpha=0.5.
    Marginals: histogram (density=True) + KDE curve overlay (seaborn).
    Lines   : y=x identity (black solid), OLS regression (red dashed).
    """
    from scipy.stats import gaussian_kde
    from matplotlib.colors import LinearSegmentedColormap

    _, raw_map, metrics_df = load_prediction_bundle("h1", horizon_filter=1)

    # Fixed display order
    MODEL_ORDER = ["LSTM", "Transformer", "ARIMA", "XGBoost", "Prophet"]
    models = [m for m in MODEL_ORDER if m in raw_map]

    # Per-model colour spec: light (histogram fill) → dark (KDE line + scatter high-density)
    STYLE = {
        "LSTM":        {"dark": "#1B7F79", "light": "#B8E0DC"},
        "Transformer": {"dark": "#2E5C8A", "light": "#B8D0E8"},
        "ARIMA":       {"dark": "#C76B2C", "light": "#F4D4B8"},
        "XGBoost":     {"dark": "#1F6FA8", "light": "#BCDCF0"},
        "Prophet":     {"dark": "#A8407C", "light": "#F0C4DC"},
    }

    PANEL_LABELS = "abcde"
    ncols, nrows = 3, 2

    # 15 cm × 10 cm — suitable for dual-column typesetting
    fig = plt.figure(figsize=(15 / 2.54, 10 / 2.54))
    fig.patch.set_facecolor("white")

    outer = fig.add_gridspec(
        nrows, ncols,
        hspace=0.35, wspace=0.30,
        left=0.09, right=0.98,
        top=0.94, bottom=0.12,
    )

    for idx, model in enumerate(models):
        r, c = divmod(idx, ncols)
        s    = STYLE[model]

        # Light → dark single-hue colormap for density scatter
        cmap = LinearSegmentedColormap.from_list(model, [s["light"], s["dark"]])

        # Each outer cell → 2×2 sub-grid
        #   [0,0] top histogram   [0,1] blank corner
        #   [1,0] main scatter    [1,1] right histogram
        inner = outer[r, c].subgridspec(
            2, 2,
            width_ratios=[5.5, 1.2],
            height_ratios=[1.2, 5.5],
            hspace=0.03, wspace=0.03,
        )
        ax_th = fig.add_subplot(inner[0, 0])
        ax_tc = fig.add_subplot(inner[0, 1])
        ax_sc = fig.add_subplot(inner[1, 0])
        ax_rh = fig.add_subplot(inner[1, 1])

        ax_tc.set_visible(False)

        df = raw_map[model]
        yt = df["y_true"].to_numpy(float)
        yp = df["y_pred"].to_numpy(float)

        # Per-panel axis range (auto-adapt; preserves Prophet's compressed range)
        lo  = min(yt.min(), yp.min())
        hi  = max(yt.max(), yp.max())
        pad = (hi - lo) * 0.06
        lo -= pad
        hi += pad

        # KDE density for point colouring
        try:
            kde_z = gaussian_kde(np.vstack([yt, yp]))(np.vstack([yt, yp]))
        except Exception:
            kde_z = np.ones(len(yt))
        si          = kde_z.argsort()
        p5, p95     = np.percentile(kde_z, [5, 95])
        norm_z      = np.clip((kde_z - p5) / max(p95 - p5, 1e-12), 0.05, 1.0)

        # ── Main scatter ──────────────────────────────────────────────
        ax_sc.scatter(
            yt[si], yp[si], c=norm_z[si], cmap=cmap,
            vmin=0.0, vmax=1.0, s=10, alpha=0.4, linewidths=0, zorder=3,
        )

        # Identity line y = x
        ax_sc.plot([lo, hi], [lo, hi], color="black", lw=1.0, zorder=4)

        # OLS regression line (red dashed)
        coef  = np.polyfit(yt, yp, 1)
        fit_x = np.linspace(lo, hi, 200)
        ax_sc.plot(fit_x, np.polyval(coef, fit_x),
                   color="#C0392B", lw=1.0, ls="--", alpha=0.8, zorder=5)

        # Light gray dashed grid
        ax_sc.set_axisbelow(True)
        ax_sc.grid(True, color="gray", alpha=0.2, linestyle="--", linewidth=0.5)

        ax_sc.set_xlim(lo, hi)
        ax_sc.set_ylim(lo, hi)
        ax_sc.set_xlabel(r"Actual PM$_{2.5}$ (μg m$^{-3}$)",   fontsize=7, labelpad=2)
        ax_sc.set_ylabel(r"Predicted PM$_{2.5}$ (μg m$^{-3}$)", fontsize=7, labelpad=2)
        ax_sc.tick_params(labelsize=6)
        sns.despine(ax=ax_sc)   # remove top + right spines only

        # R² / RMSE / MAE annotation (bottom-right, italic R²)
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mae  = float(np.mean(np.abs(yt - yp)))
        r2   = float(1.0 - np.sum((yt - yp) ** 2) /
                     max(np.sum((yt - yt.mean()) ** 2), 1e-12))
        ax_sc.text(
            0.97, 0.04,
            f"$\\it{{R}}^2$={r2:.3f}\nRMSE={rmse:.2f}\nMAE={mae:.2f}",
            transform=ax_sc.transAxes,
            ha="right", va="bottom", fontsize=6.0,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="lightgray", alpha=0.85),
        )

        # Panel label (a)(b)… — bold, upper-left
        ax_sc.text(0.04, 0.96, f"({PANEL_LABELS[idx]})",
                   transform=ax_sc.transAxes,
                   ha="left", va="top", fontsize=8, fontweight="bold")

        # ── Top histogram + manually scaled KDE (actual values) ──────
        counts_t, edges_t, _ = ax_th.hist(
            yt, bins=35, density=True,
            color=s["light"], alpha=0.65,
            edgecolor="white", linewidth=0.3, zorder=2,
        )
        # Compute KDE over [lo, hi] and scale to match histogram density heights
        bw_t     = edges_t[1] - edges_t[0]
        kde_fn_t = gaussian_kde(yt, bw_method="scott")
        x_kde_t  = np.linspace(lo, hi, 400)
        # density=True histogram and gaussian_kde both use probability density —
        # scale factor = 1 (no extra scaling needed; just clip to the visible range)
        y_kde_t  = kde_fn_t(x_kde_t)
        ax_th.plot(x_kde_t, y_kde_t, color=s["dark"], lw=0.5, alpha=0.7, zorder=3)
        ax_th.set_xlim(lo, hi)
        ax_th.set_ylim(bottom=0)
        ax_th.set_xlabel("")
        ax_th.set_ylabel("")
        ax_th.set_title(model, fontsize=8, fontweight="bold",
                        pad=3, color="#111111")
        for sp in ax_th.spines.values():
            sp.set_visible(False)
        ax_th.tick_params(bottom=False, labelbottom=False,
                          left=False,   labelleft=False)
        ax_th.set_facecolor("white")

        # ── Right histogram + manually scaled KDE (predicted values) ─
        counts_r, edges_r, _ = ax_rh.hist(
            yp, bins=35, density=True,
            color=s["light"], alpha=0.65,
            edgecolor="white", linewidth=0.3,
            orientation="horizontal", zorder=2,
        )
        bw_r     = edges_r[1] - edges_r[0]
        kde_fn_r = gaussian_kde(yp, bw_method="scott")
        y_kde_r  = np.linspace(lo, hi, 400)
        x_kde_r  = kde_fn_r(y_kde_r)   # density on x-axis for horizontal hist
        ax_rh.plot(x_kde_r, y_kde_r, color=s["dark"], lw=0.5, alpha=0.7, zorder=3)
        ax_rh.set_ylim(lo, hi)
        ax_rh.set_xlim(left=0)
        ax_rh.set_xlabel("")
        ax_rh.set_ylabel("")
        for sp in ax_rh.spines.values():
            sp.set_visible(False)
        ax_rh.tick_params(bottom=False, labelbottom=False,
                          left=False,   labelleft=False)
        ax_rh.set_facecolor("white")

    # Blank any unused grid cells (6th cell in 2×3)
    for idx in range(len(models), nrows * ncols):
        r, c = divmod(idx, ncols)
        fig.add_subplot(outer[r, c]).set_visible(False)

    save(fig, "fig03_h1_scatter_grid.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    apply_theme()

    print("── Prediction curves ─────────────────────────────────")
    fig01_h1_prediction_curve()
    fig04_h6_prediction_curve()

    print("── Model comparison (h1) ─────────────────────────────")
    fig02_h1_model_rmse()
    fig03_h1_residual_kde()
    fig03_h1_scatter_grid()

    print("── Sequence-6 analysis ───────────────────────────────")
    fig05_rmse_vs_horizon()
    fig06_seq6_avg_rmse()

    print("── Overall metric comparisons ────────────────────────")
    for metric, label, num in [("rmse", "a", 7), ("mae", "b", 8), ("mape", "c", 9)]:
        fig_overall_metric(metric, label, num)

    print("── Feature importance ────────────────────────────────")
    fig_feature_importance("h1",  "a", 10)
    fig_feature_importance("seq6", "b", 11)

    print("── AQI bucket heatmaps ───────────────────────────────")
    fig_aqi_heatmap("h1",   1, "a", 12)
    fig_aqi_heatmap("seq6", 6, "b", 13)

    print("── Feature ablation ──────────────────────────────────")
    fig_feature_ablation("h1",   "a", 14)
    fig_feature_ablation("seq6", "b", 15)

    print("── High-pollution error ──────────────────────────────")
    fig_high_pollution("h1",   1, "a", 16)
    fig_high_pollution("seq6", 6, "b", 17)

    print("── Diurnal error patterns ────────────────────────────")
    fig_error_by_hour("h1",   1, "a", 18)
    fig_error_by_hour("seq6", 6, "b", 19)

    print("── Day-period heatmaps ───────────────────────────────")
    fig_dayperiod_heatmap("h1",   1, "a", 20)
    fig_dayperiod_heatmap("seq6", 6, "b", 21)

    print("── Cross-city generalization ─────────────────────────")
    fig22_cross_city_rmse()
    fig23_generalization_gap()

    print(f"\nAll figures saved to  {FIG_DIR}")


if __name__ == "__main__":
    main()
