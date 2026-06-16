"""Generate EDA figures for the AQI project."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import BEIJING_DIR, PROCESSED_DIR, RAW_DIR

FIGURES_DIR = ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


def load_hourly_city() -> pd.DataFrame:
    """Load cleaned Beijing hourly city-level data."""
    path = BEIJING_DIR / "beijing_hourly_city.csv"
    frame = pd.read_csv(path, parse_dates=["datetime"])
    return frame.sort_values("datetime")


def load_daily_beijing() -> pd.DataFrame:
    """Load unified Beijing daily data."""
    frame = pd.read_csv(PROCESSED_DIR / "beijing.csv", parse_dates=["date"])
    return frame.sort_values("date")


def load_aligned_cities() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aligned Beijing and Shanghai daily data."""
    beijing = pd.read_csv(PROCESSED_DIR / "aligned" / "beijing.csv", parse_dates=["date"])
    shanghai = pd.read_csv(PROCESSED_DIR / "aligned" / "shanghai.csv", parse_dates=["date"])
    return beijing.sort_values("date"), shanghai.sort_values("date")


def plot_pm25_timeseries(hourly: pd.DataFrame) -> None:
    """Figure 1: PM2.5 annual time series."""
    daily = hourly.set_index("datetime")["PM2.5"].resample("D").mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    daily.plot(ax=ax, color="#2E75B6", linewidth=0.8)
    ax.set_title("Beijing PM2.5 Daily Mean Time Series (2013-2017)")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (ug/m3)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_pm25_timeseries.png", dpi=150)
    plt.close(fig)


def plot_pm25_histogram(daily: pd.DataFrame) -> None:
    """Figure 2: PM2.5 distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(daily["pm25"].dropna(), bins=40, color="#1F4E79", edgecolor="white")
    ax.set_title("Beijing PM2.5 Distribution (Daily)")
    ax.set_xlabel("PM2.5 (ug/m3)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_pm25_histogram.png", dpi=150)
    plt.close(fig)


def plot_pollutant_boxplot(hourly: pd.DataFrame) -> None:
    """Figure 3: Pollutant concentration boxplots."""
    pollutants = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    melted = hourly[pollutants].melt(var_name="pollutant", value_name="value").dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=melted, x="pollutant", y="value", ax=ax)
    ax.set_title("Beijing Pollutant Concentration Distribution")
    ax.set_xlabel("Pollutant")
    ax.set_ylabel("Concentration")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_pollutant_boxplot.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(hourly: pd.DataFrame) -> None:
    """Figure 4: Correlation heatmap across numeric fields."""
    columns = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
    corr = hourly[columns].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Beijing Air Quality Feature Correlation")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_missing_heatmap() -> None:
    """Figure 5: Missing value rates before and after cleaning."""
    import glob

    raw_file = sorted(glob.glob(str(RAW_DIR / "PRSA_Data_*.csv")))[0]
    raw = pd.read_csv(raw_file)
    cleaned = pd.read_csv(BEIJING_DIR / "beijing_hourly_all_stations.csv")

    pollutant_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    raw_missing = raw[pollutant_cols].replace("NA", np.nan).apply(pd.to_numeric, errors="coerce").isna().mean()
    clean_missing = cleaned[pollutant_cols].isna().mean()
    matrix = pd.DataFrame({"raw": raw_missing, "cleaned": clean_missing})

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix.T, annot=True, fmt=".2%", cmap="YlOrRd", ax=ax)
    ax.set_title("Missing Value Rate: Raw vs Cleaned (Single Station Sample)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_missing_heatmap.png", dpi=150)
    plt.close(fig)


def plot_city_comparison(beijing: pd.DataFrame, shanghai: pd.DataFrame) -> None:
    """Figure 6: Beijing vs Shanghai PM2.5 comparison."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(beijing["date"], beijing["pm25"], label="Beijing", linewidth=0.9)
    ax.plot(shanghai["date"], shanghai["pm25"], label="Shanghai", linewidth=0.9, alpha=0.8)
    ax.set_title("Aligned Daily PM2.5: Beijing vs Shanghai (2013-2015)")
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_city_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Generate all EDA figures."""
    hourly = load_hourly_city()
    daily = load_daily_beijing()
    aligned_bj, aligned_sh = load_aligned_cities()

    plot_pm25_timeseries(hourly)
    plot_pm25_histogram(daily)
    plot_pollutant_boxplot(hourly)
    plot_correlation_heatmap(hourly)
    plot_missing_heatmap()
    plot_city_comparison(aligned_bj, aligned_sh)
    print(f"EDA figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
