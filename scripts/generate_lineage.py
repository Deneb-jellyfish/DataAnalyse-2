"""生成数据血缘图 PNG。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "data_lineage.png"

NODES = [
    (0.5, 0.92, "Kaggle PRSA 原始数据\n(12站, 小时级)"),
    (0.5, 0.78, "站点清洗\n(3σ, ≤6h插值)"),
    (0.5, 0.64, "城市聚合\n(小时→日均)"),
    (0.5, 0.50, "beijing.csv\n(统一schema)"),
    (0.5, 0.36, "特征工程\n(滞后/滚动/时间)"),
    (0.5, 0.22, "X_train.npy / y_train.npy"),
    (0.15, 0.08, "ARIMA / Prophet"),
    (0.38, 0.08, "XGBoost"),
    (0.62, 0.08, "LSTM"),
    (0.85, 0.08, "Transformer"),
]

SIDE_NODES = [
    (0.88, 0.78, "上海 PM 原始数据"),
    (0.88, 0.64, "shanghai.csv"),
    (0.88, 0.50, "aligned/\n(迁移对齐子集)"),
]

EDGES = [
    ((0.5, 0.89), (0.5, 0.81)),
    ((0.5, 0.75), (0.5, 0.67)),
    ((0.5, 0.61), (0.5, 0.53)),
    ((0.5, 0.47), (0.5, 0.39)),
    ((0.5, 0.33), (0.5, 0.25)),
    ((0.5, 0.19), (0.15, 0.11)),
    ((0.5, 0.19), (0.38, 0.11)),
    ((0.5, 0.19), (0.62, 0.11)),
    ((0.5, 0.19), (0.85, 0.11)),
    ((0.88, 0.75), (0.88, 0.67)),
    ((0.88, 0.61), (0.88, 0.53)),
    ((0.62, 0.50), (0.88, 0.53)),
    ((0.5, 0.50), (0.62, 0.50)),
]


def setup_chinese_font() -> None:
    """Configure a Chinese-capable font for matplotlib."""
    candidates = [
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Arial Unicode MS",
        "SimHei",
        "Noto Sans CJK SC",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False


def draw_box(ax, x: float, y: float, text: str, color: str = "#DEEAF1") -> None:
    """在坐标轴比例位置绘制文本框。"""
    box = FancyBboxPatch(
        (x - 0.11, y - 0.045),
        0.22,
        0.09,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        edgecolor="#1F4E79",
        facecolor=color,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=8, transform=ax.transAxes)


def main() -> None:
    """渲染血缘图。"""
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("AQI 项目数据血缘图", fontsize=14, fontweight="bold", color="#1F4E79")

    for x, y, text in NODES:
        draw_box(ax, x, y, text)
    for x, y, text in SIDE_NODES:
        draw_box(ax, x, y, text, color="#E2EFDA")

    for start, end in EDGES:
        arrow = FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#2E75B6",
        )
        ax.add_patch(arrow)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"血缘图已保存至 {OUTPUT}")


if __name__ == "__main__":
    main()
