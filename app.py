"""AQI 项目中文演示版 Streamlit 页面。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

from utils.io import ALIGNED_DIR, BEIJING_DIR
from visualization.plot import (
    compute_metric_frame,
    load_predictions,
    plot_aqi_error_heatmap,
    plot_best_model_scatter,
    plot_metric_bars,
    plot_prediction_curves,
)

st.set_page_config(
    page_title="AQI 项目展示系统",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DESCRIPTIONS = {
    "ARIMA": "传统统计时间序列模型，适合作为经典基线。",
    "Prophet": "趋势+季节性分解模型，解释性较强，但对突发峰值不够敏感。",
    "XGBoost": "基于特征工程的树模型，是当前项目中表现最好的模型。",
    "LSTM": "循环神经网络模型，适合捕捉时间依赖关系。",
    "Informer": "面向长序列预测的稀疏注意力模型。",
    "Transformer": "标准注意力时间序列模型，用于和 LSTM/Informer 对比。",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(228, 171, 71, 0.15), transparent 28%),
                radial-gradient(circle at top right, rgba(50, 126, 118, 0.12), transparent 24%),
                linear-gradient(180deg, #f8f5ef 0%, #efe6d7 100%);
            color: #241f19;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.5rem 1.7rem;
            border-radius: 26px;
            background: linear-gradient(135deg, rgba(255, 252, 246, 0.96), rgba(249, 240, 223, 0.9));
            border: 1px solid rgba(110, 92, 62, 0.16);
            box-shadow: 0 18px 48px rgba(74, 58, 31, 0.08);
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-family: "Microsoft YaHei", "SimHei", "PingFang SC", sans-serif;
            font-size: 2.5rem;
            margin-bottom: 0.35rem;
        }
        .hero p {
            margin: 0.2rem 0;
            line-height: 1.7;
            font-size: 1rem;
            max-width: 980px;
        }
        .panel {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: rgba(255, 252, 245, 0.9);
            border: 1px solid rgba(110, 92, 62, 0.14);
            margin-bottom: 0.8rem;
        }
        .small-card {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(255, 252, 245, 0.92);
            border: 1px solid rgba(110, 92, 62, 0.14);
            min-height: 120px;
        }
        .small-card h4 {
            margin: 0 0 0.3rem 0;
            color: #7a5420;
            font-size: 1rem;
        }
        .small-card p {
            margin: 0;
            line-height: 1.55;
        }
        div[data-testid="metric-container"] {
            background: rgba(255, 252, 245, 0.92);
            border: 1px solid rgba(110, 92, 62, 0.14);
            border-radius: 18px;
            padding: 0.9rem 1rem;
        }
        /* ── 侧边栏 ── */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eeeae0 0%, #e5e1d5 100%);
            border-right: 1px solid rgba(110, 92, 62, 0.12);
        }
        section[data-testid="stSidebar"] h2 {
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #8a7a60;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(110, 92, 62, 0.15);
            margin-bottom: 0.25rem;
        }
        section[data-testid="stSidebar"] .stRadio label {
            padding: 0.45rem 0.75rem !important;
            border-radius: 10px !important;
            transition: background 0.15s;
            font-size: 0.97rem;
            color: #3a3020;
            margin: 2px 0 !important;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(82, 106, 68, 0.10) !important;
        }
        section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background: rgba(82, 106, 68, 0.20) !important;
            font-weight: 600;
            color: #2e4a20 !important;
        }
        /* ── 子页 Tab（下一天预测概览内部）── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: transparent;
            border-bottom: 2px solid rgba(110, 92, 62, 0.15);
            margin-bottom: 1.2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding: 0 22px;
            font-size: 1rem;
            font-weight: 500;
            color: #5a4e38;
            background: transparent !important;
            border: none !important;
            border-bottom: 3px solid transparent !important;
            border-radius: 0 !important;
            margin-bottom: -2px;
            transition: color 0.15s;
        }
        .stTabs [data-baseweb="tab"]:hover { color: #c94c4c; }
        .stTabs [aria-selected="true"] {
            color: #c94c4c !important;
            border-bottom: 3px solid #c94c4c !important;
        }
        .stTabs [data-baseweb="tab-highlight"],
        .stTabs [data-baseweb="tab-border"] { display: none !important; }
        .stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
          <strong>{title}</strong><br/>
          {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cards(cards: list[tuple[str, str]]) -> None:
    cols = st.columns(len(cards))
    for col, (title, text) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="small-card">
                  <h4>{title}</h4>
                  <p>{text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


@st.cache_data(show_spinner=False)
def get_predictions() -> pd.DataFrame:
    return load_predictions()


@st.cache_data(show_spinner=False)
def get_beijing_daily() -> pd.DataFrame:
    return pd.read_csv(BEIJING_DIR / "beijing_daily_city.csv", parse_dates=["date"]).sort_values("date")


@st.cache_data(show_spinner=False)
def get_shanghai_aligned() -> pd.DataFrame:
    return pd.read_csv(ALIGNED_DIR / "shanghai.csv", parse_dates=["date"]).sort_values("date")


@st.cache_data(show_spinner=False)
def get_one_day_metrics() -> pd.DataFrame:
    return compute_metric_frame(get_predictions())


def summarize_dataset(df: pd.DataFrame, date_col: str, target_col: str) -> dict[str, str | int | float]:
    return {
        "start": df[date_col].min().strftime("%Y-%m-%d"),
        "end": df[date_col].max().strftime("%Y-%m-%d"),
        "rows": int(len(df)),
        "target_mean": float(df[target_col].mean()),
        "target_max": float(df[target_col].max()),
    }


def format_metric_table(metric_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    table = metric_df.sort_values(metric_name).copy()
    best_value = float(table.iloc[0][metric_name])
    table["与最优差值"] = (table[metric_name] - best_value).round(2)
    table["RMSE"] = table["rmse"].round(2)
    table["MAE"] = table["mae"].round(2)
    table["MAPE"] = table["mape"].round(2)
    table["样本数"] = table["n"].astype(int)
    table["排名"] = range(1, len(table) + 1)
    table["模型"] = table["model"].astype(str)
    return table[["排名", "模型", "RMSE", "MAE", "MAPE", "样本数", "与最优差值"]]


def build_demo_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = df[["date", "PM2.5"]].copy()
    frame["month"] = frame["date"].dt.month
    frame["dayofweek"] = frame["date"].dt.dayofweek
    frame["dayofyear"] = frame["date"].dt.dayofyear
    frame["is_month_start"] = frame["date"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["date"].dt.is_month_end.astype(int)

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        frame[f"lag_{lag}"] = frame["PM2.5"].shift(lag)

    shifted = frame["PM2.5"].shift(1)
    for window in [3, 7, 14, 28]:
        frame[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        frame[f"roll_std_{window}"] = shifted.rolling(window).std()
        frame[f"roll_max_{window}"] = shifted.rolling(window).max()

    frame["target"] = frame["PM2.5"].shift(-1)
    feature_cols = [col for col in frame.columns if col not in {"date", "PM2.5", "target"}]
    frame = frame.dropna().reset_index(drop=True)
    return frame, feature_cols


@st.cache_resource(show_spinner=False)
def get_demo_forecast_model() -> tuple[xgb.XGBRegressor, list[str]]:
    daily = get_beijing_daily()
    train_frame, feature_cols = build_demo_training_frame(daily)
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=260,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.001,
        reg_lambda=1.0,
        n_jobs=1,
        seed=42,
        verbosity=0,
    )
    model.fit(train_frame[feature_cols], train_frame["target"], verbose=False)
    return model, feature_cols


def build_feature_row(history_df: pd.DataFrame, next_date: pd.Timestamp, feature_cols: list[str]) -> pd.DataFrame:
    pm = history_df["PM2.5"].astype(float)
    row: dict[str, float | int] = {
        "month": next_date.month,
        "dayofweek": next_date.dayofweek,
        "dayofyear": next_date.dayofyear,
        "is_month_start": int(next_date.is_month_start),
        "is_month_end": int(next_date.is_month_end),
    }
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        row[f"lag_{lag}"] = float(pm.iloc[-lag])
    for window in [3, 7, 14, 28]:
        recent = pm.iloc[-window:]
        row[f"roll_mean_{window}"] = float(recent.mean())
        row[f"roll_std_{window}"] = float(recent.std(ddof=0))
        row[f"roll_max_{window}"] = float(recent.max())
    return pd.DataFrame([[row[col] for col in feature_cols]], columns=feature_cols)


def recursive_demo_forecast(history_df: pd.DataFrame, steps: int = 7) -> pd.DataFrame:
    model, feature_cols = get_demo_forecast_model()
    history = history_df[["date", "PM2.5"]].copy().sort_values("date").reset_index(drop=True)

    results: list[dict[str, str | float | int]] = []
    for horizon in range(1, steps + 1):
        next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
        feature_row = build_feature_row(history, next_date, feature_cols)
        pred = max(0.0, float(model.predict(feature_row)[0]))
        results.append(
            {
                "日期": next_date.strftime("%Y-%m-%d"),
                "预测步长": f"第{horizon}天",
                "预测PM2.5": round(pred, 2),
            }
        )
        history.loc[len(history)] = {"date": next_date, "PM2.5": pred}
    return pd.DataFrame(results)


def plot_demo_forecast(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> plt.Figure:
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    recent_history = history_df.tail(30)
    ax.plot(recent_history["date"], recent_history["PM2.5"], color="#1f1f1f", linewidth=2.4, label="历史PM2.5")
    future_dates = pd.to_datetime(forecast_df["日期"])
    future_values = forecast_df["预测PM2.5"].astype(float)
    ax.plot(future_dates, future_values, color="#d15656", marker="o", linewidth=2.2, label="未来7天预测")
    ax.axvline(recent_history["date"].iloc[-1], linestyle="--", color="#8a7a64", linewidth=1.2)
    ax.set_title("下一天与未来七天 PM2.5 演示预测")
    ax.set_xlabel("日期")
    ax.set_ylabel("PM2.5")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    return fig


inject_css()

predictions = get_predictions()
beijing_daily = get_beijing_daily()
shanghai_aligned = get_shanghai_aligned()
metric_frame = get_one_day_metrics()

beijing_summary = summarize_dataset(beijing_daily, "date", "PM2.5")
shanghai_summary = summarize_dataset(shanghai_aligned, "date", "pm25")

st.markdown(
    """
    <div class="hero">
      <h1>AQI 项目展示系统</h1>
      <p>这是面向课程答辩和组内查看的中文版页面。当前重点展示项目总览、下一天预测结果，以及一个可交互的演示预测入口。</p>
      <p>项目主目标是利用历史空气质量与气象数据预测北京 PM2.5；现阶段已经完成下一天预测模块，未来七天预测模块预留给后续实验继续接入。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 导航")
    current_page = st.radio(
        "页面导航",
        ["📊  项目总览", "📈  下一天预测概览", "🔮  七天预测", "🤖  实际预测"],
        label_visibility="collapsed",
    )

if current_page == "📊  项目总览":
    st.subheader("项目总览")
    st.write("这个项目目前包含两条主线：一条是已经完成的“下一天 PM2.5 预测”，另一条是准备扩展的“未来七天 PM2.5 预测”。")

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("北京日级样本数", f"{beijing_summary['rows']}")
    metric_col_2.metric("北京时间范围", f"{beijing_summary['start']} ~ {beijing_summary['end']}")
    metric_col_3.metric("上海对齐样本数", f"{shanghai_summary['rows']}")
    metric_col_4.metric("当前对比模型数", f"{predictions['model'].nunique()}")

    render_cards(
        [
            (
                "北京主实验数据",
                f"时间范围为 {beijing_summary['start']} 至 {beijing_summary['end']}，共 {beijing_summary['rows']} 条日级样本，是当前建模主数据集。",
            ),
            (
                "上海对齐数据",
                f"时间范围为 {shanghai_summary['start']} 至 {shanghai_summary['end']}，共 {shanghai_summary['rows']} 条，用于跨城市泛化或迁移对比。",
            ),
            (
                "当前完成情况",
                "下一天预测已经完成 6 个模型的统一评估；未来七天预测实验计划已确定，等待新增结果接入。",
            ),
        ]
    )

    left, right = st.columns([1.1, 0.9])
    with left:
        render_panel(
            "数据集说明",
            "北京数据为项目主实验数据，目标是预测日级 PM2.5；上海对齐数据主要用于后续跨城市迁移或泛化分析。"
            "<br/>当前主要使用的建模字段包括 PM2.5、PM10、SO2、NO2、CO、O3、TEMP、PRES、DEWP、RAIN、WSPM。",
        )
        render_panel(
            "数据处理方法",
            "1. 原始小时级数据清洗与统一字段映射。"
            "<br/>2. 短缺口插值，长缺口保留，避免制造假信号。"
            "<br/>3. 小时级聚合为城市级，再聚合到日级数据。"
            "<br/>4. 构造滞后特征、滚动统计特征和时间特征。"
            "<br/>5. 按时间顺序划分训练集与测试集，后 20% 作为测试集。",
        )
    with right:
        st.subheader("数据预览")
        dataset_choice = st.radio("查看数据集", ["北京日级数据", "上海对齐数据"], horizontal=True)
        preview_rows = st.slider("预览行数", min_value=5, max_value=20, value=8, step=1, key="overview_preview")
        if dataset_choice == "北京日级数据":
            preview = beijing_daily.head(preview_rows).copy()
        else:
            preview = shanghai_aligned.head(preview_rows).copy()
        st.dataframe(preview, use_container_width=True, hide_index=True)

    st.subheader("当前项目方法概览")
    overview_table = pd.DataFrame(
        {
            "模块": ["数据处理", "单步预测", "多步预测", "评估分析"],
            "当前状态": ["已完成", "已完成", "待补充", "已完成基础版"],
            "说明": [
                "已完成清洗、聚合、对齐与特征工程",
                "已完成 ARIMA / Prophet / XGBoost / LSTM / Informer / Transformer",
                "计划新增未来7天 PM2.5 预测实验",
                "已完成整体误差、AQI分层误差与图表输出",
            ],
        }
    )
    st.dataframe(overview_table, use_container_width=True, hide_index=True)

elif current_page == “📈  下一天预测概览”:
    st.subheader(“下一天预测概览”)
    st.write('这一页展示”下一天 PM2.5 预测”结果，共四个子页面，参数调节项均放在图表旁边。')

    min_date = predictions[“date”].min().date()
    max_date = predictions[“date”].max().date()
    all_models = metric_frame[“model”].astype(str).tolist()

    # 共享筛选器（对所有子页生效）
    fc1, fc2 = st.columns([1.5, 1.5])
    with fc1:
        selected_models = st.multiselect(“筛选模型”, all_models, default=all_models, key=”nd_models”)
    with fc2:
        selected_dates = st.date_input(
            “时间范围”,
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=”nextday_dates”,
        )

    if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
        start_date, end_date = selected_dates[0], selected_dates[1]
    else:
        start_date, end_date = min_date, max_date

    active_models = selected_models if selected_models else all_models
    filtered = predictions[
        (predictions[“model”].astype(str).isin(active_models))
        & (predictions[“date”].dt.date >= start_date)
        & (predictions[“date”].dt.date <= end_date)
    ].copy()

    st.markdown(“---”)

    if filtered.empty:
        st.warning(“当前筛选条件下没有可展示的预测结果。”)
    else:
        filtered_metric = compute_metric_frame(filtered)

        tab_story, tab_compare, tab_error, tab_city = st.tabs([
            “  Story  “,
            “  Model Compare  “,
            “  Error Diagnose  “,
            “  City Transfer  “,
        ])

        # ── Story：概览卡片 + 排名表 ──────────────────────────────
        with tab_story:
            left_s, right_s = st.columns([2.2, 0.8])
            with right_s:
                ranking_metric = st.radio(
                    “排序依据”,
                    [“rmse”, “mae”, “mape”],
                    format_func=lambda x: x.upper(),
                    key=”story_rank”,
                )
            ranking_table = format_metric_table(filtered_metric, ranking_metric)
            best_model = str(ranking_table.iloc[0][“模型”])
            with left_s:
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric(“当前最优模型”, best_model)
                mc2.metric(f”最优 {ranking_metric.upper()}”, f”{float(ranking_table.iloc[0][ranking_metric.upper()]):.2f}”)
                mc3.metric(“参与对比模型数”, str(len(ranking_table)))
                st.markdown(“<br/>”, unsafe_allow_html=True)
                render_cards([
                    (“当前最优模型”, f”{best_model}，按 {ranking_metric.upper()} 排名第一。”),
                    (“模型简介”, MODEL_DESCRIPTIONS.get(best_model, “”)),
                    (“使用建议”, “先看 Model Compare 折线图，再看 Error Diagnose，最后看 City Transfer 散点。”),
                ])
                st.markdown(“<br/>”, unsafe_allow_html=True)
                st.markdown(“**模型排名总表**”)
                st.dataframe(ranking_table, use_container_width=True, hide_index=True)

        # ── Model Compare：多模型预测折线图 ──────────────────────
        with tab_compare:
            chart_col, ctrl_col = st.columns([2.2, 0.8])
            with ctrl_col:
                st.markdown(“#### 图表控制”)
                recent_days = st.slider(
                    “展示最近多少天”,
                    min_value=45, max_value=180, value=90, step=15,
                    key=”cmp_days”,
                )
                st.caption(f”当前展示最近 **{recent_days}** 天。”)
                st.markdown(“---”)
                st.caption(“建议答辩时选 2~4 个模型，折线图会更清晰。”)
            with chart_col:
                st.markdown(“#### 真实值与多模型预测对比”)
                st.caption(“观察不同模型在测试集上能否跟住真实 PM2.5 的波动趋势。”)
                fig = plot_prediction_curves(filtered, days=recent_days)
                st.pyplot(fig, use_container_width=True)

        # ── Error Diagnose：指标柱状图 + AQI 热力图 ──────────────
        with tab_error:
            st.markdown(“#### 图2：模型误差指标对比”)
            st.caption(“对比 RMSE / MAE / MAPE 三个维度下各模型的整体表现。”)
            fig = plot_metric_bars(filtered_metric)
            st.pyplot(fig, use_container_width=True)

            st.markdown(“#### 图4：AQI 分层误差热力图”)
            st.caption(“观察模型在不同污染等级区间上的误差表现，高污染区间通常更难预测。”)
            fig = plot_aqi_error_heatmap(filtered)
            st.pyplot(fig, use_container_width=True)

        # ── City Transfer：散点拟合图 ─────────────────────────────
        with tab_city:
            chart_col2, ctrl_col2 = st.columns([2.2, 0.8])
            with ctrl_col2:
                st.markdown(“#### 模型选择”)
                focus_model = st.selectbox(
                    “重点查看模型”,
                    active_models,
                    key=”city_focus”,
                )
                st.markdown(f”**{focus_model}**”)
                st.caption(MODEL_DESCRIPTIONS.get(focus_model, “”))
            with chart_col2:
                st.markdown(f”#### {focus_model} 预测散点图”)
                st.caption(“散点越靠近对角线，说明预测值越接近真实值。”)
                focus_metric = filtered_metric[
                    filtered_metric[“model”].astype(str) == focus_model
                ].reset_index(drop=True)
                fig = plot_best_model_scatter(filtered, focus_metric)
                st.pyplot(fig, use_container_width=True)

elif current_page == "🔮  七天预测":
    st.subheader("七天预测")
    st.info("这个页面预留给“未来七天 PM2.5 预测”实验。等组员把 7 天预测实验跑好后，这里会接入新的结果图、误差表和分析结论。")

    render_panel(
        "后续这里会放什么",
        "1. 未来7天真实值 vs 预测值折线图。"
        "<br/>2. Day+1 到 Day+7 的 RMSE / MAE 曲线。"
        "<br/>3. 模型 × Horizon 的误差热力图。"
        "<br/>4. AQI 分层下的多步预测误差分析。"
        "<br/>5. 七天预测实验结论与模型对比。",
    )

    roadmap = pd.DataFrame(
        {
            "预留模块": ["7天预测结果总览", "按天误差曲线", "AQI分层分析", "模型对比表", "实验结论"],
            "当前状态": ["待接入", "待接入", "待接入", "待接入", "待接入"],
            "说明": [
                "展示未来7天预测曲线",
                "展示 Day+1 ~ Day+7 的误差变化",
                "分析高污染区间误差",
                "对比 XGBoost / LSTM / Transformer 等",
                "总结哪类模型远期更稳定",
            ],
        }
    )
    st.dataframe(roadmap, use_container_width=True, hide_index=True)

elif current_page == "🤖  实际预测":
    st.subheader("实际预测")
    st.write("这个页面提供一个演示版的实际预测入口：你可以决定输入多少天的历史数据，并基于这些历史值预测下一天和未来七天的 PM2.5。")

    render_panel(
        "说明",
        "当前这一版是演示预测器，使用项目内北京日级数据训练的自回归 XGBoost 来做“下一天 + 未来七天”的实时推演。"
        "<br/>后续如果 7 天正式实验完成，这里可以替换成正式多步模型。",
    )

    input_days = st.slider("输入历史天数", min_value=30, max_value=90, value=30, step=5)
    default_history = beijing_daily[["date", "PM2.5"]].tail(input_days).copy().reset_index(drop=True)
    default_history["date"] = default_history["date"].dt.strftime("%Y-%m-%d")
    default_history = default_history.rename(columns={"date": "日期", "PM2.5": "PM2.5"})

    st.markdown("#### 可编辑输入数据")
    st.caption("你可以直接在下面修改最近若干天的 PM2.5 数值，然后点击按钮重新预测。")
    edited_history = st.data_editor(
        default_history,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="history_editor",
    )

    if st.button("开始预测", type="primary"):
        try:
            history_for_pred = edited_history.copy()
            history_for_pred["date"] = pd.to_datetime(history_for_pred["日期"])
            history_for_pred["PM2.5"] = pd.to_numeric(history_for_pred["PM2.5"])
            history_for_pred = history_for_pred.sort_values("date").reset_index(drop=True)

            if history_for_pred["PM2.5"].isna().any():
                st.error("输入数据中存在空值，请补全后再预测。")
            elif len(history_for_pred) < 30:
                st.error("为了保证滞后特征足够，输入历史天数至少需要 30 天。")
            else:
                forecast_df = recursive_demo_forecast(history_for_pred, steps=7)
                next_day_pm25 = float(forecast_df.iloc[0]["预测PM2.5"])
                avg_7day_pm25 = float(forecast_df["预测PM2.5"].mean())

                metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
                metric_col_1.metric("下一天预测值", f"{next_day_pm25:.2f}")
                metric_col_2.metric("未来7天平均预测值", f"{avg_7day_pm25:.2f}")
                metric_col_3.metric("未来7天峰值预测", f"{float(forecast_df['预测PM2.5'].max()):.2f}")

                st.markdown("#### 预测结果表")
                st.dataframe(forecast_df, use_container_width=True, hide_index=True)

                st.markdown("#### 预测结果图")
                fig = plot_demo_forecast(history_for_pred, forecast_df)
                st.pyplot(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"预测失败：{exc}")
