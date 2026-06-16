# AQI Hourly PM2.5 Forecasting

Git 仓库链接：`https://github.com/Deneb-jellyfish/DataAnalyse-2.git`



## 1. 顶层提交内容

- `README.md`
  提交说明总览，介绍仓库结构、运行方式和关键产出位置。

- `完整代码/`
  项目的全部代码、数据、脚本、模型、实验输出和过程文档。

- `项目报告.pdf`
  最终项目报告，建议优先查看。

- `团队分工声明.pdf`
  团队成员分工、工作量和对应产出说明。

- `PM25预测_课程汇报.pptx`
  答辩 PPT。

- `项目演示视频.mp4`
  项目功能与结果展示视频。

- `项目海报.png`
  项目海报。

## 2. 项目主题

本项目聚焦小时级 PM2.5 浓度预测，主要完成两类任务：

- `h1`
  预测未来 1 小时 PM2.5。

- `seq6`
  连续预测未来 6 小时 PM2.5。

项目当前保留并整理好的主线模型包括：

- `ARIMA`
- `Prophet`
- `XGBoost`
- `LSTM`
- `Transformer`

其中实验结果、误差分析、图表和模型输出均已保存在 `完整代码/outputs/hourly/` 下。

## 3. 完整代码目录说明

`完整代码/` 内是完整可运行项目，主要结构如下：

- `完整代码/data/`
  原始数据与预处理后的数据文件。

- `完整代码/scripts/`
  训练、分析、汇总、出图脚本。

- `完整代码/models/`
  各模型封装实现。

- `完整代码/utils/`
  预处理、特征工程、路径管理等辅助模块。

- `完整代码/evaluation/`
  评价指标计算逻辑。

- `完整代码/outputs/`
  全部实验输出文件。

- `完整代码/docs/`
  接口说明、特征说明、预处理日志和补充图。

- `完整代码/run_all.py`
  一键主流程入口。

- `完整代码/main.py`
  预处理和特征工程入口。

- `完整代码/app.py`
  Streamlit 可视化界面入口。

## 4. 重要代码入口

如果需要快速理解项目代码，建议先看这些文件：

- `完整代码/run_all.py`
  一键串联整体流程，是最重要的运行入口。

- `完整代码/main.py`
  负责预处理和特征工程。

- `完整代码/scripts/train_xgboost.py`
  树模型训练主脚本。

- `完整代码/scripts/train_lstm.py`
  LSTM 训练脚本。

- `完整代码/scripts/train_transformer.py`
  Transformer 训练脚本。

- `完整代码/scripts/run_hourly_analysis.py`
  误差分析、AQI 分层分析、特征消融分析。

- `完整代码/scripts/assemble_hourly_outputs.py`
  汇总指标表、实验注册表和简要结论生成。

- `完整代码/scripts/generate_hourly_figures.py`
  生成报告用图。

## 5. 重要文档文件

`完整代码/docs/` 中比较关键的说明材料如下：

- `完整代码/docs/interface_contract.md`
  项目接口与交付物说明，可用于快速核对 A/B 工作内容与产物。

- `完整代码/docs/feature_doc_hourly.md`
  小时级特征工程说明。

- `完整代码/docs/preprocess_log.md`
  预处理日志和数据对齐结果记录。

- `完整代码/docs/figures/00_hourly_pipeline_overview.png`
  小时级实验流程图。

## 6. 重要实验输出文件

下面这些是最值得查看的结果文件，均位于 `完整代码/outputs/hourly/` 下。

### 6.1 汇总结论

- `完整代码/outputs/hourly/summaries/overall_metrics_summary.csv`
  所有模型在 `h1` 和 `seq6` 任务上的总体指标汇总，是最核心的结果表。

- `完整代码/outputs/hourly/summaries/analysis_notes.md`
  自动生成的简要实验结论，适合快速浏览。

- `完整代码/outputs/hourly/summaries/experiment_registry.csv`
  记录关键实验产物是否存在，便于验收。

### 6.2 模型指标

- `完整代码/outputs/hourly/metrics/arima_metrics.csv`
- `完整代码/outputs/hourly/metrics/prophet_metrics.csv`
- `完整代码/outputs/hourly/metrics/xgboost_metrics.csv`
- `完整代码/outputs/hourly/metrics/lstm_metrics.csv`
- `完整代码/outputs/hourly/metrics/transformer_metrics.csv`

这些文件分别保存各模型在不同任务和预测步长上的误差指标。

### 6.3 预测结果

- `完整代码/outputs/hourly/predictions/xgboost_predictions_h1.csv`
- `完整代码/outputs/hourly/predictions/xgboost_predictions_seq6.csv`
- `完整代码/outputs/hourly/predictions/lstm_predictions_h1.csv`
- `完整代码/outputs/hourly/predictions/lstm_predictions_seq6.csv`
- `完整代码/outputs/hourly/predictions/transformer_predictions_h1.csv`
- `完整代码/outputs/hourly/predictions/transformer_predictions_seq6.csv`

这些文件保存逐时刻预测值与真实值，可用于复核模型表现。

### 6.4 误差分析与扩展分析

- `完整代码/outputs/hourly/analyses/feature_ablation.csv`
  XGBoost 特征消融结果。

- `完整代码/outputs/hourly/analyses/aqi_bucket_metrics.csv`
  按 AQI 区间分层的误差结果。

- `完整代码/outputs/hourly/analyses/high_pollution_error_analysis.csv`
  高污染样本误差分析。

- `完整代码/outputs/hourly/analyses/hourly_error_by_hour.csv`
  按小时统计误差。

- `完整代码/outputs/hourly/analyses/hourly_error_by_dayperiod.csv`
  按时段统计误差。

- `完整代码/outputs/hourly/analyses/cross_city_metrics.csv`
  跨城市泛化测试结果。

- `完整代码/outputs/hourly/analyses/xgboost_feature_importance_h1.csv`
- `完整代码/outputs/hourly/analyses/xgboost_feature_importance_seq6.csv`
  XGBoost 特征重要性结果。

### 6.5 图表文件

图表统一保存在 `完整代码/outputs/hourly/figures/` 中，共 23 张，主要包括：

- `fig01_h1_prediction_curve.png`
  单步预测曲线。

- `fig04_h6_prediction_curve.png`
  多步预测曲线。

- `fig07_overall_rmse.png`
  各模型总体 RMSE 对比。

- `fig10_shap_h1.png`
  XGBoost 单步任务 SHAP 特征解释图。

- `fig11_shap_seq6.png`
  XGBoost 多步任务 SHAP 特征解释图。

- `fig14_feature_ablation_h1.png`
- `fig15_feature_ablation_seq6.png`
  特征消融图。

- `fig22_cross_city_rmse.png`
- `fig23_generalization_gap.png`
  跨城市泛化结果图。

## 7. 运行方式

建议先进入代码目录再运行：

```powershell
cd .\完整代码
py -3 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 7.1 一键主流程

```powershell
cd .\完整代码
py -3 run_all.py --dry-run
py -3 run_all.py
```

### 7.2 只运行预处理和特征工程

```powershell
cd .\完整代码
py -3 main.py --steps all --granularity hourly
```

### 7.3 单独运行某一阶段

```powershell
cd .\完整代码
py -3 scripts\train_xgboost.py --task both
py -3 scripts\run_hourly_analysis.py
py -3 scripts\assemble_hourly_outputs.py
py -3 scripts\generate_hourly_figures.py
py -3 scripts\generate_shap_figures.py
```

### 7.4 启动可视化界面

```powershell
cd .\完整代码
py -3.12 -m streamlit run app.py
```

## 8. 建议查看顺序

如果是老师或答辩评审，建议按下面顺序查看：

1. `项目报告.pdf`
2. `PM25预测_课程汇报.pptx`
3. `团队分工声明.pdf`
4. `完整代码/outputs/hourly/summaries/analysis_notes.md`
5. `完整代码/outputs/hourly/summaries/overall_metrics_summary.csv`
6. `完整代码/outputs/hourly/analyses/feature_ablation.csv`
7. `完整代码/outputs/hourly/analyses/cross_city_metrics.csv`
8. `完整代码/outputs/hourly/figures/`
9. `完整代码/run_all.py` 与 `完整代码/scripts/`

## 9. 说明

- 仓库内部相对路径已经保持一致，整理后仍可从 `完整代码/` 内继续运行脚本。
- 当前小时级实验输出目录已经整理为：
  `analyses/`、`configs/`、`figures/`、`histories/`、`metrics/`、`models/`、`predictions/`、`summaries/`。