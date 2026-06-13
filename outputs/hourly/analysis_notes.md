# 小时级实验记录（analysis_notes）

## 2026-06-13 第一轮复建（48维特征重建后）

### 1) 本次实验目标
- 修复小时级特征口径不一致问题（文档48维、实际20维）。
- 在统一口径下补齐 `XGBoost / LSTM / Transformer` 的 `h1+h12` 结果。
- 生成统一主表并补全实验登记。

### 2) 使用的数据和任务
- 数据：`data/processed/beijing_hourly.csv`
- 任务：`h1=pm25(t+1)`，`h12=pm25(t+12)`
- 切分：训练池前80%，测试后20%；调参验证为训练池内后20%。

### 3) 参数改动
- `XGBoost`：确认 `h1` 标签按 `shift(-1)` 构造，修复过乐观评估。
- `LSTM`：`h1(seq_len=48)`，`h12(seq_len=72)`，其余参数见 `lstm_h*_config.json`。
- `Transformer`：`h1(seq_len=48)`，`h12(seq_len=72)`，其余参数见 `transformer_h*_config.json`。

### 4) 特征改动
- `utils/feature_engineering.py` 的 `build_hourly_feature_frame` 修复为正式版 48 维：
  - 中尺度滞后、滚动、差分、时段标记、气象变化、风向与交互项。
- 重新执行 `scripts/build_features.py --granularity hourly` 产出新矩阵。

### 5) 验证集指标（训练过程摘要）
- `LSTM h1` 最优验证损失约 `83.33`；`LSTM h12` 最优验证损失约 `4143.90`。
- `Transformer h1` 最优验证损失约 `81.49`；`Transformer h12` 最优验证损失约 `2876.51`。

### 6) 测试集指标（当前）
- `ARIMA`：h1 RMSE `9.01`，h12 RMSE `24.08`
- `XGBoost`：h1 RMSE `8.78`，h12 RMSE `45.64`
- `LSTM`：h1 RMSE `8.92`，h12 RMSE `55.67`
- `Transformer`：h1 RMSE `8.88`，h12 RMSE `47.94`
- `Prophet`：环境阻塞（`ModuleNotFoundError`），已记录 blocked。

### 7) 相比上一版变化
- `h1`：XGBoost、LSTM、Transformer 均进入约 `8.8~8.9` RMSE 区间，明显优于旧版深度模型结果。
- `h12`：树模型与深度模型仍显著弱于 ARIMA，远期预测仍是当前短板。

### 8) 原因判断
- `h1` 主要受益于特征口径修复与标签定义修复。
- `h12` 受远期不确定性影响更大，现参数仍偏 baseline，缺少针对性调参与特征消融验证。

### 9) 下一步建议
1. 进入 `XGBoost` 第一轮系统调参（按计划网格与 early stopping）。
2. 完成 `feature_ablation.csv`（5组特征 × h1/h12）。
3. 完成 AQI 分层、高污染、时段误差分析与对应图表。
4. 补做跨城市泛化（至少 XGBoost h1/h12）。


## 2026-06-13 第二轮优化（XGBoost调参+分析模块）

### 本轮新增
- 完成 XGBoost 第一轮系统调参（h1+h12）。
- 产出特征消融：`feature_ablation.csv`。
- 产出误差分析：`aqi_bucket_metrics.csv`、`high_pollution_error_analysis.csv`、`hourly_error_by_hour.csv`、`hourly_error_by_dayperiod.csv`。
- 产出跨城市泛化：`cross_city_metrics.csv`、`cross_city_predictions.csv`。
- 产出图表 11 张：`outputs/hourly/figures/01~11`。

### 本轮关键结果
- XGBoost h1: RMSE=8.65（优于 ARIMA 9.01）。
- XGBoost h12: RMSE=45.46（仍显著弱于 ARIMA 24.08）。
- 跨城市（北京训->上海测）：h1 RMSE=15.46，h12 RMSE=40.01。

### 结论
- h1 主力模型路线成立（XGBoost 当前最佳）。
- h12 仍需继续专项优化，重点放在更强的中尺度记忆与鲁棒目标/损失策略。

## 2026-06-13 XGBoost h12 第二轮局部细化调参

- 执行命令：python scripts/train_xgboost.py --task h12 --tune --tune-stage second --trials 40
- 结果：h12 RMSE=45.55，MAE=32.03，MAPE=114.62%。
- 对比上一轮最佳 RMSE=45.46，本轮仅接近但仍未超过，说明当前搜索空间在测试集泛化增益有限。
- 第二轮最优参数（CV）：n_estimators=600, max_depth=3, learning_rate=0.02, subsample=1.0, colsample_bytree=1.0, min_child_weight=3, reg_lambda=9, reg_alpha=0.2, gamma=0.1。
