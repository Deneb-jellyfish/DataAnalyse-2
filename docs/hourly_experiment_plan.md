# 小时级 PM2.5 预测实验方案（h1 + 未来6小时逐小时预测版）

本文档用于统一后续小时级 PM2.5 实验的正式执行口径。

本版方案不再采用：
- `+12h` 单点预测

而是改为：
- `h1` 单点预测
- `未来 6 小时逐小时预测`（multi-step / multi-output）

目标是让实验既保留一个稳定、清晰、容易做好的短期基准任务，又能体现“预测未来一段时间”的时序建模能力。

---

## 1. 本版核心调整

旧版正式方案的问题在于：
- `+12h` 单点预测对当前这套“仅历史观测、无未来气象预报”的信息条件过难
- 结果虽然能比较，但不够符合“预测未来一段时间”的直觉
- `h12` 在报告里不如“未来 6 小时轨迹预测”直观

因此本版做如下调整：
- 保留 `h1 = pm25(t+1)` 作为主基准任务
- 将扩展任务改为：`输入 t 及之前信息，输出 t+1 到 t+6 共 6 个小时的预测值`
- 深入分析模块也围绕 `h1` 与 `未来6小时预测` 重新组织

---

## 2. 本次实验目标

本次实验围绕两个任务展开。

### 任务 A：未来 1 小时 PM2.5 预测

- 输入：时刻 `t` 及之前的小时级污染与气象信息
- 输出：`pm25(t+1)`
- 作用：衡量模型对超短期污染变化的跟踪能力

### 任务 B：未来 6 小时逐小时预测

- 输入：时刻 `t` 及之前的小时级污染与气象信息
- 输出：
  - `pm25(t+1)`
  - `pm25(t+2)`
  - `pm25(t+3)`
  - `pm25(t+4)`
  - `pm25(t+5)`
  - `pm25(t+6)`
- 作用：衡量模型对短中期污染演化轨迹的预测能力

### 为什么改成“未来6小时逐小时预测”

- 比 `+12h` 单点预测更贴近当前特征可支撑的预测跨度
- 仍然比 `h1` 难，能够体现步长拉长后的性能退化
- 更符合“预测未来一段时间”的表述
- 图表展示更自然，便于答辩时解释模型对未来轨迹的把握能力

---

## 3. 参与对比的模型

正式主对比仍包含以下 5 个模型：

1. `ARIMA`
2. `Prophet`
3. `XGBoost`
4. `LSTM`
5. `Transformer`

### 模型定位

| 模型 | 类型 | 本次角色 |
|---|---|---|
| ARIMA | 传统统计模型 | 经典时间序列基线 |
| Prophet | 趋势/季节分解模型 | 可解释时间序列模型 |
| XGBoost | 特征工程型机器学习 | 主力非深度模型 |
| LSTM | 循环神经网络 | 深度学习时序代表 |
| Transformer | 注意力模型 | 深度学习对照模型 |

### Prophet 说明

- `Prophet` 必须是真正的 Prophet
- 如果本机环境中 Prophet 不能安装或不能运行，必须明确记录为 `blocked`
- 不允许再用 STL fallback 冒充 `Prophet` 主结果

---

## 4. 数据与代码基础

### 4.1 主数据文件

- [data/processed/beijing_hourly.csv](/d:/数据挖掘期末/data/processed/beijing_hourly.csv)

### 4.2 特征工程代码

- [utils/feature_engineering.py](/d:/数据挖掘期末/utils/feature_engineering.py)
- [docs/feature_doc_hourly.md](/d:/数据挖掘期末/docs/feature_doc_hourly.md)

### 4.3 特征产物目录

- [data/processed/features_hourly](/d:/数据挖掘期末/data/processed/features_hourly)

### 4.4 跨城市测试数据

- [data/processed/aligned_hourly/shanghai.csv](/d:/数据挖掘期末/data/processed/aligned_hourly/shanghai.csv)

---

## 5. 当前特征工程方向

当前小时级特征工程保持统一特征集，重点支持：
- `h1` 单点预测
- `未来6小时逐小时预测`

特征重点包括：

### 5.1 短中期滞后

- `pm25_lag1`
- `pm25_lag2`
- `pm25_lag3`
- `pm25_lag4`
- `pm25_lag6`
- `pm25_lag8`
- `pm25_lag12`
- `pm25_lag24`

### 5.2 短中期滚动统计

- `pm25_roll_mean_3`
- `pm25_roll_mean_6`
- `pm25_roll_mean_12`
- `pm25_roll_mean_24`
- `pm25_roll_std_3`
- `pm25_roll_std_6`
- `pm25_roll_std_12`
- `pm25_roll_std_24`

### 5.3 趋势变化

- `pm25_diff_1`
- `pm25_diff_2`
- `pm25_diff_3`
- `pm25_diff_6`

### 5.4 时间与时段特征

- `hour_sin`
- `hour_cos`
- `month_sin`
- `month_cos`
- `weekday`
- `is_weekend`
- `is_holiday`
- `is_daytime`
- `is_rush_hour`

### 5.5 气象与交互特征

- `temp`
- `pres`
- `dewp`
- `humidity`
- `wind_speed`
- `precipitation`
- `wind_dir_sin`
- `wind_dir_cos`
- `precipitation_flag`
- `dewp_temp_gap`
- `temp_diff_1`
- `temp_diff_2`
- `pres_diff_1`
- `pres_diff_2`
- `wind_speed_diff_1`
- `wind_speed_diff_2`
- `temp_x_humidity`
- `wind_speed_x_wind_dir_sin`
- `wind_speed_x_pm25_lag1`

---

## 6. 统一实验规则

### 6.1 统一任务定义

- `h1`：预测 `pm25(t+1)`
- `seq6`：预测 `pm25(t+1:t+6)`

### 6.2 统一评估时间点

- 各模型的最终主表必须建立在同一批 `forecast_origin_time / target_time` 上
- 多步任务下，每个步长也必须对齐后再比较

### 6.3 统一输出字段

所有预测结果文件统一字段：

```csv
forecast_origin_time,target_time,horizon_hours,city,model,y_true,y_pred,split
```

说明：
- `h1` 文件中 `horizon_hours` 固定为 `1`
- `seq6` 文件中 `horizon_hours` 取值为 `1,2,3,4,5,6`

所有指标文件统一字段：

```csv
model,horizon_hours,rmse,mae,mape,n_samples,notes
```

说明：
- `h1` 任务只有一行 `horizon_hours=1`
- `seq6` 任务建议至少提供 `horizon_hours=1..6` 的分步指标
- 如需总览，可额外增加 `notes` 标记 `aggregate_seq6`

### 6.4 实验记录必须完整

所有正式实验同步更新：
- `outputs/hourly/experiment_registry.csv`
- `outputs/hourly/analysis_notes.md`

---

## 7. 数据切分方案

总体切分继续采用时间顺序：
- 前 `80%`：训练池
- 后 `20%`：测试集

调参时在训练池内部再划分：
- 前 `80%`：子训练集
- 后 `20%`：验证集

即总体约为：
- `64%`：训练
- `16%`：验证
- `20%`：测试

对于序列模型：
- `h1` 与 `seq6` 由于 `seq_len` 不同，可用样本数允许略少
- 但最终比较必须能通过 `target_time` 对齐

---

## 8. 各模型任务设计

## 8.1 XGBoost

### 任务定义

- `h1`：目标列 `pm25_target_h1 = pm25.shift(-1)`
- `seq6`：
  - `pm25_target_h1 = pm25.shift(-1)`
  - `pm25_target_h2 = pm25.shift(-2)`
  - ...
  - `pm25_target_h6 = pm25.shift(-6)`

### 实现建议

有两种可接受实现：

1. `6 个独立直接模型`
   - `XGBoost_h1`
   - `XGBoost_h2`
   - ...
   - `XGBoost_h6`

2. `共享特征 + 分步独立回归器`
   - 本质仍是 6 个直接预测头

建议优先使用方案 1，最稳定、最容易复现。

### 必交文件

- `outputs/hourly/xgboost_predictions_h1.csv`
- `outputs/hourly/xgboost_predictions_seq6.csv`
- `outputs/hourly/xgboost_metrics.csv`
- `outputs/hourly/xgboost_feature_importance_h1.csv`
- `outputs/hourly/xgboost_feature_importance_seq6.csv`
- `outputs/hourly/xgboost_h1_best_params.json`
- `outputs/hourly/xgboost_seq6_best_params.json`

---

## 8.2 ARIMA

### 任务定义

- `h1`：直接预测 `t+1`
- `seq6`：对每个 forecast origin 直接生成未来 `1~6` 小时预测

### 实现建议

- 允许逐步取 `forecast(steps=6)` 的前 6 个输出
- 但最终保存时必须展开为逐小时记录
- 不允许抽样评估进入主榜

### 必交文件

- `outputs/hourly/arima_predictions_h1.csv`
- `outputs/hourly/arima_predictions_seq6.csv`
- `outputs/hourly/arima_metrics.csv`
- `outputs/hourly/arima_best_order_h1.json`
- `outputs/hourly/arima_best_order_seq6.json`

---

## 8.3 Prophet

### 任务定义

- `h1`
- `seq6`

### 实现建议

- 训练完成后，对每个 forecast origin 生成未来 `6` 小时 Prophet 预测
- 展开保存为逐小时记录
- 如果 Prophet 不可用，写 `blocked`

### 必交文件

- `outputs/hourly/prophet_predictions_h1.csv`
- `outputs/hourly/prophet_predictions_seq6.csv`
- `outputs/hourly/prophet_metrics.csv`
- `outputs/hourly/prophet_config_h1.json`
- `outputs/hourly/prophet_config_seq6.json`

---

## 8.4 LSTM

### 任务定义

- `h1`：单输出
- `seq6`：建议改为 `6 维多输出`

### 推荐序列方案

| 任务 | 建议 `seq_len` |
|---|---|
| `h1` | `48` |
| `seq6` | `48` 或 `72` |

### 实现建议

- `h1` 保持现有单输出结构
- `seq6` 建议改为：
  - 输入过去 `seq_len` 小时
  - 输出未来 `6` 个小时的预测向量

### 必交文件

- `outputs/hourly/lstm_predictions_h1.csv`
- `outputs/hourly/lstm_predictions_seq6.csv`
- `outputs/hourly/lstm_metrics.csv`
- `outputs/hourly/lstm_train_history.json`
- `outputs/hourly/lstm_h1_config.json`
- `outputs/hourly/lstm_seq6_config.json`

---

## 8.5 Transformer

### 任务定义

- `h1`：单输出
- `seq6`：建议改为 `6 维多输出`

### 推荐序列方案

| 任务 | 建议 `seq_len` |
|---|---|
| `h1` | `48` |
| `seq6` | `48` 或 `72` |

### 实现建议

- `h1` 保持现有单输出
- `seq6` 改为未来 6 小时多输出头

### 必交文件

- `outputs/hourly/transformer_predictions_h1.csv`
- `outputs/hourly/transformer_predictions_seq6.csv`
- `outputs/hourly/transformer_metrics.csv`
- `outputs/hourly/transformer_train_history.json`
- `outputs/hourly/transformer_h1_config.json`
- `outputs/hourly/transformer_seq6_config.json`

---

## 9. 深入分析模块

## 9.1 总体指标主表

生成：
- `outputs/hourly/overall_metrics_summary.csv`

建议内容：
- `h1` 单步结果
- `seq6` 中 `horizon_hours=1..6` 的分步结果
- 可附加 `seq6_mean` 作为未来 6 小时平均表现

---

## 9.2 特征消融（重点做 XGBoost）

至少做 5 组：

1. `base_short_lag`
2. `base_plus_daily_cycle`
3. `mid_range_memory`
4. `trend_and_roll`
5. `full_48`

分别在：
- `h1`
- `seq6`

其中 `seq6` 建议至少汇报：
- 各步长平均 RMSE
- 或 `horizon=1..6` 的分步 RMSE

必交：
- `outputs/hourly/feature_ablation.csv`

---

## 9.3 AQI 分层误差分析

建议分层：
- `0-35`
- `35-75`
- `75-115`
- `115-150`
- `>150`

对：
- `h1`
- `seq6 (1..6步)`

分别统计。

必交：
- `outputs/hourly/aqi_bucket_metrics.csv`

---

## 9.4 高污染场景误差分析

筛选：
- `pm25 > 150`

分别统计：
- `h1`
- `seq6`

必交：
- `outputs/hourly/high_pollution_error_analysis.csv`

---

## 9.5 昼夜 / 时段误差分析

仍保留两层：

### 按小时统计

- `0~23` 每个小时分别统计误差

### 按时段统计

- `凌晨`
- `早高峰`
- `白天平峰`
- `晚高峰`
- `夜间`

对：
- `h1`
- `seq6`

分别分析。

必交：
- `outputs/hourly/hourly_error_by_hour.csv`
- `outputs/hourly/hourly_error_by_dayperiod.csv`

---

## 9.6 跨城市泛化

训练：
- 北京数据

测试：
- 上海对齐小时级数据

建议至少做：
- `XGBoost h1`
- `XGBoost seq6`

必交：
- `outputs/hourly/cross_city_metrics.csv`
- `outputs/hourly/cross_city_predictions.csv`

---

## 10. 图表清单

图统一输出到：
- `outputs/hourly/figures/`

建议图表：

1. `01_prediction_curve_h1.png`
2. `02_prediction_curve_seq6.png`
3. `03_overall_metrics_bar.png`
4. `04_xgboost_feature_importance_h1.png`
5. `05_xgboost_feature_importance_seq6.png`
6. `06_aqi_bucket_error_heatmap.png`
7. `07_feature_ablation_comparison.png`
8. `08_high_pollution_error.png`
9. `09_error_by_hour_curve.png`
10. `10_dayperiod_error_bar.png`
11. `11_cross_city_generalization.png`

---

## 11. 必交结果文件清单

### 11.1 模型预测结果

- `arima_predictions_h1.csv`
- `arima_predictions_seq6.csv`
- `prophet_predictions_h1.csv`
- `prophet_predictions_seq6.csv`
- `xgboost_predictions_h1.csv`
- `xgboost_predictions_seq6.csv`
- `lstm_predictions_h1.csv`
- `lstm_predictions_seq6.csv`
- `transformer_predictions_h1.csv`
- `transformer_predictions_seq6.csv`

### 11.2 模型指标文件

- `arima_metrics.csv`
- `prophet_metrics.csv`
- `xgboost_metrics.csv`
- `lstm_metrics.csv`
- `transformer_metrics.csv`

### 11.3 汇总与分析文件

- `overall_metrics_summary.csv`
- `feature_ablation.csv`
- `aqi_bucket_metrics.csv`
- `high_pollution_error_analysis.csv`
- `hourly_error_by_hour.csv`
- `hourly_error_by_dayperiod.csv`
- `cross_city_metrics.csv`
- `cross_city_predictions.csv`

### 11.4 配置与训练历史

- `xgboost_h1_best_params.json`
- `xgboost_seq6_best_params.json`
- `arima_best_order_h1.json`
- `arima_best_order_seq6.json`
- `prophet_config_h1.json`
- `prophet_config_seq6.json`
- `lstm_h1_config.json`
- `lstm_seq6_config.json`
- `transformer_h1_config.json`
- `transformer_seq6_config.json`
- `lstm_train_history.json`
- `transformer_train_history.json`
- `experiment_registry.csv`
- `analysis_notes.md`

---

## 12. 推荐执行顺序

1. 确认输入产物与特征矩阵
2. 先保留现有 `h1`
3. 先完成 `XGBoost seq6`
4. 再补 `ARIMA seq6`
5. 再补 `Prophet seq6`
6. 再补 `LSTM seq6`
7. 再补 `Transformer seq6`
8. 做特征消融
9. 做 AQI / 高污染 / 时段分析
10. 做跨城市泛化
11. 统一出图与总表

---

## 13. 最终需要回答的问题

1. `h1` 上哪个模型最好？
2. 未来 6 小时中，误差会不会随着步长 `1 -> 6` 逐渐上升？
3. 哪些特征对未来 6 小时轨迹预测最重要？
4. 高污染场景下，多步预测误差是否显著放大？
5. 白天 / 夜晚 / 高峰时段，多步预测的误差结构是否不同？
6. 北京训练的模型迁移到上海后，多步预测还能保持多少有效性？

---

## 14. 一句话执行结论

本版小时级实验主线调整为：

- `+1h` 单点预测
- `未来6小时逐小时预测`
- `统一48维特征`
- `5模型统一口径对比`
- `误差分层 + 特征消融 + 时段分析 + 泛化测试`

作为最终可提交、可答辩、可复现的小时级 PM2.5 预测方案。
