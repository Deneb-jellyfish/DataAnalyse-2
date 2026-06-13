# 小时级 PM2.5 预测实验方案（正式执行版）

本文档用于直接交付给负责建模与实验实现的同学，作为后续小时级 PM2.5 预测主线实验的统一执行说明。

本方案对应课程作业“选项 B：算法系统性对比研究”，目标是在**同一预测任务**上完成**至少 5 种不同类型算法**的系统比较，并补足误差分析、特征分析、场景分析与可视化交付。

---

## 1. 为什么要重做这一版

第一次小时级实验已经说明两个事实：

1. 小时级数据是对的，方向也对。
   旧版 `hour-experiment` 分支已经证明小时级建模比原来的日级更合理，数据量足够，`XGBoost +1h` 也已经能跑出可用结果。

2. 旧版 `+24h` 任务不适合作为当前主扩展任务。
   原因不是“模型完全不会做”，而是旧版存在以下问题：

- `+24h` 信息跨度过长，现有输入主要是历史观测，没有未来气象预报，导致长步长预测普遍退化。
- `ARIMA` 在旧版主结果里只抽样评估部分测试点，不能与其他模型直接公平比较。
- `Prophet` 旧版实际上退化成 STL fallback，不是真正的 Prophet，不能直接作为正式对比结果。
- `LSTM/Transformer` 旧版没有把双任务完整跑齐，实验矩阵没有闭环。
- `experiment_registry.csv` 旧版记录不完整，范围、参数、口径混乱，不利于复现和答辩。

因此本次正式方案做如下调整：

- 保留 `+1h` 作为主任务。
- 将原计划中的 `+24h` 改为 `+12h`，作为中短期扩展任务。
- 强调**统一评估口径、统一输出格式、统一实验记录**。
- 在模型数量满足要求的前提下，追求“任务聚焦 + 分析深入”，而不是盲目堆很多 horizon。

---

## 2. 本次正式实验目标

本次实验围绕两个任务展开：

### 任务 A：未来 1 小时 PM2.5 预测

- 输入：时刻 `t` 及之前的小时级污染与气象信息
- 输出：`pm25(t+1)`
- 作用：衡量模型对超短期污染变化的跟踪能力

### 任务 B：未来 12 小时 PM2.5 预测

- 输入：时刻 `t` 及之前的小时级污染与气象信息
- 输出：`pm25(t+12)`
- 作用：衡量模型对中短期污染演化的预测能力

### 为什么从 `24h` 改成 `12h`

- `12h` 仍然有明显难度，不是简单任务。
- `12h` 仍能体现“预测步长拉长后性能下降”的规律。
- `12h` 仍受日内周期、昼夜切换、气象演化影响，具有分析价值。
- 相比 `24h`，`12h` 对当前这套历史观测特征更友好，更有希望把 5 个模型都做成“能分析、能交付、能答辩”的结果。

---

## 3. 参与对比的模型

正式主对比必须包含以下 5 个模型：

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
| XGBoost | 特征工程型机器学习 | 当前最有希望的主力模型 |
| LSTM | 循环神经网络 | 深度学习时序代表 |
| Transformer | 注意力模型 | 深度学习对照模型 |

### 重要说明

- `Prophet` 必须是真正的 Prophet。
- 如果本机环境中 Prophet 装不上或运行失败，必须在实验记录中明确写为“阻塞”，不能继续拿 STL fallback 冒充 Prophet 主结果。
- 如果后续确实想保留 STL，可单独作为附录模型，命名必须写成 `STL-Fallback`，不能写成 `Prophet`。

---

## 4. 本次实验使用的数据与代码文件

### 4.1 主数据文件

- [data/processed/beijing_hourly.csv](/d:/数据挖掘期末/data/processed/beijing_hourly.csv)

字段包括：

- `datetime`
- `city`
- `pm25`
- `pm10`
- `so2`
- `no2`
- `co`
- `o3`
- `temp`
- `pres`
- `dewp`
- `humidity`
- `wind_dir`
- `wind_speed`
- `precipitation`

### 4.2 小时级原始清洗结果参考

- [data/processed/beijing/beijing_hourly_city.csv](/d:/数据挖掘期末/data/processed/beijing/beijing_hourly_city.csv)

### 4.3 当前小时级特征工程代码

- [utils/feature_engineering.py](/d:/数据挖掘期末/utils/feature_engineering.py)
- [docs/feature_doc_hourly.md](/d:/数据挖掘期末/docs/feature_doc_hourly.md)

### 4.4 当前特征产物目录

- [data/processed/features_hourly](/d:/数据挖掘期末/data/processed/features_hourly)

其中已有：

- `X_train.npy`
- `X_test.npy`
- `y_train.npy`
- `y_test.npy`
- `dates_train.npy`
- `dates_test.npy`
- `feature_names.json`
- `scaler.pkl`

### 4.5 跨城市泛化测试数据

- [data/processed/aligned_hourly/shanghai.csv](/d:/数据挖掘期末/data/processed/aligned_hourly/shanghai.csv)

---

## 5. 当前特征工程基础（已完成）

本轮特征工程已经从旧版 20 维升级到**48 维**，重点针对 `+12h` 做了增强。

### 5.1 旧版保留的核心特征

- 短期滞后：`pm25_lag1`、`pm25_lag3`、`pm25_lag24`
- 长周期统计：`pm25_roll_mean_24`、`pm25_roll_mean_168`
- 周期特征：`hour_sin`、`hour_cos`、`month_sin`、`month_cos`
- 基础气象：`temp`、`pres`、`dewp`、`humidity`、`wind_speed`、`precipitation`

### 5.2 为 `+12h` 新增的重点特征

#### 中尺度滞后

- `pm25_lag6`
- `pm25_lag12`
- `pm25_lag18`
- `pm25_lag36`
- `pm25_lag48`

#### 中尺度滚动统计

- `pm25_roll_mean_6`
- `pm25_roll_mean_12`
- `pm25_roll_std_6`
- `pm25_roll_std_12`

#### 趋势变化特征

- `pm25_diff_1`
- `pm25_diff_3`
- `pm25_diff_6`
- `pm25_diff_12`

#### 昼夜/时段特征

- `is_weekend`
- `is_daytime`
- `is_rush_hour`

#### 气象变化特征

- `temp_diff_1`
- `temp_diff_3`
- `pres_diff_1`
- `pres_diff_3`
- `wind_speed_diff_1`
- `wind_speed_diff_3`

#### 风向与交互特征

- `wind_dir_sin`
- `wind_dir_cos`
- `dewp_temp_gap`
- `precipitation_flag`
- `wind_speed_x_pm25_lag1`

### 5.3 当前样本规模

根据重新生成后的当前产物：

- 有效样本数：`26352`
- 训练集：`21081`
- 测试集：`5271`
- 特征数：`48`

说明：新增特征后没有额外损失有效样本，这一点很好，可以直接作为正式实验基础。

---

## 6. 统一实验规则（必须遵守）

这一部分是本次重做最重要的地方，用来避免第一次实验再次出现口径问题。

### 6.1 所有模型必须使用同一任务定义

- `+1h` 统一定义为：预测 `pm25(t+1)`
- `+12h` 统一定义为：预测 `pm25(t+12)`

### 6.2 所有模型必须使用统一的评估时间点

- 最终主表中的各模型必须在同一批 `target_time` 上比较。
- 不能再出现某个模型只评估一部分测试点、另一个模型评估全量测试点的情况。
- `ARIMA` 如果因时间过长只能抽样评估，则该结果不能进入主榜单，只能作为附录。

### 6.3 所有模型必须采用直接预测

本次正式版**不做递归 12 步预测**，统一采用：

- 输入时刻 `t` 以前的信息
- 直接预测 `t+1` 或 `t+12`

原因：

- 避免误差逐步累积
- 方便所有模型公平比较
- 更适合形成统一输出格式

### 6.4 所有模型统一输出格式

所有 `*_predictions_*.csv` 必须统一字段：

```csv
forecast_origin_time,target_time,horizon_hours,city,model,y_true,y_pred,split
```

所有 `*_metrics.csv` 必须统一字段：

```csv
model,horizon_hours,rmse,mae,mape,n_samples,notes
```

### 6.5 实验记录必须完整

所有正式实验都必须同步更新：

- `outputs/hourly/experiment_registry.csv`
- `outputs/hourly/analysis_notes.md`

其中 `experiment_registry.csv` 建议字段为：

```csv
experiment_id,model,horizon_hours,feature_set,train_range,val_range,test_range,params,best_score,run_time,status,author,notes
```

---

## 7. 数据切分与验证方案

### 7.1 总体切分

继续沿用当前小时级特征工程的时间顺序切分：

- 前 `80%`：训练池
- 后 `20%`：测试集

参考时间范围：

- 训练：`2013-03-14 00:00:00 -> 2016-05-14 08:00:00`
- 测试：`2016-05-14 09:00:00 -> 2017-02-26 23:00:00`

### 7.2 调参时的验证集划分

在训练池内部再按时间顺序划分：

- 前 `80%`：子训练集
- 后 `20%`：验证集

也就是整个数据的大致结构为：

- `64%`：训练
- `16%`：验证
- `20%`：测试

### 7.3 序列模型的对齐说明

LSTM / Transformer 由于有 `seq_len`，可用样本数会略少于树模型，这是正常现象。

但正式比较时必须保证：

- 输出的 `target_time` 明确可追踪
- 最终总表可通过 `target_time` 交集进行公平比较

---

## 8. 每个模型怎么做

## 8.1 XGBoost

### 任务定义

- `h1`：目标列 `pm25_target_h1 = pm25.shift(-1)`
- `h12`：目标列 `pm25_target_h12 = pm25.shift(-12)`

### 输入文件

- [data/processed/beijing_hourly.csv](/d:/数据挖掘期末/data/processed/beijing_hourly.csv)
- [utils/feature_engineering.py](/d:/数据挖掘期末/utils/feature_engineering.py)

### 实现方式

1. 读取 `beijing_hourly.csv`
2. 调用 `build_hourly_feature_frame(df)` 生成 48 维特征
3. 分别构造 `h1` 与 `h12` 的目标列
4. 删除因目标平移带来的尾部空值
5. 按时间顺序切分 train/val/test
6. 训练两个独立模型：`XGBoost+h1`、`XGBoost+h12`

### 调参建议

先用较小网格，再做局部细化：

| 参数 | 第一轮范围 |
|---|---|
| `max_depth` | `4, 6, 8` |
| `n_estimators` | `200, 400, 800` |
| `learning_rate` | `0.03, 0.05, 0.1` |
| `subsample` | `0.8, 1.0` |
| `colsample_bytree` | `0.8, 1.0` |
| `min_child_weight` | `1, 3, 5` |
| `reg_lambda` | `1, 3, 5` |

调参方法：

- 优先使用验证集 early stopping
- 如时间允许，可再加 `TimeSeriesSplit`

### 必交文件

- `outputs/hourly/xgboost_predictions_h1.csv`
- `outputs/hourly/xgboost_predictions_h12.csv`
- `outputs/hourly/xgboost_metrics.csv`
- `outputs/hourly/xgboost_feature_importance_h1.csv`
- `outputs/hourly/xgboost_feature_importance_h12.csv`
- `outputs/hourly/xgboost_h1_best_params.json`
- `outputs/hourly/xgboost_h12_best_params.json`

### 备注

XGBoost 是本次主力模型，后续的特征消融、特征重要性、误差分层分析都优先围绕它展开。

---

## 8.2 ARIMA

### 任务定义

- `h1`：直接预测 `t+1`
- `h12`：直接预测 `t+12`

### 输入文件

- [data/processed/beijing_hourly.csv](/d:/数据挖掘期末/data/processed/beijing_hourly.csv)

### 实现原则

- 仅使用 `pm25` 单变量序列
- 必须输出完整测试区间上的预测结果
- 正式主表禁止“每 6 个点测一次”这种抽样评估

### 参数搜索建议

先做小网格搜索：

- `p ∈ {1, 3, 6, 12}`
- `d ∈ {0, 1}`
- `q ∈ {0, 1, 3}`

如果可承受，可尝试季节项：

- `P ∈ {0, 1}`
- `D ∈ {0, 1}`
- `Q ∈ {0, 1}`
- `s = 24`

选择依据：

- 先看验证集 RMSE
- 再参考 AIC / BIC

### 必交文件

- `outputs/hourly/arima_predictions_h1.csv`
- `outputs/hourly/arima_predictions_h12.csv`
- `outputs/hourly/arima_metrics.csv`
- `outputs/hourly/arima_best_order_h1.json`
- `outputs/hourly/arima_best_order_h12.json`

### 备注

ARIMA 主要作为传统统计基线，不要求它一定最优，但要求口径公平、结果可解释。

---

## 8.3 Prophet

### 任务定义

- `h1`
- `h12`

### 输入文件

- [data/processed/beijing_hourly.csv](/d:/数据挖掘期末/data/processed/beijing_hourly.csv)

### 实现方式

1. `datetime -> ds`
2. `pm25 -> y`
3. 在训练集上拟合 Prophet
4. 使用未来时间框架得到预测
5. 通过 `target_time` 对齐回测试集真实值

### 推荐配置

- 开启日周期：`daily_seasonality=True`
- 视情况开启周周期：`weekly_seasonality=True`
- 如实现顺利，可加入外生回归量：
  - `temp`
  - `pres`
  - `humidity`
  - `wind_speed`

### 必须注意

- 这次必须验证 Prophet 是否真正安装并可运行
- 如果还是 fallback，就不能把它放进正式主对比结果
- 阻塞也要记录，不能静默跳过

### 必交文件

- `outputs/hourly/prophet_predictions_h1.csv`
- `outputs/hourly/prophet_predictions_h12.csv`
- `outputs/hourly/prophet_metrics.csv`
- `outputs/hourly/prophet_config_h1.json`
- `outputs/hourly/prophet_config_h12.json`

---

## 8.4 LSTM

### 任务定义

- `h1`
- `h12`

### 输入文件

- [data/processed/beijing_hourly.csv](/d:/数据挖掘期末/data/processed/beijing_hourly.csv)
- [utils/feature_engineering.py](/d:/数据挖掘期末/utils/feature_engineering.py)

### 建议序列方案

| 任务 | 建议 `seq_len` |
|---|---|
| `h1` | `48` |
| `h12` | `72` 或 `96` |

### 网络与训练建议

第一轮建议：

- `hidden_dim`: `64` 或 `128`
- `num_layers`: `2`
- `dropout`: `0.1` 或 `0.2`
- `batch_size`: `64`
- `lr`: `1e-3` 或 `5e-4`
- `epochs`: `80~150`
- `patience`: `10~15`

### 调参优先级

先调这几个，不要一开始就乱改很多：

1. `seq_len`
2. `hidden_dim`
3. `lr`
4. `dropout`

### 必交文件

- `outputs/hourly/lstm_predictions_h1.csv`
- `outputs/hourly/lstm_predictions_h12.csv`
- `outputs/hourly/lstm_metrics.csv`
- `outputs/hourly/lstm_train_history.json`
- `outputs/hourly/lstm_h1_config.json`
- `outputs/hourly/lstm_h12_config.json`

---

## 8.5 Transformer

### 任务定义

- `h1`
- `h12`

### 输入文件

- 同 LSTM

### 建议序列方案

| 任务 | 建议 `seq_len` |
|---|---|
| `h1` | `48` |
| `h12` | `72` 或 `96` |

### 第一轮推荐参数

- `d_model`: `64`
- `nhead`: `4`
- `num_layers`: `2` 或 `3`
- `dim_feedforward`: `128` 或 `256`
- `dropout`: `0.1`
- `batch_size`: `64`
- `lr`: `1e-3`
- `patience`: `10~15`

### 调参优先级

1. `seq_len`
2. `d_model`
3. `num_layers`
4. `lr`

### 必交文件

- `outputs/hourly/transformer_predictions_h1.csv`
- `outputs/hourly/transformer_predictions_h12.csv`
- `outputs/hourly/transformer_metrics.csv`
- `outputs/hourly/transformer_train_history.json`
- `outputs/hourly/transformer_h1_config.json`
- `outputs/hourly/transformer_h12_config.json`

---

## 9. 必做深入分析模块

## 9.1 总体指标主表

生成：

- `outputs/hourly/overall_metrics_summary.csv`

字段建议：

```csv
model,horizon_hours,rmse,mae,mape,n_samples,notes
```

该表是最终 PPT 和 README 的总表来源。

---

## 9.2 特征消融（重点做 XGBoost）

本次特征工程已经扩到 48 维，不做消融就无法说明“为什么改进有效”。

至少做 5 组：

1. `base_short_lag`
   - `lag1/lag3`
   - 基础气象
2. `base_plus_daily_cycle`
   - 上一组 + `hour_sin/hour_cos`
3. `mid_range_memory`
   - 上一组 + `lag6/lag12/lag18`
4. `trend_and_roll`
   - 上一组 + `roll6/roll12/roll24` + `diff`
5. `full_48`
   - 全部特征

分别在：

- `h1`
- `h12`

两个任务下比较。

必交：

- `outputs/hourly/feature_ablation.csv`

---

## 9.3 AQI 分层误差分析

根据真实 `pm25` 分层，统计各模型误差。

建议分层：

- `0-35`
- `35-75`
- `75-115`
- `115-150`
- `>150`

必交：

- `outputs/hourly/aqi_bucket_metrics.csv`

---

## 9.4 高污染场景误差分析

单独筛选：

- `pm25 > 150`

观察不同模型在高污染样本上的误差是否显著放大。

必交：

- `outputs/hourly/high_pollution_error_analysis.csv`

---

## 9.5 昼夜/时段误差分析

这部分是小时级实验必须保留的亮点分析。

至少做两层：

### 按小时统计

- `0~23` 每个小时分别统计误差

### 按时段统计

建议分成：

- `凌晨`：`00:00-05:59`
- `早高峰`：`06:00-09:59`
- `白天平峰`：`10:00-16:59`
- `晚高峰`：`17:00-20:59`
- `夜间`：`21:00-23:59`

必交：

- `outputs/hourly/hourly_error_by_hour.csv`
- `outputs/hourly/hourly_error_by_dayperiod.csv`

---

## 9.6 跨城市泛化（建议至少做 XGBoost）

训练：

- 北京数据

测试：

- 上海对齐小时级数据

建议至少做：

- `XGBoost +1h`
- `XGBoost +12h`

如时间允许再补：

- `LSTM +1h`

必交：

- `outputs/hourly/cross_city_metrics.csv`
- `outputs/hourly/cross_city_predictions.csv`

---

## 10. 必交图表清单

所有图建议统一输出到：

- `outputs/hourly/figures/`

### 图 1：`+1h` 多模型预测曲线对比

- 文件名：`01_prediction_curve_h1.png`

### 图 2：`+12h` 多模型预测曲线对比

- 文件名：`02_prediction_curve_h12.png`

### 图 3：整体误差对比柱状图

- 文件名：`03_overall_metrics_bar.png`
- 内容：5 个模型在 `h1/h12` 下的 `RMSE/MAE/MAPE`

### 图 4：XGBoost 特征重要性（`h1`）

- 文件名：`04_xgboost_feature_importance_h1.png`

### 图 5：XGBoost 特征重要性（`h12`）

- 文件名：`05_xgboost_feature_importance_h12.png`

### 图 6：AQI 分层误差热力图

- 文件名：`06_aqi_bucket_error_heatmap.png`

### 图 7：特征消融对比图

- 文件名：`07_feature_ablation_comparison.png`

### 图 8：高污染场景误差图

- 文件名：`08_high_pollution_error.png`

### 图 9：按小时误差变化曲线

- 文件名：`09_error_by_hour_curve.png`

### 图 10：按时段误差柱状图

- 文件名：`10_dayperiod_error_bar.png`

### 图 11：跨城市泛化结果图

- 文件名：`11_cross_city_generalization.png`

---

## 11. 必交结果文件清单

## 11.1 模型预测结果

- `arima_predictions_h1.csv`
- `arima_predictions_h12.csv`
- `prophet_predictions_h1.csv`
- `prophet_predictions_h12.csv`
- `xgboost_predictions_h1.csv`
- `xgboost_predictions_h12.csv`
- `lstm_predictions_h1.csv`
- `lstm_predictions_h12.csv`
- `transformer_predictions_h1.csv`
- `transformer_predictions_h12.csv`

## 11.2 模型指标文件

- `arima_metrics.csv`
- `prophet_metrics.csv`
- `xgboost_metrics.csv`
- `lstm_metrics.csv`
- `transformer_metrics.csv`

## 11.3 汇总与分析文件

- `overall_metrics_summary.csv`
- `feature_ablation.csv`
- `aqi_bucket_metrics.csv`
- `high_pollution_error_analysis.csv`
- `hourly_error_by_hour.csv`
- `hourly_error_by_dayperiod.csv`
- `cross_city_metrics.csv`
- `cross_city_predictions.csv`

## 11.4 配置与训练历史

- `xgboost_h1_best_params.json`
- `xgboost_h12_best_params.json`
- `arima_best_order_h1.json`
- `arima_best_order_h12.json`
- `prophet_config_h1.json`
- `prophet_config_h12.json`
- `lstm_h1_config.json`
- `lstm_h12_config.json`
- `transformer_h1_config.json`
- `transformer_h12_config.json`
- `lstm_train_history.json`
- `transformer_train_history.json`
- `experiment_registry.csv`
- `analysis_notes.md`

---

## 12. 推荐执行顺序

### 第 0 步：确认输入产物

确认以下文件存在且可读：

- [data/processed/beijing_hourly.csv](/d:/数据挖掘期末/data/processed/beijing_hourly.csv)
- [data/processed/features_hourly/feature_names.json](/d:/数据挖掘期末/data/processed/features_hourly/feature_names.json)
- [docs/feature_doc_hourly.md](/d:/数据挖掘期末/docs/feature_doc_hourly.md)

### 第 1 步：先重做 XGBoost `h1`

目标：

- 用新 48 维特征重跑
- 拿到新的 `h1` 基准线

### 第 2 步：做 XGBoost `h12`

目标：

- 用直接预测 `t+12` 的方式建立主扩展任务
- 先验证这版改造是否真的优于旧的 `24h`

### 第 3 步：补 ARIMA `h1/h12`

要求：

- 全量评估
- 不允许抽样口径混入主榜

### 第 4 步：补 Prophet `h1/h12`

要求：

- 必须是真 Prophet
- 不成功就明确记录阻塞

### 第 5 步：补 LSTM `h1/h12`

### 第 6 步：补 Transformer `h1/h12`

### 第 7 步：做 XGBoost 特征消融

### 第 8 步：做 AQI 分层、高污染、时段分析

### 第 9 步：做跨城市泛化

### 第 10 步：统一出图、汇总总表、整理实验记录

---

## 13. 实验记录要求

每次正式实验必须在 `analysis_notes.md` 中写清楚：

1. 本次实验目标
2. 使用的数据和任务
3. 参数改动
4. 特征改动
5. 验证集指标
6. 测试集指标
7. 相比上一版是变好还是变坏
8. 原因判断
9. 下一步建议

这样做的目的不是形式化，而是确保后续任何成员接手时，不需要重新猜整个实验链条。

---

## 14. 最终需要回答的核心问题

这次实验最后不能只给一堆图和表，而要明确回答：

1. 小时级 PM2.5 预测中，哪个模型在 `+1h` 上整体最好？
2. 将 horizon 从 `1h` 拉长到 `12h` 后，性能下降有多明显？
3. 新增的中尺度特征是否真的提升了 `+12h` 预测？
4. 哪些特征对 `+12h` 最重要？
5. 高污染场景是否显著放大预测误差？
6. 白天、夜晚、早晚高峰的误差是否存在稳定差异？
7. 北京训练的模型迁移到上海后还能保持多少有效性？

---

## 15. 最低完成标准 / 推荐完成标准

### 最低完成标准

- 5 个模型全部完成 `h1`
- `XGBoost + ARIMA + LSTM` 至少完成 `h12`
- 主表、主图、实验记录齐全

### 推荐完成标准

- 5 个模型全部完成 `h1 + h12`
- 完成 XGBoost 特征消融
- 完成 AQI 分层与时段误差分析
- 完成 XGBoost 跨城市泛化

### 理想完成标准

- 5 个模型双任务全部完成
- 深度模型与树模型都有较完整分析
- 图、表、CSV、记录可直接用于 README、PPT、报告与 Streamlit 展示

---

## 16. 一句话执行结论

这次正式版实验不再以旧 `+24h` 为主线，而是以：

- `+1h` 主任务
- `+12h` 扩展任务
- `48 维新特征`
- `5 模型统一口径对比`
- `误差分层 + 特征消融 + 时段分析 + 泛化测试`

作为最终可提交、可答辩、可复现的小时级 PM2.5 实验主线。
