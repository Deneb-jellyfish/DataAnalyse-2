# 小时级 PM2.5 预测实验方案（算法系统性对比研究）

本文档用于直接交付给负责实验实现的同学，作为后续小时级建模主线的执行说明。  
本方案对应课程作业要求中的“**选项 B：算法系统性对比研究**”，目标是在**同一预测任务**上实现并对比**至少 5 种不同类型算法**，并进行较深入的理论与实验分析。

---

## 1. 实验目标

将当前项目主线从“日级 PM2.5 下一天预测”调整为“**小时级 PM2.5 预测**”，并围绕同一任务对多种算法进行系统比较。  
本次实验强调：

1. 使用**小时级数据**，避免日级数据样本量不足的问题。
2. 保证至少 **5 个模型**参与同任务对比。
3. 实验不追求铺很多 horizon，而追求**任务聚焦 + 模型丰富 + 分析深入**。
4. 除整体误差外，还要加入**特征消融、AQI 分层误差、高污染场景分析、跨城市泛化**等更有深度的内容。

---

## 2. 主任务定义

本次主实验只做两个预测任务，不再扩太多预测步长：

### 任务 A：未来 1 小时 PM2.5 预测

- 输入：当前时刻之前一段时间的小时级污染与气象信息
- 输出：未来 `+1h` 的 `PM2.5`
- 任务意义：考察模型对超短期污染变化的跟踪能力

### 任务 B：未来 24 小时 PM2.5 预测

- 输入：当前时刻之前一段时间的小时级污染与气象信息
- 输出：未来 `+24h` 的 `PM2.5`
- 任务意义：考察模型对提前一天预警的能力

### 为什么只做 `+1h` 和 `+24h`

不再做 `+3h/+6h/+12h/+24h` 一整排 horizon，原因是：

1. 当前要求重点是“系统性对比至少 5 种算法”，不是横向铺满很多预测步长。
2. `+1h` 和 `+24h` 已经足够体现短期预测与提前预警的差异。
3. 这样可以把时间放到更有深度的分析上，如特征作用、AQI 分层、高污染时段、跨城市泛化等。

---

## 3. 参与对比的模型

必须至少完成以下 5 个模型：

1. `ARIMA`
2. `Prophet`
3. `XGBoost`
4. `LSTM`
5. `Transformer`

可选加分模型：

6. `Informer`

### 模型角色定位

- `ARIMA`：传统统计时间序列模型基线
- `Prophet`：趋势 + 周期分解模型
- `XGBoost`：特征工程型机器学习主力模型
- `LSTM`：循环神经网络代表
- `Transformer`：注意力模型代表
- `Informer`：如时间允许，可作为更长序列建模加分项

本次实验推荐将 **XGBoost 作为主结果模型**，但其余 4 个模型必须完整参与对比。

---

## 4. 数据与文件使用说明

### 4.1 主实验数据（北京小时级）

主建模数据使用：

- [data/processed/beijing_hourly.csv](/d:/数据挖掘期末/data/processed/beijing_hourly.csv)

该文件为统一字段后的北京小时级主实验数据。

字段结构（统一格式）：

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

### 4.2 原始小时级参考数据

如果需要核对原始北京城市小时级表，可参考：

- [data/processed/beijing/beijing_hourly_city.csv](/d:/数据挖掘期末/data/processed/beijing/beijing_hourly_city.csv)

### 4.3 小时级特征矩阵

小时级特征工程产物目录：

- [data/processed/features_hourly](/d:/数据挖掘期末/data/processed/features_hourly)

其中包括：

- `X_train.npy`
- `X_test.npy`
- `y_train.npy`
- `y_test.npy`
- `dates_train.npy`
- `dates_test.npy`
- `feature_names.json`
- `scaler.pkl`

### 4.4 上海小时级补充数据（用于跨城市实验）

建议用于迁移/泛化测试的数据：

- [data/processed/aligned_hourly/shanghai.csv](/d:/数据挖掘期末/data/processed/aligned_hourly/shanghai.csv)

如果需要查看上海 overlap 清洗结果，也可以参考：

- [data/processed/five_city/shanghai_hourly_overlap_20130301_20151231.csv](/d:/数据挖掘期末/data/processed/five_city/shanghai_hourly_overlap_20130301_20151231.csv)

---

## 5. 当前小时级特征工程基础

项目里已经设计并生成了小时级特征，参考文档：

- [docs/feature_doc_hourly.md](/d:/数据挖掘期末/docs/feature_doc_hourly.md)
- [utils/feature_engineering.py](/d:/数据挖掘期末/utils/feature_engineering.py)

当前小时级特征包括：

### 滞后特征

- `pm25_lag1`
- `pm25_lag3`
- `pm25_lag24`

### 滚动统计特征

- `pm25_roll_mean_24`
- `pm25_roll_mean_168`
- `pm25_roll_std_24`
- `pm25_roll_std_168`

### 时间周期特征

- `hour_sin`
- `hour_cos`
- `month_sin`
- `month_cos`
- `weekday`
- `is_holiday`

### 气象特征

- `temp`
- `pres`
- `dewp`
- `wind_speed`
- `precipitation`

### 交互特征

- `temp_x_humidity`
- `wind_speed_x_wind_dir_sin`

### 当前小时级样本规模

根据现有特征工程结果：

- 有效样本数：`26352`
- 训练集：`21081`
- 测试集：`5271`
- 特征数：`20`

这个规模已经明显优于原先日级实验，可以作为主建模基础。

---

## 6. 训练/验证/测试划分规则

### 6.1 基本原则

必须按**时间顺序**切分，禁止随机打乱。

### 6.2 当前已有切分

小时级特征工程默认采用：

- 前 `80%`：训练集
- 后 `20%`：测试集

训练测试时间范围可直接读取：

- `dates_train.npy`
- `dates_test.npy`

### 6.3 验证集建议

如果模型需要调参，建议在训练集内部再按时间顺序切一个验证集：

- 前 `64%`：训练
- 中间 `16%`：验证
- 后 `20%`：测试

树模型调参建议：

- 使用 `TimeSeriesSplit`

深度模型调参建议：

- 在训练集末尾切验证集，不允许随机抽样

---

## 7. 各模型实施方案

下面按模型分别说明建议实现方式。

---

### 7.1 ARIMA

#### 任务

- `+1h`：对 `pm25` 小时序列做单步滚动预测
- `+24h`：做 24 小时 ahead 预测

#### 输入

- 使用 [data/processed/beijing_hourly.csv](/d:/数据挖掘期末/data/processed/beijing_hourly.csv)
- 主要目标列：`pm25`

#### 方法建议

- `+1h`：参考现有 [models/arima_model.py](/d:/数据挖掘期末/models/arima_model.py) 的思路改成小时级版本
- `+24h`：可采用固定 horizon 预测或递归 24 次的方式

#### 输出文件

- `outputs/hourly/arima_predictions_h1.csv`
- `outputs/hourly/arima_predictions_h24.csv`
- `outputs/hourly/arima_metrics.csv`

---

### 7.2 Prophet

#### 任务

- `+1h`
- `+24h`

#### 输入

- 使用小时级 `datetime` + `pm25`

#### 方法建议

- `datetime` 改成 Prophet 所需的 `ds`
- `pm25` 改成 `y`
- 可保留趋势和周期建模
- `+24h` 可通过未来时间框架预测实现

#### 注意

- 如果本地 Prophet 依赖环境不稳定，要优先保证可运行性
- 若确实无法在该机器稳定运行，需明确记录原因，不能直接空缺

#### 输出文件

- `outputs/hourly/prophet_predictions_h1.csv`
- `outputs/hourly/prophet_predictions_h24.csv`
- `outputs/hourly/prophet_metrics.csv`

---

### 7.3 XGBoost

#### 任务

- `+1h`
- `+24h`

#### 输入

- 主输入直接使用 `features_hourly/` 下的矩阵
- 如果要做 `+24h`，需要重新构造目标列 `pm25.shift(-24)` 并生成对应特征矩阵

#### 方法建议

- `+1h`：直接使用现有 `features_hourly` 产物
- `+24h`：基于同一小时级特征集重新构建目标标签
- 训练一个 `+1h` 模型和一个 `+24h` 模型
- 作为主结果模型，必须做完整误差分析与特征重要性分析

#### 输出文件

- `outputs/hourly/xgboost_predictions_h1.csv`
- `outputs/hourly/xgboost_predictions_h24.csv`
- `outputs/hourly/xgboost_metrics.csv`
- `outputs/hourly/xgboost_feature_importance_h1.csv`
- `outputs/hourly/xgboost_feature_importance_h24.csv`

---

### 7.4 LSTM

#### 任务

- `+1h`
- `+24h`

#### 输入

- 从小时级统一数据构造滑动窗口

#### 建议窗口

- 对 `+1h`：输入过去 `24` 小时
- 对 `+24h`：输入过去 `48` 或 `72` 小时

#### 方法建议

- 不建议直接做非常大的 seq2seq
- 第一版优先做：
  - `24 -> 1`
  - `48/72 -> 1`（目标是 `t+24`）
- 这样更稳，更适合当前样本规模

#### 输出文件

- `outputs/hourly/lstm_predictions_h1.csv`
- `outputs/hourly/lstm_predictions_h24.csv`
- `outputs/hourly/lstm_metrics.csv`
- `outputs/hourly/lstm_train_history.json`

---

### 7.5 Transformer

#### 任务

- `+1h`
- `+24h`

#### 输入

- 小时级滑动窗口

#### 方法建议

- 与 LSTM 采用同一任务定义，保证公平对比
- 不要先做复杂多输出结构
- 先做固定 horizon 单目标预测即可

#### 输出文件

- `outputs/hourly/transformer_predictions_h1.csv`
- `outputs/hourly/transformer_predictions_h24.csv`
- `outputs/hourly/transformer_metrics.csv`
- `outputs/hourly/transformer_train_history.json`

---

### 7.6 Informer（可选加分）

#### 任务

- `+1h`
- `+24h`

#### 说明

- 若时间不够，可暂不纳入正式五模型主对比
- 若实现，则单独作为第六模型加分项

#### 输出文件

- `outputs/hourly/informer_predictions_h1.csv`
- `outputs/hourly/informer_predictions_h24.csv`
- `outputs/hourly/informer_metrics.csv`
- `outputs/hourly/informer_train_history.json`

---

## 8. 必做的深入分析

这是本次实验“深刻性”的核心，不能只停留在跑 5 个模型和报 RMSE。

---

### 8.1 整体指标对比

每个模型在两个任务下都必须输出：

- `RMSE`
- `MAE`
- `MAPE`
- 样本数

最终形成总表：

- `outputs/hourly/overall_metrics_summary.csv`

建议字段：

```csv
model,horizon_hours,rmse,mae,mape,n_samples
ARIMA,1,...
ARIMA,24,...
XGBoost,1,...
XGBoost,24,...
```

---

### 8.2 特征消融实验（重点针对 XGBoost）

至少做以下 4 组特征集：

1. 基础污染物 + 气象特征
2. 基础特征 + 时间周期特征
3. 基础特征 + 时间周期 + lag 特征
4. 基础特征 + 时间周期 + lag + rolling + 交互特征

分别在：

- `+1h`
- `+24h`

两个任务下比较指标变化。

#### 输出文件

- `outputs/hourly/feature_ablation.csv`

建议字段：

```csv
feature_set,horizon_hours,rmse,mae,mape,notes
base,1,...
base_plus_time,1,...
full,24,...
```

---

### 8.3 特征重要性分析（重点针对 XGBoost）

必须分别输出：

- `+1h` 特征重要性
- `+24h` 特征重要性

要比较：

- 短期预测是否更依赖 `lag1/lag3`
- 24 小时预测是否更依赖 `lag24/rolling/hour_sin/hour_cos`

#### 输出文件

- `outputs/hourly/xgboost_feature_importance_h1.csv`
- `outputs/hourly/xgboost_feature_importance_h24.csv`

---

### 8.4 AQI 分层误差分析

按照真实 PM2.5 值进行 AQI 等级划分：

- `Good`
- `Moderate`
- `Unhealthy-Sensitive`
- `Unhealthy`
- `Very Unhealthy`

统计：

- 各模型在 `+1h` 下的分层误差
- 各模型在 `+24h` 下的分层误差

#### 输出文件

- `outputs/hourly/aqi_bucket_metrics.csv`

建议字段：

```csv
model,horizon_hours,aqi_bucket,rmse,mae,mape,n_samples
```

---

### 8.5 高污染样本误差分析

单独挑出高污染时段（如 `pm25 > 150`）：

- 看每个模型在这些样本上的误差
- 对比 `+1h` 和 `+24h`

#### 输出文件

- `outputs/hourly/high_pollution_error_analysis.csv`

---

### 8.6 昼夜时段误差分析（新增重点分析模块）

由于本次实验改为小时级预测，因此必须考虑昼夜周期对预测结果的影响。  
这一部分是小时级实验区别于原先日级实验的重要深入点之一。

#### 分析目的

回答以下问题：

- 模型在白天和夜晚的误差是否存在明显差异？
- 模型是否在早高峰、晚高峰、深夜等时段更容易失准？
- `+24h` 任务是否比 `+1h` 更依赖时间段规律？

#### 建议分组方式

至少做两层分组：

##### 第一层：按小时分组

- `0 ~ 23` 点分别统计误差

##### 第二层：按昼夜时段分组

建议分为以下 5 段：

- `凌晨`：`00:00 - 05:59`
- `早高峰`：`06:00 - 09:59`
- `白天平峰`：`10:00 - 16:59`
- `晚高峰`：`17:00 - 20:59`
- `夜间`：`21:00 - 23:59`

如果实现难度较大，最少也要做：

- `白天`：`06:00 - 17:59`
- `夜晚`：`18:00 - 05:59`

#### 分析内容

对于每个模型，在 `+1h` 和 `+24h` 两个任务下分别统计：

- 每个小时上的 `RMSE`
- 每个小时上的 `MAE`
- 每个昼夜时段上的 `RMSE`
- 每个昼夜时段上的 `MAE`

重点建议至少保证：

- `XGBoost`
- `LSTM`
- `Transformer`

如果工作量受限，最少先完成：

- `XGBoost`

#### 输出文件

- `outputs/hourly/hourly_error_by_hour.csv`
- `outputs/hourly/hourly_error_by_dayperiod.csv`

建议字段：

```csv
model,horizon_hours,hour,rmse,mae,n_samples
XGBoost,1,0,...
```

```csv
model,horizon_hours,day_period,rmse,mae,n_samples
XGBoost,24,早高峰,...
```

#### 图表要求

- `outputs/hourly/figures/10_error_by_hour_curve.png`
  - 横轴 `0~23` 点，纵轴误差，展示不同模型在一天内各小时的误差变化

- `outputs/hourly/figures/11_dayperiod_error_bar.png`
  - 比较白天/夜晚或各时段的误差差异

#### 预期分析方向

这一部分最终要尝试回答：

- 哪些模型在昼夜周期变化中更稳定
- 哪些模型在高峰时段更容易失准
- 对于提前 24 小时预测，时间周期特征是否更重要

---

### 8.7 跨城市泛化实验（建议至少对 XGBoost 和 LSTM 做）

任务：

- 北京训练
- 上海测试

分别做：

- `+1h`
- `+24h`

目标：

- 评估模型是否学到可迁移规律
- 看跨城市后误差放大多少

#### 输入文件

- 训练：北京小时级统一数据或小时级特征矩阵
- 测试：上海小时级对齐数据 [data/processed/aligned_hourly/shanghai.csv](/d:/数据挖掘期末/data/processed/aligned_hourly/shanghai.csv)

#### 输出文件

- `outputs/hourly/cross_city_metrics.csv`
- `outputs/hourly/cross_city_predictions.csv`

---

## 9. 必须交付的图

以下图表建议全部产出到：

- `outputs/hourly/figures/`

### 图 1：下一小时真实值与预测值对比

- 文件名：`01_next_hour_prediction_curve.png`
- 内容：展示多个模型在 `+1h` 任务上的预测曲线

### 图 2：未来 24 小时真实值与预测值对比

- 文件名：`02_next_24h_prediction_curve.png`
- 内容：展示多个模型在 `+24h` 任务上的预测曲线或样例窗口预测

### 图 3：模型整体误差柱状图

- 文件名：`03_overall_metrics_bar.png`
- 内容：比较 5 个模型在 `+1h` 和 `+24h` 的 RMSE / MAE / MAPE

### 图 4：XGBoost 特征重要性（+1h）

- 文件名：`04_xgboost_feature_importance_h1.png`

### 图 5：XGBoost 特征重要性（+24h）

- 文件名：`05_xgboost_feature_importance_h24.png`

### 图 6：AQI 分层误差热力图

- 文件名：`06_aqi_bucket_error_heatmap.png`

### 图 7：特征消融对比图

- 文件名：`07_feature_ablation_comparison.png`

### 图 8：跨城市泛化结果图

- 文件名：`08_cross_city_generalization.png`

### 图 9：高污染样本误差分析图

- 文件名：`09_high_pollution_error.png`

### 图 10：按小时误差变化曲线

- 文件名：`10_error_by_hour_curve.png`
- 内容：展示模型在 `0~23` 点上的误差差异

### 图 11：昼夜/时段误差柱状图

- 文件名：`11_dayperiod_error_bar.png`
- 内容：比较不同时段（凌晨、早高峰、白天、晚高峰、夜间）的误差

---

## 10. 必须交付的结果文件清单

全部建议输出到：

- `outputs/hourly/`

### 模型预测结果

- `arima_predictions_h1.csv`
- `arima_predictions_h24.csv`
- `prophet_predictions_h1.csv`
- `prophet_predictions_h24.csv`
- `xgboost_predictions_h1.csv`
- `xgboost_predictions_h24.csv`
- `lstm_predictions_h1.csv`
- `lstm_predictions_h24.csv`
- `transformer_predictions_h1.csv`
- `transformer_predictions_h24.csv`
- `informer_predictions_h1.csv`（可选）
- `informer_predictions_h24.csv`（可选）

### 模型指标结果

- `arima_metrics.csv`
- `prophet_metrics.csv`
- `xgboost_metrics.csv`
- `lstm_metrics.csv`
- `transformer_metrics.csv`
- `informer_metrics.csv`（可选）

### 汇总与分析结果

- `overall_metrics_summary.csv`
- `feature_ablation.csv`
- `xgboost_feature_importance_h1.csv`
- `xgboost_feature_importance_h24.csv`
- `aqi_bucket_metrics.csv`
- `high_pollution_error_analysis.csv`
- `hourly_error_by_hour.csv`
- `hourly_error_by_dayperiod.csv`
- `cross_city_metrics.csv`
- `cross_city_predictions.csv`

### 训练历史与实验记录

- `lstm_train_history.json`
- `transformer_train_history.json`
- `informer_train_history.json`（可选）
- `experiment_registry.csv`
- `analysis_notes.md`

---

## 11. 统一结果文件格式要求

### 11.1 预测结果文件格式

所有 `*_predictions_*.csv` 建议统一字段：

```csv
forecast_origin_time,target_time,horizon_hours,city,model,y_true,y_pred,split
2016-05-14 08:00:00,2016-05-14 09:00:00,1,Beijing,XGBoost,82.4,79.2,test
```

字段解释：

- `forecast_origin_time`：预测起点时刻
- `target_time`：被预测的目标时刻
- `horizon_hours`：预测步长（1 或 24）
- `city`：城市
- `model`：模型名
- `y_true`：真实值
- `y_pred`：预测值
- `split`：train / val / test

### 11.2 模型指标文件格式

所有 `*_metrics.csv` 建议统一字段：

```csv
model,horizon_hours,rmse,mae,mape,n_samples,notes
```

---

## 12. 实施流程（按顺序执行）

下面是推荐的实际执行顺序，必须尽量按顺序来，不要一开始就同时做很多事情。

### 第 0 步：确认输入文件完整

必须确认以下文件存在：

- [data/processed/beijing_hourly.csv](/d:/数据挖掘期末/data/processed/beijing_hourly.csv)
- [data/processed/features_hourly](/d:/数据挖掘期末/data/processed/features_hourly)

若不存在，先运行：

```powershell
python scripts/clean_data.py
python scripts/build_features.py --granularity hourly
```

### 第 1 步：先完成 XGBoost 小时级 `+1h`

原因：

- 最稳
- 最容易先跑通
- 同时还能验证小时级特征矩阵是否可用

要求输出：

- `xgboost_predictions_h1.csv`
- `xgboost_metrics.csv`

### 第 2 步：扩展 XGBoost 到 `+24h`

要求：

- 在现有小时级特征工程基础上构造 `pm25.shift(-24)` 标签
- 训练 `+24h` 模型
- 输出预测结果与指标

### 第 3 步：完成 ARIMA 和 Prophet

要求：

- 两个模型都分别做 `+1h` 和 `+24h`
- 输出统一格式的结果文件

### 第 4 步：完成 LSTM

要求：

- 先做 `+1h`
- 再做 `+24h`
- 保存训练历史

### 第 5 步：完成 Transformer

要求：

- 与 LSTM 使用相同任务定义
- 保证对比公平

### 第 6 步：做特征消融

要求：

- 至少基于 XGBoost 完成 4 组特征消融
- 输出 `feature_ablation.csv`

### 第 7 步：做 AQI 分层与高污染分析

要求：

- 输出 `aqi_bucket_metrics.csv`
- 输出 `high_pollution_error_analysis.csv`

### 第 8 步：做昼夜时段误差分析

要求：

- 输出 `hourly_error_by_hour.csv`
- 输出 `hourly_error_by_dayperiod.csv`
- 生成按小时误差曲线图和昼夜时段误差柱状图

### 第 9 步：做跨城市泛化

要求：

- 至少对 XGBoost 和 LSTM 做北京训练、上海测试
- 输出 `cross_city_metrics.csv`

### 第 10 步：统一出图与总结

要求：

- 补齐所有图
- 完成 `analysis_notes.md`
- 保证后续同学拿到文件就能继续写报告和做展示

---

## 13. 实验记录要求

每次正式实验都必须在：

- `outputs/hourly/experiment_registry.csv`

里追加记录。

建议字段：

```csv
experiment_id,model,horizon_hours,feature_set,train_range,val_range,test_range,params,best_score,run_time,status,author,notes
```

同时必须在：

- `outputs/hourly/analysis_notes.md`

中写清楚：

1. 本次实验目的
2. 改了哪些参数或特征
3. 指标有没有变好
4. 主要发现是什么
5. 后续下一位同学应该接着做什么

---

## 14. 最终需要回答的核心问题

这份实验最终不能只给出一堆表格，而要清楚回答以下问题：

1. 小时级 PM2.5 预测中，哪种模型整体表现最好？
2. `+1h` 与 `+24h` 预测之间，模型性能下降有多明显？
3. 对 `+24h` 预测最关键的特征是什么？
4. 高污染时段是否显著拉大预测误差？
5. 白天、夜晚、高峰时段的误差是否显著不同？
6. 机器学习模型与深度学习模型在小时级任务上的优势分别是什么？
7. 北京训练的模型迁移到上海后还能否保持有效预测能力？

---

## 15. 最低完成标准与推荐完成标准

### 最低完成标准

- 5 个模型全部完成 `+1h`
- 至少 XGBoost、ARIMA、LSTM 完成 `+24h`
- 产出统一格式预测结果和总指标文件

### 推荐完成标准

- 5 个模型全部完成 `+1h` 和 `+24h`
- 完成特征消融
- 完成 AQI 分层误差分析
- 完成至少 XGBoost 的跨城市泛化实验

### 理想完成标准

- 5 个模型全部完成双任务
- XGBoost 与 LSTM 完成跨城市泛化
- 图表和 CSV 可直接用于报告与答辩展示
- 后续整合成员不需要返工实验设计

---

## 16. 补充说明

1. 原有日级实验结果不要删，可保留为项目背景和前期尝试。
2. 当前小时级实验应成为项目主线。
3. 若某个模型在 `+24h` 任务中明显退化，不要简单删除结果，必须记录并分析原因。
4. 所有结果必须能被其他成员复用，不能只停留在控制台输出。

---

## 17. 本次任务的直接交付结论

负责实验的同学拿到这份说明后，优先顺序应当是：

1. 确认小时级统一数据与小时级特征矩阵存在
2. 先跑通 `XGBoost +1h`
3. 再跑 `XGBoost +24h`
4. 再补齐 ARIMA、Prophet、LSTM、Transformer
5. 然后做特征消融、AQI 分层、高污染误差、跨城市泛化
6. 最后统一整理图、表、实验记录和总结文档

如果时间不足，优先保证：

- 5 个模型的双任务结果
- 总指标汇总
- 至少 1 组深入分析（优先做特征消融）

如果时间充足，则全部做完。
