# 数据接口约定

本仓库现在以**小时级 PM2.5 预测实验**为主，下面只保留当前主线真正使用的数据接口。

## 1. 主实验输入文件

| 用途 | 文件 |
|:--|:--|
| 北京主实验小时数据 | `data/processed/beijing_hourly.csv` |
| 上海小时数据 | `data/processed/shanghai_hourly.csv` |
| 跨城市对齐小时数据（北京） | `data/processed/aligned_hourly/beijing.csv` |
| 跨城市对齐小时数据（上海） | `data/processed/aligned_hourly/shanghai.csv` |

## 2. 小时级特征矩阵

目录：

- `data/processed/features_hourly/`

主要文件：

| 文件 | 说明 |
|:--|:--|
| `X_train.npy` | 训练特征矩阵，形状为 `(n_train, n_features)` |
| `X_test.npy` | 测试特征矩阵，形状为 `(n_test, n_features)` |
| `y_train.npy` | 训练标签，形状为 `(n_train,)` |
| `y_test.npy` | 测试标签，形状为 `(n_test,)` |
| `dates_train.npy` | 训练集对应时间戳 |
| `dates_test.npy` | 测试集对应时间戳 |
| `feature_names.json` | 特征名列表，顺序与 `X_*` 列一致 |
| `scaler.pkl` | 在训练集上拟合的标准化器 |

划分方式：

- 按时间顺序切分
- 滞后与滚动特征构造完成后，后 20% 作为测试集

## 3. 模型预测输出

主实验预测文件统一写入：

- `outputs/hourly/`

约定：

| 文件 | 说明 |
|:--|:--|
| `*_predictions_h1.csv` | 单步 `h1` 预测结果 |
| `*_predictions_seq6.csv` | 多步 `seq6` 预测结果 |
| `*_metrics.csv` | 各模型指标文件 |

常见列：

- `datetime`
- `task`
- `horizon`
- `split`
- `y_true`
- `y_pred`

不同模型的个别附加列允许存在，但以上字段是分析脚本默认依赖的核心列。

## 4. 汇总与分析输出

下面这些文件会被报告和展示层直接消费：

| 文件 | 说明 |
|:--|:--|
| `summaries/overall_metrics_summary.csv` | 总体指标汇总 |
| `analyses/feature_ablation.csv` | 特征消融结果 |
| `analyses/aqi_bucket_metrics.csv` | AQI 分层误差 |
| `analyses/high_pollution_error_analysis.csv` | 高污染样本误差 |
| `analyses/hourly_error_by_hour.csv` | 按小时误差 |
| `analyses/hourly_error_by_dayperiod.csv` | 按时段误差 |
| `analyses/cross_city_metrics.csv` | 跨城市泛化指标 |
| `figures/*.png` | 报告图表 |

## 5. 当前推荐入口

### 预处理 + 特征工程

```powershell
py -3 main.py --steps all --granularity hourly
```

### 一键总流程

```powershell
py -3 run_all.py
```
