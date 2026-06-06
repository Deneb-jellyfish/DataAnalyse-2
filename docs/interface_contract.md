# 数据接口约定

本文档定义组员 A（数据工程）、组员 B（模型）、组员 C（工程交付）之间的数据交接格式。

## 统一日级城市数据 schema

文件：`data/processed/beijing.csv`、`data/processed/shanghai.csv`，以及 `data/processed/aligned/` 下的对齐副本。

| 字段 | 类型 | 北京来源 | 上海来源 | 说明 |
|------|------|----------|----------|------|
| `date` | `YYYY-MM-DD` | PRSA 日聚合 | 小时 → 日均 | 主关联键 |
| `city` | 字符串 | `"Beijing"` | `"Shanghai"` | |
| `pm25` | 浮点 | `PM2.5` | `pm25_city` | **预测目标** |
| `pm10` | 浮点 | `PM10` | 缺失 | 上海无污染物列 |
| `so2` | 浮点 | `SO2` | 缺失 | |
| `no2` | 浮点 | `NO2` | 缺失 | |
| `co` | 浮点 | `CO` | 缺失 | |
| `o3` | 浮点 | `O3` | 缺失 | |
| `temp` | 浮点 | `TEMP` | `TEMP` | 摄氏度 |
| `pres` | 浮点 | `PRES` | `PRES` | 百帕 hPa |
| `dewp` | 浮点 | `DEWP` | `DEWP` | 摄氏度 |
| `humidity` | 浮点 | 由温度+露点估算 | `HUMI` | 相对湿度 % |
| `wind_dir` | 字符串 | 缺失 | `cbwd` | 类别型风向 |
| `wind_speed` | 浮点 | `WSPM` | `Iws` | 米/秒 |
| `precipitation` | 浮点 | `RAIN` | `precipitation` | 毫米 |

## 文件选用指南

| 使用场景 | 文件 |
|----------|------|
| 北京主实验（全时间范围） | `data/processed/beijing.csv` |
| 上海独立分析 | `data/processed/shanghai.csv` |
| 跨城市迁移实验 | `data/processed/aligned/beijing.csv` + `aligned/shanghai.csv` |
| 小时级 EDA / 完整污染物 | `data/processed/beijing/beijing_hourly_city.csv` |
| XGBoost 特征矩阵 | `data/processed/features/X_train.npy` 等 |

## 特征矩阵约定（组员 B）

目录：`data/processed/features/`

| 文件 | 说明 |
|------|------|
| `X_train.npy` | `(n_train, n_features)` 标准化后的训练特征 |
| `X_test.npy` | `(n_test, n_features)` 用训练集 scaler 变换 |
| `y_train.npy` | `(n_train,)` PM2.5 目标值 |
| `y_test.npy` | `(n_test,)` PM2.5 目标值 |
| `dates_train.npy` | 训练集日期字符串 |
| `dates_test.npy` | 测试集日期字符串 |
| `feature_names.json` | 与 `X_*` 列顺序一致的特征名列表 |
| `scaler.pkl` | 在训练集上拟合的标准化器 |

训练/测试划分：按时间顺序，有效样本的后 20% 作为测试集。

## 模型预测交接（组员 B → 组员 C）

`predictions.csv` 字段（由组员 B 定义，组员 C 消费）：

- `date`、`city`、`model`、`y_true`、`y_pred`

## 提交规范

- Commit 前缀：`[A]`、`[B]`、`[C]`
- 不要将 `data/` 目录内容提交到 GitHub
