# 特征说明文档

## 概述

- 预测目标：`pm25`（PM2.5 日均浓度，单位 μg/m³）
- 输入数据：`data/processed/beijing.csv`
- 输出目录：`data/processed/features/`

## 训练/测试划分

- 滞后/滚动特征构造后的有效行数：1091
- 因滞后产生的丢弃行数：14
- 训练样本：872（2013-03-25 至 2016-05-14）
- 测试样本：219（2016-05-15 至 2017-02-26）
- 划分方式：按时间顺序，后 20% 为测试集

## 特征列表

| 特征名 | 说明 |
|--------|------|
| pm25_lag1 | 前 1 日 PM2.5 |
| pm25_lag3 | 前 3 日 PM2.5 |
| pm25_lag7 | 前 7 日 PM2.5 |
| pm25_roll_mean_7 | 7 日滚动均值（滞后 1 日） |
| pm25_roll_mean_14 | 14 日滚动均值（滞后 1 日） |
| pm25_roll_std_7 | 7 日滚动标准差（滞后 1 日） |
| pm25_roll_std_14 | 14 日滚动标准差（滞后 1 日） |
| month_sin | 月份正弦周期编码 |
| month_cos | 月份余弦周期编码 |
| weekday | 星期几（0=周一） |
| is_holiday | 中国法定节假日标记（0/1） |
| temp | 日均温度（℃） |
| pres | 日均气压（hPa） |
| dewp | 日均露点（℃） |
| wind_speed | 日均风速（m/s） |
| precipitation | 日降水量（mm） |
| temp_x_humidity | 温度 × 湿度交互项 |
| wind_speed_x_wind_dir_sin | 风速 × sin(风向角) 交互项 |

## 矩阵维度

- X_train: (872, 18)
- X_test: (219, 18)
- y_train: (872,)
- y_test: (219,)

## 加载示例

```python
import json
import numpy as np

X_train = np.load('data/processed/features/X_train.npy')
y_train = np.load('data/processed/features/y_train.npy')
with open('data/processed/features/feature_names.json') as f:
    feature_names = json.load(f)
```

特征已使用 `utils/feature_engineering.py` 中的 StandardScaler 在训练集上拟合标准化，
测试集使用同一 scaler 变换。推理新样本时请加载 `scaler.pkl`。
