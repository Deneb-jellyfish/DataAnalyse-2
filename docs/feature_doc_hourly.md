# 小时级特征说明文档

## 概述

- 预测目标：`pm25`（PM2.5 小时浓度，单位 μg/m³）
- 输入数据：`data/processed/beijing_hourly.csv`
- 输出目录：`data/processed/features_hourly/`

## 训练/测试划分

- 滞后/滚动特征构造后的有效行数：34773
- 因滞后产生的丢弃行数：168
- 训练样本：27818（2013-03-08 00:00:00 至 2016-05-13 17:00:00）
- 测试样本：6955（2016-05-13 18:00:00 至 2017-02-28 23:00:00）
- 划分方式：按时间顺序，后 20% 为测试集

## 特征列表

| 特征名 | 说明 |
|--------|------|
| pm25_lag1 | 前 1 小时 PM2.5 |
| pm25_lag3 | 前 3 小时 PM2.5 |
| pm25_lag6 | 前 6 小时 PM2.5 |
| pm25_lag12 | 前 12 小时 PM2.5 |
| pm25_lag18 | 前 18 小时 PM2.5 |
| pm25_lag24 | 前 24 小时 PM2.5 |
| pm25_lag36 | 前 36 小时 PM2.5 |
| pm25_lag48 | 前 48 小时 PM2.5 |
| pm25_roll_mean_6 | 6 小时滚动均值（滞后 1 小时） |
| pm25_roll_mean_12 | 12 小时滚动均值（滞后 1 小时） |
| pm25_roll_mean_24 | 24 小时滚动均值（滞后 1 小时） |
| pm25_roll_mean_168 | 168 小时（7 天）滚动均值（滞后 1 小时） |
| pm25_roll_std_6 | 6 小时滚动标准差（滞后 1 小时） |
| pm25_roll_std_12 | 12 小时滚动标准差（滞后 1 小时） |
| pm25_roll_std_24 | 24 小时滚动标准差（滞后 1 小时） |
| pm25_roll_std_168 | 168 小时滚动标准差（滞后 1 小时） |
| pm25_diff_1 | 当前 PM2.5 相对 1 小时前变化 |
| pm25_diff_3 | 当前 PM2.5 相对 3 小时前变化 |
| pm25_diff_6 | 当前 PM2.5 相对 6 小时前变化 |
| pm25_diff_12 | 当前 PM2.5 相对 12 小时前变化 |
| hour_sin | 小时正弦周期编码 |
| hour_cos | 小时余弦周期编码 |
| month_sin | 月份正弦周期编码 |
| month_cos | 月份余弦周期编码 |
| weekday | 星期几（0=周一） |
| is_weekend | 是否周末（0/1） |
| is_holiday | 中国法定节假日标记（0/1） |
| is_daytime | 是否白天（06:00-17:59） |
| is_rush_hour | 是否通勤高峰（7-9,17-19） |
| temp | 小时温度（℃） |
| pres | 小时气压（hPa） |
| dewp | 小时露点（℃） |
| humidity | 小时湿度（%） |
| wind_speed | 小时风速（m/s） |
| precipitation | 小时降水量（mm） |
| wind_dir_sin | 风向正弦编码 |
| wind_dir_cos | 风向余弦编码 |
| precipitation_flag | 是否有降水（0/1） |
| dewp_temp_gap | 温度与露点差（temp-dewp） |
| temp_diff_1 | 温度相对 1 小时前变化 |
| temp_diff_3 | 温度相对 3 小时前变化 |
| pres_diff_1 | 气压相对 1 小时前变化 |
| pres_diff_3 | 气压相对 3 小时前变化 |
| wind_speed_diff_1 | 风速相对 1 小时前变化 |
| wind_speed_diff_3 | 风速相对 3 小时前变化 |
| temp_x_humidity | 温度 × 湿度交互项 |
| wind_speed_x_wind_dir_sin | 风速 × sin(风向角) 交互项 |
| wind_speed_x_pm25_lag1 | 风速 × PM2.5(t-1) 交互项 |

## 矩阵维度

- X_train: (27818, 48)
- X_test: (6955, 48)
- y_train: (27818,)
- y_test: (6955,)

## 加载示例

```python
import json
import numpy as np

X_train = np.load('data/processed/features_hourly/X_train.npy')
y_train = np.load('data/processed/features_hourly/y_train.npy')
with open('data/processed/features_hourly/feature_names.json') as f:
    feature_names = json.load(f)
```

特征已使用 `utils/feature_engineering.py` 中的 StandardScaler 在训练集上拟合标准化，
测试集使用同一 scaler 变换。推理新样本时请加载 `scaler.pkl`。
