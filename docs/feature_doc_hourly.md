# Hourly Feature Reference

## Overview

- Target: `pm25`
- Input dataset: `data/processed/beijing_hourly.csv`
- Output directory: `data/processed/features_hourly/`

## Train/Test Split

- Valid rows after feature construction: 26352
- Dropped rows caused by lag/rolling windows: 168
- Train rows: 21081 (2013-03-14 00:00:00 -> 2016-05-14 08:00:00)
- Test rows: 5271 (2016-05-14 09:00:00 -> 2017-02-26 23:00:00)
- Split rule: chronological split, last 20% used as the test set

## Feature List

| Feature | Description |
|---|---|
| pm25_lag1 | PM2.5 one hour ago. |
| pm25_lag3 | PM2.5 three hours ago. |
| pm25_lag6 | PM2.5 six hours ago. |
| pm25_lag12 | PM2.5 twelve hours ago. |
| pm25_lag18 | PM2.5 eighteen hours ago. |
| pm25_lag24 | PM2.5 one day ago at the same hour. |
| pm25_lag36 | PM2.5 thirty-six hours ago. |
| pm25_lag48 | PM2.5 two days ago at the same hour. |
| pm25_roll_mean_6 | Trailing 6-hour PM2.5 mean using only past values. |
| pm25_roll_mean_12 | Trailing 12-hour PM2.5 mean using only past values. |
| pm25_roll_mean_24 | Trailing 24-hour PM2.5 mean using only past values. |
| pm25_roll_mean_168 | Trailing 7-day PM2.5 mean using only past values. |
| pm25_roll_std_6 | Trailing 6-hour PM2.5 standard deviation. |
| pm25_roll_std_12 | Trailing 12-hour PM2.5 standard deviation. |
| pm25_roll_std_24 | Trailing 24-hour PM2.5 standard deviation. |
| pm25_roll_std_168 | Trailing 7-day PM2.5 standard deviation. |
| pm25_diff_1 | Current PM2.5 minus PM2.5 one hour ago. |
| pm25_diff_3 | Current PM2.5 minus PM2.5 three hours ago. |
| pm25_diff_6 | Current PM2.5 minus PM2.5 six hours ago. |
| pm25_diff_12 | Current PM2.5 minus PM2.5 twelve hours ago. |
| hour_sin | Hour-of-day cyclic encoding (sine). |
| hour_cos | Hour-of-day cyclic encoding (cosine). |
| month_sin | Month-of-year cyclic encoding (sine). |
| month_cos | Month-of-year cyclic encoding (cosine). |
| weekday | Weekday index, Monday=0. |
| is_weekend | Weekend indicator. |
| is_holiday | Chinese public holiday indicator. |
| is_daytime | Indicator for 06:00-17:59 local time. |
| is_rush_hour | Indicator for commute-heavy hours (7-9, 17-19). |
| temp | Current temperature. |
| pres | Current pressure. |
| dewp | Current dew point. |
| humidity | Current humidity. |
| wind_speed | Current wind speed. |
| precipitation | Current precipitation. |
| wind_dir_sin | Wind direction encoded as sine. |
| wind_dir_cos | Wind direction encoded as cosine. |
| precipitation_flag | Indicator for any precipitation in the current hour. |
| dewp_temp_gap | Temperature minus dew point. |
| temp_diff_1 | Current temperature minus temperature one hour ago. |
| temp_diff_3 | Current temperature minus temperature three hours ago. |
| pres_diff_1 | Current pressure minus pressure one hour ago. |
| pres_diff_3 | Current pressure minus pressure three hours ago. |
| wind_speed_diff_1 | Current wind speed minus wind speed one hour ago. |
| wind_speed_diff_3 | Current wind speed minus wind speed three hours ago. |
| temp_x_humidity | Temperature multiplied by humidity. |
| wind_speed_x_wind_dir_sin | Wind speed scaled by wind-direction sine. |
| wind_speed_x_pm25_lag1 | Wind speed multiplied by PM2.5 one hour ago. |

## Matrix Shapes

- X_train: (21081, 48)
- X_test: (5271, 48)
- y_train: (21081,)
- y_test: (5271,)

## Loading Example

```python
import json
import numpy as np

X_train = np.load('data/processed/features_hourly/X_train.npy')
y_train = np.load('data/processed/features_hourly/y_train.npy')
with open('data/processed/features_hourly/feature_names.json', encoding='utf-8') as f:
    feature_names = json.load(f)
```

The scaler is fit on the training split only and stored as `scaler.pkl`.