# AQI Data Cleaning

这个仓库现在包含两套原始空气质量数据：

- `data/raw/PRSA_Data_*.csv`
  - 北京 `12` 个站点
  - 小时级
  - 时间范围：`2013-03-01` 到 `2017-02-28`
  - 字段包含 `PM2.5 / PM10 / SO2 / NO2 / CO / O3` 以及气象信息
- `data/raw/othercity/*PM20100101_20151231.csv`
  - 北京、上海、广州、成都、沈阳
  - 小时级
  - 时间范围：`2010-01-01` 到 `2015-12-31`
  - 每个城市有多个 `PM_*` 监测点列，例如上海有 `PM_Jingan / PM_US Post / PM_Xuhui`

## 清洗规则

脚本 `scripts/clean_data.py` 会做下面这些事情：

- 把 `NA` 统一识别成缺失值
- 合成 `datetime` 列
- 删除无用编号列 `No`
- 对数值列只插值连续不超过 `6` 小时的短缺口
- 对风向列做前向填充，并用众数兜底
- 把明显不合理的负值转成缺失
- 把五城市数据里的多个 `PM_*` 列按行求均值，生成统一的 `pm25_city`
- 为跨城市实验额外裁剪一份与北京主数据重叠的时间段：`2013-03-01` 到 `2015-12-31`

## 运行方式

在仓库根目录执行：

```powershell
py -3 scripts/clean_data.py
```

## 输出文件

### 北京主数据

输出到 `data/processed/beijing/`：

- `beijing_hourly_all_stations.csv`
  - 北京 `12` 站点清洗后的小时级明细
- `beijing_hourly_city.csv`
  - 按小时对 `12` 站点做城市级均值聚合
- `beijing_daily_city.csv`
  - 在城市小时级基础上继续聚合成日均值

### 五城市补充数据

输出到 `data/processed/five_city/`：

- `{city}_hourly_clean.csv`
  - 每个城市完整时间段的清洗结果
- `{city}_hourly_overlap_20130301_20151231.csv`
  - 和北京主数据时间重叠的版本
- `five_city_hourly_overlap_combined.csv`
  - 所有城市拼在一起的重叠时间段总表

## 字段说明

### 北京输出

- `datetime`
- `station`
- `PM2.5, PM10, SO2, NO2, CO, O3`
- `TEMP, PRES, DEWP, RAIN`
- `wd`
- `WSPM`

### 五城市输出

- `datetime`
- `city`
- `season`
- `pm25_city`
- `pm_site_count`
- `DEWP, HUMI, PRES, TEMP`
- `wind_dir`
- `wind_speed`
- `precipitation`
- `cum_precipitation`

## 建议怎么用

- 做主实验：优先使用 `data/processed/beijing/beijing_hourly_city.csv` 或 `beijing_daily_city.csv`
- 做跨城市补充实验：使用 `data/processed/five_city/five_city_hourly_overlap_combined.csv`
- 如果要把北京和其他城市统一建模，建议先统一字段名，再只保留交集特征，例如：
  - `datetime`
  - `city`
  - `pm25`
  - `TEMP`
  - `PRES`
  - `DEWP`
  - `wind_dir`
  - `wind_speed`
  - `precipitation`
