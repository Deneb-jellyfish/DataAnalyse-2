# AQI Data Cleaning

这个仓库现在已经整理出一套可直接运行的数据清洗流程，目标是把两类原始空气质量数据统一清洗到 `data/processed/`，方便后续做 EDA、特征工程、时间序列预测和跨城市对比实验。

## 仓库里现在有哪些数据

### 1. 北京 PRSA 12 站点数据

原始文件位置：

- `data/raw/PRSA_Data_Aotizhongxin_20130301-20170228.csv`
- `data/raw/PRSA_Data_Changping_20130301-20170228.csv`
- `data/raw/PRSA_Data_Dingling_20130301-20170228.csv`
- `data/raw/PRSA_Data_Dongsi_20130301-20170228.csv`
- `data/raw/PRSA_Data_Guanyuan_20130301-20170228.csv`
- `data/raw/PRSA_Data_Gucheng_20130301-20170228.csv`
- `data/raw/PRSA_Data_Huairou_20130301-20170228.csv`
- `data/raw/PRSA_Data_Nongzhanguan_20130301-20170228.csv`
- `data/raw/PRSA_Data_Shunyi_20130301-20170228.csv`
- `data/raw/PRSA_Data_Tiantan_20130301-20170228.csv`
- `data/raw/PRSA_Data_Wanliu_20130301-20170228.csv`
- `data/raw/PRSA_Data_Wanshouxigong_20130301-20170228.csv`

数据特征：

- 监测站数量：`12`
- 每个文件行数：`35,064`
- 原始总行数：`420,768`
- 时间粒度：小时级
- 时间范围：`2013-03-01 00:00:00` 到 `2017-02-28 23:00:00`
- 主要字段：
  - `PM2.5, PM10, SO2, NO2, CO, O3`
  - `TEMP, PRES, DEWP, RAIN`
  - `wd, WSPM`
  - `station`

这套数据是你项目的主数据，适合做北京空气质量预测主实验。

### 2. 五城市 PM 数据

原始文件位置：

- `data/raw/othercity/BeijingPM20100101_20151231.csv`
- `data/raw/othercity/ShanghaiPM20100101_20151231.csv`
- `data/raw/othercity/GuangzhouPM20100101_20151231.csv`
- `data/raw/othercity/ChengduPM20100101_20151231.csv`
- `data/raw/othercity/ShenyangPM20100101_20151231.csv`

数据特征：

- 城市数量：`5`
- 每个文件行数：`52,584`
- 原始总行数：`262,920`
- 时间粒度：小时级
- 时间范围：`2010-01-01 00:00:00` 到 `2015-12-31 23:00:00`
- 城市列表：
  - `Beijing`
  - `Shanghai`
  - `Guangzhou`
  - `Chengdu`
  - `Shenyang`

这套数据和北京 PRSA 的结构不完全一样。它不是统一的 `PM2.5` 一列，而是一个城市内多个监测点的 PM 列，例如：

- 上海：`PM_Jingan`, `PM_US Post`, `PM_Xuhui`
- 北京：`PM_Dongsi`, `PM_Dongsihuan`, `PM_Nongzhanguan`, `PM_US Post`
- 广州：`PM_City Station`, `PM_5th Middle School`, `PM_US Post`

另外还包含一些通用气象字段：

- `DEWP, HUMI, PRES, TEMP`
- `cbwd`
- `Iws`
- `precipitation, Iprec`
- `season`

这套数据更适合做跨城市补充实验，而不是直接替代北京主数据。

## 清洗脚本

脚本位置：

- `scripts/clean_data.py`

运行命令：

```powershell
py -3 scripts/clean_data.py
```

## 清洗规则

脚本会对两类数据统一做下面这些处理：

1. 把 `NA`、空字符串等统一识别成缺失值
2. 删除无意义的编号列 `No`
3. 由 `year/month/day/hour` 合成标准 `datetime`
4. 按时间排序
5. 对数值列只插值连续不超过 `6` 小时的短缺口
6. 对风向列做前向填充，再用众数兜底
7. 把明显不合理的负值转成缺失
8. 保留长缺口，不强行硬补，避免制造假信号

### 北京 PRSA 的额外处理

- 保留原始站点列 `station`
- 先输出 12 站点小时级清洗结果
- 再按小时对 12 个站点求均值，得到北京城市级小时表
- 再从城市级小时表聚合出北京城市级日表

### 五城市数据的额外处理

- 自动识别每个城市文件中的多个 `PM_*` 列
- 先分别对这些 `PM_*` 列做短缺口插值
- 再按行求均值，生成统一列 `pm25_city`
- 额外裁剪出与北京 PRSA 主数据重叠的时间段：
  - `2013-03-01 00:00:00` 到 `2015-12-31 23:00:00`

## 数据量统计

### 原始数据量

#### 北京 PRSA

| 文件 | 行数 |
|---|---:|
| `PRSA_Data_Aotizhongxin_20130301-20170228.csv` | 35,064 |
| `PRSA_Data_Changping_20130301-20170228.csv` | 35,064 |
| `PRSA_Data_Dingling_20130301-20170228.csv` | 35,064 |
| `PRSA_Data_Dongsi_20130301-20170228.csv` | 35,064 |
| `PRSA_Data_Guanyuan_20130301-20170228.csv` | 35,064 |
| `PRSA_Data_Gucheng_20130301-20170228.csv` | 35,064 |
| `PRSA_Data_Huairou_20130301-20170228.csv` | 35,064 |
| `PRSA_Data_Nongzhanguan_20130301-20170228.csv` | 35,064 |
| `PRSA_Data_Shunyi_20130301-20170228.csv` | 35,064 |
| `PRSA_Data_Tiantan_20130301-20170228.csv` | 35,064 |
| `PRSA_Data_Wanliu_20130301-20170228.csv` | 35,064 |
| `PRSA_Data_Wanshouxigong_20130301-20170228.csv` | 35,064 |
| 合计 | 420,768 |

#### 五城市

| 文件 | 行数 |
|---|---:|
| `BeijingPM20100101_20151231.csv` | 52,584 |
| `ShanghaiPM20100101_20151231.csv` | 52,584 |
| `GuangzhouPM20100101_20151231.csv` | 52,584 |
| `ChengduPM20100101_20151231.csv` | 52,584 |
| `ShenyangPM20100101_20151231.csv` | 52,584 |
| 合计 | 262,920 |

### 清洗后输出数据量

#### 北京输出

| 文件 | 行数 | 说明 |
|---|---:|---|
| `data/processed/beijing/beijing_hourly_all_stations.csv` | 420,768 | 北京 12 站点小时级明细 |
| `data/processed/beijing/beijing_hourly_city.csv` | 35,064 | 北京城市级小时均值 |
| `data/processed/beijing/beijing_daily_city.csv` | 1,461 | 北京城市级日均值 |

#### 五城市输出

| 文件 | 行数 | 说明 |
|---|---:|---|
| `data/processed/five_city/beijing_hourly_clean.csv` | 52,584 | 北京完整时段清洗结果 |
| `data/processed/five_city/shanghai_hourly_clean.csv` | 52,584 | 上海完整时段清洗结果 |
| `data/processed/five_city/guangzhou_hourly_clean.csv` | 52,584 | 广州完整时段清洗结果 |
| `data/processed/five_city/chengdu_hourly_clean.csv` | 52,584 | 成都完整时段清洗结果 |
| `data/processed/five_city/shenyang_hourly_clean.csv` | 52,584 | 沈阳完整时段清洗结果 |
| `data/processed/five_city/beijing_hourly_overlap_20130301_20151231.csv` | 24,864 | 与北京主数据时间重叠部分 |
| `data/processed/five_city/shanghai_hourly_overlap_20130301_20151231.csv` | 24,864 | 与北京主数据时间重叠部分 |
| `data/processed/five_city/guangzhou_hourly_overlap_20130301_20151231.csv` | 24,864 | 与北京主数据时间重叠部分 |
| `data/processed/five_city/chengdu_hourly_overlap_20130301_20151231.csv` | 24,864 | 与北京主数据时间重叠部分 |
| `data/processed/five_city/shenyang_hourly_overlap_20130301_20151231.csv` | 24,864 | 与北京主数据时间重叠部分 |
| `data/processed/five_city/five_city_hourly_overlap_combined.csv` | 124,320 | 五城市重叠时段总表 |

## 缺失值情况

### 北京 PRSA 原始缺失较多的字段

脚本运行时统计到的主要缺失量如下：

- `CO`: 20,701
- `O3`: 13,277
- `NO2`: 12,116
- `SO2`: 9,021
- `PM2.5`: 8,739
- `PM10`: 6,449
- `wd`: 1,822

经过短缺口插值后，仍然保留了一部分长缺口：

- `CO`: 14,073
- `O3`: 7,279
- `NO2`: 6,943
- `SO2`: 5,090
- `PM2.5`: 4,452
- `PM10`: 3,104

这表示脚本没有过度补值，保留了较长时间段的真实缺失。

## 输出字段说明

### 北京 12 站点明细表

文件：

- `data/processed/beijing/beijing_hourly_all_stations.csv`

字段：

- `datetime`
- `station`
- `PM2.5, PM10, SO2, NO2, CO, O3`
- `TEMP, PRES, DEWP, RAIN`
- `wd`
- `WSPM`

### 北京城市级小时表

文件：

- `data/processed/beijing/beijing_hourly_city.csv`

字段：

- `datetime`
- `PM2.5, PM10, SO2, NO2, CO, O3`
- `TEMP, PRES, DEWP, RAIN, WSPM`
- `pm25_station_count`

其中 `pm25_station_count` 表示该小时内实际参与均值计算的站点数量。

### 五城市清洗表

文件示例：

- `data/processed/five_city/shanghai_hourly_clean.csv`

字段：

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

其中：

- `pm25_city` 是该城市多个 `PM_*` 监测点按行求平均后的统一 PM2.5 指标
- `pm_site_count` 是这一时刻参与平均的 PM 监测点数量

## 推荐怎么使用这些表

### 主实验

优先使用：

- `data/processed/beijing/beijing_hourly_city.csv`
- `data/processed/beijing/beijing_daily_city.csv`

适合做：

- 北京 AQI / PM2.5 单城市预测
- 时间序列特征工程
- ARIMA、Prophet、XGBoost、LSTM、Transformer 等模型实验

### 补充实验

优先使用：

- `data/processed/five_city/five_city_hourly_overlap_combined.csv`

适合做：

- 多城市泛化能力比较
- 跨城市迁移实验
- 城市间空气质量变化差异分析

### 如果要做统一建模

建议先把北京和五城市都整理成统一字段，例如：

- `datetime`
- `city`
- `pm25`
- `TEMP`
- `PRES`
- `DEWP`
- `wind_dir`
- `wind_speed`
- `precipitation`

不要直接把北京 PRSA 原表和五城市原表生拼，因为它们的字段结构不同。

## 目录结构

```text
data/
  raw/
    PRSA_Data_*.csv
    othercity/
      BeijingPM20100101_20151231.csv
      ShanghaiPM20100101_20151231.csv
      GuangzhouPM20100101_20151231.csv
      ChengduPM20100101_20151231.csv
      ShenyangPM20100101_20151231.csv
  processed/
    beijing/
      beijing_hourly_all_stations.csv
      beijing_hourly_city.csv
      beijing_daily_city.csv
    five_city/
      beijing_hourly_clean.csv
      shanghai_hourly_clean.csv
      guangzhou_hourly_clean.csv
      chengdu_hourly_clean.csv
      shenyang_hourly_clean.csv
      beijing_hourly_overlap_20130301_20151231.csv
      shanghai_hourly_overlap_20130301_20151231.csv
      guangzhou_hourly_overlap_20130301_20151231.csv
      chengdu_hourly_overlap_20130301_20151231.csv
      shenyang_hourly_overlap_20130301_20151231.csv
      five_city_hourly_overlap_combined.csv
scripts/
  clean_data.py
```

## 下一步建议

如果你接下来要继续做建模，推荐顺序是：

1. 先基于 `beijing_hourly_city.csv` 做 EDA 和特征工程
2. 再选 `beijing_daily_city.csv` 做传统时间序列模型
3. 最后再引入 `five_city_hourly_overlap_combined.csv` 做跨城市实验
