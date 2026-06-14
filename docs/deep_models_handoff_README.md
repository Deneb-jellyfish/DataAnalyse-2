# 深度模型补跑说明

本文档只给接手跑实验的同学用，目标是：

- 不重复跑已经完成的任务
- 只补齐 `LSTM seq6`、`Transformer h1`、`Transformer seq6`
- 跑完后再做一次分析汇总

---

## 1. 先确认当前目录

先进入仓库根目录：

```powershell
cd /d D:\数据挖掘期末
```

后面的命令都默认在这个目录下执行。

---

## 2. 当前已经完成的部分

下面这些任务已经有产物了，不要重复跑：

### 2.1 已完成

- `XGBoost h1`
- `XGBoost seq6`
- `ARIMA h1`
- `ARIMA seq6`
- `Prophet h1`
- `Prophet seq6`
- `LSTM h1`

### 2.2 LSTM h1 已存在的产物

- `outputs/hourly/lstm_h1.pkl`
- `outputs/hourly/lstm_h1_config.json`
- `outputs/hourly/lstm_predictions_h1.csv`

所以：

- 不要再跑 `py -3 scripts/train_lstm.py --task h1 ...`

---

## 3. 如果小时级特征矩阵不存在，先重建

如果下面这些文件已经存在，就跳过这一步：

- `data/processed/features_hourly/X_train.npy`
- `data/processed/features_hourly/X_test.npy`
- `data/processed/features_hourly/y_train.npy`

如果不存在，执行：

```powershell
py -3 scripts\build_features.py --granularity hourly
```

---

## 4. 现在需要补跑的命令

按下面顺序直接运行。

### 4.1 跑 LSTM seq6

```powershell
py -3 scripts\train_lstm.py --task seq6 --profile handoff
```

说明：

- `--task seq6` 表示只跑未来 6 小时逐小时预测
- `--profile handoff` 是为了控制训练时长，避免默认配置过重

预期输出：

- `outputs/hourly/lstm_predictions_seq6.csv`
- `outputs/hourly/lstm_seq6.pkl`
- `outputs/hourly/lstm_seq6_config.json`
- `outputs/hourly/lstm_metrics.csv`
- `outputs/hourly/lstm_train_history.json`

### 4.2 跑 Transformer h1

```powershell
py -3 scripts\train_transformer.py --task h1 --epochs 60 --patience 8 --batch-size 64 --lr 5e-4
```

预期输出：

- `outputs/hourly/transformer_predictions_h1.csv`
- `outputs/hourly/transformer_predictions_h1_val.csv`
- `outputs/hourly/transformer_h1.pt`
- `outputs/hourly/transformer_h1_config.json`
- `outputs/hourly/transformer_metrics.csv`
- `outputs/hourly/transformer_train_history.json`

### 4.3 跑 Transformer seq6

```powershell
py -3 scripts\train_transformer.py --task seq6 --epochs 80 --patience 8 --batch-size 64 --lr 5e-4
```

预期输出：

- `outputs/hourly/transformer_predictions_seq6.csv`
- `outputs/hourly/transformer_predictions_seq6_val.csv`
- `outputs/hourly/transformer_seq6.pt`
- `outputs/hourly/transformer_seq6_config.json`
- `outputs/hourly/transformer_metrics.csv`
- `outputs/hourly/transformer_train_history.json`

---

## 5. 深度模型补完后，再跑分析汇总

这一步等上面 3 条命令全部成功后再执行。

### 5.1 误差分析

```powershell
py -3 scripts\run_hourly_analysis.py
```

### 5.2 跨城市泛化

```powershell
py -3 scripts\run_cross_city_xgboost.py
```

### 5.3 汇总总表

```powershell
py -3 scripts\assemble_hourly_outputs.py
```

### 5.4 生成图

```powershell
py -3 scripts\generate_hourly_figures.py
```

---

## 6. 建议的最小执行顺序

如果只想补齐深度模型，最少跑这 3 条：

```powershell
py -3 scripts\train_lstm.py --task seq6 --profile handoff
py -3 scripts\train_transformer.py --task h1 --epochs 60 --patience 8 --batch-size 64 --lr 5e-4
py -3 scripts\train_transformer.py --task seq6 --epochs 80 --patience 8 --batch-size 64 --lr 5e-4
```

如果想把整套实验收尾，就再加上这 4 条：

```powershell
py -3 scripts\run_hourly_analysis.py
py -3 scripts\run_cross_city_xgboost.py
py -3 scripts\assemble_hourly_outputs.py
py -3 scripts\generate_hourly_figures.py
```

---

## 7. 如果要快速检查是否跑成功

至少看下面这些文件是否生成：

- `outputs/hourly/lstm_predictions_seq6.csv`
- `outputs/hourly/transformer_predictions_h1.csv`
- `outputs/hourly/transformer_predictions_seq6.csv`
- `outputs/hourly/lstm_metrics.csv`
- `outputs/hourly/transformer_metrics.csv`

全部补齐后，再看：

- `outputs/hourly/overall_metrics_summary.csv`
- `outputs/hourly/experiment_registry.csv`
- `outputs/hourly/analysis_notes.md`
- `outputs/hourly/figures/`

---

## 8. 一句话版

已经跑完的不要动，当前只补：

```powershell
py -3 scripts\train_lstm.py --task seq6 --profile handoff
py -3 scripts\train_transformer.py --task h1 --epochs 60 --patience 8 --batch-size 64 --lr 5e-4
py -3 scripts\train_transformer.py --task seq6 --epochs 80 --patience 8 --batch-size 64 --lr 5e-4
```
