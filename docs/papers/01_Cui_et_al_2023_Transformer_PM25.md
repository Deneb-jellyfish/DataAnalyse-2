# [1] Cui et al. (2023) — Transformer vs CNN-LSTM-Attention for PM2.5

## 完整引用

Cui, B., Liu, M., Li, S., Jin, Z., Zeng, Y., & Lin, X. (2023). Deep learning methods for atmospheric PM2.5 prediction: A comparative study of transformer and CNN-LSTM-attention. *Atmospheric Pollution Research*, 14(9), 101833.

- **DOI**: https://doi.org/10.1016/j.apr.2023.101833
- **期刊**: Atmospheric Pollution Research (IF 3.9)
- **引用量**: 88+
- **出版商**: Elsevier

## 摘要

（来自论文原文）该研究首次将 Transformer 架构应用于长时间序列 PM2.5 浓度预测，在北京 12 个空气质量监测站 2013–2016 年的小时级数据上进行实验。研究系统比较了 Transformer 与 CNN-LSTM-Attention 两种深度学习方法的预测性能。结果表明，Transformer 模型在所有评价指标上均显著优于 CNN-LSTM-Attention，尤其在秋冬季复杂污染场景下表现突出。

## 关键发现

| 指标 | Transformer | CNN-LSTM-Attention | 提升 |
|------|-------------|---------------------|------|
| R²（预测） | **94.4%** | 83.6% | ~13% |
| EVS | — | — | +12% |
| MAE | — | — | +9% |
| MSE | — | — | +6% |

- Transformer 擅长捕捉短期污染突变（由气象条件突变引起）和长期季节趋势
- 自注意力机制能有效建模空气污染物与气象变量之间的长程依赖关系
- 这是首个将 Transformer 应用于 PM2.5 浓度长期预测的研究

## 与本项目的关联

- **数据集一致**：同样使用北京 12 站点小时级数据（2013–2016）
- **任务一致**：均为小时级 PM2.5 预测
- **模型对比重合**：本项目包含 Transformer 和 LSTM 变体
- **核心参考价值**：可作为本项目 Transformer 模型表现的直接参照基准
- 本项目在 `single-step h1` 任务中 LSTM 略优于 Transformer（RMSE 11.046 vs 11.167），与此文 Transformer 最优的结论形成有趣对照——可能与任务设置差异（单步 vs 长序列）有关

## 获取原文

- **ScienceDirect**: https://www.sciencedirect.com/science/article/abs/pii/S1309104223001873
- **DOI**: https://doi.org/10.1016/j.apr.2023.101833
- **Semantic Scholar**: https://www.semanticscholar.org/paper/72c16e09cf96ce0f3646fddffc85cf30225a8c24
- 注：此文为 Elsevier 付费期刊，需通过机构访问或 Sci-Hub 获取全文
