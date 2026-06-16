# [5] Taylor & Letham (2018) — Prophet: Forecasting at Scale

## 完整引用

Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37–45.

- **DOI**: https://doi.org/10.1080/00031305.2017.1380080
- **期刊**: The American Statistician (Taylor & Francis)
- **引用量**: 2770+
- **预印本**: PeerJ Preprints, https://doi.org/10.7287/peerj.preprints.3190v2

## 摘要

（来自论文原文）Forecasting is a common data science task that helps organizations with capacity planning, goal setting, and anomaly detection. Despite its importance, there are serious challenges associated with producing reliable and high-quality forecasts — especially when there are a variety of time series and analysts with expertise in time series modeling are rare. To address these challenges, we describe a practical approach to forecasting "at scale" that combines configurable models with analyst-in-the-loop performance analysis. We propose a modular regression model with interpretable parameters that can be intuitively adjusted by analysts with domain knowledge about the time series. We describe performance analyses to compare and evaluate forecasting procedures, and automatically flag forecasts for manual review and adjustment. Tools that help analysts use their expertise most effectively enable reliable, practical forecasting of business time series.

## 关键贡献

Prophet 模型将时间序列分解为三个主要成分：

$$y(t) = g(t) + s(t) + h(t) + \varepsilon_t$$

- **$g(t)$（趋势项）**：分段线性或 logistic 增长曲线，自动检测趋势变化点
- **$s(t)$（季节项）**：傅里叶级数建模的多周期季节性（日/周/年）
- **$h(t)$（假日效应）**：节假日及特殊事件的独立影响项
- **$\varepsilon_t$（误差项）**：服从正态分布的随机误差

Prophet 的核心设计理念：
- 分析师在回路（analyst-in-the-loop）：允许领域专家直观调整参数
- 对缺失值和趋势变化具有鲁棒性
- 自动生成不确定性区间

## 与本项目的关联

- 本项目使用增强版 Prophet，在传统 Prophet 基础上加入 48 维外生回归项
- 传统 Prophet 在小时级 PM2.5 上严重退化为"平滑均值线"
- 增强后，Prophet 的 `single-step h1` RMSE 降至 11.800，`seq6-average` 升至第二名（RMSE 26.480）
- 此实验直接证明了 Prophet 原文中"分析师在回路"理念的重要性——特征工程就是本项目的"分析师输入"

## 获取原文

- **Taylor & Francis**: https://www.tandfonline.com/doi/full/10.1080/00031305.2017.1380080
- **PeerJ 预印本**: https://peerj.com/preprints/3190/
- **DOI**: https://doi.org/10.1080/00031305.2017.1380080
