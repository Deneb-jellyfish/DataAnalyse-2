# [2] Wu et al. (2025) — PM2.5 Forecasting Systematic Review

## 完整引用

Wu, C., Wang, R., Lu, S., Tian, J., Yin, L., Wang, L., & Zheng, W. (2025). Time-series data-driven PM2.5 forecasting: From theoretical framework to empirical analysis. *Atmosphere*, 16(3), 292.

- **DOI**: https://doi.org/10.3390/atmos16030292
- **期刊**: Atmosphere (MDPI, IF 2.9)
- **引用量**: 51+
- **许可**: CC BY 4.0（开放获取）

## 摘要

PM2.5 in air pollution poses a significant threat to public health and the ecological environment. There is an urgent need to develop accurate PM2.5 prediction models to support decision-making and reduce risks. This review comprehensively explores the progress of PM2.5 concentration prediction, covering bibliometric trends, time series data characteristics, deep learning applications, and future development directions. This article obtained data on 2327 journal articles published from 2014 to 2024 from the WOS database. Bibliometric analysis shows that research output is growing rapidly, with China and the United States playing a leading role, and recent research is increasingly focusing on data-driven methods such as deep learning. Key data sources include ground monitoring, meteorological observations, remote sensing, and socioeconomic activity data. Deep learning models (including CNN, RNN, LSTM, and Transformer) perform well in capturing complex temporal dependencies. With its self-attention mechanism and parallel processing capabilities, Transformer is particularly outstanding in addressing the challenges of long sequence modeling. Despite these advances, challenges such as data integration, model interpretability, and computational cost remain. Emerging technologies such as meta-learning, graph neural networks, and multi-scale modeling offer promising solutions while integrating prediction models into real-world applications such as smart city systems can enhance practical impact.

## 关键发现

- 基于 WOS 数据库 2327 篇文献的系统计量分析
- **CNN、RNN、LSTM、Transformer** 已成为主流架构
- **混合模型**（CNN-LSTM、CNN-BiLSTM、CNN-GRU）持续优于单一模型
- 多源数据融合（地面监测 + 气象 + 卫星 AOD + 社会经济）显著提升精度
- Transformer 的自注意力机制在长序列建模中表现最佳
- 未来方向：图神经网络、元学习、物理驱动深度学习、多尺度建模、在线学习
- 模型可解释性和跨区域泛化仍是核心挑战

## 与本项目的关联

- **项目背景支撑**：为项目引言提供了权威的领域全景
- **方法选型依据**：本项目的五模型选择（ARIMA/Prophet/XGBoost/LSTM/Transformer）与此综述的主流架构高度一致
- **后续工作方向**：TFT、Informer、N-BEATS 等建议直接来自此文
- **跨区域泛化挑战**：此文明确指出的挑战正是本项目跨城市实验的核心问题

## 获取原文

- **MDPI 开放获取**: https://www.mdpi.com/2073-4433/16/3/292
- **PDF 直接下载**: https://mdpi.com/2073-4433/16/3/292/pdf
- **DOI**: https://doi.org/10.3390/atmos16030292
