# [3] Poelzl et al. (2025) — Transfer Learning for PM10 Prediction

## 完整引用

Poelzl, M., Kern, R., Kecorius, S., & Lovrić, M. (2025). Exploration of transfer learning techniques for the prediction of PM10. *Scientific Reports*, 15, 2919.

- **DOI**: https://doi.org/10.1038/s41598-025-86550-6
- **期刊**: Scientific Reports (Nature Portfolio, IF 4.6)
- **引用量**: 10+
- **许可**: CC BY 4.0（开放获取）

## 摘要

Modelling of pollutants provides valuable insights into air quality dynamics, aiding exposure assessment where direct measurements are not viable. Machine learning (ML) models can be employed to explore such dynamics, including the prediction of air pollution concentrations, yet demanding extensive training data. To address this, techniques like transfer learning (TL) leverage knowledge from a model trained on a rich dataset to enhance one trained on a sparse dataset, provided there are similarities in data distribution. In our experimental setup, we utilize meteorological and pollutant data from multiple governmental air quality measurement stations in Graz, Austria, supplemented by data from one station in Zagreb, Croatia to simulate data scarcity. Common ML models such as Random Forests, Multilayer Perceptrons, Long-Short-Term Memory, and Convolutional Neural Networks are explored to predict particulate matter in both cities. Our detailed analysis of PM10 suggests that similarities between the cities and the meteorological features exist and can be further exploited. Hence, TL appears to offer a viable approach to enhance PM10 predictions for the Zagreb station, despite the challenges posed by data scarcity. Our results demonstrate the feasibility of different TL techniques to improve particulate matter prediction on transferring a ML model trained from all stations of Graz and transferred to Zagreb. Through our investigation, we discovered that selectively choosing time spans based on seasonal patterns not only aids in reducing the amount of data needed for successful TL but also significantly improves prediction performance. Specifically, training a Random Forest model using data from all measurement stations in Graz and transferring it with only 20% of the labelled data from Zagreb resulted in a 22% enhancement compared to directly testing the trained model on Zagreb.

## 关键发现

- **跨城市场景**：Graz（奥地利，数据丰富）→ Zagreb（克罗地亚，数据稀缺）
- **迁移学习有效性**：仅用 20% 目标城市标注数据，实现 **22% 性能提升**
- **季节模式选择**：基于季节模式选择训练时段，可进一步减少所需数据量并提升性能
- **多模型验证**：Random Forest、MLP、LSTM、CNN 四种模型均验证了迁移学习的可行性
- **关键成功因素**：城市间气象相似性是迁移学习成功的前提

## 与本项目的关联

- **最直接的跨城市实验参照**：与本项目"北京训练→上海测试"的跨城市泛化实验高度一致
- **方法论验证**：此文证明了迁移学习在空气质量预测中的可行性
- **结果对比基础**：本项目的跨城市泛化误差增量（h1: +6.57 RMSE）可与此文的结论对照
- **改进方向**：此文的季节模式选择策略可应用于本项目后续工作

## 获取原文

- **Nature 开放获取**: https://www.nature.com/articles/s41598-025-86550-6
- **PMC 全文**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11757726
- **DOI**: https://doi.org/10.1038/s41598-025-86550-6
