# [6] Chen & Guestrin (2016) — XGBoost

## 完整引用

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).

- **DOI**: https://doi.org/10.1145/2939672.2939785
- **会议**: KDD 2016（数据挖掘顶级会议）
- **引用量**: 55,404+（机器学习领域引用量最高的论文之一）
- **arXiv**: https://arxiv.org/abs/1603.02754

## 摘要

Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable end-to-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.

## 关键贡献

**正则化目标函数**（核心创新）：

$$\mathcal{L}(\phi) = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k)$$

其中 $\Omega(f) = \gamma T + \frac{1}{2}\lambda\|w\|^2$，通过控制树的叶子节点数（$T$）和叶子权重（$w$）防止过拟合。

**算法创新**：
- **稀疏感知算法**：自动处理缺失值，学习最优的缺失值分裂方向
- **加权分位数草图**：支持近似树学习，大幅降低计算复杂度
- **缓存感知访问**：优化内存访问模式，提升硬件利用效率
- **数据压缩与分片**：支持分布式计算，可扩展到数十亿样本

## 与本项目的关联

- XGBoost 是本项目的核心强基线模型
- 在 `single-step h1` 任务中：RMSE=11.293，MAPE=11.418%（MAPE 最低）
- 在 `seq6-average` 任务中：RMSE=25.188，**排名第一**，优于所有深度模型
- XGBoost 的特征重要性分析和 SHAP 分析是本项目特征工程有效性验证的核心工具
- 本项目采用 6 个独立直接模型实现 seq6（XGBoost_h1 至 XGBoost_h6）

## 获取原文

- **arXiv 开放获取**: https://arxiv.org/pdf/1603.02754.pdf
- **ACM Digital Library**: https://dl.acm.org/doi/10.1145/2939672.2939785
- **DOI**: https://doi.org/10.1145/2939672.2939785
