# [9] Lundberg & Lee (2017) — SHAP (Model Interpretability)

## 完整引用

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765–4774.

- **会议**: NeurIPS 2017
- **引用量**: 40,000+
- **arXiv**: https://arxiv.org/abs/1705.07874
- **GitHub**: https://github.com/shap/shap

## 摘要

（来自论文原文）Understanding why a model makes a certain prediction can be as crucial as the prediction's accuracy in many applications. However, the highest accuracy for large modern datasets is often achieved by complex models that even experts struggle to interpret, such as ensemble or deep learning models, creating a tension between accuracy and interpretability. In response, various methods have recently been proposed to help users interpret the predictions of complex models, but it is often unclear how these methods are related and when one method is preferable over another. To address this problem, we present a unified framework for interpreting predictions, SHAP (SHapley Additive exPlanations). SHAP assigns each feature an importance value for a particular prediction. Its novel components include: (1) the identification of a new class of additive feature importance measures, and (2) theoretical results showing there is a unique solution in this class with a set of desirable properties. The new class unifies six existing methods, notable because several recent methods in the class lack the proposed desirable properties. SHAP shows strong practical advantages across multiple problems.

## 关键贡献

**SHAP 值**（基于 Shapley 博弈论）：

- 特征 $i$ 的 SHAP 值 = 该特征在所有可能的特征子集上的边际贡献的加权平均
- 满足三个理想性质：局部精度（Local Accuracy）、缺失性（Missingness）、一致性（Consistency）
- 统一了六种已有的特征归因方法：LIME、DeepLIFT、Layer-Wise Relevance Propagation、Shapley 回归值、Shapley 采样值、定量输入影响

**可视化工具**：
- SHAP Summary Plot：全局特征重要性排名 + 影响方向
- SHAP Dependence Plot：单特征依赖关系
- SHAP Force Plot：单个预测的解释

## 与本项目的关联

- 本项目使用 SHAP 对 XGBoost 模型进行全局可解释性分析
- `single-step h1` 任务中：排名靠前特征为 `pm25_lag1`、滚动均值、短差分与近时刻气象变量
- `seq6` 任务中：风速变化、湿度、时段周期和交互项的重要性上升
- SHAP 分析验证了本项目的核心结论：多步预测对"趋势 + 气象扰动"的依赖比单步更明显

## 获取原文

- **arXiv 开放获取**: https://arxiv.org/pdf/1705.07874.pdf
- **NeurIPS Proceedings**: https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
- **SHAP 文档**: https://shap.readthedocs.io/
