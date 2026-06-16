# [10] Diebold & Mariano (1995) — DM Test (Predictive Accuracy)

## 完整引用

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.

- **DOI**: https://doi.org/10.1080/07350015.1995.10524599
- **期刊**: Journal of Business & Economic Statistics (Taylor & Francis)
- **引用量**: 10,000+

## 摘要

（来自论文原文）We propose and evaluate explicit tests of the null hypothesis of no difference in the accuracy of two competing forecasts. In contrast to previously developed tests, a wide variety of accuracy measures can be used (in particular, the loss function need not be quadratic, nor need it be symmetric), and forecast errors can be non-Gaussian, nonzero mean, serially correlated, and contemporaneously correlated. As a result, our tests are applicable to a wide variety of forecast evaluation problems, ranging from those involving forecasts based on complicated parametric models estimated from long time series to those involving forecasts based on simple survey data. The tests themselves are easy to compute; furthermore, we propose a regression-based implementation that is even easier. Monte Carlo analysis indicates that the tests perform well in realistic sample sizes, and an empirical example provides an illustration.

## 关键贡献

**Diebold-Mariano 检验统计量**：

$$DM = \frac{\bar{d}}{\sqrt{\frac{2\pi\hat{f}_d(0)}{T}}}$$

其中：
- $d_t = L(e_{1t}) - L(e_{2t})$ 为两个模型预测损失的差分序列
- $L(\cdot)$ 为损失函数（不限于二次损失）
- $\hat{f}_d(0)$ 为差分序列在零频率处的谱密度估计

**关键特性**：
- 损失函数可以是任意形式（MSE、MAE、非对称损失等）
- 预测误差允许非正态、非零均值、序列相关和同期相关
- 零假设：两个预测精度无显著差异
- 适用于固定参数和递归估计参数的预测比较

## 与本项目的关联

- 本项目在评估框架中使用了 Diebold-Mariano 检验（`evaluation/dm_test.py`）
- 可用于判断 LSTM 与 XGBoost 在 h1 任务上的 RMSE 差异（11.046 vs 11.293）是否具有统计显著性
- 结合本项目多种误差指标（RMSE/MAE/MAPE）进行统计显著性检验

## 获取原文

- **Taylor & Francis**: https://www.tandfonline.com/doi/abs/10.1080/07350015.1995.10524599
- **DOI**: https://doi.org/10.1080/07350015.1995.10524599
- 注：此文为付费期刊，可通过机构访问或 Sci-Hub 获取全文。
- 也可参考 Diebold 个人网站上的工作论文版本。
