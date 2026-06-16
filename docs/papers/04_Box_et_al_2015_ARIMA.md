# [4] Box et al. (2015) — ARIMA 经典教材

## 完整引用

Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

- **ISBN**: 978-1-118-67502-1
- **出版社**: John Wiley & Sons
- **首版**: 1970 年（Box & Jenkins）
- **最新版**: 2015 年第 5 版

## 内容概要

本书是时间序列分析的经典教科书，由 Box 和 Jenkins 于 1970 年首次出版，开创了 ARIMA（自回归积分滑动平均）模型的完整理论框架。第 5 版由 Reinsel 和 Ljung 修订，新增了以下内容：

- 多元时间序列分析方法
- 状态空间模型与卡尔曼滤波
- 异方差时间序列模型（ARCH/GARCH）
- 单位根检验与协整理论
- 干预分析与异常值检测

## ARIMA 模型核心内容

ARIMA(p,d,q) 模型的基本形式为：

$$\phi(B)(1-B)^d y_t = c + \theta(B)\varepsilon_t$$

其中：
- $B$：滞后算子
- $\phi(B)$：自回归多项式
- $\theta(B)$：滑动平均多项式
- $\varepsilon_t$：白噪声序列
- $d$：差分阶数

模型构建的三阶段迭代策略：
1. **识别（Identification）**：通过 ACF/PACF 图确定 p, d, q
2. **估计（Estimation）**：最大似然法或最小二乘法估计参数
3. **诊断（Diagnostic Checking）**：残差白噪声检验

## 与本项目的关联

- 本项目使用 auto_arima 自动定阶功能实现 ARIMA 模型
- ARIMA 在 `single-step h1` 任务中 RMSE=11.271，排名第三
- ARIMA 在 `seq6-average` 任务中 RMSE=31.154，排名第五（多步预测能力有限）
- 此书是 ARIMA 模型的权威方法学来源

## 获取方式

- **出版社页面**: https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021
- 注：本书为商业教材（约 700 页），无在线免费版本。建议通过图书馆借阅或购买。
