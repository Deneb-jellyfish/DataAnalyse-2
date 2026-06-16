# [7] Hochreiter & Schmidhuber (1997) — LSTM

## 完整引用

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

- **DOI**: https://doi.org/10.1162/neco.1997.9.8.1735
- **期刊**: Neural Computation (MIT Press)
- **引用量**: 100,000+（深度学习领域引用量最高的论文之一）

## 摘要

（来自论文原文）Learning to store information over extended time intervals by recurrent backpropagation takes a very long time, mostly because of insufficient, decaying error backflow. We briefly review Hochreiter's (1991) analysis of this problem, then address it by introducing a novel, efficient, gradient-based method called long short-term memory (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete-time steps by enforcing constant error flow through "constant error carousels" (CECs) within special units. Multiplicative gate units learn to open and close access to the constant error flow. LSTM is local in space and time; its computational complexity per time step and weight is O(1). Our experiments with artificial data involve local, distributed, real-valued, and noisy pattern representations. In comparisons with real-time recurrent learning, back propagation through time, recurrent cascade correlation, Elman nets, and neural sequence chunking, LSTM leads to many more successful runs, and learns much faster. LSTM also solves complex, artificial long-time-lag tasks that have never been solved by previous recurrent network algorithms.

## 关键贡献

**门控机制**（核心创新）：

- **遗忘门（Forget Gate）**：$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$ —— 控制历史信息的保留程度
- **输入门（Input Gate）**：$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$ —— 控制新信息的写入
- **输出门（Output Gate）**：$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$ —— 控制当前状态的输出
- **细胞状态（Cell State）**：$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ —— 恒定误差传送带（CEC）

LSTM 通过 CEC 机制解决了传统 RNN 的梯度消失问题，能够学习跨越 1000+ 时间步的长程依赖。

## 与本项目的关联

- LSTM 是本项目的核心深度学习模型
- 在 `single-step h1` 任务中：RMSE=11.046，**排名第一**，验证了 LSTM 对超短期局部依赖的强建模能力
- 在 `seq6-average` 任务中：RMSE=30.814，排名第四（多步预测不如 XGBoost 和增强版 Transformer）
- 本项目 LSTM 使用 seq_len=48（h1）和 seq_len=48/72（seq6），配合 Early Stopping 防止过拟合

## 获取原文

- **MIT Press Journals**: https://direct.mit.edu/neco/article/9/8/1735/6109/Long-Short-Term-Memory
- **DOI**: https://doi.org/10.1162/neco.1997.9.8.1735
- 注：此文为 MIT Press 期刊，需机构订阅。可通过 Sci-Hub 或 arXiv 上的后续相关论文获取。
