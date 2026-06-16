# [8] Vaswani et al. (2017) — Transformer (Attention Is All You Need)

## 完整引用

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998–6008.

- **会议**: NeurIPS 2017（机器学习顶级会议）
- **引用量**: 150,000+（AI 领域引用量最高的论文之一）
- **arXiv**: https://arxiv.org/abs/1706.03762

## 摘要

（来自论文原文）The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show that these models are superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.

## 关键贡献

**自注意力（Scaled Dot-Product Attention）**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **多头注意力（Multi-Head Attention）**：并行学习不同子空间的注意力模式
- **位置编码（Positional Encoding）**：通过正弦/余弦函数注入序列位置信息
- **编码器-解码器架构**：完全基于注意力机制，无需递归或卷积
- **并行训练**：训练效率远超 RNN/LSTM

## 与本项目的关联

- Transformer 是本项目的深度学习对照模型
- 在 `single-step h1` 任务中：RMSE=11.167，排名第二，接近 LSTM
- 在 `seq6-average` 任务中：RMSE=26.669，排名第三
- 与 Cui 等[1] 在同一北京数据集上验证的 R²=94.4% 形成对比——本项目使用更精简的架构和不同的任务定义
- 本项目 Transformer 使用 seq_len=48（h1）和 seq_len=48/72（seq6），最后通过全连接层输出单点或 6 维多步预测

## 获取原文

- **arXiv 开放获取**: https://arxiv.org/pdf/1706.03762.pdf
- **NeurIPS Proceedings**: https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html
- **GitHub 源码**: https://github.com/tensorflow/tensor2tensor
