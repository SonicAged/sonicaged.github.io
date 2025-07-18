---
title: Gugugu Neralashun
date: 2025-07-16 22:16:14
categories:
  - model
  - attention
  - category
tags:
  - CDR
  - model
  - Basic
  - deep learning
---

# 具有一般性的 Attention

<div style="display: flex; align-items: center;">
  <img src="\img\Attention\Gugugu Neralashun.png" style="width: 200px; margin-right: 20px;">
  <p>详细讨论了可以应用于任何类型注意力模型的基础机制，这些机制不依赖于特定的特征模型或查询模型。这一部分构成了注意力模型的核心计算框架，主要包括三个关键子方面：注意力评分函数(Attention Scoring)、注意力对齐(Attention Alignment)和注意力维度(Attention Dimensionality)。</p>
</div>



在阅读这篇博客前请先阅读 {% post_link Attention %}

<!-- more -->

## 回顾

<img src="/img/Attention/GeneralAttentionModule.png" alt="GeneralAttentionModule" width="50%" height="auto">

如果还记得这张图的话，那便是极好的捏 ~~不记得了就回去看捏~~ 本文主要就是在说明每一个模块常见的具体实现有什么

## 注意力评分函数(Attention Scoring)

注意力评分函数是计算查询向量$\mathbf{q}$与键向量$\mathbf{k}_l$之间相关性得分的核心组件：

1. **加性评分(Additive/Concatenate)**：

   $$
   \text{score}(\mathbf{q},\mathbf{k}\_l) = \mathbf{w}^\top \text{act}(\mathbf{W}\_1\mathbf{q} + \mathbf{W}\_2\mathbf{k}\_l + \mathbf{b})
   $$

   其中$\mathbf{w} \in \mathbb{R}^{d_w}$, $\mathbf{W}_1 \in \mathbb{R}^{d_w \times d_q}$, $\mathbf{W}_2 \in \mathbb{R}^{d_w \times d_k}$和$\mathbf{b} \in \mathbb{R}^{d_w}$是可训练参数。

2. **乘性评分(Multiplicative/Dot-Product)**：

   $$
   \text{score}(\mathbf{q},\mathbf{k}\_l) = \mathbf{q}^\top \mathbf{k}\_l
   $$

3. **缩放乘性评分(Scaled Multiplicative)**：

   $$
   \text{score}(\mathbf{q},\mathbf{k}\_l) = \frac{\mathbf{q}^\top \mathbf{k}\_l}{\sqrt{d_k}}
   $$

4. **通用评分(General)**：

   $$
   \text{score}(\mathbf{q},\mathbf{k}\_l) = \mathbf{k}\_l^\top \mathbf{W} \mathbf{q}
   $$

   其中$\mathbf{W} \in \mathbb{R}^{d_k \times d_q}$是权重矩阵。

5. **带偏置的通用评分(Biased General)**：

   $$
   \text{score}(\mathbf{q},\mathbf{k}\_l) = \mathbf{k}\_l^\top (\mathbf{W}\mathbf{q} + \mathbf{b})
   $$

6. **激活通用评分(Activated General)**：
   $$
   \text{score}(\mathbf{q},\mathbf{k}\_l) = \text{act}(\mathbf{k}\_l^\top \mathbf{W} \mathbf{q} + b)
   $$

## 注意力对齐(Attention Alignment)

对齐函数将原始注意力分数$\mathbf{e} = [e_1, \ldots, e_{n_f}]$转换为标准化权重：

1. **软对齐/全局对齐(Soft/Global Alignment)**：

   $$
   a\_l = \frac{\exp(e_l)}{\sum\_{j=1}^{n_f} \exp(e_j)}
   $$

2. **硬对齐(Hard Alignment)**：
   从多项式分布采样：

   $$
   m \sim \text{Multinomial}(a\_1, \ldots, a\_{n_f})
   $$

   然后：

   $$
   \mathbf{c} = \mathbf{v}\_m
   $$

3. **局部对齐(Local Alignment)**：
   窗口位置$p$的确定：
   $$
   p = S \times \text{sigmoid}(\mathbf{w}\_p^\top \tanh(\mathbf{W}\_p \mathbf{q}))
   $$
   然后计算：
   $$
   a\_l = \frac{\exp(e\_l)}{\sum\_{j=p-D}^{p+D} \exp(e\_j)} \exp\left(-\frac{(l-p)^2}{2\sigma^2}\right)
   $$

## 注意力维度(Attention Dimensionality)

1. **单维注意力(Single-Dimensional Attention)**：

   $$
   \mathbf{c} = \sum\_{l=1}^{n_f} a\_l \mathbf{v}\_l
   $$

2. **多维注意力(Multi-Dimensional Attention)**：
   调整评分函数产生向量分数：

   $$
   \mathbf{e}\_l = \mathbf{W}\_d^\top \text{act}(\mathbf{W}\_1 \mathbf{q} + \mathbf{W}\_2 \mathbf{k}\_l + \mathbf{b})
   $$

   然后计算：

   $$
   a\_{l,i} = \frac{\exp(e\_{l,i})}{\sum\_{j=1}^{n\_f} \exp(e\_{j,i})}
   $$

   最终上下文向量：

   $$
   \mathbf{c} = \sum\_{l=1}^{n\_f} \mathbf{a}\_l \circ \mathbf{v}\_l
   $$

   其中$\circ$表示逐元素乘法。

## 4. 实际应用与选择建议

论文提供了关于不同机制选择的实用建议：

1. **评分函数选择**：

   - 计算效率优先：乘性或缩放乘性评分
   - 性能优先：加性评分或通用评分
   - 大维度键向量：必须使用缩放乘性评分防止梯度问题

2. **对齐方式选择**：

   - 标准情况：软对齐
   - 需要严格稀疏性：硬对齐（但要注意训练难度）
   - 序列数据：考虑局部对齐
   - 需要动态选择关注区域：强化对齐

3. **维度选择**：
   - 大多数情况：单维注意力足够
   - 需要细粒度控制：考虑多维注意力

这些通用机制可以自由组合，例如可以设计一个使用加性评分、软对齐和多维注意力的模型。论文特别指出，Transformer 模型成功的关键在于巧妙地组合了缩放乘性评分、软对齐和单维注意力（通过多头机制实现类似多维注意力的效果）。

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/Brauwers和Frasincar - 2023 - A General Survey on Attention Mechanisms in Deep Learning.pdf" target="_blank">📄 Brauwers 和 Frasincar - 2023 - A General Survey on Attention Mechanisms in Deep Learning</a>

{% post_link Attention %}
