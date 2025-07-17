---
title: Query-Related-Attention
date: 2025-07-17 00:25:30
categories:
  - model
  - attention
  - query-related
tags:
  - CDR
  - model
  - Basic
  - deep learning
---

# 在查询上做文章的 Attention

查询是任何注意力模型的重要组成部分，因为它们直接决定了从特征向量中提取哪些信息。这些查询基于任务模型的期望输出，并可以解释为字面问题。一些查询具有特定的特征，需要特定类型的机制来处理它们。因此，这一类别封装了处理特定类型查询特征的注意力机制。这一类别的机制处理以下两种查询特征之一：查询的类型或查询的多重性。

在阅读这篇博客前请先阅读 {% post_link Attention %}

<!-- more -->

## 查询类型机制(Query Type Mechanisms)

### 基本查询与特殊查询

查询(Query)在注意力机制中扮演着关键角色，它决定了模型关注输入数据的哪些部分。3.3 节首先区分了两种基本查询类型：

1. **基本查询(Basic Queries)**：这类查询通常直接来源于模型结构或数据特征。例如：

   - RNN 中的隐藏状态作为序列生成过程中的查询
   - 辅助变量（如医疗图像分类中的患者特征）作为查询
   - 图像处理中 CNN 提取的特征向量作为查询

2. **特殊查询(Specialized Queries)**：用于特定注意力架构的查询，如：
   - 旋转注意力(Rotatory Attention)中使用上下文向量作为查询
   - 交互式协同注意力(Interactive Co-attention)中使用平均键向量作为查询
   - 注意力堆叠(Attention-over-Attention)中的多层查询

### 自注意力机制(Self-Attention)

自注意力（或称内部注意力）是查询相关机制中最重要的创新之一，它允许模型通过数据自身生成查询：

1. **自注意力的两种解释**：

   - **恒定查询解释**：将查询视为固定问题（如"文档属于哪类？"）
   - **可学习查询解释**：将查询作为可训练参数，随模型优化

2. **技术实现**：
   自注意力通过线性变换从特征矩阵 F 生成查询矩阵 Q：

   $$
   Q = W_Q \times F
   $$

   其中 $W_Q \in \mathbb{R}^{d_q \times d_f}$ 是可训练权重矩阵。

3. **自注意力的应用价值**：

   - 揭示特征向量间的关系（如词语依赖、图像区域关联）
   - 生成改进的特征表示，可通过两种方式：
  
     $$
     f^{(new)} = c \quad \text{或} \quad f^{(new)} = \text{Normalize}(f^{(old)} + c)
     $$

   - 在 Transformer 架构中作为核心组件

4. **领域应用**：
   - 计算机视觉：图像识别、GANs 中的区域聚焦
   - 视频处理：时空关系建模
   - 语音处理：语音识别
   - 自然语言处理：情感分析、机器翻译
   - 图网络：节点关系建模

## 多重查询机制(Multi-Query Mechanisms)

### 多头注意力(Multi-Head Attention)

<img src="/img/Attention/MultiheadAttention.png" alt="Multi-head Attention" width="60%" height="auto">

多头注意力是处理多重查询的核心技术，其关键特点包括：

1. **并行注意力头**：

   - 每个头有独立的$W_Q^{(j)}, W_K^{(j)}, W_V^{(j)}$矩阵
   - 生成不同的查询表示：$q^{(j)} = W_Q^{(j)} \times q$
   - 允许模型同时关注不同方面的信息

2. **实现细节**：

   - 每个头产生独立的上下文向量$c^{(j)}$
   - 最终输出通过线性变换合并 $c = W_O \times \text{concat}(c^{(1)},...,c^{(d)})$

3. **优势**：
   - 增强模型捕捉多样化关系的能力
   - 在 Transformer 中实现并行计算
   - 可解释性强（不同头可学习不同关注模式）

### 多跳注意力(Multi-Hop Attention)

<img src="/img/Attention/MultihopAttention.png" alt="Multi-hop Attention" width="60%" height="auto">

多跳注意力通过序列化处理逐步精炼查询和上下文：

1. **工作机制**：

   - 迭代更新查询：$q^{(s+1)} = \text{transform}(q^{(s)}, c^{(s)})$
   - 逐步积累信息：使用$[q^{(s+1)}, c^{(s)}]$作为新查询
   - 可视为信息传递的"深度"处理

2. **与多头注意力的区别**：
   | 特性 | 多头注意力 | 多跳注意力 |
   |------|-----------|-----------|
   | 处理方式 | 并行 | 串行 |
   | 计算效率 | 高 | 较低 |
   | 信息整合 | 拼接 | 迭代精炼 |
   | 典型应用 | Transformer 编码层 | Transformer 解码层 |

3. **变体实现**：
   - 使用相同权重矩阵的轻量级版本
   - 结合自注意力机制的增强版本
   - 在 LCR-Rot-hop 等模型中的应用

#### 胶囊注意力(Capsule-Based Attention)

<img src="/img/Attention/CapsuleAttention.png" alt="Capsule Attention" width="60%" height="auto">

Capsule-based attention通过将注意力机制与胶囊网络相结合，在多分类任务中展现出独特的优势，特别是在需要可解释性和细粒度分类的场景中。其模块化设计也便于扩展和调整，是注意力机制研究中的一个重要分支。

##### 基本概念与核心思想

Capsule-based attention是一种特殊的注意力机制，它通过为每个类别创建独立的"胶囊"(capsule)来处理多分类问题。每个胶囊本质上是一个独立的注意力模块，专门负责识别和提取与该类别相关的特征信息。

###### 胶囊网络基础

Capsule-based attention源自胶囊网络(Capsule Networks)的概念，由Hinton等人于2017年提出。与传统神经网络使用标量神经元不同，胶囊网络使用向量形式的"胶囊"来表示实体及其属性。这种表示方式能够更好地捕捉特征间的空间层次关系。

###### 注意力机制的结合

将注意力机制与胶囊网络结合，形成了capsule-based attention。这种机制的核心思想是：
- 每个类别对应一个独立的注意力模块(胶囊)
- 每个胶囊学习自己独特的"查询"(query)，用于从输入特征中提取相关信息
- 通过注意力权重明确显示哪些特征对分类决策最重要

##### 架构与工作流程

Capsule-based attention模型通常由三个主要组件构成：

1. **注意力模块**

    每个胶囊c的注意力模块计算过程如下：

    1. **键值生成**：将输入特征向量 $f_l$ 转换为键 $k_l$ 和值 $v_l$ ： $k_l = W_K^{(c)} f_l, \quad v_l = W_V^{(c)} f_l$
    其中$W_K^{(c)}$和$W_V^{(c)}$是类别特定的可训练权重矩阵

    2. **注意力分数计算**：使用查询向量 $q_l$ 计算注意力分数 $e_{c,l} = q_c^T k_l$

    3. **注意力权重计算**：通过softmax归一化 $a_{c,l} = \frac{\exp(e_{c,l})}{\sum_j \exp(e_{c,j})}$

    4. **上下文向量生成**：加权求和得到类别特定的上下文向量 $
    c_c = \sum_l a_{c,l} v_l$

2. **概率模块**

    上下文向量经过一个简单的分类层，输出该类别的预测概率 $p_c = \sigma(w_c^T c_c + b_c)$ ,其中 $\sigma$ 是sigmoid激活函数， $w_c$ 和 $b_c$ 是可训练参数。

3. **重构模块**

    最后将概率与上下文向量结合，生成类别表示向量 $r_c = p_c \times c_c$



##### 训练机制与损失函数

Capsule-based attention采用联合训练策略，优化两个目标：

1. **分类损失**

    使用标准的交叉熵损失函数确保分类准确性 $\mathcal{L}_{cls} = -\sum_c y_c \log p_c$ ，其中 $y_c$ 是类别 $c$ 的真实标签。

2. **重构损失**

    引入重构损失促使模型学习有意义的表示 $\mathcal{L}_{rec} = \sum_c \|r_c - \bar{f}\|^2$ ，其中$\bar{f}$是所有输入特征向量的平均值。

3. **联合训练**

    总损失函数是两者的加权和 $\mathcal{L} = \mathcal{L}\_{cls} + \lambda \mathcal{L}\_{rec}$ ，其中 $\lambda$ 是超参数，控制重构损失的重要性。

##### 技术特点与优势

1.  **多查询机制**：与传统注意力不同，capsule-based attention为每个类别学习独立的查询向量 $q_c$ ，使得模型能够并行处理多个分类任务，捕捉不同类别关注的不同特征，减少类别间的干扰。

2. **可解释性**：通过分析注意力权重a_{c,l}，可以直观理解哪些输入特征对特定类别最重要、模型做出决策的依据以及不同类别关注的特征差异

3. **鲁棒性**：重构损失项使模型学习到的表示更加鲁棒，迫使胶囊学习有意义的特征组合，减少对噪声特征的依赖，提高对抗样本的抵抗力

##### 变体与扩展

1. **动态路由capsule**

    引入动态路由机制，迭代调整胶囊间的连接强度：

    $$
    \text{for iteration } t:
    \quad b\_{i,j} \leftarrow b\_{i,j} + \hat{v}\_j^T u\_{j|i}
    \quad c\_{i,j} = \text{softmax}(b\_{i,j})
    \quad s\_j = \sum_i c\_{i,j} u\_{j|i}
    \quad v\_j = \text{squash}(s\_j)
    $$

2. **多头capsule**

每个胶囊使用多个注意力头，捕获不同方面的信息 $r_c = \text{concat}(r_c^{(1)}, ..., r_c^{(h)})$

3. **层次化capsule**

构建多级胶囊结构，从低级特征到高级语义逐步抽象：

$$
\text{低级胶囊} \rightarrow \text{中级胶囊} \rightarrow \text{高级胶囊}
$$

##### 与其他注意力机制的比较

| 特性 | Capsule-based Attention | 传统注意力 | 多头注意力 |
|------|------------------------|------------|------------|
| 查询数量 | 每个类别一个 | 通常一个 | 固定数量 |
| 参数共享 | 胶囊间不共享 | 完全共享 | 头间共享 |
| 可解释性 | 高 | 中等 | 低 |
| 计算成本 | 较高 | 低 | 中等 |
| 适用任务 | 多分类/多标签 | 通用 | 序列处理 |



## Transformer 架构中的查询机制

3.3 节特别强调了 Transformer 模型如何整合多种查询机制：

1. **关键整合**：

   - 多头自注意力并行处理
   - 层间多跳式信息传递
   - 查询-键-值分离的灵活设计

2. **变体发展**：

   - **Transformer-XL**：通过循环机制扩展上下文窗口
   - **Reformer**：通过 LSH 哈希提升计算效率
   - **Linformer**：低秩近似实现线性复杂度
   - **Synthesizer**：探索非成对注意力权重

3. **应用领域扩展**：
   - 图像描述生成（Image Captioning）
   - 医学图像分割
   - 对话系统响应生成
   - 推荐系统（如 BERT4Rec）

## 查询机制的选择与实践建议

1. **机制选择准则**：
   | 任务需求 | 推荐机制 | 优势 |
   |---------|---------|------|
   | 并行处理 | 多头注意力 | 计算效率高 |
   | 深度特征交互 | 多跳注意力 | 信息整合深入 |
   | 细粒度分类 | 胶囊注意力 | 可解释性强 |
   | 长序列处理 | Transformer-XL | 上下文扩展 |
   | 资源受限 | Linformer | 线性复杂度 |

2. **实现注意事项**：

   - 查询维度$d_q$应与键维度$d_k$匹配
   - 多头注意力的头数需平衡效果与计算成本
   - 多跳注意力的深度可能导致梯度消失
   - 胶囊注意力适合中等规模类别数

3. **新兴研究方向**：
   - 动态查询路由机制
   - 查询生成的条件控制
   - 跨模态查询对齐
   - 基于能量的查询选择

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/Brauwers和Frasincar - 2023 - A General Survey on Attention Mechanisms in Deep Learning.pdf" target="_blank">📄 Brauwers 和 Frasincar - 2023 - A General Survey on Attention Mechanisms in Deep Learning</a>

{% post_link Attention %}
