---
title: Fufufu Relashinala
categories:
  - Deep Learning
  - Model
  - Attention
  - Category
tags:
  - model
  - Basic
  - deep learning
  - attention
date: 2025-07-14 22:30:57
---

# 在输入特征上做文章的 Attention

本文将接着详细说明一种基于输入特征分类 Attention 的方式，并介绍在这种分类方式下关注到的不同的 Attention 的架构。

<div style="display: flex; align-items: center;">
  <img src="\img\Attention\Fufufu Relashinala.png" style="width: 200px; margin-right: 20px;">
  <p>
  具体来说，本文主要探讨了基于输入特征特性的注意力机制变体。本节根据输入特征的不同特性，将特征相关的注意力机制分为三类：特征多重性(Multiplicity of Features)、特征层级(Levels of Features)和特征表示(Feature Representations)。
  </p>
</div>



在阅读这篇博客前请先阅读 {% post_link Attention %}

<!-- more -->

## 特征多重性(Multiplicity of Features)

这部分讨论了如何处理多个输入源的特征，主要分为单一特征注意力和多特征注意力机制。

### 单一特征注意力(Singular Features Attention)

大多数任务模型只处理单一输入(如图像、句子或声音序列)，使用单一特征注意力机制。这种机制直接对单个输入的特征向量进行注意力计算。

### 多特征注意力机制

当模型需要同时处理多个输入源时，需要特殊的多特征注意力机制：

**协同注意力(Co-attention)**

- 分为 **粗科粒度(Coarse-grained)** 和 **细颗粒度(Fine-grained)** 两种
- **粗颗粒度协同**注意力使用一个输入的*紧凑表示*作为查询来关注另一个输入
- **细颗粒度协同**注意力使用一个输入的所有特征向量作为查询

#### 粗颗粒度协同

论文给出的粗颗粒度协同的实例是**alternating co-attention**

##### alternating co-attention

<img src="/img/Attention/AlternatingCo-Attention.png" alt="alternating co-attention" width="60%" height="auto">

如上图所示，这是 alternating co-attention 的架构图，该机制交替使用两个输入的特征矩阵，先计算第一个输入的注意力，将其上下文向量作为查询计算第二个输入的注意力，然后再用第二个输入的上下文向量重新计算第一个输入的注意力。

这里现给出他的 score 函数

对于有序列输入的 Attention：

$$
\mathrm{score}(\underset{d\_{q}\times1}{\boldsymbol{q}},\underset{d\_{k}\times1}{\boldsymbol{k}\_{l}})=\underset{1\times d\_{w}}{\boldsymbol{w}^{T}}\times\mathrm{act}(\underset{d\_{w}\times d\_{q}}{\boldsymbol{W}\_{1}}\times\underset{d\_{q}\times1}{\boldsymbol{q}}+\underset{d\_{w}\times d\_{k}}{\boldsymbol{W}\_{2}}\times\underset{d\_{k}\times1}{\boldsymbol{k}\_{l}}+\underset{d\_{w}\times1}{\boldsymbol{b}})
$$

对于无序列输入的 Attention ~~（这是一种自注意力机制，后面会提到）~~ ：

$$
\underset{1\times1}{e\_{l}^{(0)}}=\underset{1\times d\_{w}}{\boldsymbol{w}^{(1)T}}\times\operatorname{act}(\underset{d\_{w}\times d\_{k}^{(1)}}{\boldsymbol{W}^{(1)}}\times\underset{d\_{k}^{(1)}\times1}{\boldsymbol{k}\_{l}^{(1)}}+\underset{d\_{w}\times1}{\boldsymbol{b}^{(1)}})
$$

对于第二层 Attention：

$$
\underset{1\times1}{e\_{l}^{(2)}}=\mathrm{score}(\underset{d\_{v}^{(1)}\times 1}{\boldsymbol{c}^{(0)}},\underset{d\_{k}^{(2)}\times1}{\boldsymbol{k}\_{l}^{(2)}})
$$

对于第三层 Attention：

$$
\underset{1\times1}{e\_{l}^{(1)}}=\mathrm{score}(\underset{d\_{v}^{(2)}\times 1}{\boldsymbol{c}^{(2)}},\underset{d\_{k}^{(1)}\times1}{\boldsymbol{k}\_{l}^{(1)}})
$$

生成的上下文向量$\boldsymbol{c}^{(1)}$和$\boldsymbol{c}^{(2)}$被连接起来，并在输出模型中用于预测。交替协同注意力由于需要一个接一个地计算上下文向量，因此本质上包含了*一种顺序性*。这可能会带来计算上的劣势，因为*无法并行*化。

##### interactive co-attention

<img src="/img/Attention/InteractiveCo-Attention.png" alt="interactive co-attention" width="60%" height="auto">

- 并行计算两个输入的注意力
- 使用未加权平均的关键向量作为查询
- 计算效率更高，可以并行处理

$$
\underset{d\_k^{(i)}\times1}{\bar{\boldsymbol{k}}^{(i)}}=\frac{1}{n\_f^{(i)}}\sum\limits\_{l=1}^{n\_f^{(i)}}\underset{d\_k^{(i)}\times1}{\boldsymbol{k}\_l^{(i)}}, \quad \underset{1\times1}{e\_{l}^{(i)}}=\mathrm{score}(\underset{d\_{k}^{(3-i)}\times1}{\bar{\boldsymbol{k}}^{(3-i)}},\underset{d\_{k}^{(i)}\times1}{\boldsymbol{k}\_{l}^{(i)}}) , \qquad i=1,2
$$

#### 细颗粒度协同

虽然粗粒度的共同注意力机制使用一个输入的紧凑表示作为查询，以计算另一个输入的注意力，但细粒度的共同注意力在计算注意力分数时会单独考虑每个输入的每个元素。在这种情况下，查询变成了一个矩阵。

##### 并行协同注意力(Parallel Co-attention)

<img src="/img/Attention/ParallelCo-Attention.png" alt="parallel co-attention" width="60%" height="auto">

- 同时计算两个输入的注意力
- 使用亲和矩阵(Affinity Matrix)转换关键向量空间
- 通过聚合形式计算注意力分数

我们有两种方式生成矩阵 $\mathbf{A}$

$$
\underset{n\_{f}^{(1)}\times n\_{f}^{(2)}}{\mathbf{A}}=\operatorname{act}(\underset{n\_{f}^{(1)}\times d\_{k}^{(1)}}{\begin{array}{c}K^{(1)^{T}}\end{array}}\times\underset{d\_{k}^{(1)}\times d\_{k}^{(2)}}{\begin{array}{c}W\_{A}\end{array}}\times\underset{d\_{k}^{(2)}\times n\_{f}^{(2)}}{\begin{array}{c}K^{(2)}\end{array}})
$$

$$
\underset{1\times1}{A\_{i,j}}=\underset{1\times3d\_{k}}{\boldsymbol{w}\_{A}^{T}}\times\mathrm{concat}(\underset{d\_{k}\times1}{\boldsymbol{k}\_{i}^{(1)}},\underset{d\_{k}\times1}{\boldsymbol{k}\_{j}^{(2)}},\underset{d\_{k}\times1}{\boldsymbol{k}\_{i}^{(1)}}\circ\underset{d\_{k}\times1}{\boldsymbol{k}\_{j}^{(2)}})
$$

其中，$\circ$表示哈德曼积

Affinity Matrix 可以解释为两个键矩阵列的相似性矩阵，并有助于将图像键转换到与句子中单词的键相同的空间，反之亦然。

由于现在查询由向量变成了矩阵，因此 score 函数也发生了变化

$$
e^{(1)} =\boldsymbol{w}\_{1}\times\mathrm{act}(\boldsymbol{W}\_{2}\times\boldsymbol{K}^{(2)}\times\boldsymbol{A}^{T}+\boldsymbol{W}\_{1}\times\boldsymbol{K}^{(1)})
$$

$$
e^{(2)} =\boldsymbol{w}\_{2}\times\mathrm{act}(\boldsymbol{W}\_{1}\times\boldsymbol{K}^{(1)}\times\boldsymbol{A}^{\:\:}+\boldsymbol{W}\_{2}\times\boldsymbol{K}^{(2)})
$$

值得一提的是，之前的 score 函数实际是现在这一形式的特殊表达，也就是说，这个表达更具一般性

如前所述，亲和矩阵本质上是两个注意力模块 1 和 2 的关键向量的相似性矩阵。这个意味着一种不同的确定注意力分数的方法。即，可以将一行或一列中的最大相似度值作为注意力分数。

$$
e\_{i}^{(1)}=\max\_{j=1,\ldots,n\_{f}^{(2)}}A\_{i,j},\quad e\_{j}^{(2)}=\max\_{i=1,\ldots,n\_{f}^{(1)}}A\_{i,j}.
$$

##### 旋转注意力(Rotatory Attention)

Rotatory Attention 是一种用于处理多输入数据的注意力机制，特别适用于情感分析任务中同时考虑目标短语、左上下文和右上下文的场景。该机制通过交替关注不同输入来构建更丰富的表示。

- 主要用于情感分析任务
- 处理三个输入：目标短语 $\boldsymbol{F}^{t} = [ \boldsymbol{f}\_{1}^{t}, \ldots , \boldsymbol{f}\_{n\_{f}^{t}}^{t}] \in \mathbb{R} ^{d\_{f}^{t}\times n\_{f}^{t}}$ 、左上下文 $\boldsymbol{F^{l}} = [ \boldsymbol{f\_{1}^{l}}, \ldots , \boldsymbol{f\_{n\_{f}^{l}}^{l}}]\in\mathbb{R} ^{d\_{f}^{l}\times n\_{f}^{l}}$ 和右上下文 $\boldsymbol{F^{r}} = [ \boldsymbol{f\_{1}^{r}}, \ldots , \boldsymbol{f\_{n\_{f}^{r}}^{r}}]\in\mathbb{R}^{d\_f^r\times n\_f^r}$
- 通过注意力机制迭代改进表示

其大致的过程如下：

1. **初始特征提取**

   首先，模型从三个输入中提取特征向量目标短语特征矩阵 $\boldsymbol{F}^{t}$ 左上下文特征矩阵 $\boldsymbol{F^{l}}$ 右上下文特征矩阵 $\boldsymbol{F^{r}}$ 

2. **目标短语初始表示**

   计算目标短语的初始表示向量$r^{t}$，通过对特征向量取平均：

   $$
   r^{t}=\frac{1}{n\_{f}^{t}}\sum\_{i=1}^{n\_{f}^{t}} f\_{i}^{t}
   $$

3. **左上下文注意力计算**

   使用$r^{t}$作为查询，计算左上下文的注意力：

   1. 提取键向量 $k\_{1}^{l},\ldots,k\_{n\_{f}^{l}}^{l}\in \mathbb{R}^{d\_{k}^{l}}$ 和值向量 $v\_{1}^{l},\ldots,v\_{n\_{f}^{l}}^{l}\in \mathbb{R}^{d\_{v}^{l}}$

   2. 计算注意力分数 $e\_{i}^{l}=\operatorname{score}\left(r^{t}, k\_{i}^{l}\right)$

   3. 使用 softmax 对齐函数计算注意力权重$a\_{i}^{l}$

   4. 计算左上下文表示向量 $r^{l}=\sum\_{i=1}^{n\_{f}^{l}} a\_{i}^{l}v\_{i}^{l}$

4. **右上下文注意力计算**

   类似地，计算右上下文的表示向量$r^{r}$：

   1. 提取键向量 $k\_{1}^{r},\ldots,k\_{n\_{f}^{r}}^{r}\in \mathbb{R}^{d\_{k}^{r}}$ 和值向量 $v\_{1}^{r},\ldots,v\_{n\_{f}^{r}}^{r}\in \mathbb{R}^{d\_{v}^{r}}$

   2. 计算注意力分数 $e\_{i}^{r}=\operatorname{score}\left(r^{t}, k\_{i}^{r}\right)$

   3. 使用 softmax 对齐函数计算注意力权重$a\_{i}^{r}$

   4. 计算右上下文表示向量 $r^{r}=\sum\_{i=1}^{n\_{f}^{r}} a\_{i}^{r}v\_{i}^{r}$

5. **目标短语更新表示**

   使用左上下文表示$r^{l}$和右上下文表示$r^{r}$来更新目标短语的表示：

   1. 提取目标短语的键向量 $k\_{1}^{t},\ldots,k\_{n\_{f}^{t}}^{t}\in \mathbb{R}^{d\_{k}^{t}}$ 和值向量 $v\_{1}^{t},\ldots,v\_{n\_{f}^{t}}^{t}\in \mathbb{R}^{d\_{v}^{t}}$

   2. 计算左感知目标表示 $r^{l\_{t}}$：

      - 注意力分数：$e\_{i}^{l\_{t}}=\operatorname{score}\left(r^{l}, k\_{i}^{t}\right)$
      - 使用 softmax 对齐函数计算注意力权重 $a\_{i}^{l\_{t}}$
      - 计算表示向量：$r^{l\_{t}}=\sum\_{i=1}^{n\_{f}^{t}} a\_{i}^{l\_{t}}v\_{i}^{t}$

   3. 计算右感知目标表示 $r^{r\_{t}}$：

      - 注意力分数：$e\_{i}^{r\_{t}}=\operatorname{score}\left(r^{r}, k\_{i}^{t}\right)$
      - 使用 softmax 对齐函数计算注意力权重 $a\_{i}^{r\_{t}}$
      - 计算表示向量：$r^{r\_{t}}=\sum\_{i=1}^{n\_{f}^{t}} a\_{i}^{r\_{t}}v\_{i}^{t}$

6. 最终表示为 $r=\operatorname{concat}\left(r^{l},r^{r},r^{l\_{t}},r^{r\_{t}}\right)$

Rotatory Attention 具有以下特点：

1. **双向信息流动**：通过从目标短语到上下文，再从上下文回到目标短语的信息流动，实现了双向的信息交互。

2. **层次化表示**：构建了多层次的特征表示，从原始特征到上下文感知特征。

3. **特定任务优化**：特别适合情感分析任务，能够捕捉目标短语与上下文之间的复杂关系。

Rotatory Attention 通过这种交替关注的方式，能够更好地理解目标短语在特定上下文中的情感倾向，从而提高了情感分类的准确性。

## 特征层级(Levels of Features)

这部分讨论了如何处理具有层级结构的特征，主要分为单层级注意力和多层级注意力机制。多层级注意力能够捕捉**不同粒度**上的重要信息。

### 单层级注意力(Single-Level Attention)

传统注意力机制通常在单一层级上处理特征，如只关注单词级别或句子级别。

### 多层级注意力机制

#### **注意力叠加(Attention-via-Attention)**

<img src="/img/Attention/AttentionViaAttention.png" alt="attention-via-attention" width="60%" height="auto">

- 同时处理字符级和词级特征
- 先计算词级注意力，用其上下文向量辅助计算字符级注意力
- 最终拼接两个层级的上下文向量



用于机器翻译任务，同时利用字符级和词级信息。其核心思想是在预测字符时，先通过词级注意力确定重要词语，再在这些词语对应的字符上施加注意力。



其大致过程如下：

  1. 输入句子被编码为字符级特征矩阵 $F^{(c)}\in \mathbb{R}^{d\_{f}^{(c)}\times n\_{f}^{(c)}}$ 和词级特征矩阵 $F^{(w)}\in \mathbb{R}^{d\_{f}^{(w)}\times n\_{f}^{(w)}}$
  2. 字符级查询 $q^{(c)}\in \mathbb{R}^{d\_{q}}$ 通过查询模型生成
  3. 先计算词级注意力，生成词级上下文向量 $c^{(w)}\in \mathbb{R}^{d\_{v}^{(w)}}$
  4. 将 $q^{(c)}$ 和 $c^{(w)}$ 拼接作为字符级注意力的查询
  5. 最终输出是字符级和词级上下文向量的拼接


#### **层级注意力(Hierarchical Attention)**

<img src="/img/Attention/HierarchicalAttention.png" alt="hierarchical attention" width="60%" height="auto">

- 从最低层级开始，逐步构建高层级表示
- 常用于文档分类：词 → 句 → 文档
- 每个层级通过注意力机制生成摘要表示

用于文档分类。该方法自底向上构建层级表示：从词级表示构建句级表示，再从句级表示构建文档级表示。



其大致过程如下：

1. 文档包含 $n\_S$ 个句子，第 $s$ 个句子包含 $n\_s$ 个词

2. 对每个句子计算词级注意力，生成句表示 $c^{(s)}\in \mathbb{R}^{d\_{v}^{(S)}}$

3. 将所有句表示 $C=[c^{(1)},\ldots,c^{(n\_{S})}]\in \mathbb{R}^{d\_{v}^{(S)} \times n\_{S}}$ 作为文档级注意力的输入

4. 文档级注意力输出 $c^{(D)}\in \mathbb{R}^{d\_{v}^{(D)}}$ 用于分类

#### 应用领域

多层级注意力已成功应用于 ~~懒得做链接了捏，可以去原文找捏~~ ：
- 推荐系统：建模用户长短期偏好(Ying et al., 2018)
- 视频动作识别：捕捉不同时间尺度的运动信息(Wang et al., 2016)
- 跨领域情感分类：学习领域共享和特定特征(Li et al., 2018)
- 聊天机器人：生成更连贯的响应(Xing et al., 2018)
- 人群计数：处理不同尺度的人群密度(Sindagi & Patel, 2019)

## 特征表示(Feature Representations)

这部分讨论了特征表示相关的注意力机制（Feature Representations），主要关注如何利用注意力机制来处理和组合不同的特征表示。这部分内容可以分为两类：单表示注意力（Single-representational attention）和多表示注意力（Multi-representational attention）。

### 单一表示注意力(Single-Representational Attention)

单表示注意力是最基础的注意力形式，它使用单一的特征表示模型（如词嵌入、CNN特征提取器等）来生成特征向量。这些特征向量随后被送入注意力模块进行处理。

### 多表示注意力(Multi-Representational Attention)

多表示注意力是一种更高级的技术，它允许模型同时考虑多种不同的特征表示，并通过注意力机制来学习如何最优地组合这些表示。

#### 元嵌入(Meta-embeddings)

这种方法可以创建所谓的"元嵌入"（meta-embeddings）。

   - 整合多个嵌入表示
   - 通过注意力机制加权平均不同表示
   - 生成更高质量的特征表示


元嵌入的创建过程大致如下：

1. **输入表示**：对于一个输入 $x$ （如一个词），我们有 $E$ 种不同的嵌入表示： $x^{(e\_1)}, \ldots, x^{(e\_E)}$ , 其中每种嵌入 $x^{(e\_i)}$ 的维度为 $d\_{e\_i}$（$i=1,\ldots,E$ ）。

2. **维度标准化**：由于不同嵌入可能有不同维度，首先通过线性变换将它们映射到统一维度 $d\_t$ ： $x^{(t\_i)} = W\_{e\_i} \times x^{(e\_i)} + b\_{e\_i}$ ， 其中 $W\_{e\_i} \in \mathbb{R}^{d\_t \times d\_{e\_i}}$ 和 $b\_{e\_i} \in \mathbb{R}^{d\_t}$ 是可训练的参数。

3. **注意力加权组合**：最终的元嵌入是这些标准化表示的加权和： $x^{(e)} = \sum\_{i=1}^E a\_i \times x^{(t\_i)}$ 其中权重 $a\_i$ 通过注意力机制计算得到。

##### 注意力计算

在多表示注意力中，注意力权重的计算可以视为一种自注意力机制，其查询可以理解为"哪些表示对当前任务最重要"。具体计算过程如下：

1. 将标准化后的表示 $x^{(t\_1)}, \ldots, x^{(t\_E)}$ 作为特征矩阵F的列向量
2. 由于没有显式查询，这相当于自注意力机制
3. 使用适当的注意力评分函数计算权重
4. 通过对齐函数（如softmax）得到归一化的注意力权重

### 技术优势

1. **灵活性**：可以整合来自不同来源或不同粒度的特征表示
2. **适应性**：通过注意力权重自动学习不同表示的重要性
3. **可解释性**：注意力权重可以提供关于哪些特征表示对任务更重要的见解


# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/Brauwers和Frasincar - 2023 - A General Survey on Attention Mechanisms in Deep Learning.pdf" target="_blank">📄 Brauwers 和 Frasincar - 2023 - A General Survey on Attention Mechanisms in Deep Learning</a>

{% post_link Attention %}
