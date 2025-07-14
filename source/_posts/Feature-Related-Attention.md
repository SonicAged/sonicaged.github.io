---
title: Feature-Related Attention
date: 2025-07-14 22:30:57
categories:
  - CDR
  - model
  - attention
  - feature related
tags:
  - CDR
  - model
  - Basic
  - deep learning
  - 还没写完捏
---

# 在输入特征上做文章的Attention

本文将接着详细说明一种基于输入特征分类Attention的方式，并介绍在这种分类方式下关注到的不同的Attention的架构。

具体来说，本文主要探讨了基于输入特征特性的注意力机制变体。本节根据输入特征的不同特性，将特征相关的注意力机制分为三类：特征多重性(Multiplicity of Features)、特征层级(Levels of Features)和特征表示(Feature Representations)。

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

如上图所示，这是alternating co-attention的架构图，该机制交替使用两个输入的特征矩阵，先计算第一个输入的注意力，将其上下文向量作为查询计算第二个输入的注意力，然后再用第二个输入的上下文向量重新计算第一个输入的注意力。

这里现给出他的score函数

对于有序列输入的Attention：

$$
\mathrm{score}(\underset{d\_{q}\times1}{\boldsymbol{q}},\underset{d\_{k}\times1}{\boldsymbol{k}\_{l}})=\underset{1\times d\_{w}}{\boldsymbol{w}^{T}}\times\mathrm{act}(\underset{d\_{w}\times d\_{q}}{\boldsymbol{W}\_{1}}\times\underset{d\_{q}\times1}{\boldsymbol{q}}+\underset{d\_{w}\times d\_{k}}{\boldsymbol{W}\_{2}}\times\underset{d\_{k}\times1}{\boldsymbol{k}\_{l}}+\underset{d\_{w}\times1}{\boldsymbol{b}})
$$

对于无序列输入的Attention ~~（这是一种自注意力机制，后面会提到）~~ ：

$$
\underset{1\times1}{e\_{l}^{(0)}}=\underset{1\times d\_{w}}{\boldsymbol{w}^{(1)T}}\times\operatorname{act}(\underset{d\_{w}\times d\_{k}^{(1)}}{\boldsymbol{W}^{(1)}}\times\underset{d\_{k}^{(1)}\times1}{\boldsymbol{k}\_{l}^{(1)}}+\underset{d\_{w}\times1}{\boldsymbol{b}^{(1)}})
$$

对于第二层Attention：

$$
\underset{1\times1}{e\_{l}^{(2)}}=\mathrm{score}(\underset{d\_{v}^{(1)}\times 1}{\boldsymbol{c}^{(0)}},\underset{d\_{k}^{(2)}\times1}{\boldsymbol{k}\_{l}^{(2)}})
$$

对于第三层Attention：

$$
\underset{1\times1}{e\_{l}^{(1)}}=\mathrm{score}(\underset{d\_{v}^{(2)}\times 1}{\boldsymbol{c}^{(2)}},\underset{d\_{k}^{(1)}\times1}{\boldsymbol{k}\_{l}^{(1)}})
$$

生成的上下文向量$\boldsymbol{c}^{(1)}$和$\boldsymbol{c}^{(2)}$被连接起来，并在输出模型中用于预测。交替协同注意力由于需要一个接一个地计算上下文向量，因此本质上包含了*一种顺序性*。这可能会带来计算上的劣势，因为*无法并行*化。 

##### interactive co-attention

   - 并行计算两个输入的注意力
   - 使用未加权平均的关键向量作为查询
   - 计算效率更高，可以并行处理



##### 并行协同注意力(Parallel Co-attention)
   - 同时计算两个输入的注意力
   - 使用亲和矩阵(Affinity Matrix)转换关键向量空间
   - 通过聚合形式计算注意力分数



##### 旋转注意力(Rotatory Attention)
   - 主要用于情感分析任务
   - 处理三个输入：目标短语、左上下文和右上下文
   - 通过注意力机制迭代改进表示

## 特征层级(Levels of Features)

这部分讨论了如何处理具有层级结构的特征，主要分为单层级注意力和多层级注意力机制。

### 单层级注意力(Single-Level Attention)

传统注意力机制通常在单一层级上处理特征，如只关注单词级别或句子级别。

### 多层级注意力机制

1. **注意力叠加(Attention-via-Attention)**
   - 同时处理字符级和词级特征
   - 先计算词级注意力，用其上下文向量辅助计算字符级注意力
   - 最终拼接两个层级的上下文向量



2. **层级注意力(Hierarchical Attention)**
   - 从最低层级开始，逐步构建高层级表示
   - 常用于文档分类：词→句→文档
   - 每个层级通过注意力机制生成摘要表示



## 特征表示(Feature Representations)

这部分讨论了特征表示方式的注意力机制变体，主要分为单一表示注意力和多表示注意力。

### 单一表示注意力(Single-Representational Attention)

传统方法使用单一嵌入或表示模型生成特征表示。

### 多表示注意力(Multi-Representational Attention)

1. **元嵌入(Meta-embeddings)**
   - 整合多个嵌入表示
   - 通过注意力机制加权平均不同表示
   - 生成更高质量的特征表示

2. **自注意力机制**
   - 学习特征向量之间的关系
   - 通过注意力改进特征表示
   - 常用于Transformer架构中

## 应用领域

3.1节讨论的特征相关注意力机制在多个领域有广泛应用：
- 医学数据分析(多特征协同注意力)
- 推荐系统(多层级注意力)
- 情感分析(旋转注意力)
- 文档分类(层级注意力)
- 多语言处理(多表示注意力)

## 总结

3.1节系统性地分类了基于输入特征特性的注意力机制变体，为研究者提供了清晰的框架来选择适合特定任务和数据类型的最佳注意力机制。这些机制通过充分利用输入特征的多重性、层级结构和表示多样性，显著提升了模型在各种任务上的表现。



图3展示了完整的注意力机制分类体系，其中3.1节讨论的特征相关注意力机制是该体系的重要组成部分。

$$
\underset{n\_{f}^{(1)}\times n\_{f}^{(2)}}{A}=\operatorname{act}(\underset{n\_{f}^{(1)}\times d\_{k}^{(1)}}{\begin{array}{c}K^{(1)^{T}}\end{array}}\times\underset{d\_{k}^{(1)}\times d\_{k}^{(2)}}{\begin{array}{c}W\_{A}\end{array}}\times\underset{d\_{k}^{(2)}\times n\_{f}^{(2)}}{\begin{array}{c}K^{(2)}\end{array}})
$$

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/Brauwers和Frasincar - 2023 - A General Survey on Attention Mechanisms in Deep Learning.pdf" target="_blank">📄 Brauwers和Frasincar - 2023 - A General Survey on Attention Mechanisms in Deep Learning</a>

{% post_link Attention %}