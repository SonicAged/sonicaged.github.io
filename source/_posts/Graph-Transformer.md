---
title: Trans?!and Former?!
date: 2025-07-1900:26:57
categories:
  - model
  - attention
  - graph
tags:
  - CDR
  - model
  - Basic
  - deeplearning
  - PyTorch
  - graphtheory
  - attention
---

# Graph Transformer

Graph Transformer 是传统Transformer架构在图数据上的泛化，旨在处理任意结构的图数据（如社交网络、分子结构等）。其既保留了Transformer的强表示能力，又继承了GNN对图结构的归纳偏置，成为图表示学习领域的重要基线模型。

<!--more-->

在过去，Transformer（Lin等人，2021年）在许多NLP、CV和GRL任务中取得了卓越的性能。Graph Transformer将Transformer架构推广到图表示学习中，捕捉长距离依赖关系（Ying等人，2021年）。与之前使用局部注意力的方法不同，Graph Transformer通过全局注意力直接学习高阶图属性。Graph Transformer在图深度学习领域发展迅速，特别是在中小型图的图分类任务中。进一步将Graph Transformer分为两个子类别，即标准Transformer（Ying等人，2021年）和GNNTransformer（Nguyen等人，2019年）。标准Transformer通常对输入图的所有节点使用自注意力机制，忽略节点之间的邻接关系，而GNNTransformer则使用GNN层来获取邻接信息。 

<img src="/img/Attention/GraphTransformer.png" alt="GraphTransformer" width="60%" height="auto">

现在介绍Graph Transformer层和具有边特征的Graph Transformer层。该层架构上图所示。第一个模型是为没有显式边属性的图设计的，而第二个模型则保持一个指定的边特征管道，以整合可用的边信息，并在每一层中保持它们的抽象表示。

### 输入特征线性投影公式

输入首先，准备将输入节点和边嵌入传递到Graph Transformer层。对于一个具有每个节点i的节点特征 $\alpha\_i \in \mathbb{R}^{d\_n\times 1}$ 和每个节点$i$和节点$j$之间的边特征$\beta\_{ij} \in \mathbb{R}^{d\_e\times 1}$的图G，通过线性投影将输入节点特征αi和边特征$\beta\_{ij}$传递以嵌入到$d-$维隐藏特征$h^0\_i$和$e^0\_{ij}$。

**节点特征投影**：$\hat{h}\_{i}^{0} = A^{0}\alpha\_{i} + a^{0}\qquad$ **边特征投影**：$e\_{ij}^{0} = B^{0}\beta\_{ij} + b^{0}$

其中，$A^{0} \in \mathbb{R}^{d \times d\_{n}}$ 和 $B^{0} \in \mathbb{R}^{d \times d\_{e}}$ 是投影矩阵，$a^{0}, b^{0} \in \mathbb{R}^{d}$ 是偏置，$\alpha\_i$ 和 $\beta\_{ij}$ 分别是原始节点和边特征

### 2. 位置编码融合公式

现在通过线性投影嵌入预先计算的节点位置编码$k$，并将其添加到节点特征$\hat{h}^0\_i$。

**位置编码投影**$\lambda\_{i}^{0} = C^{0}\lambda\_{i} + c^{0}\qquad$ **节点特征更新**$h\_{i}^{0} = \hat{h}\_{i}^{0} + \lambda\_{i}^{0}$

其中，$C^{0} \in \mathbb{R}^{d \times k}$ 是位置编码投影矩阵，$c^{0} \in \mathbb{R}^{d}$ 是偏置项，$\lambda\_i$ 是预计算的Laplacian特征向量。请注意，拉普拉斯位置编码仅在输入层添加到节点特征，而不是在中间的Graph Transformer层。

### 3. 基础图Transformer层公式

与最初在(Vaswani等人，2017)中提出的Transformer架构非常相似。现在开始定义一层的节点更新方程。

**多头注意力输出**$\hat{h}\_{i}^{\ell+1} = O\_{h}^{\ell} \|\_{k=1}^{H} \left( \sum\_{j \in \mathcal{N}\_{i}} w\_{ij}^{k，\ell} V^{k，\ell} h\_{j}^{\ell} \right)$

**注意力权重计算**$w\_{ij}^{k，\ell} = \text{softmax}\_{j} \left( \frac{Q^{k，\ell} h\_{i}^{\ell} \cdot K^{k，\ell} h\_{j}^{\ell}}{\sqrt{d\_{k}}} \right)$

其中，$\|$ 表示拼接操作，$Q^{k，\ell}, K^{k，\ell}, V^{k，\ell} \in \mathbb{R}^{d\_{k} \times d}$ 是各头的查询、键、值矩阵，$O\_{h}^{\ell} \in \mathbb{R}^{d \times d}$ 是输出投影矩阵

### 4. 前馈网络与归一化

为了数值稳定性，softmax内部项的指数输出被夹在$−5$到$+5$之间。然后将注意力输出$\hat{h}^{\ell +1}\_i$传递给一个前接和后接残差连接和规范化层的前馈网络(FFN)，如下所示:

**归一化步骤**$\hat{\hat{h}}\_{i}^{\ell+1} = \text{Norm}(h\_{i}^{\ell} + \hat{h}\_{i}^{\ell+1})\quad$ **前馈网络**$\hat{\hat{\hat{h}}}\_{i}^{\ell+1} = W\_{2}^{\ell} \text{ReLU}(W\_{1}^{\ell} \hat{\hat{h}}\_{i}^{\ell+1})\quad$ **最终输出**$h\_{i}^{\ell+1} = \text{Norm}(\hat{\hat{h}}\_{i}^{\ell+1} + \hat{\hat{\hat{h}}}\_{i}^{\ell+1})$

其中，$W\_{1}^{\ell} \in \mathbb{R}^{2d \times d}$ 和 $W\_{2}^{\ell} \in \mathbb{R}^{d \times 2d}$ 是前馈网络参数，Norm可以是BatchNorm或LayerNorm.为了清晰起见，省略了偏差项。

### 5. 带边特征的图Transformer层

带有边特征的Graph Transformer层带有边特征的Graph Transformer是为更好地利用多种图数据集中的丰富特征信息而设计的，这些信息以边属性的形式存在。由于的目标仍然是更好地利用边特征，这些特征是对应于节点对的成对得分，将这些可用的边特征与通过成对注意力计算的隐式边得分联系起来。

换句话说，当一个节点$i$在 *query* 和 *key* 特征投影相乘后关注节点$j$时，中间注意力得分$\hat{w}\_{ij}$在softmax之前被计算出来。让将这个得分$\hat{w}\_{ij}$视为关于边$<i，j>$的隐式信息。现在尝试注入边$<i，j>$的可用边信息，并改进已经计算出的隐式注意力得分$\hat{w}\_{ij}$。

这是通过简单地将两个值$\hat{w}\_{ij}$和$e\_{ij}$相乘来完成的。这种信息注入在NLPTransformer中并没有被广泛探索或应用，因为在两个单词之间通常没有可用的特征信息。

然而，在分子图或社交媒体图等图数据集中，边交互上往往有一些可用的特征信息，因此设计一种架构来利用这些信息变得自然。对于边，还维护了一个指定的节点对称边特征表示管道，用于从一层到另一层传播边属性。现在继续定义一层的层更新方程。

**注意力分数计算**$\hat{w}\_{ij}^{k，\ell} = \left( \frac{Q^{k，\ell} h\_{i}^{\ell} \cdot K^{k，\ell} h\_{j}^{\ell}}{\sqrt{d\_{k}}} \right) \cdot E^{k，\ell} e\_{ij}^{\ell}\quad$ **边特征更新**$\hat{e}\_{ij}^{\ell+1} = O\_{e}^{\ell} \prod\_{k=1}^{H} (\hat{w}\_{ij}^{k，\ell})$

其中，$E^{k，\ell} \in \mathbb{R}^{d\_{k} \times d}$ 是边特征投影矩阵，$O\_{e}^{\ell} \in \mathbb{R}^{d \times d}$ 是边特征输出投影矩阵

### 6. 边特征的前馈网络



**边特征归一化**$\hat{\hat{e}}\_{ij}^{\ell+1} = \text{Norm}(e\_{ij}^{\ell} + \hat{e}\_{ij}^{\ell+1})\quad$ **边特征变换**$\hat{\hat{\hat{e}}}\_{ij}^{\ell+1} = W\_{e，2}^{\ell} \text{ReLU}(W\_{e，1}^{\ell} \hat{\hat{e}}\_{ij}^{\ell+1})\quad$ **最终边输出**$e\_{ij}^{\ell+1} = \text{Norm}(\hat{\hat{e}}\_{ij}^{\ell+1} + \hat{\hat{\hat{e}}}\_{ij}^{\ell+1})\quad$

其中，$W\_{e，1}^{\ell} \in \mathbb{R}^{2d \times d}$ 和 $W\_{e，2}^{\ell} \in \mathbb{R}^{d \times 2d}$ 是边特征前馈网络参数

### 总结

这项工作提出了一种简单而有效的方法，将Transformer网络推广到任意图上，并引入了相应的架构。

实验一致表明，存在：
- Laplacian特征向量作为节点位置编码
- batch normalization，代替层归一化，
则在Transformer前馈层周围增强了Transformer在所有实验中的普遍性。

鉴于这个架构的简单性和通用性以及与标准GNN相比的竞争性能，提出的模型可以作为进一步改进跨图应用中使用节点注意力的基线。

# 📚𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒