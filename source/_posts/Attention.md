---
title: Attention Overview
date: 2025-07-10 19:56:23
categories:
  - CDR
  - model
  - attention
tags:
  - CDR
  - model
  - Basic
  - 还没写完捏
  - PyTorch
---

# Is Attention All My Need ?

> 注意力机制在图神经网络中扮演着越来越重要的角色。~~但鼠鼠现在连正常的Attention有哪些都不清楚捏~~本文鼠鼠将从一般的Attention出发，给出Attention的总体结构，然后按分类介绍现有的主要的Attention

本文主要来自于一篇论文，基本可以看作[那篇论文](/paper/Brauwers和Frasincar%20-%202023%20-%20A%20General%20Survey%20on%20Attention%20Mechanisms%20in%20Deep%20Learning.pdf)的阅读笔记

<!-- more -->

## 🎯 引言

在深度学习领域，注意力机制已经成为一个革命性的创新，特别是在处理序列数据和图像数据方面取得了巨大成功。而在图神经网络中，注意力机制的引入不仅提高了模型的表现力，还增强了模型的可解释性。

在图结构数据中应用注意力机制主要有以下优势：
1. 自适应性：能够根据任务动态调整不同邻居节点的重要性
2. 可解释性：通过注意力权重可以直观理解模型的决策过程
3. 长程依赖：有效缓解了传统GNN中的过平滑问题
4. 异质性处理：更好地处理异质图中的不同类型节点和边

## 📚 总览Attention

本章节主要参考了论文[📄 Brauwers和Frasincar - 2023 - A General Survey on Attention Mechanisms in Deep Learning](/paper/Brauwers和Frasincar%20-%202023%20-%20A%20General%20Survey%20on%20Attention%20Mechanisms%20in%20Deep%20Learning.pdf)有兴趣的话可以看看原文捏

<embed src="/paper/Brauwers和Frasincar%20-%202023%20-%20A%20General%20Survey%20on%20Attention%20Mechanisms%20in%20Deep%20Learning.pdf" width="45%" height="400" type="application/pdf">

### Attention的一般结构

<img src="/img/Attention/TotalModel.png" alt="TotalModel" width="60%" height="auto">

上图是从总体上看Attention在整个任务模型框架中的位置

框架包含四个核心组件：
1. **特征模型**：负责输入数据的特征提取
2. **查询模型**：生成注意力查询向量
3. **注意力模型**：计算注意力权重
4. **输出模型**：生成最终预测结果

接下来，我们会从 *输入* 的角度来看**特征模型**和**查询模型**，从 *输出* 的角度来看**注意力模型**和**输出模型**

#### 输入处理机制

1. **特征模型**，即将任务的输入进行embedding
   
    对于输入矩阵$ X \in \mathbb{R}^{d_x \times n_x} $，特征模型提取特征向量：$\boldsymbol{F} = [f_1, \ldots, f_{n_f}] \in \mathbb{R}^{d_f \times n_f}$

2. **查询模型**，查询模型产生查询向量$ \boldsymbol{q} \in \mathbb{R}^{d_q} $，用以告诉注意力模型哪一个特征是重要的

一般情况下，这两个模型可以用CNN或RNN

#### 输出计算机制

<img src="/img/Attention/GeneralAttentionModule.png" alt="GeneralAttentionModule" width="50%" height="auto">

上图是Attention模型总体结构的说明，下面对这张图进行详细的说明

1. 特征矩阵$\boldsymbol{F} = [\boldsymbol{f}\_1, \ldots, \boldsymbol{f}\_{n\_f}] \in \mathbb{R}^{d\_f \times n\_f}$，通过*某些方法*将其分为Keys矩阵$\boldsymbol{K} = [\boldsymbol{k}\_1, \ldots, \boldsymbol{k}\_{n\_f}] \in \mathbb{R}^{d\_k \times n\_f}$和Values矩阵$\boldsymbol{V} = [\boldsymbol{v}_1, \ldots, \boldsymbol{v}\_{n\_f}] \in \mathbb{R}^{d\_v \times n\_f}$，这里的*某些方法*，一般情况下，按以下的方式通过**线性变换**得到：

$$
\underset{d\_{k} \times n\_{f}}{\boldsymbol{K}}=\underset{d\_{k} \times d\_{f}}{\boldsymbol{W}\_{K}} \times \underset{d\_{f} \times n\_{f}}{\boldsymbol{F}}, \quad \underset{d\_{v} \times n\_{f}}{\boldsymbol{V}}=\underset{d\_{v} \times d\_{f}}{\boldsymbol{W}\_{V}} \times \underset{d\_{f} \times n\_{f}}{\boldsymbol{F}} .
$$

2. `Attention Scores`模块根据 $\boldsymbol{q}$ 计算每一个key向量对应的分数$\boldsymbol{e} = [e_1, \ldots, e_{n_f}] \in \mathbb{R}^{n_f}$：

    $$
    \underset{1\times 1}{e\_l} = \text{score}(\underset{d\_q \times 1}{\boldsymbol{q}}, \underset{d\_k \times 1}{\boldsymbol{k}\_l}) 
    $$

    如前所述，查询象征着对信息的请求。注意力分数$e_l$表示根据查询，关键向量$\boldsymbol{k}_l$中包含的信息的重要性。如果查询和关键向量的维度相同，则得分函数的一个例子是取向量的点积。

3. 由于经过这么一堆操作之后，分数有很大的可能已经飞起来了捏，这个时候就需要`Attention Alignment`模块对其进行**归一化**之类的操作了捏

    $$
    \underset{1\times 1}{a\_l} = \text{align}(\underset{d\_q \times 1}{\boldsymbol{e\_l}}, \underset{n\_f \times 1}{\boldsymbol{e}}) 
    $$

注意力权重$\boldsymbol{a} = [a_1, \ldots, a_{n_f}] \in \mathbb{R}^{n_f}$为注意力模块提供了一个相当直观的解释。每个权重直接表明了每个特征向量相对于其他特征向量对于这个问题的重要性。 

4. 在`Weight Average`模块完成**上下文生成**：

    $$
    \underset{d\_v \times 1}{\boldsymbol{c}} = \sum\_{l = 1}^{n\_f} \underset{1 \times 1}{a\_l}\times \underset{d\_v \times 1}{\boldsymbol{v}\_l}
    $$

5. 输出处理就想怎么搞就怎么搞了捏，例如 用于分类

    $$
    \underset{d\_y \times 1}{\hat{\boldsymbol{y}}} = \text{softmax}( \underset{d\_y \times d\_v}{\boldsymbol{W}\_c}\times \underset{d\_v \times 1}{\boldsymbol{c}} + \underset{d\_y \times 1}{\boldsymbol{b}\_c})
    $$

### Attention分类

<img src="/img/Attention/Taxonomy.png" style="max-width: 100%; height: auto;">

论文按照上图的方式给Attention进行了分类


#### 基本注意力机制

最基本的注意力机制可以表示为：

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

其中：
- $Q$：查询矩阵（Query）
- $K$：键矩阵（Key）
- $V$：值矩阵（Value）
- $d_k$：键向量的维度

#### 自注意力机制

自注意力是一种特殊的注意力机制，其中Q、K、V都来自同一个源序列：

$$SelfAttention(X) = Attention(XW_Q, XW_K, XW_V)$$

### 图注意力机制

#### GAT（Graph Attention Networks）

GAT通过引入注意力机制来加权邻居节点的特征。对于节点i，其更新公式为：

$$h\_i^{(l+1)} = \sigma(\sum\_{j \in \mathcal{N}\_i} \alpha\_{ij}W^{(l)}h\_j^{(l)})$$

其中注意力系数$\alpha_{ij}$的计算：

$$\alpha_{ij} = \frac{exp(LeakyReLU(a^T[Wh_i || Wh_j]))}{\sum_{k \in \mathcal{N}_i} exp(LeakyReLU(a^T[Wh_i || Wh_k]))}$$

#### 多头注意力

为了提高模型的稳定性和表达能力，GAT使用了多头注意力机制：

$$h\_i^{(l+1)} = \sigma(\frac{1}{K} \sum\_{k=1}^K \sum\_{j \in \mathcal{N}\_i} \alpha\_{ij}^k W^k h\_j^{(l)})$$

### 变体与扩展

#### 边注意力

除了节点之间的注意力，一些模型还引入了边注意力机制：

$$e_{ij} = a^T[Wh_i || Wh_j || We_{ij}]$$

其中$e_{ij}$是边的特征。

#### 全局注意力

通过引入全局节点或池化操作，可以实现全局注意力：

$$g = \sum_{i \in V} \beta_i h_i$$

其中$\beta_i$是全局注意力权重。

## 💻 实现细节

### PyTorch实现的GAT层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 变换矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # 注意力向量
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        # x: 节点特征矩阵 [N, in_features]
        # adj: 邻接矩阵 [N, N]
        
        # 线性变换
        h = torch.mm(x, self.W)  # [N, out_features]
        N = h.size()[0]

        # 计算注意力分数
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        a_input = a_input.view(N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # 掩码机制
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 聚合特征
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # 多头注意力层
        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for _ in range(nheads)
        ])
        
        # 输出层
        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        # 多头注意力
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=1)
```

### 实际应用示例

```python
# 模型初始化
model = GAT(nfeat=input_dim,
           nhid=8,
           nclass=num_classes,
           dropout=0.6,
           alpha=0.2,
           nheads=8)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

# 训练循环
def train():
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = F.nll_loss(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    return loss.item()
```

## 🔍 注意事项与最佳实践


## 📈 未来展望


# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒
<a href="/paper/Brauwers和Frasincar - 2023 - A General Survey on Attention Mechanisms in Deep Learning.pdf" target="_blank">📄 Brauwers和Frasincar - 2023 - A General Survey on Attention Mechanisms in Deep Learning</a>
<a href="/paper/Lee 等 - 2018 - Attention Models in Graphs A Survey.pdf" target="_blank">📄 Lee 等 - 2018 - Attention Models in Graphs A Survey</a>
<a href="https://github.com/xmu-xiaoma666/External-Attention-pytorch" target="_blank">github: External-Attention-pytorch</a>