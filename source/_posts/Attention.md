---
title: Attention in Graph
date: 2025-07-10 19:56:23
categories:
  - CDR
  - model
tags:
  - CDR
  - model
  - Basic
  - 还没写完捏
  - PyTorch
  - graph theory
js:
  - https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js
---

# Is Attention All My Need ?

> 注意力机制在图神经网络中扮演着越来越重要的角色。本文将深入探讨注意力机制在图结构数据处理中的应用，从基础概念到实际实现。

本文鼠鼠计划从一般的Attention出发，到图结构中的Attention的作用，最后讨论Attention是否真的可以按照期望解决DRP中鼠鼠期待它能够解决的所有问题。

$$
\underset{d\_{k} \times n\_{f}}{\boldsymbol{K}}=\underset{d\_{k} \times d\_{f}}{\boldsymbol{W}\_{K}} \times \underset{d\_{f} \times n\_{f}}{\boldsymbol{F}}, \quad \underset{d\_{v} \times n\_{f}}{\boldsymbol{V}}=\underset{d\_{v} \times d\_{f}}{\boldsymbol{W}\_{V}} \times \underset{d\_{f} \times n\_{f}}{\boldsymbol{F}} .
$$
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

<embed src="/paper/1609.02907v4.pdf" width="40%" height="40%" type="application/pdf">

### 注意力机制回顾

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