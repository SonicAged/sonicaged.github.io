---
title: What Is GNN and GCN ?
date: 2025-07-10 15:36:46
categories:
  - CDR
  - model
tags:
  - CDR
  - model
  - embedding
  - PyTorch
  - graph theory
  - Basic
---

# GNN 与 GCN

> 图神经网络（Graph Neural Networks, GNN）和图卷积网络（Graph Convolutional Networks, GCN）是处理图数据的强大工具。本文将从理论到实践，全面介绍这两种重要的深度学习模型。

本文主要介绍了*GNN和GCN的大致原理*，*GCN在PyG和PyTorch的实现* 以及它们在*DRP中的应用*

<!-- more -->

## 🎯 Intro

在深度学习领域，处理图结构数据一直是一个具有挑战性的任务。传统的深度学习模型（如CNN、RNN）在处理欧几里得空间中的数据表现出色，但对于图这种非欧几里得结构的数据却显得力不从心。GNN和GCN的出现，为我们提供了处理图数据的有力工具。

而在DRP领域，由于涉及到大量的Embedding，GCN现在几乎已经成为了必不可少的模块。

但在开始各种各样的奇形怪状的GCN之前，了解GNN和GCN本身的实现仍然是非常必要的。~~于鼠鼠而言~~大致有以下理由：
1. 部分抽象的基于GCN的模块第三方库不一定支持
2. 由于反应表示数据的不平衡，我们可以构建的模型的层数是非常有限的（因为会过平滑）。因此对层内的改造就显得非常必要了。而这一切的前提便是理解原理捏

在这里强烈建议去看一下[Distill](https://distill.pub/)的两篇有关图神经网络的博客，非常易懂。

---

## 📚 理论基础

### 图的基本概念

在开始之前，我们需要理解图的基本表示：
- 图 $G = (V, E)$，其中 $V$ 是节点集合，$E$ 是边集合
- 邻接矩阵 $A \in \mathbb{R}^{n \times n}$
- 度矩阵 $D = diag(d_1,...,d_n)$，其中 $d_i = \sum_j A_{ij}$
- 节点特征矩阵 $X \in \mathbb{R}^{n \times d}$

### GNN框架

GNN的基本框架遵循消息传递范式（Message Passing Neural Network, MPNN），可以用以下数学公式表示：

1. **消息传递阶段**（Message Passing）：
   
   对于节点 $v$，从其邻居节点 $u \in \mathcal{N}(v)$ 收集信息：
   
   $$m_v^{(l)} = \sum_{u \in \mathcal{N}(v)} M_l(h_v^{(l-1)}, h_u^{(l-1)}, e_{uv})$$

   其中：
   - $h_v^{(l-1)}$ 是节点 $v$ 在第 $l-1$ 层的特征
   - $e_{uv}$ 是边 $(u,v)$ 的特征
   - $M_l$ 是可学习的消息函数

2. **消息聚合阶段**（Aggregation）：
   
   将收集到的消息进行聚合：

   $$a_v^{(l)} = AGG(\{m_v^{(l)} | u \in \mathcal{N}(v)\})$$

   常见的聚合函数包括：
   - 求和：$AGG_{sum} = \sum_{u \in \mathcal{N}(v)} m_u$
   - 平均：$AGG_{mean} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} m_u$
   - 最大：$AGG_{max} = max_{u \in \mathcal{N}(v)} m_u$

3. **节点更新阶段**（Update）：
   
   更新节点的表示：

   $$h_v^{(l)} = U_l(h_v^{(l-1)}, a_v^{(l)})$$

   其中 $U_l$ 是可学习的更新函数，通常是MLP或其他神经网络。

### GCN实现

#### 拉普拉斯矩阵 🔍

拉普拉斯矩阵是图信号处理中的核心概念，有多种形式：

1. **组合拉普拉斯矩阵**：$L = D - A$

2. **标准化拉普拉斯矩阵**：$L_{sym} = D^{-\frac{1}{2}}LD^{-\frac{1}{2}} = I - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$

3. **随机游走拉普拉斯矩阵**：$L_{rw} = D^{-1}L = I - D^{-1}A$

拉普拉斯矩阵的特性：
- 对称性：$L = L^T$
- 半正定性：所有特征值非负
- 最小特征值为0，对应的特征向量是常数向量
- 特征值的重数对应图的连通分量数

#### 从传统卷积到图卷积 🔄

##### 传统卷积回顾

在欧几里得空间中，卷积操作定义为：

$$(f * g)(p) = \sum_{q \in \mathcal{N}(p)} f(q) \cdot g(p-q)$$

这里的关键特点是：
- 平移不变性
- 局部性
- 参数共享

##### 图上的卷积定义

在图域中，我们需要重新定义这些特性：

1. **空间域卷积**：
   $$h_v = \sum_{u \in \mathcal{N}(v)} W(e_{u,v})h_u$$
   其中 $W(e_{u,v})$ 是边的权重函数

2. **谱域卷积**：
   $$g_\theta * x = Ug_\theta U^T x$$
   其中 $U$ 是拉普拉斯矩阵的特征向量矩阵

#### GCN的数学推导 ⚙️

Kipf & Welling提出的GCN模型中，单层传播规则为：

$$H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})$$

其中：
- $\tilde{A} = A + I_N$ 是添加了自环的邻接矩阵
- $\tilde{D}\_{ii} = \sum\_{j} \tilde{A}\_{ij}$ 是对应的度矩阵
- $H^{(l)}$ 是第 $l$ 层的激活值
- $W^{(l)}$ 是可学习的权重矩阵
- $\sigma$ 是非线性激活函数

~~一些自己的理解~~
1. 引入$L_{sym} = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$作为聚合（AGG）部分
   - 添加自环：$\tilde{A} = A + I_N$
   - 计算归一化系数：$\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$
2. 特征变换：$H^{(l)}W^{(l)}$
3. 邻域聚合：$\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}$
4. 非线性变换：$\sigma(\cdot)$

---

## 💻 实现细节

基于这个理论框架的简单实现如下：

```python
def message_passing(nodes, edges):
    messages = {}
    for edge in edges:
        src, dst = edge
        msg = compute_message(nodes[src], nodes[dst])
        messages.setdefault(dst, []).append(msg)
    return messages

def aggregate_messages(messages):
    aggregated = {}
    for node, msgs in messages.items():
        aggregated[node] = sum(msgs) / len(msgs)  # 平均聚合
    return aggregated

def update_nodes(nodes, aggregated):
    updated = {}
    for node, agg_msg in aggregated.items():
        updated[node] = nodes[node] + agg_msg  # 残差连接
    return updated
```

### PyTorch Geometric实现 🚀

> 本节代码基于 PyTorch 2.1.0 和 PyTorch Geometric 2.4.0 版本

使用PyTorch Geometric库的GCN实现：

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

### 原生PyTorch实现 🔧

> 本节代码基于 PyTorch 2.1.0、NumPy 1.24.0 和 SciPy 1.11.0 版本

不使用PyG，手动实现GCN~~主要是目前不太清楚主流的HGCN的实现方式捏~~：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W)

    def forward(self, x, adj):
        # adj: 归一化的邻接矩阵
        support = torch.mm(x, self.W)
        output = torch.sparse.mm(adj, support)
        return output

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

def normalize_adj(adj):
    """归一化邻接矩阵"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
```

---


## 🎮 应用场景

~~由于鼠鼠就是个臭写DRP的捏~~ 这里只给出GNN在DRP中的应用


1. **药物表示**
   - *分子图构建*：将药物SMILES字符串转换为图结构，节点表示原子（含原子类型、电荷等特征），边表示化学键（如键类型、距离）。  
   - *GNN编码*：使用图卷积网络（GCN）、图注意力网络（GAT）或图同构网络（GIN）等层迭代聚合邻域信息，生成药物嵌入（embedding）。例如，GraTransDRP（2022）结合GAT和Transformer提升药物表征能力。

2. **癌症表示**
   - *生物网络构建*：基于基因互作（如STRING数据库的蛋白-蛋白互作）、基因共表达或通路信息构建异质图。例如，AGMI（2021）整合多组学数据和PPI网络，通过GNN学习癌症样本的联合表征。  
   - *多组学融合*：部分模型（如TGSA）利用GNN整合基因组、转录组等数据，通过跨模态注意力机制增强特征交互。

3. **异构图与联合建模**
   - *细胞系-药物异构图*：如GraphCDR（2021）将细胞系和药物作为两类节点，通过边连接已知响应对，直接学习跨实体关系。  
   - *知识增强*：预训练GNN于大规模生物化学属性预测（如Zhu et al., 2021），再迁移至DRP任务，提升泛化性。

## 🎯 总结与展望

- **动态图建模**：捕捉治疗过程中动态变化的生物网络。  
- **三维分子图**：结合几何深度学习（如SchNet）提升立体化学感知。  
- **基准测试**：需统一评估协议（如固定数据集和指标）以公平比较GNN与其他方法。

~~之后应该会写一些具体模型的博客，有相关的会直接上链接的捏jrm~~

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒
<a href="/paper/1609.02907v4.pdf" target="_blank">📄 Thomas - SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS</a>
<a href="https://pytorch-geometric.readthedocs.io/" target="_blank">PyTorch Geometric 官方文档</a>
<a href="https://distill.pub/2021/gnn-intro/" target="_blank">Distill: A Gentle Introduction to Graph Neural Networks</a>
<a href="https://distill.pub/2021/understanding-gnns/" target="_blank">Distill: Understanding Convolutions on Graphs</a>
<a href="https://www.zhihu.com/tardis/zm/art/107162772" target="_blank">知乎：图卷积网络（GCN）入门详解</a>
<a href="https://github.com/tkipf/gcn" target="_blank">GCN 论文官方代码（GitHub）</a>