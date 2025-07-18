---
title: Woc?! GAT? We're Saved!
date: 2025-07-18 23:16:58
categories:
  - model
  - attention
  - graph
tags:
  - CDR
  - model
  - Basic
  - deep learning
  - PyTorch
  - graph theory
---

# GAT（Graph Attention Networks）

图神经网络（GNNs）旨在为下游任务学习在低维空间中训练良好的表示，同时保留拓扑结构。近年来，注意力机制在自然语言处理和计算机视觉领域表现出色，被引入到GNNs中，以自适应地选择判别特征并自动过滤噪声信息。本博客将重点介绍GAT 

<!-- more -->

## 层内 GAT (Intra-Layer GAT)

层内GATs是指在单个神经网络层内应用注意力机制的图神经网络，主要特点是：

- 注意力机制作用于局部邻居节点
- 在特征聚合步骤中为不同节点分配不同权重
- 能够自适应地关注图中最相关的部分

考虑到不同的局部邻域和不同的功能，层内GATs可以进一步分为六个子类，包括邻居注意力、高阶注意力、关系感知注意力、层次注意力、注意力采样/池化和超注意力。

### 邻居注意力 (Neighbor Attention)

#### 邻居注意力核心思想

邻居注意力机制的核心是通过学习的方式动态确定图中每个节点对其邻居的重要性权重，而非传统GNN中采用的固定权重策略。其关键创新点包括：

1. **动态权重分配**：根据节点特征相似性动态计算注意力权重
2. **局部聚焦**：仅考虑一阶邻居节点的注意力计算
3. **端到端训练**：注意力权重与网络参数共同优化

#### 基础GAT模型

GAT 可以简单概括为一种将 Attention 机制引入 GCN（图卷积网络）的方法，而 GCN 是一种能够在深度学习中进行图结构分类等任务的方法。Attention 机制是指一种根据输入数据动态决定关注焦点的机制，通过将这种 Attention 机制引入 GCN（图卷积网络），可以提高分类、识别和预测的精度。因此，如果对 GCN 的理解不够深入，可能会难以完全理解 GAT。

GCN 与 GAT 的主要区别在于节点卷积时的系数（这就是所谓的 Attention 系数）存在显著差异。

在常规 GCN 中，当计算某节点的下一层特征（潜在变量）时，会通过将相邻节点的特征与线性权重相乘后的和，再通过激活函数（如 ReLU 等）传递至下一层。在计算相邻节点的特征量的线性组合时，传统方法将所有相邻节点平等对待，而 GAT 则引入了重要性（Attention 系数）的概念，不再将相邻节点视为同等。

其概念可以形象地理解为如下：

<img src="/img/GAT/AttentionInGraphVisualize.png" alt="AttentionInGraphVisualize" width="60%" height="auto">

>上图展示了节点 1 与其相邻的节点 2、3、4 之间，通过粗线表示的边，节点 3 最为重要，其次为节点 4，然后是节点 2，按重要性顺序排列的示意图。

GAT 正是通过这种方式，将这种概念应用于相邻节点。

GAT通过引入注意力机制来加权邻居节点的特征。对于节点i，其更新公式为：

$$h\_i^{(l+1)} = \sigma(\sum\_{j \in \mathcal{N}\_i} \alpha\_{ij}W^{(l)}h\_j^{(l)})$$

其中注意力系数$\alpha\_{ij}$的计算：

$$\alpha\_{ij} = \frac{exp(LeakyReLU(a^T[Wh\_i || Wh\_j]))}{\sum\_{k \in \mathcal{N}\_i} exp(LeakyReLU(a^T[Wh\_i || Wh\_k]))}$$

##### 注意力计算过程
基础GAT模型(Velickovic et al. 2018)的注意力计算包含三个关键步骤：
1. **线性变换**：对节点特征进行共享线性变换 $h\_i' = W h\_i$
2. **注意力分数计算**：使用单层前馈网络计算注意力分数 $e\_{ij} = a^T[Wh\_i||Wh\_j]$
3. **归一化处理**：通过softmax进行归一化 $\alpha\_{ij} = \text{softmax}(e\_{ij})$

##### 多头注意力机制
GAT引入多头注意力以稳定学习过程：
- **连接式多头**：各头输出直接拼接 $h\_i' = \|\_{k=1}^K \sigma(\sum\_{j\in N\_i} \alpha\_{ij}^k W^k h\_j)$
- **平均式多头**：各头输出取平均 $h\_i' = \sigma(\frac{1}{K}\sum\_{k=1}^K \sum\_{j\in N\_i} \alpha\_{ij}^k W^k h\_j)$

#### 关键改进模型

##### GATv2
GATv2(Brody et al. 2021)解决了原始GAT的静态注意力问题：
- 调整计算顺序：先非线性变换再线性投影
$e\_{ij} = a^T \text{LeakyReLU}(W[h\_i||h\_j])$
- 证明原始GAT的注意力排名与查询无关
- 在多个基准数据集上表现优于GAT

##### PPRGAT
PPRGAT(Choi 2022)整合个性化PageRank信息：
- 在注意力计算中加入PPR分数
$e\_{ij} = \text{LeakyReLU}(a^T[Wh\_i||Wh\_j||\pi\_{ij}])$
- 保留原始GAT结构的同时利用全局图信息
- 在节点分类任务中表现优异

##### SuperGAT
SuperGAT(Kim and Oh 2021)提出两种自监督变体：
1. 边预测任务：利用注意力分数预测边存在性
2. 标签一致性任务：利用注意力分数预测节点标签一致性
3.  在噪声图上表现更鲁棒

#### 💻 实现细节

##### PyTorch实现的GAT层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def \_\_init\_\_(self, in\_features, out\_features, dropout, alpha, concat=True):
        super(GATLayer, self).\_\_init\_\_()
        self.in\_features = in\_features
        self.out\_features = out\_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        ### 变换矩阵
        self.W = nn.Parameter(torch.zeros(size=(in\_features, out\_features)))
        nn.init.xavier\_uniform\_(self.W.data, gain=1.414)
        
        ### 注意力向量
        self.a = nn.Parameter(torch.zeros(size=(2*out\_features, 1)))
        nn.init.xavier\_uniform\_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        ### x: 节点特征矩阵 [N, in\_features]
        ### adj: 邻接矩阵 [N, N]
        
        ### 线性变换
        h = torch.mm(x, self.W)  ### [N, out\_features]
        N = h.size()[0]

        ### 计算注意力分数
        a\_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        a\_input = a\_input.view(N, N, 2 * self.out\_features)
        e = self.leakyrelu(torch.matmul(a\_input, self.a).squeeze(2))

        ### 掩码机制
        zero\_vec = -9e15 * torch.ones\_like(e)
        attention = torch.where(adj > 0, e, zero\_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        ### 聚合特征
        h\_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h\_prime)
        else:
            return h\_prime

class GAT(nn.Module):
    def \_\_init\_\_(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).\_\_init\_\_()
        self.dropout = dropout
        
        ### 多头注意力层
        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) 
            for \_ in range(nheads)
        ])
        
        ### 输出层
        self.out\_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        ### 多头注意力
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out\_att(x, adj)
        return F.log\_softmax(x, dim=1)
```

##### 实际应用示例

```python
### 模型初始化
model = GAT(nfeat=input\_dim,
           nhid=8,
           nclass=num\_classes,
           dropout=0.6,
           alpha=0.2,
           nheads=8)

### 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight\_decay=5e-4)

### 训练循环
def train():
    model.train()
    optimizer.zero\_grad()
    output = model(features, adj)
    loss = F.nll\_loss(output[idx\_train], labels[idx\_train])
    loss.backward()
    optimizer.step()
    return loss.item()
```

### 高阶注意力 (High-Order Attention)

#### 高阶注意力核心思想

高阶注意力机制突破了传统GAT仅关注直接邻居的限制，通过以下方式扩展了注意力范围：

1. 多跳邻居聚合：考虑k-hop范围内的节点（k>1）
2. 路径感知：关注节点间的路径信息而不仅是直接连接
3. 全局感受野：部分模型实现近似全局注意力

<img src="\img\GAT\HighOrderAttention.png" alt="HighOrderAttention" width="60%" height="auto">

#### 关键技术路线

##### 基于路径的注意力

SPAGAN(Yang et al. 2019)开创性地提出路径注意力机制：
- 计算节点间最短路径作为注意力传播路径
- 路径特征通过LSTM编码 $p\_{ij} = \text{LSTM}([h\_i, h\_{i1}, ..., h\_{j}])$
- 注意力分数计算 $\alpha\_{ij} = \text{softmax}(W\_p p\_{ij})$

##### 基于随机游走的注意力
PaGNN(Yang et al. 2021d)采用个性化PageRank：
- 定义转移矩阵P=AD⁻¹
- 计算PPR分数矩阵 $\Pi = \alpha(I - (1-\alpha)P)^{-1}$
- 将PPR分数融入注意力计算

##### 谱域注意力
MAGNA(Wang et al. 2021)结合谱理论：
- 使用低通滤波器平滑高频噪声
- 注意力传播公式 $H^{(l+1)} = \sigma(\sum\_{k=0}^K \beta\_k T\_k(\tilde{L})H^{(l)}W\_k)$ ，其中$T\_k$为切比雪夫多项式

### 关系感知注意力 (Relation-Aware Attention)

关系感知注意力机制主要解决图中不同类型关系的差异化处理问题，其核心思想包括：
1. **关系类型区分**：识别并利用图中不同类型的关系（边）
2. **关系特定参数**：为每种关系类型学习独立的注意力参数
3. **关系特征融合**：结合关系特征与节点特征进行注意力计算

<img src="\img\GAT\RelationAwareAttention.png" alt="RelationAwareAttention" width="60%" height="auto">

#### 关键技术方法
##### 带符号图注意力
SiGAT(Huang et al. 2019b)针对带符号图提出：
- 平衡理论建模：朋友的朋友是朋友
- 状态理论建模：敌人的朋友是敌人
- 注意力分数计算 $α\_{ij} = \text{softmax}(\text{MLP}([h\_i‖h\_j‖r\_{ij}]))$, 其中$r\_{ij}$表示边类型（正/负）

##### 异构图注意力
RGAT(Busbridge et al. 2019)处理异构图：
- 关系特定变换矩阵 $h\_i^r = W\_r h\_i$
- 关系感知注意力 $e\_{ij}^r = a\_r^T[h\_i^r‖h\_j^r]$

### 层次注意力 (Hierarchical Attention)

层次注意力机制通过构建多级注意力结构来处理图数据中的复杂关系，其核心思想包括：
1. **多层次信息整合**：同时考虑节点级、路径级和图级信息
2. **分层注意力计算**：在不同层次应用不同的注意力机制
3. **信息流动控制**：设计信息从低层向高层的传递方式

<img src="\img\GAT\HierachicalAttention.png" alt="HierachicalAttention" width="60%" height="auto">

#### 关键技术方法

##### 节点-语义双层次注意力

HAN(Wang et al. 2019d)提出：
- **节点级注意力**：学习同一元路径下节点的重要性 $h\_i' = \sum\_{j\in N\_i} \alpha\_{ij}h\_j$
- **语义级注意力**：学习不同元路径的重要性 $Z = \sum\_{m=1}^M \beta\_m \cdot Z\_m$

##### 多粒度层次注意力
GraphHAM(Lin et al. 2022)设计：
- 局部粒度注意力：捕捉节点邻域特征
- 全局粒度注意力：捕捉图结构特征
- 跨粒度注意力：协调不同粒度信息

##### 动态层次注意力
RGHAT(Zhang et al. 2020c)实现：
- **底层实体注意力**：处理节点特征 $e\_{ij} = a(h\_i,h\_j)$
- **高层关系注意力**：处理边类型特征 $r\_k = \sum\_{i,j} \beta\_{ij}^k e\_{ij}$

## 层间注意力 (Inter-Layer GAT)

在神经网络层之间，跨层GAT通过特征融合方法将不同特征空间的表示结合起来。根据不同的融合方法，我们将这些基于注意力的GNN分为五个子类别，包括多级注意力、多通道注意力、多视图注意力和时空注意力。

### 多级注意力 (Multi-Level Attention)

多级注意力机制通过构建多级注意力结构来处理图数据中的复杂关系，其核心思想包括：
1. **层次化信息处理**：将图数据分解为不同层次的抽象表示
2. **跨层次信息交互**：在不同层次间建立注意力连接
3. **自适应权重分配**：自动学习各层次对最终任务的重要性

#### 关键技术方法

##### 节点-路径双层次注意力
DAGNN(Liu et al. 2020)提出：
- **节点级注意力**：学习局部邻域内节点的重要性 $h\_i^{(l)} = \sum\_{j\in N\_i} \alpha\_{ij}^{(l)} h\_j^{(l-1)}$
- **路径级注意力**：学习不同传播深度的重要性 $z\_i = \sum\_{l=0}^L \beta\_l h\_i^{(l)}$

##### 自适应深度注意力
TDGNN(Wang and Derr 2021)设计：
- 动态调整各层的注意力范围
- 基于树分解的层次化注意力
- 跨层信息融合机制

##### 跳跃知识架构
GAMLP(Zhang et al. 2022c)实现：
- 底层局部注意力
- 中层区域注意力
- 高层全局注意力
- 跳跃连接整合各层特征

### 多通道注意力 (Multi-Channel Attention)

多通道注意力机制通过构建并行注意力通道来处理图数据中的多样化特征，其核心思想包括：
1. **通道多样化**：将输入特征分解为多个特征通道
2. **通道特异性**：为每个通道设计独立的注意力机制
3. **通道融合**：自适应地整合不同通道的信息
#### 关键技术方法

##### 频率自适应注意力

FAGCN(Bo et al. 2021)提出：
- 低频通道：捕捉节点相似性 $h\_{low} = \sum\_{j\in N\_i} \frac{1}{\sqrt{d\_i d\_j}} h\_j$
- 高频通道：捕捉节点差异性 $h\_{high} = \sum\_{j\in N\_i} -\frac{1}{\sqrt{d\_i d\_j}} h\_j$
- 自适应融合： $h\_i = \alpha\_{low} h\_{low} + \alpha\_{high} h\_{high}$

##### 自适应通道混合
ACM(Luan et al. 2021)设计：
- 多通道特征提取
- 通道间注意力交互
- 动态通道权重分配

##### 不确定性感知通道
UAG(Feng et al. 2021)实现：
- 通道不确定性估计
- 基于不确定性的通道注意力
- 鲁棒性通道融合

### 多视角注意力 (Multi-View Attention)

多视角注意力机制通过构建并行注意力视角来处理图数据中的多样化信息，其核心思想包括：
1. **视角多样化**：从不同角度（如拓扑结构、节点特征、时间序列等）构建多个视角
2. **视角特异性**：为每个视角设计独立的注意力机制
3. **视角融合**：自适应地整合不同视角的信息

#### 关键技术方法

##### 结构-特征双视角注意力
AM-GCN(Wang et al. 2020b)提出：
- 拓扑视角：基于图结构的邻接矩阵 $A\_{topo} = A$
- 特征视角：基于节点特征的相似度矩阵 $A\_{feat} = \text{sim}(X,X^T)$
- 注意力融合： $A\_{final} = \sum\_{v\in\{topo,feat\}} w\_v A\_v$

### 时空注意力 (Spatio-Temporal Attention)

时空注意力机制通过构建并行注意力模块来处理图数据中的空间和时间依赖性，其核心思想包括：
1. 空间依赖性：捕捉节点间的拓扑关系
2. 时间依赖性：建模动态图中的时序演化模式
3. 时空交互：联合建模时空维度的相互影响
#### 关键技术方法
##### 空间-时间双注意力
DySAT(Sankar et al. 2018)提出：
- 空间注意力：基于当前快照的图结构 $A\_{spatial} = \text{softmax}(Q\_s K\_s^T/\sqrt{d})$
- 时间注意力：基于节点的时间序列 $A\_{temporal} = \text{softmax}(Q\_t K\_t^T/\sqrt{d})$
- 联合建模： $Z = (A\_{spatial} \oplus A\_{temporal})V$

### 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/Lee 等 - 2018 - Attention Models in Graphs A Survey.pdf" target="_blank">📄 Lee 等 - 2018 - Attention Models in Graphs A Survey</a>

<a href="https://github.com/xmu-xiaoma666/External-Attention-pytorch" target="_blank">
  <span style="display: inline-block; vertical-align: middle;">
    <img src="/icon/github.svg" alt="github" style="height: 1.3em; vertical-align: middle; margin-top: 16px;">
  </span>
  External-Attention-pytorch
</a>

<a href="/paper/Sun 等 - 2023 - Attention-based graph neural networks a survey.pdf" target="_blank">📄 Sun 等 - 2023 - Attention-based graph neural networks a survey</a>

<a href="https://disassemble-channel.com/graph-attention-network-gat/" target="_blank">
  <span style="display: inline-block; vertical-align: middle;">
    <img src="/icon/google.svg" alt="github" style="height: 1.3em; vertical-align: middle; margin-top: 16px;">
  </span>
  【深層学習】Graph Attention Networks(GAT)を理解する
</a> 
