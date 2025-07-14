---
title: GAT
date: 2025-07-14 22:26:58
categories:
tags:
---

# 

<!-- more -->

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

<a href="/paper/Lee 等 - 2018 - Attention Models in Graphs A Survey.pdf" target="_blank">📄 Lee 等 - 2018 - Attention Models in Graphs A Survey</a>
<a href="https://github.com/xmu-xiaoma666/External-Attention-pytorch" target="_blank">github: External-Attention-pytorch</a> 