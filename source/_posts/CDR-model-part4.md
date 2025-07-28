---
title: CDR-model In 2024 | Part 2
categories:
  - CDR
  - Model
tags:
  - CDR
  - model framework
  - practice
  - GNN
  - attention
  - contrast learning
  - GAN
  - dynamic graph
  - hypergraph
date: 2025-07-27 23:12:02
---

# 2024年的CDR模型~easy go~

原来没有剩多少捏，蚌

想看2020到2022的可以前往{% post_link CDR-model-part1 %}

想看2023的可以前往{% post_link CDR-model-part2 %}

想看2024（前半截）的可以前往{% post_link CDR-model-part3 %}

<!-- more -->

## 2024

### <a href="/paper/CDR/2024/Wang - 2024 - A hierarchical attention network integrating multi-scale relationship for drug response prediction.pdf" target="_blank">MultiDRP</a>

好空虚，感觉什么都没有捏

### <a href="/paper/CDR/2024/Shi - MCMVDRP a multi-channel multi-view deep learning framework for cancer drug response prediction.pdf" target="_blank">MCMVDRP</a>

又是一篇典型的一看架构图就知道在干什么的东西捏

<img src="/img/CDR/Framework/CDR-model-part2/MCMVDRP.png" alt="MCMVDRP" style="zoom:33%;" />

### <a href="/paper/CDR/2024/Sun - 2024 - DAPM-CDR A domain adaptation prompting model for drug response prediction.pdf" target="_blank">DAPM CDR</a>

和提示词有关的，目前没有兴趣捏

#### <a href="/paper/CDR/2024/Wang - 2024 - XGraphCDS An explainable deep learning model for predicting drug sensitivity from gene pathways and.pdf" target="_blank">XGraphCDS</a>

将模型可解释性的，但这不是鼠鼠想关注的捏

### <a href="/paper/CDR/2024/Xu 等 - RedCDR Dual Relation Distillation for Cancer Drug Response Prediction.pdf" target="_blank">RedCDR</a>

不想看了捏（直接掏之前的笔记了捏）

#### RedCDR模型构造过程详解

RedCDR（Dual Relation Distillation for Cancer Drug Response Prediction）是一个用于癌症药物反应预测的双关系蒸馏模型。模型基于多组学数据和药物分子结构，通过并行双分支架构独立学习细胞线/药物属性信息与细胞线-药物关联信息，并使用融合模块实现知识蒸馏。以下详细说明模型的构造过程，包括核心组件、公式和关键步骤。

##### 1. 模型总体框架

RedCDR由三个主要模块组成：**属性分支**（Attribute Branch）、**关联分支**（Association Branch）和**融合模块**（Fusion Module）。整体架构如图1所示：
![RedCDR](/img/CDR/Framework/CDR-model-part2/RedCDR.png)

- **输入**：
  - 细胞线多组学数据：基因组突变（$x\_i^g \in \mathbb{R}^{1 \times \text{Dim}\_g}$）、转录组表达（$x\_i^t \in \mathbb{R}^{1 \times \text{Dim}\_t}$）、表观基因组甲基化（$x\_i^e \in \mathbb{R}^{1 \times \text{Dim}\_e}$）。
  - 药物分子图：$\mathcal{G}\_j = (X\_j^d, A\_j^d)$，其中$X\_j^d \in \mathbb{R}^{N\_j^d \times \text{Dim}\_d}$为原子特征矩阵，$A\_j^d \in \mathbb{R}^{N\_j^d \times N\_j^d}$为邻接矩阵。
  - 细胞线-药物响应矩阵：$R \in \mathbb{R}^{N\_C \times N\_D}$，$R\_{ij} = 1$表示敏感响应，$R\_{ij} = 0$表示抵抗响应。
- **输出**：预测响应概率$\hat{R}\_{ij}^{\text{Fu}}$。

##### 2. 属性分支（Attribute Branch）

属性分支独立学习细胞线和药物的属性表示，包括多组学编码器和药物编码器。

###### 2.1 自适应交互多组学编码器（Adaptive Interacting Multi-omics Encoder）

此模块整合细胞线的多组学数据，考虑组学间交互和重要性权重。结构如图2：
<img src="/img/CDR/Framework/CDR-model-part2/RedCDR-multi-omics-encoder.png" alt="RedCDR-multi-omics-encoder" style="zoom:50%;" />

- **步骤1：基础多组学表示**  
  使用独立网络提取各组学潜在表示（统一维度$F$）：
  $$
  e\_i^g = f\_g(x\_i^g), \quad e\_i^t = f\_t(x\_i^t), \quad e\_i^e = f\_e(x\_i^e)
  $$
  其中$f\_g(\cdot)$为1D卷积网络（适用于基因组数据），$f\_t(\cdot)$和$f\_e(\cdot)$为全连接网络。拼接后得到基础表示：
  $$
  h\_i^b = \text{LeakyReLU}([e\_i^g \| e\_i^t \| e\_i^e] W\_b + b\_b)
  $$
  这里$W\_b \in \mathbb{R}^{3F \times F}$, $b\_b \in \mathbb{R}^{1 \times F}$，$\|$表示向量拼接。

- **步骤2：组学特定表示学习**  
  构建多关系图，利用细胞线间相似性增强交互：
  - 计算组学$m \in \{g, t, e\}$的相似矩阵$S^m$：
    $$
    S\_{ij}^m = k(e\_i^m, e\_j^m) = \frac{e\_i^m \cdot e\_j^{mT}}{\|e\_i^m\| \|e\_j^m\|}
    $$
  - 基于$S^m$构建邻接矩阵$A^m$（选择top-$k$相似细胞线）。
  - 应用多层GCN传播信息：
    $$
    H^{m,(l+1)} = \text{LeakyReLU}(\widetilde{A}^m H^{m,(l)} W\_m^{(l)})
    $$
    其中$\widetilde{A}^m = (\hat{D}^m)^{-1/2} (A^m + I) (\hat{D}^m)^{-1/2}$为归一化邻接矩阵，$H^{m,(0)} = [e\_i^g \| e\_i^t \| e\_i^e]$。

- **步骤3：自适应融合**  
  使用点积注意力机制加权融合组学表示，以$h\_i^b$作为查询，组学特定表示$[h\_i^g, h\_i^t, h\_i^e]$作为键和值：
  $$
  h\_i^c = \text{Softmax}(q\_i k\_i^T) v\_i, \quad q\_i = h\_i^b, \quad k\_i = v\_i = [h\_i^g, h\_i^t, h\_i^e]
  $$
  最终细胞线表示：
  $$
  h\_i^{\text{At},C} = \text{ReLU}(h\_i^c W\_C^{\text{At}} + b\_C^{\text{At}})
  $$
  其中$W\_C^{\text{At}} \in \mathbb{R}^{F \times F}$, $b\_C^{\text{At}} \in \mathbb{R}^{1 \times F}$。

###### 2.2 药物编码器（Drug Encoder）

处理药物分子图以提取全局表示：
- 应用多层GCN学习原子节点表示$H\_j^d$：
  $$
  H\_j^{d,(l+1)} = \text{LeakyReLU}(\widetilde{A}\_j^d H\_j^{d,(l)} W\_d^{(l)})
  $$
  其中$\widetilde{A}\_j^d$为归一化邻接矩阵。
- 使用全局最大池化（GMP）生成药物级表示：
  $$
  h\_j^{\text{At},D} = \text{GMP}(H\_j^d)
  $$

##### 3. 关联分支（Association Branch）

关联分支从细胞线-药物响应中学习表示，区分敏感和抵抗响应。

###### 3.1 二分图构建

- 敏感图$\mathcal{G}^s = (X^s, A^s)$和抵抗图$\mathcal{G}^r = (X^r, A^r)$：
  - 节点嵌入$X^s = X^r = [X^C \| X^D] \in \mathbb{R}^{(N\_C + N\_D) \times F}$为随机初始化可训练向量。
  - 邻接矩阵：
    $$
    A^s = \begin{bmatrix} 0 & R^s \\ (R^s)^T & 0 \end{bmatrix}, \quad A^r = \begin{bmatrix} 0 & R^r \\ (R^r)^T & 0 \end{bmatrix}
    $$
    其中$R\_{ij}^s = M\_{ij} R\_{ij}$, $R\_{ij}^r = M\_{ij} (1 - R\_{ij})$，$M\_{ij}$为指示矩阵（存在响应时为1）。

###### 3.2 表示学习

- 对敏感图应用多层GCN：
  $$
  H^{s,(l+1)} = \text{LeakyReLU}\left( (\hat{D}^s)^{-1/2} A^s (\hat{D}^s)^{-1/2} H^{s,(l)} W\_s^{(l)} \right)
  $$
  其中$H^{s,(0)} = X^s$。拼接所有层输出：
  $$
  h\_i^{S,C} = [h\_i^{s,(0)} \| h\_i^{s,(1)} \| \cdots \| h\_i^{s,(L)}], \quad h\_j^{S,D} = [h\_j^{s,(0)} \| h\_j^{s,(1)} \| \cdots \| h\_j^{s,(L)}]
  $$
- 同样方式学习抵抗图表示$h\_i^{R,C}$和$h\_j^{R,D}$。
- 整合敏感和抵抗表示：
  $$
  h\_i^{\text{As},C} = \text{ReLU}([h\_i^{S,C} \| h\_i^{R,C}] W\_C^{\text{As}} + b\_C^{\text{As}})
  $$
  $$
  h\_j^{\text{As},D} = \text{ReLU}([h\_j^{S,D} \| h\_j^{R,D}] W\_D^{\text{As}} + b\_D^{\text{As}})
  $$

##### 4. 融合模块（Fusion Module）与双关系蒸馏

融合模块整合双分支表示，并通过蒸馏机制迁移知识。

###### 4.1 分支融合

拼接属性分支和关联分支的表示：
$$
h\_i^{\text{Fu},C} = \text{ReLU}([h\_i^{\text{At},C} \| h\_i^{\text{As},C}] W\_C^{\text{Fu}} + b\_C^{\text{Fu}})
$$
$$
h\_j^{\text{Fu},D} = \text{ReLU}([h\_j^{\text{At},D} \| h\_j^{\text{As},D}] W\_D^{\text{Fu}} + b\_D^{\text{Fu}})
$$

###### 4.2 响应预测

使用线性相关系数解码器重建响应矩阵：
$$
\rho(h\_i, h\_j) = \frac{(h\_i - \mu\_i \mathbf{1}) (h\_j - \mu\_j \mathbf{1})^T}{\sqrt{(h\_i - \mu\_i \mathbf{1})(h\_i - \mu\_i \mathbf{1})^T} \sqrt{(h\_j - \mu\_j \mathbf{1})(h\_j - \mu\_j \mathbf{1})^T}}
$$
应用sigmoid函数处理不平衡样本：
$$
\phi(x) = \frac{1}{1 + e^{-\gamma x}}
$$
最终预测：
$$
\hat{R}\_{ij}^K = \phi(\rho(h\_i^{K,C}, h\_j^{K,D})), \quad K \in \{\text{At}, \text{As}, \text{Fu}\}
$$

###### 4.3 双关系蒸馏机制

促进知识从分支迁移到融合模块：
- **表示蒸馏（Representation Distillation）**：保持细胞线和药物间的相似关系：
  $$
  S\_{ij}^{K,C} = k(h\_i^{K,C}, h\_j^{K,C}), \quad S\_{ij}^{K,D} = k(h\_i^{K,D}, h\_j^{K,D})
  $$
  损失函数（以属性分支为例）：
  $$
  \mathcal{L}\_{rd}^{\text{At}} = \frac{1}{N\_C^2} \sum\_{c\_i \in C} \sum\_{c\_j \in C} \| S\_{ij}^{\text{At},C} - S\_{ij}^{\text{Fu},C} \|\_2^2 + \frac{1}{N\_D^2} \sum\_{d\_i \in D} \sum\_{d\_j \in D} \| S\_{ij}^{\text{At},D} - S\_{ij}^{\text{Fu},D} \|\_2^2
  $$
  总表示蒸馏损失：$\mathcal{L}\_{rd} = \mathcal{L}\_{rd}^{\text{At}} + \mathcal{L}\_{rd}^{\text{As}}$。
- **预测蒸馏（Prediction Distillation）**：对齐分支和融合模块的预测：
  $$
  \mathcal{L}\_{pd} = \frac{1}{N\_C N\_D} \sum\_{c\_i \in C} \sum\_{d\_j \in D} \left( \| \hat{R}\_{ij}^{\text{At}} - \hat{R}\_{ij}^{\text{Fu}} \|\_2^2 + \| \hat{R}\_{ij}^{\text{As}} - \hat{R}\_{ij}^{\text{Fu}} \|\_2^2 \right)
  $$

##### 5. 优化目标

总损失函数结合预测任务和蒸馏损失：
$$
\mathcal{L} = \mathcal{L}\_{ce} + \lambda\_{rd} \mathcal{L}\_{rd} + \lambda\_{pd} \mathcal{L}\_{pd}
$$
其中$\mathcal{L}\_{ce}$为监督交叉熵损失：
$$
\mathcal{L}\_{ce} = -\frac{1}{N\_M} \sum\_{c\_i \in C} \sum\_{d\_j \in D} M\_{ij} \left[ R\_{ij} \log(\hat{R}\_{ij}^{\text{Fu}}) + (1 - R\_{ij}) \log(1 - \hat{R}\_{ij}^{\text{Fu}}) \right]
$$
$\lambda\_{rd}$和$\lambda\_{pd}$为超参数平衡权重。

此构造过程使RedCDR能自适应整合多组学数据，并行学习属性与关联信息，并通过蒸馏提升预测鲁棒性。

### <a href="/paper/CDR/2024/Zhou - 2024 - Multi-omics fusion based on attention mechanism for survival and drug response prediction in Digesti.pdf" target="_blank">MFGAN</a>

感觉没什么东西捏

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/CDR/2024/Shi - MCMVDRP a multi-channel multi-view deep learning framework for cancer drug response prediction.pdf" target="_blank">📄 Shi - MCMVDRP a multi-channel multi-view deep learning framework for cancer drug response prediction</a>

<a href="/paper/CDR/2024/Sun - 2024 - DAPM-CDR A domain adaptation prompting model for drug response prediction.pdf" target="_blank">📄 Sun - 2024 - DAPM-CDR A domain adaptation prompting model for drug response prediction</a>

<a href="/paper/CDR/2024/Wang - 2024 - A hierarchical attention network integrating multi-scale relationship for drug response prediction.pdf" target="_blank">📄 Wang - 2024 - A hierarchical attention network integrating multi-scale relationship for drug response prediction</a>

<a href="/paper/CDR/2024/Wang - 2024 - XGraphCDS An explainable deep learning model for predicting drug sensitivity from gene pathways and.pdf" target="_blank">📄 Wang - 2024 - XGraphCDS An explainable deep learning model for predicting drug sensitivity from gene pathways and</a>

<a href="/paper/CDR/2024/Xu 等 - RedCDR Dual Relation Distillation for Cancer Drug Response Prediction.pdf" target="_blank">📄 Xu 等 - RedCDR Dual Relation Distillation for Cancer Drug Response Prediction</a>

<a href="/paper/CDR/2024/Zhou - 2024 - Multi-omics fusion based on attention mechanism for survival and drug response prediction in Digesti.pdf" target="_blank">📄 Zhou - 2024 - Multi-omics fusion based on attention mechanism for survival and drug response prediction in Digesti</a>