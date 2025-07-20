---
title: Hypergraph Introduction
date: 2025-07-19 13:38:32
categories:
  - model
  - hypergraph
tags:
  - CDR
  - model
  - embedding
  - PyTorch
  - graph theory
  - Basic
  - hypergraph
  - GNN
  - attention
---

# 超图的定义以及构造方式

随着CDR预测的精度越来越高，越来越多的人意识到全局信息的提取将会起到非常重要的作用~~没活硬整了捏~~。引入超图也就限得十分自然了。也正因如此，我们需要像了解图一样了解超图。

这篇博客主要讨论了超图的定义和构造超图的方式，接下来应该会有专门的一篇博客介绍HGCN和HGAT

<!-- more -->

## 什么是超图捏？

首先简要介绍超图的初步知识。一个超图表示为 $\mathcal G$ ，由一组顶点 $\mathcal V$ 和一组超边 $\mathcal E$ 组成。在加权超图中，每个超边 $e \in \mathcal E$ 都被分配了一个权重 $w(e)$ .表示整个超图中连接关系的重要性。令 $\mathbf W$ 表示超边权重的对角矩阵，即 $\text{diag}(\mathbf W)=[w(e\_1), w(e\_2),...,w(e\_{|\mathcal E |})]$ 。给定一个超图${\mathcal G}=({\mathcal V},{\mathcal E},{\mathbf W})$, 超图的结构通常用一个发生矩阵$\mathbf H \in \{0, 1\}^{|\mathcal{V}|\times|\mathcal{E}|}$ 表示，其中每个条目 $\mathbf H(v, e)$ 表示顶点 $v$ 是否在超边 $e$ 中


$$
\mathbf{H}(v,e)=\begin{cases} 1 &\text{if}\quad v \in e \\\\ 0 &\text{if}\quad v \notin e \end{cases}
$$

在某些情况下，关联矩阵不是简单的$(0,1)$-矩阵，而是元素在0到1范围内的连续矩阵，其中 $\mathbf H(v, e)$ 项可以表示为顶点$v$分配给超边$e$的可能性，或者顶点$v$对超边$e$的重要性，

我们可以定义超边$e$和顶点$v$的度数为

$$\delta(e)=\sum\_{v\in\mathcal{V}}\mathbf{H}(v,e),$$

以及

$$d(v)=\sum\_{e\in\mathcal{E}}w(e)*\mathbf{H}(v,e),$$

分别记为$\mathbf D\_{e} \in \mathbb{R} ^{| \mathcal{E} | \times | \mathcal{E} | }$ 和$\mathbf{D}\_{v} \in \mathbb{R} ^{| \mathcal{V} | \times | \mathcal{V} | }$ ，其中$D\_{e}$ 是超边权重的对角矩阵， $D\_{v}$ 是顶点度数的对角矩阵。

拉普拉斯矩阵在图论中起着重要的作用。例如，基于拉普拉斯矩阵的图的谱分析（谱聚类、谱分区）是基于找到图的拉普拉斯矩阵的特征值和特征向量。对于一个简单图，拉普拉斯矩阵定义为$\Delta=\mathbf{D}-\mathbf{A}$ ，其中 $\mathbf D$ 是顶点度的对角矩阵， $\mathbf A$ 是邻接矩阵。对于超图，拉普拉斯矩阵更复杂，定义为

$$\Delta=\mathbf{D}\_{v}-\mathbf{HWD}\_{e}^{-1}\mathbf{H}^{T}.$$

这个拉普拉斯矩阵可以被归一化为

$$\Delta=\mathbf{I}-\mathbf{D}\_{v}^{-1/2}\mathbf{HWD}\_{e}^{-1}\mathbf{H}^{T}\mathbf{D}\_{v}^{-1/2}.$$

下表对刚才的公式进行了总结：

| Notation        | Definition                                                   |
| --------------- | ------------------------------------------------------------ |
| $X$             | The feature vectors of subject set $X(v)$ indicates the feature vector for vertex $v$. |
| ${\mathcal G}=({\mathcal V},{\mathcal E},{\mathbf W})$ | $\mathcal G$ expresses the hypergraph structure, and $\mathcal V$, $\mathcal E$, and $W$ represent the set of vertices, the set of hyperedges, and diagonal matrix of vertex degrees, respectively. |
| $d(v)$          | The degree of vertex $v$.                                    |
| $w(e)$          | The weight of hyperedge $e$.                                 |
| $\mathbf H$             | The degree of hyperedge $e$.                                 |
| $\mathbf H\_{i,j}$       | $H\_{i,j}$ indicates the incidence strength between the vertex and the hyperedge. |
| $\mathbf D\_v$           | The diagonal matrix of vertex degrees.                       |
| $\mathbf D\_e$           | The diagonal matrix of hyperedge degrees.                    |
| $\Delta$        | The Laplacian of the hypergraph.                             |

## 超图生成

**超图学习，一定要有超图**🤚😭✋。

生成的超图结构的质量直接影响数据关联建模的有效性。如何构建有效的超图的问题并不简单，因此已经进行了广泛的研究。超图生成方法通常可以分为四类，即基于距离的方法、基于表示的方法、基于属性的方法和基于网络的方法。接下来，我们重新审视这些超图生成方法，总结如下表所示。

<table>
    <tr>
        <td><b>Hyperedge Category</b></td>
        <td><b>Hyperedge Subtypes</b></td>
        <td><b>Methods</b></td>
    </tr>
    <tr>
        <td rowspan="2" style="text-align: center;">Implicit Hyperedge</td>
        <td>Distance Based</td>
        <td>Nearest Neighbor (k-NN), k-ball<br/>Clustering (k-means)</td>
    </tr>
    <tr>
        <td>Representation Based</td>
        <td>l1 Reconstruction (l1-hypergraph)<br/>Elastic Net (Elastic hypergraph)<br/>l2-hypergraph</td>
    </tr>
    <tr>
        <td rowspan="2" style="text-align: center;">Explicit Hyperedge</td>
        <td>Attribute Based</td>
        <td>Attribute Hypergraph,</td>
    </tr>
    <tr>
        <td>Network Based</td>
        <td>Social Media Network<br/>Brain Network<br/>Reaction Network<br/>Cellular Network</td>
    </tr>
</table>


### 基于距离的超图生成

<img src="\img\Hypergraph\Distance-based hyperedge generation.png" alt="Distance-based hyperedge generation" width="80%" height="auto">

基于距离的超图生成方法利用特征空间中的距离来探索顶点之间的关系。其主要目标是在特征空间中找到邻近的顶点， 并构造一个超边来连接它们。通常，有两类方法可以构造这个超边，即最近邻搜索和聚类。在基于最近邻的方法中，超边的构造需要为每个顶点找到最近的邻居。也就是说，给定一个顶点（即质心），超边可以将自己和特征空间中的最近邻居连接起来。所选邻居的数量是一个预先定义的参数$k$或由距离范围内的顶点数$e$决定。这种类型的超边可以连接一组连接到相同质心的相似顶点。与基于最近邻的方法不同，基于聚类的方法的目标是直接通过聚类算法（例如$k$均值）将所有顶点分组到簇中，并通过超边连接同一簇中的所有顶点（请注意，可以使用不同规模的聚类结果来生成多个超边）。

对于这些超边，如何衡量有效性至关重要。每个超边的权重或置信度可以通过下式建立

$$w(e)=\sum\_{u,v\in e}\exp\biggl(-\frac{d(\mathbf{X}(u),\mathbf{X}(v))^2}{\sigma^2}\biggr),$$

其中，$u$和$v$表示超边$e$中的一对顶点，$d(\mathbf X(u), \mathbf X(v))$ 是 $u$ 和 $v$ 之间的距离。参数 $\sigma$ 可以经验地设置为所有顶点对的距离的中位数

除了特征空间的相似性外，位置（空间）信息也可以用于基于距离的超边生成即每个顶点可以找到其空间部居。在这个例子中，选定的粉红色顶点及其四个橙色邻居可以通过超边连接。

基于距离的超边可以表示特征空间中的顶点连接，该方法的主要局限性在于由于噪声和离群点的存在，顶点之间的距离在某些情况下可能不准确；另一个问题是应该连接多少邻居，因为最近邻的规模可能会影响超图学习的性能，但在实践中自适应地学习这个数量并不容易。

### 基于表示的超图生成

基于表示的超图生成方法通过特征重构来制定顶点之间的关系，例如$l1$-超图是一种使用稀疏表示来建模超图生成中顶点之间关系的工作。更具体地说，在$l1$-超图中，每个顶点充当质心，并目可以由其最近邻表示为

$$\underset{\mathbf{z}}{\operatorname*{argmin}}\|\mathbf{Bz}-\mathbf{X}(v\_{c})\|\_{2}^{2}+\gamma\|\mathbf{z}\|\_{1}$$

$$\mathrm{s.t.}\:\forall i,\mathbf{z}\_{\mathrm{i}}\geq0,$$

其中，$\mathbf{X}(v\_{c})$ 和 $\mathbf B$ 分别表示质心样本 $v\_c$ 及其 $k$ 个最近邻的特征向量， $\mathbf z\_{i}$ 是与第i个最近邻关联的已学习重建系数。基于这种表示，具有非零重建系数的邻居中的顶点被分组以生成超边并且可以将超边和每个顶点之间的连接强度设置为系数$\mathbf z\_{i}$



$l1$-超图由于使用了$l1$范数正则化，因此在揭示样本的分组信息方面能力较弱。为解决这一问题，引入弹性网通过将$l2$范数惩罚与$l1$范数约束相结合，以增强分组效果，这可以表示为

$$\operatorname*{argmin}\_{\mathbf{z}}\|\mathbf{Bz}-\mathbf{X}(v\_{c})\|\_{2}^{2}+\gamma\|\mathbf{z}\|\_{1}+\beta\|\mathbf{z}\|\_{2}^{2}.\\\mathrm{s.t.}\:\forall i,\mathbf{z}\_{i}\geq0.$$

通过利用$l2$范数和$l1$范数惩罚，弹性网可以将更多的相关邻居分组，以构建超边、其权重可以丛重建系数中导出。

值得注意的是，这两种基于表示的方法有两个局限性：

1. 这些方法使用基于$l2$-范数的度量来测量重建误差，这伊然对稀疏重建误差敏感。
2. 这些方法通过求解线性问题生成超边，这不能处理非线性数据。

为了解决这些问题，$l2$-超图通过将稀疏噪声分量从原始数据中分离出来，并将局部性整合到线性回归框架中，保留了约束。

$$
\mathrm{argmin}\_{\mathbf{C},\mathbf{E}}\|\mathbf{X}-\mathbf{X}\mathbf{C}-\mathbf{E}\|\_{F}^{2}+\frac{\gamma\_{1}}{2}\|\mathbf{C}\|\_{F}^{2}+\frac{\gamma\_{1}}{2}\|\mathbf{Q}\odot\mathbf{C}\|\_{F}^{2}+\beta\|\mathbf{E}\|\_{1}
\\\\
\mathrm{s.t.}\:\mathbf{C}^{T}\mathbf{1}=\mathbf{1},\mathrm{diag}(\mathbf{C})=\mathbf{0},
$$

其中，$\mathbf C$是系数矩阵，$\mathbf E$ 是数据误差矩阵，$\mathbf Q$是用于保持局部流形结构的邻域适配器矩阵，$\mathbf Q\_{ij}=\frac{\text{exp}(\|\mathbf{x}\_i-\mathbf{x}\_j\|\_2)}{\sum\_{t \neq i} \text{exp}(\|\mathbf{x}\_i-\mathbf{x}\_j\|\_2)}$ ， $\odot$ 表示逐元素乘法。然后，系数矩阵$\mathbf C$可用以生成超边。

基于表示的超边可以评估持机空间中每个原点的电键能通过计算特征向量之间的相关性来建立顶点间的连接与基于距离的方法类似，该方法也可能会受到数据噪声和离截值的影响，此外，该方法的数据采样过程也存在局限性：仅选择部分相关样本进行重建，生成的超边可能无法充分代表数据的相关性。

### 基于属性的超图生成

<img src="\img\Hypergraph\Attribute-based hypergraph generation.png" alt="Attribute-based hypergraph generation" width="30%" height="auto">


基于属性的超图生成方法是那些利用属性信息来构建基于属性的超图。在这个例子中，共享相同属性的顶点由一个超边连接。

在基于属性的超图中、每个超边被视为一个团，然后可以使用该团中成对边的热核权重的平均值作为超边权重

$$
w(e)=\frac{1}{\delta(e)(\delta(e)-1)}\sum\_{u,v\in e}\exp\left(-\frac{\left\|\mathbf{X}(u)-\mathbf{X}(v)\right\|\_2^2}{\sigma^2}\right)
$$

其中，$\delta(e)$ 表示超边 $e$ 的度。由于属性可能是分层的，因此生成的超边也可以具有不同的级别，从而导致不同超边表示多尺度属性连接，尽管属性信息在表示数据方面具有显著优垫，但在某些情况下，这种信息可能不可用。一种可能的解决方案是定义一组可以从现有数据中学习的属性。

### 基于网络的超图生成

<img src="\img\Hypergraph\Network-based hypergraph generation.png" alt="Network-based hypergraph generation" width="30%" height="auto">

网络数据在许多应用中可用，例如社交网络、反应网络、细胞网络和脑网络。对于这些数据，可以使用网络信息来生成主题相关性。例如，在中的顶点表示月户和图像。超边，包括同质和异质超边，用于捕获多类型关系，包括图像之间的视觉-文本内容关系，以及用户和图像之间的社会链接。通过使用基于最近邻和基于属性的超边生成方法，构建表示图像之间视觉和文本关系的同质超边。异局超边是使用连接图像和用户的社交链接关系构建的。

在基于位置的社交网络中，Yang等人同时使用用户友谊和移动信息来构建超图。具体来说，它在社交领域内生成经典的友谊超边，在社交、语义、时间和空间域之间生成签到超边。在蛋白质-蛋白质相互作用网络中、串联亲和纯化(TAP)数据自然跨越一个超图.其中复合物可以表示蛋白质基本集的子集(超边)。

一般来说、除了第一阶相关性之外、更高阶（例如第二和第三阶）相关性在网络也可以用于超边生成。图2e说明了网络数据的超边示例顶点连接。直观地说，如果注意力集中在网络中顶点的局部连接上，我们只需要考虑其低阶邻居。例如，在推荐网络中，用户根据对项目的相似偏好进行连接，仅使用一阶和二阶相关性来生成超图并进行协同过滤以进行推荐。相反，如果顶点的信息在网络中传播很长距离，我们需要使用更高阶的相关性来生成超边。

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/Bai 等 - 2021 - Hypergraph convolution and hypergraph attention.pdf" target="_blank">📄 Bai 等 - 2021 - Hypergraph convolution and hypergraph attention</a>

<a href="/paper/Feng 等 - 2019 - Hypergraph Neural Networks.pdf" target="_blank">📄 Feng 等 - 2019 - Hypergraph Neural Networks</a>

<a href="/paper/Gao 等 - 2021 - Hypergraph Learning Methods and Practices.pdf" target="_blank">📄 Gao 等 - 2021 - Hypergraph Learning Methods and Practices</a>
