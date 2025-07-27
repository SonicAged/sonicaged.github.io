---
title: CDR models From 2020 To 2022
date: 2025-07-20 23:52:07
categories:
  - CDR
  - models
tags:
  - CDR
  - model framework
  - practice
  - GNN
  - attention
  - contrast learning
  - graph signal processing
  - graph theory
  - random walk
  - multi-view
---

# 2020到2022年的CDR模型

本博客将简单分析从2020年到2022年的CDR论文中所有的图神经相关的模型架构，使用数据，模型和实验亮点。由于篇幅有限~~（主要是鼠鼠懒捏）~~，所有的模型都不会有详细的说明。但鼠鼠都将原文贴在了最后，想下细了解的可以去看原文捏。

另外，鼠鼠已经在之前大致地看过一遍截止目前的相关文章，奈何因为一些SB事情，再加上没有留下任何的笔记~~（主要原因）~~，所有已经忘得差不多了。这也是决定写一篇博客的原因

还有就是，这里的*大致*指的是：2023之后的全部和2022之前的顺眼的捏，所以这篇中的大多数模型是没看过的捏

<!-- more -->

## 2020

### <a href="/paper/CDR/2020-2022/Liu 等 - DeepCDR a hybrid graph convolutional network for predicting cancer drug response.pdf" target="_blank">DeepCDR</a>

这是鼠鼠初入CDR时看的第一篇论文，这篇论文现在看来虽然有些过时~~（过时到源代码都是用tensorflow1.x写的，现在跑起来用鼠鼠的蒟蒻4060还会爆显存）~~。但不可否认的是它给之后的模型给出了方向，也成为了几乎所有的之后的模型的基线模型。

[Source Code](https://github.com/kimmo1019/DeepCDR)

#### Framework

<img src="/img/CDR/Framework/CDR-model-part1/DeepCDR.png" alt="DeepCDR" width="80%" height="auto">

这个架构现在看来是十分朴素的，他大致就分为了两条线：

1. **药物**：将其转换为**图结构**处理，这一整条线成为了后来大多数药物的GCN Encoder
2. **组学**：没有组学融合，没有防止过平滑，就是纯粹的FCN，淳朴得像刚走出大山的理塘王子

<img src="/img/难绷/丁真.jpg" alt="理塘王子" width="20%" height="auto">

另外，值得一提的是，DeepCDR是一个回归模型，因此之后选用的指标是Pearson’s correlation，Spearman’s correlation以及RMSE

#### 数据集

GDSC，CCLE和TCGA

## 2021

### <a href="/paper/CDR/2020-2022/Feng 等 - AGMI Attention-Guided Multi-omics Integration for Drug Response Prediction with Graph Neural Networ.pdf" target="_blank">AGMI</a>

鼠鼠看过之后的模型之后回头看到的模型捏，现在看来，应该是第一个在组学那边构造图模型的捏。但是在简图的过程中，引入了一些大多数模型没有用到的先验知识，也许这不是一些希望看到的事情捏

[Source Code](https://github.com/yivan-WYYGDSG/AGMI)

#### Framework

<img src="/img/CDR/Framework/CDR-model-part1/AGMI.png" alt="AGMI" width="80%" height="auto">

还是分为两条线来看：

1. **药物**：将DeepCDR的UGCN换成了GIN
2. **组学**：主要有两个新颖的地方：
    - 开始的时候，将组学信息利用先验知识将其连成一张图
    - 处理这张图的是一个叫做MPNN（消息传递神经网络）的带有注意力引导的NN，这里就只贴一张图捏~~（鼠鼠对RNN和门控单元不感兴趣捏）~~
    - <img src="/img/CDR/Framework/CDR-model-part1/AGMI-MPNN.png" alt="AGMI-MPNN" width="80%" height="auto">

仍然是一个回归模型，没有什么新颖的实验捏

#### 数据集

CCLE和GDSC

## 2022

### <a href="/paper/CDR/2020-2022/Hostallero 等 - Looking at the BiG picture incorporating bipartite graphs in drug response prediction.pdf" target="_blank">BiG-DRP</a>

首次引入异构二分图将IC50值作为一种输入，而不只是label看待，这也就意味着，各式各样的针对处理异构图的图神经网络将会开始大规模的出现了捏

[Source Code](https://github.com/ddhostallero/BiG-DRP)

#### Framework

<img src="/img/CDR/Framework/CDR-model-part1/BiG-DRP.png" alt="BiG-DRP" width="80%" height="auto">

可以看到，只有药物那条线没了，取而代之的是一个与细胞系结合的异构网络。但实际上，可以看出来，这里的H-GCN本质上于GCN没有什么区别，大致理由是对于一个二部图而言的邻居不可能是同类的节点，换句话说，即使是正常的GCN，一层中药物聚合的也只有细胞系的信息，细胞系聚合也只有药物的信息，但是由于药物和细胞系特征的维数是不一样的，所以原本的权重矩阵不能再是方阵了

仍然是一个回归模型。值得一提的是，其在BiG-DRP+中引入了早停机制，应该是第一次捏

#### 数据集

GDSC和TCGA

### <a href="/paper/CDR/2020-2022/Jiang 等 - DeepTTA a transformer-based model for predicting cancer drug response.pdf" target="_blank">DeepTTA</a>

甚至单组学模型。逆天的是直接往药物线里硬套了一个Transformer，真是一种战斗爽的感觉捏

[Source Code](https://github.com/jianglikun/DeepTTC)

~~为什么一个叫DeepTTA的模型的仓库名叫DeepTTC捏~~

#### Framework

<img src="/img/CDR/Framework/CDR-model-part1/DeepTTA.png" alt="DeepTTA" width="80%" height="auto">

单组学，回归模型（虽然叫Classiffier）

然后就是当时读到的时候有一些小问题：

1. 同一药物的子结构序列理论上来讲是有多个，但他们的上下文差距理论上来讲是很大的，只学习其中一个是否会有些不妥呢
2. 分子式非常接近的不同药物理论上是可能产生相同子结构序列的，这样是否会有些奇怪捏

#### 数据集

GDSC和CCLE

### <a href="/paper/CDR/2020-2022/Liu 等 - GraphCDR a graph neural network method with contrastive learning for cancer drug response predictio.pdf" target="_blank">GraphCDR</a>

鼠鼠读到的第一篇对比学习的文章，虽然一些~~很多~~模型是为了加入某个模块而讲故事，起初鼠鼠也认为这是没有硬整的东西，但是现在看来对比学习实际上对于CDR是十分有用的捏。

其中一部分原因便是在{% post_link CDR-data-analysis %}中提到的，如果用分类的方式构造异构体，那么这张图将会是一张非常稀疏的图，也就会有过平滑的风险，而对比学习恰好可以缓解这一点捏。

关于对比学习，鼠鼠一直都想写一篇博客，奈何能力太捞，还是感觉对这个很不熟悉捏

[Source Code](https://github.com/BioMedicalBigDataMiningLab/GraphCDR)

#### Framework

<img src="/img/CDR/Framework/CDR-model-part1/GraphCDR.png" alt="GraphCDR" width="80%" height="auto">

先说明架构图中可能引起误解的地方：

1. Corrupt的箭头没有什么意义，这里的腐化图使之通过IC50值得出的耐药性关系，并不是在原图的基础上生成的
2. 两张图实际上是被两个不同的GNN处理的，只不过这两个GNN的配置一样而已捏

值得说明的是，模型中Readout那一坨其实就是一个self-attention捏

另外，这是一个分类模型~~（好像一句废话捏）~~

#### 数据集

CCLE和GDSC

### <a href="/paper/CDR/2020-2022/Ma - 2022 - DualGCN a dual graph convolutional network model to predict cancer drug response.pdf" target="_blank">DualGCN</a>

~~疑似是有点太捞了捏~~所以就放一个架构图就行了捏

<img src="/img/CDR/Framework/CDR-model-part1/DualCDR.png" alt="DualCDR" width="80%" height="auto">

### <a href="/paper/CDR/2020-2022/Nguyen 等 - Integrating Molecular Graph Data of Drugs and Multiple -Omic Data of Cell Lines for Drug Response Pr.pdf" target="_blank">GraOmicDRP</a>

~~这位更是如此捏~~

<img src="/img/CDR/Framework/CDR-model-part1/GraOmicDRP.png" alt="GraOmicDRP" width="80%" height="auto">

### <a href="/paper/CDR/2020-2022/Nguyen 等 - Graph Convolutional Networks for Drug Response Prediction.pdf" target="_blank">GraphDRP</a>

~~又来一个捏~~

<img src="/img/CDR/Framework/CDR-model-part1/GraphDRP.png" alt="GraphDRP" width="80%" height="auto">

### <a href="/paper/CDR/2020-2022/Peng 等 - Predicting Drug Response Based on Multi-Omics Fusion and Graph Convolution.pdf" target="_blank">MOFGCN</a>

这是鼠鼠第一篇完整读过源代码的模型，现在还是觉得这个模型的代码写的很好看捏。同时这个模型的论文写得也非常清楚，没有那些谜语人似的弯弯绕绕，就是模块、作用、公式。同时，本论文中做的实验也很多很全面（常规的实验都做了一遍捏）。总之，如果想入门图神经的CDR，先看着一篇会让你取得很大的提升。

[Source Code](https://github.com/weiba/MOFGCN)

#### Framework

<img src="/img/CDR/Framework/CDR-model-part1/MOFGCN.png" alt="MOFGCN" width="80%" height="auto">

关于细胞系相似度矩阵，可以去看一下引用的原文捏[Similarity network fusion for aggregating data types on a genomic scale | Nature Methods](https://www.nature.com/articles/nmeth.2810)，~~由于学校爆的💰好像不太够，鼠鼠好像看不了捏~~，看看原文有没有对全核矩阵和稀疏核矩阵数学背景的说明捏，在这里简单谈一下鼠鼠的理解捏。

>在这里先把全核矩阵和稀疏核矩阵的定义式写在这里捏
>$$
>F_{ij}=\begin{cases} \frac{E_{ij}}{2\sum_j E_{ij}}  &\text{if}\quad i \neq j \\\\ \frac{1}{2} &\text{if}\quad i = j \end{cases}
>$$
>
>$$
>S_{ij}=\begin{cases} \frac{E_{ij}}{\sum_j E_{ij}}  &\text{if}\quad j \in N_i \\\\ \frac{1}{2} &\text{if}\quad j \notin N_i \end{cases}
>$$
>
>我们从概率的角度出发（为什么是概率，你会发现全核矩阵和稀疏核矩阵一定保证了一样的和一定是1！），那么这是否可以被视作两种不同的状态转移的规则呢？
>
>然后就查到了一个叫做**随机游走**的东西捏，大概就有以下的解释：
>
>1. 全核矩阵可以理解为**连续时间随机游走**的转移概率矩阵：
>    - 每个状态(细胞系)以相同概率1/2保持当前状态
>    - 以概率1/2转移到其他状态，转移概率与相似度成正比
>    - 确保马尔可夫链的遍历性和平稳分布存在
>
>2. 稀疏核矩阵则对应**受限随机游走**：
>    - 游走者只能移动到拓扑邻居节点
>    - 模拟生物系统中局部相互作用占主导的特性
>    - 避免远距离间接关联引入的虚假信号
>
>顺手的~~实际上并不顺手~~，我查到了这玩意儿在**图信号处理**中也有涉及：
>
>>之前在{% post_link GNN-and-GCN %}中提到过拉普拉斯矩阵$L=D-A$，下面简单的说明一些图信号相关的知识：
>>
>>- **图结构**：定义为$G = (V, E)$，$V$是节点集合($∣V∣ = N$)，E是边集合
>>
>>- **图信号**：定义为映射$f : V → ℝ$，可以表示为向量$f ∈ ℝᴺ$，其中fᵢ表示节点$vᵢ$上的信号值
>>
>>- **图滤波器与矩阵表示**：在图信号处理中，**图滤波器**是对图信号进行变换操作的核心工具，可以表示为矩阵H ∈ ℝᴺˣᴺ。滤波操作可表示为$\hat{f} = Hf$
>>
>>    其中H可以表示为拉普拉斯矩阵的函数$H = h(\mathcal{L}) = \sum_{k=0}^K h_k \mathcal{L}^k$
>
>##### 全核矩阵的频域解释
>
>全核矩阵$F$在图信号处理视角下是一个**低通滤波器**，其作用机理可以从图傅里叶变换理解：
>
>1. **图傅里叶变换**：将图信号投影到拉普拉斯矩阵的特征基
>    $$
>    \hat{f} = U^T f
>    $$
>    
>
>    其中$U$是$\mathcal L$的特征向量矩阵
>
>2. **全核矩阵的滤波作用**：
>    $$
>    F = g(\mathcal{L}) = U \cdot diag(g(λ₁),...,g(λₙ)) \cdot U^T
>    $$
>    
>
>    其中$g(λ)$是单调递减函数，保留低频成分
>
>3. **具体实现**：
>    $$
>    F_{ij} = \begin{cases} 
>    \frac{W_{ij}}{2\sum_j W_{ij}} & i \neq j \\ 
>    \frac{1}{2} & i = j 
>    \end{cases}
>    $$
>    
>
>    这种设计保证了：
>
>    - 行标准化(每行和=1)
>    - 对角线强化(自环保留)
>    - 全局信息平滑
>
>##### 稀疏核矩阵的频域解释
>
>稀疏核矩阵$S$是一个**带通滤波器**，其数学表示为：
>
>1. **拓扑约束**：
>    $$
>    S_{ij} = \begin{cases} 
>    \frac{W_{ij}}{\sum_{j \in N_i} W_{ij}} & j \in N_i \\ 
>    0 & j \notin N_i 
>    \end{cases}
>    $$
>    
>
>    其中$N_i$是节点$i$的邻居集合
>
>2. **频域特性**：
>
>    - 通过限制局部连接($N_i$)产生频带选择效应
>
>    - 特征值分布更集中，突出局部模式
>
>    - 相当于频域中的窗函数：
>        $$
>        S = U \cdot diag(s(λ₁),...,s(λₙ)) \cdot U^T
>        $$
>        其中$s(λ)$是峰值在中间频段的函数
>

~~这里顺便说明一下融合过程捏~~

##### 多尺度分析与矩阵组合

全核矩阵$F$和稀疏核矩阵$S$的组合实现了**多尺度分析**：

混合滤波器设计

组合滤波器可表示为：
$$
H = αF + (1-α)S \quad α ∈ [0,1]
$$


对应的频域响应：
$$
h(λ) = αg(λ) + (1-α)s(λ)
$$


这种组合能够：

- 低频：由F主导，捕获全局趋势
- 中高频：由S主导，提取局部特征

##### 矩阵迭代的谱分析

MOFGCN中的矩阵迭代：
$$
\begin{aligned}
GF_t &= GS \cdot \frac{(CF_{t-1} + MF_{t-1})}{2} \cdot GS^T \\\\
CF_t &= CS \cdot \frac{(GF_{t-1} + MF_{t-1})}{2} \cdot CS^T \\\\
MF_t &= MS \cdot \frac{(CF_{t-1} + GF_{t-1})}{2} \cdot MS^T
\end{aligned}
$$


从图信号处理看，这是**谱调制过程**：

1. 每次迭代相当于在频域进行$\hat{f}^{(t)} = h(λ) \cdot \hat{f}^{(t-1)}$
2. 不同组学源的矩阵($GS,CS,MS$)提供互补频带
3. 平均操作(1/2)防止频谱畸变



对于后面的异构图，论文中详细的说明了异构图的GCN构造方式，这里就不在赘述了捏

关于实验，他一共做了4个：

- **测试1**：在随机将矩阵中的值置零时，将我们的模型与HNMDRP和SRMF进行比较，以重建细胞系-药物关联矩阵。
- **测试2**：在随机屏蔽一行或一列时，将我们的模型与HNMDRP和SRMF进行比较，以重建细胞系-药物关联矩阵。
- **测试3**：在预测单一药物的反应时，将我们的方法与HNMDRP、SRMF、MOLI、DeepForest和DeepCDR进行比较。
- **测试4**：在预测一些靶向药物的反应时，将我们的模型与HNMDRP、SRMF、MOLI、DeepForest和DeepCDR进行比较。 

### <a href="/paper/CDR/2020-2022/Peng 等 - Predicting cancer drug response using parallel heterogeneous graph convolutional networks with neigh.pdf" target="_blank">NIHGCN</a>

又是一个力大砖飞的典型代表捏，看看架构图就行了捏

<img src="/img/CDR/Framework/CDR-model-part1/NIHGCN.png" alt="NIHGCN" width="80%" height="auto">

### <a href="/paper/CDR/2020-2022/Wang - A multi-view multi-omics model for cancer drug response prediction.pdf" target="_blank">MvMo</a>

首次引入了Multi-view attention，可以去看一下{% post_link GAT %}，其他的没什好说的，还是直接一个架构图捏

<img src="/img/CDR/Framework/CDR-model-part1/MvMo.png" alt="MvMo" width="80%" height="auto">

#### <a href="/paper/CDR/2020-2022/Xie - 2022 - Drug response prediction using graph representation learning and Laplacian feature selection.pdf" target="_blank">📄 Xie - 2022 - Drug response prediction using graph representation learning and Laplacian feature selection</a>

这篇看着有点奇怪捏，不想看了捏~~（虽然有很大一部分原因是看到现在有点无聊了捏）~~

### <a href="/paper/CDR/2020-2022/Zhu 等 - TGSA protein–protein association-based twin graph neural networks for drug response prediction with.pdf" target="_blank">TGSA</a>

又是先验知识捏，不想看了捏

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/CDR/2020-2022/Feng 等 - AGMI Attention-Guided Multi-omics Integration for Drug Response Prediction with Graph Neural Networ.pdf" target="_blank">📄 Feng 等 - AGMI Attention-Guided Multi-omics Integration for Drug Response Prediction with Graph Neural Networ</a>

<a href="/paper/CDR/2020-2022/Hostallero 等 - Looking at the BiG picture incorporating bipartite graphs in drug response prediction.pdf" target="_blank">📄 Hostallero 等 - Looking at the BiG picture incorporating bipartite graphs in drug response prediction</a>

<a href="/paper/CDR/2020-2022/Jiang 等 - DeepTTA a transformer-based model for predicting cancer drug response.pdf" target="_blank">📄 Jiang 等 - DeepTTA a transformer-based model for predicting cancer drug response</a>

<a href="/paper/CDR/2020-2022/Liu 等 - DeepCDR a hybrid graph convolutional network for predicting cancer drug response.pdf" target="_blank">📄 Liu 等 - DeepCDR a hybrid graph convolutional network for predicting cancer drug response</a>

<a href="/paper/CDR/2020-2022/Liu 等 - GraphCDR a graph neural network method with contrastive learning for cancer drug response predictio.pdf" target="_blank">📄 Liu 等 - GraphCDR a graph neural network method with contrastive learning for cancer drug response predictio</a>

<a href="/paper/CDR/2020-2022/Ma - 2022 - DualGCN a dual graph convolutional network model to predict cancer drug response.pdf" target="_blank">📄 Ma - 2022 - DualGCN a dual graph convolutional network model to predict cancer drug response</a>

<a href="/paper/CDR/2020-2022/Nguyen 等 - Graph Convolutional Networks for Drug Response Prediction.pdf" target="_blank">📄 Nguyen 等 - Graph Convolutional Networks for Drug Response Prediction.pdf</a>

<a href="/paper/CDR/2020-2022/Nguyen 等 - Integrating Molecular Graph Data of Drugs and Multiple -Omic Data of Cell Lines for Drug Response Pr.pdf" target="_blank">📄 Nguyen 等 - Integrating Molecular Graph Data of Drugs and Multiple -Omic Data of Cell Lines for Drug Response Pr</a>

<a href="/paper/CDR/2020-2022/Peng 等 - Predicting cancer drug response using parallel heterogeneous graph convolutional networks with neigh.pdf" target="_blank">📄 Peng 等 - Predicting cancer drug response using parallel heterogeneous graph convolutional networks with neigh</a>

<a href="/paper/CDR/2020-2022/Peng 等 - Predicting Drug Response Based on Multi-Omics Fusion and Graph Convolution.pdf" target="_blank">📄 Peng 等 - Predicting Drug Response Based on Multi-Omics Fusion and Graph Convolution</a>

<a href="/paper/CDR/2020-2022/Wang - A multi-view multi-omics model for cancer drug response prediction.pdf" target="_blank">📄 Wang - A multi-view multi-omics model for cancer drug response prediction</a>

<a href="/paper/CDR/2020-2022/Xie - 2022 - Drug response prediction using graph representation learning and Laplacian feature selection.pdf" target="_blank">📄 Xie - 2022 - Drug response prediction using graph representation learning and Laplacian feature selection</a>

<a href="/paper/CDR/2020-2022/Zhang 等 - Multi-Scale Dynamic Convolutional Network for Knowledge Graph Embedding.pdf" target="_blank">📄 Zhang 等 - Multi-Scale Dynamic Convolutional Network for Knowledge Graph Embedding</a>

<a href="/paper/CDR/2020-2022/Zhu 等 - TGSA protein–protein association-based twin graph neural networks for drug response prediction with.pdf" target="_blank">📄 Zhu 等 - TGSA protein–protein association-based twin graph neural networks for drug response prediction with</a>