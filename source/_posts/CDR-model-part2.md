---
title: CDR models In 2023
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
  - multi-view
date: 2025-07-26 20:11:14
---

# 2023年的CDR模型

想看2020到2022的可以前往{% post_link CDR-model-part1 %}

本文将介绍的是2023年的CDR模型，总的来说23年的模型在之前的基础上变得更复杂了，但感觉和鼠鼠想要的创新还有一些小距离捏~~（主要是一连看到好多篇拼好模捏）~~

<!-- more -->

## 2023

### <a href="/paper/CDR/2023/Chu 等 - Graph Transformer for Drug Response Prediction.pdf" target="_blank">GraTransDRP</a>

这应该是第一次将Graph Transformer引入CDR中吧~~应该吧~~。

关于Graph Transformer，鼠鼠专门写过一篇博客{% post_link Graph-Transformer %}。像了解的可以去看一下捏

#### Framework

~~这个架构图让鼠鼠想到一位故人<a href="/paper/CDR/2020-2022/Nguyen 等 - Integrating Molecular Graph Data of Drugs and Multiple -Omic Data of Cell Lines for Drug Response Pr.pdf" target="_blank">(GraOmicDRP)</a>捏，一看果然是一个团队做的捏，该说不说，这里就直接把两个架构图放在一起捏dddd~~

<img src="/img/CDR/Framework/CDR-model-part2/GraTransDRP.png" alt="GraTransDRP"  width="80%" height="auto">

<img src="/img/CDR/Framework/CDR-model-part1/GraOmicDRP.png" alt="GraOmicDRP" width="80%" height="auto">

#### 数据集

经典出装（CCLE和GDSC）

### <a href="/paper/CDR/2023/Li 等 - MMCL-CDR enhancing cancer drug response prediction with multi-omics and morphology images contrasti.pdf" target="_blank">MMCL CDR</a>

这玩意儿引入了**形态学图像**并将其用于与之前的组学数据进行对比学习，鼠鼠并不喜欢先验知识，但鼠鼠现在想知道另一件事（过两天会进行实验，到时候补充）：是否可以直接在组学内部经行对比学习以提高一些东西呢？

鼠鼠现在设想的改造如下：

1. 将形态学图像改成另一组组学信息
2. 改造对比学习公式使其可以对三个组学进行对比学习，所作的修改类似于从二元交叉熵到多元交叉熵

#### Framework

<img src="\img\CDR\Framework\CDR-model-part2\MMCL-CDR.png" alt="MMCL CDR" style="zoom:50%;" />

#### 数据集

GDSC、PubChem、DMSZ

### <a href="/paper/CDR/2023/Liu和Zhang - A subcomponent-guided deep learning method for interpretable cancer drug response prediction.pdf" target="_blank">SubCDR</a>

这里主要是将药物做成官能团序列一样的东西用于训练

#### Framework

<img src="\img\CDR\Framework\CDR-model-part2\SubCDR.png" alt="SubCDR" style="zoom:33%;" />

#### 数据集

GDSC和CCLE

### <a href="/paper/CDR/2023/Liu 等 - 2023 - HMM-GDAN Hybrid multi-view and multi-scale graph duplex-attention networks for drug response predic.pdf" target="_blank">HMM GDAN</a>

多的不说少的不唠，这其实是鼠鼠看的第一篇Multi-view Attention的论文，之前有过一篇~~（但感觉只是形式上的）~~

#### Framework

<img src="\img\CDR\Framework\CDR-model-part2\HMM-GDAN.png" alt="HMM-GDAN" style="zoom: 33%;" />

这里给出其multi-view的具体构造过程：

##### 1. 多视图图构建

###### 输入数据

<img src="\img\CDR\Framework\CDR-model-part2\multi-view.png" alt="image-20250727105814311" style="zoom: 80%;" />

- **多组学数据**：构成三维张量  
  $$
  C \in \mathbb{R}^{N\_{cl} \times N\_g \times (K-1)}
  $$
  - $N\_{cl}$: 细胞系数量
  - $N\_g$: 癌症相关基因数量  
  - $K-1$: 组学类型数

###### 构建步骤

1. **节点特征生成**  
   细胞系$i$的基因特征矩阵：  
   $$
   X\_c^i = C(i, :, :) \in \mathbb{R}^{N\_g \times (K-1)}
   $$

2. **邻接矩阵生成**
   
   - **组学视图**（$v=1,...,K-1$）:  
     $$
     \rho\_{ij}^{(v)} = \frac{\sum\_{q=1}^{N\_g} (Y\_{i,q}^{(v)} - \bar{Y}\_i^{(v)})(Y\_{j,q}^{(v)} - \bar{Y}\_j^{(v)})}{\sqrt{\sum\_{q=1}^{N\_g} (Y\_{i,q}^{(v)} - \bar{Y}\_i^{(v)})^2}\sqrt{\sum\_{q=1}^{N\_g} (Y\_{j,q}^{(v)} - \bar{Y}\_j^{(v)})^2}}
     $$

     阈值化：

     $$
     E\_{ij}^{(v)} = \begin{cases} 
     1 & \text{if } |\rho\_{ij}^{(v)}| > \tau^{(v)} \\ 
     0 & \text{otherwise}
     \end{cases}
     $$
     
   - **STRING视图**（$v=K$）:  
     $$
     E\_{ij}^{(K)} = \begin{cases} 
     1 & \text{if } s\_{ij}^{(K)} > \tau^{(K)} \\ 
     0 & \text{otherwise}
     \end{cases}
     $$
   
3. **最终图结构**  
   $$
   \mathcal{G}\_c = \langle V, \{ E^{(v)} \}\_{v=1}^K \rangle
   $$

##### 2. 双注意力机制

<img src="\img\CDR\Framework\CDR-model-part2\Duplex-attention.png" alt="image-20250727105814311" style="zoom: 25%;" />

###### 多视图自注意力

1. 共享参数投影：  
   $$
   \Theta = [H^{(1)}\theta, ..., H^{(K)}\theta] \in \mathbb{R}^{N\_g \times K}
   $$
   
2. 相似度计算：  
   $$
   \bar{\alpha}\_{ij} = \text{LeakyReLU}\left( \frac{\langle \Theta\_i, \Theta\_j \rangle}{\|\Theta\_i\|\_2 \|\Theta\_j\|\_2}, \beta \right)
   $$
   
3. 注意力系数：  
   $$
   \alpha\_{ij} = \frac{\exp(\bar{\alpha}\_{ij})}{\sum\_{j=1}^K \exp(\bar{\alpha}\_{ij})}
   $$
   
4. 视图表示更新：  
   $$
   H\_s^{(v)} = \sum\_{j=1}^K \alpha\_{vj} H^{(j)}
   $$

###### 视图级注意力

1. 表示拼接：  
   $$
   H\_s = [H\_s^{(1)}, ..., H\_s^{(K)}] \in \mathbb{R}^{N\_g \times K d\_c}
   $$
   
2. 视图权重计算：  
   $$
   \bar{w}^{(v)} = \text{Avg}(H\_s z^{(v)} + b)
   $$
   
3. 归一化权重：  
   $$
   w^{(v)} = \frac{\exp(\bar{w}^{(v)})}{\sum\_{v=1}^K \exp(\bar{w}^{(v)})}
   $$
   
4. 最终融合：  
   $$
   H = \sum\_{v=1}^K w^{(v)} H\_s^{(v)}
   $$

此外，其还引入了multi-scale学习~（比较简单，可以直接去看原文捏）~

### <a href="/paper/CDR/2023/Liu - 2023 - Emden A novel method integrating graph and transformer representations for predicting the effect of.pdf" target="_blank">Emden</a>

这感觉没什么有用的信息捏

#### Framework

<img src="\img\CDR\Framework\CDR-model-part2\Emden.png" alt="Emden" style="zoom:50%;" />

### <a href="/paper/CDR/2023/Liu和Zhang - A subcomponent-guided deep learning method for interpretable cancer drug response prediction.pdf" target="_blank">AutoCDPR</a>

采用了类似于AutoML的方式，但对鼠鼠而言，这东西就像将之前应该是超参数的东西也加入了训练，但实际用到CDR的部分只占很小一点~~（是不是有点舍本逐末了捏）~~

#### Framework

<img src="\img\CDR\Framework\CDR-model-part2\AutoCDPR.png" alt="AutoCDPR" style="zoom: 33%;" />



#### 数据集

GDSC和CCLE外加PubChem

### <a href="/paper/CDR/2023/Peng - 2023 - Improving drug response prediction based on two-space graph convolution.pdf" target="_blank">TSGCNN</a>

由于之后有比这个玩意儿更加激进的RedCDR，所以这里只放一个架构图捏

<img src="\img\CDR\Framework\CDR-model-part2\TSGCNN.png" alt="image-20250727111745618" style="zoom:50%;" />

### <a href="/paper/CDR/2023/Rajendran和Sivannarayna - Multi Head Graph Attention for Drug Response Predicton.pdf" target="_blank">📄 Rajendran和Sivannarayna - Multi Head Graph Attention for Drug Response Predicton</a>

从各个方面来讲都过于《淳朴》了捏，略过

### <a href="/paper/CDR/2023/Sagingalieva 等 - Hybrid Quantum Neural Network For Drug Response Prediction.pdf" target="_blank">HQNN</a>

将其他模型最后的FCN换成了一个QNN

<img src="\img\CDR\Framework\CDR-model-part2\HQNN.png" alt="image-20250727112841044" style="zoom: 50%;" />

### <a href="/paper/CDR/2023/Wang 等 - GADRP graph convolutional networks and autoencoders for cancer drug response prediction.pdf" target="_blank">GADRP</a>

没什么特别的捏

<img src="/img/CDR/Framework/CDR-model-part2/GADPR.png" alt="GADPR" style="zoom:50%;" />

### <a href="/paper/CDR/2023/Yang - 2023 - GPDRP a multimodal framework for drug response prediction with graph transformer.pdf" target="_blank">GPDRP</a>

这位更是如此捏

<img src="/img/CDR/Framework/CDR-model-part2/GPDRP.png" alt="image-20250727113526356" style="zoom:50%;" />

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/CDR/2023/Chu 等 - Graph Transformer for Drug Response Prediction.pdf" target="_blank">📄 Chu 等 - Graph Transformer for Drug Response Prediction</a>

<a href="/paper/CDR/2023/Li 等 - MMCL-CDR enhancing cancer drug response prediction with multi-omics and morphology images contrasti.pdf" target="_blank">📄 Li 等 - MMCL-CDR enhancing cancer drug response prediction with multi-omics and morphology images contrasti</a>

<a href="/paper/CDR/2023/Liu - 2023 - Emden A novel method integrating graph and transformer representations for predicting the effect of.pdf" target="_blank">📄 Liu - 2023 - Emden A novel method integrating graph and transformer representations for predicting the effect of</a>

<a href="/paper/CDR/2023/Liu 等 - 2023 - HMM-GDAN Hybrid multi-view and multi-scale graph duplex-attention networks for drug response predic.pdf" target="_blank">📄 Liu 等 - 2023 - HMM-GDAN Hybrid multi-view and multi-scale graph duplex-attention networks for drug response predic</a>

<a href="/paper/CDR/2023/Liu和Zhang - A subcomponent-guided deep learning method for interpretable cancer drug response prediction.pdf" target="_blank">📄 Liu和Zhang - A subcomponent-guided deep learning method for interpretable cancer drug response prediction</a>

<a href="/paper/CDR/2023/Oloulade 等 - Cancer drug response prediction with surrogate modeling-based graph neural architecture search.pdf" target="_blank">📄 Oloulade 等 - Cancer drug response prediction with surrogate modeling-based graph neural architecture search</a>

<a href="/paper/CDR/2023/Peng - 2023 - Improving drug response prediction based on two-space graph convolution.pdf" target="_blank">📄 Peng - 2023 - Improving drug response prediction based on two-space graph convolution</a>

<a href="/paper/CDR/2023/Rajendran和Sivannarayna - Multi Head Graph Attention for Drug Response Predicton.pdf" target="_blank">📄 Rajendran和Sivannarayna - Multi Head Graph Attention for Drug Response Predicton</a>

<a href="/paper/CDR/2023/Sagingalieva 等 - Hybrid Quantum Neural Network For Drug Response Prediction.pdf" target="_blank">📄 Sagingalieva 等 - Hybrid Quantum Neural Network For Drug Response Prediction</a>

<a href="/paper/CDR/2023/Wang 等 - GADRP graph convolutional networks and autoencoders for cancer drug response prediction.pdf" target="_blank">📄 Wang 等 - GADRP graph convolutional networks and autoencoders for cancer drug response prediction</a>

<a href="/paper/CDR/2023/Yang - 2023 - GPDRP a multimodal framework for drug response prediction with graph transformer.pdf" target="_blank">📄 Yang - 2023 - GPDRP a multimodal framework for drug response prediction with graph transformer</a>