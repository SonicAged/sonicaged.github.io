---
title: HGCN and HGAT
date: 2025-07-20 17:33:51
categories:
  - model
  - hypergraph
tags:
  - CDR
  - model
  - embedding
  - graph theory
  - Basic
  - hypergraph
  - GNN
  - PyTorch
  - attention
  - GNN
---

# 超图的卷积和注意力机制

在{% post_link Hypergraph %}中鼠鼠介绍了超图的定义以及构造方式~~实际上是复述捏~~，接下来将会介绍在图神经网络中如何处理超图，主要~~纯粹~~介绍一下HGCN和HGAT。

在阅读该博客前，确保已经了解了GCN和GAT~之前也有相应的博客，可以看一下捏~。

<!-- more -->

## HGCN

~~这个玩意儿实际上非常简单，让我们先回忆一下：~~

>$$
>H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})
>$$
>
>1. 引入$L_{sym} = I - \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$作为聚合（AGG）部分
>       - 添加自环：$\tilde{A} = A + I_N$
>       - 计算归一化系数：$\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$
>2. 特征变换：$H^{(l)}W^{(l)}$
>3. 邻域聚合：$\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}$
>4. 非线性变换：$\sigma(\cdot)$

而超图也有对应的拉普拉斯矩阵

$$
\Delta=\mathbf{I}-\mathbf{D}_{v}^{-1/2}\mathbf{HWD}_{e}^{-1}\mathbf{H}^{T}\mathbf{D}_{v}^{-1/2}
$$

我们只需要把前半截一砍，换成超图的，就是HGCN了捏

$$
\mathbf X^{(l+1)} = \sigma(\mathbf{D}_{v}^{-1/2}\mathbf{HWD}_{e}^{-1}\mathbf{H}^{T}\mathbf{D}_{v}^{-1/2}\mathbf X^{(l)}\mathbf P^{(l)})
$$

为了避免命名的重复，我们用 $\mathbf X^{(\cdot)}$ 和 $\mathbf P^{(\cdot)}$ 代替GCN中的  $H^{(\cdot)}$ 和 $W^{(\cdot)}$

## HGAT

~~感觉和GAT没区别捏~~

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/Bai 等 - 2021 - Hypergraph convolution and hypergraph attention.pdf" target="_blank">📄 Bai 等 - 2021 - Hypergraph convolution and hypergraph attention</a>

<a href="/paper/Feng 等 - 2019 - Hypergraph Neural Networks.pdf" target="_blank">📄 Feng 等 - 2019 - Hypergraph Neural Networks</a>

<a href="/paper/Gao 等 - 2021 - Hypergraph Learning Methods and Practices.pdf" target="_blank">📄 Gao 等 - 2021 - Hypergraph Learning Methods and Practices</a>
