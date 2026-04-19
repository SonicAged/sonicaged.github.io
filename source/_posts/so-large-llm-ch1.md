---
title: so-large-lm Day 1
categories:
  - LLMs and Agents
  - LLMs
  - so-large-lm
tags:
  - DIY
  - LLMs
date: 2026-04-19 19:38:02
---

# 大模型基础 ｜ Day 1

好像是一个LLM的入门讲解捏

<!--more-->

## 引言

我们将语言模型是为一种**对token序列的概率分布**。假设我们有一个词元集的词汇表 $V$ 。语言模型p为每个词元序列 $x_{1},...,x_{L}$ ∈ $V$ 分配一个概率（介于0和1之间的数字）：

$$
p(x_1, \dots, x_L)
$$

### 自回归语言模型(Autoregressive Language Models)

将序列  $x_{1:L}$  的联合分布  $p(x_{1:L})$  的常见写法是使用概率的链式法则：

$$
p(x_{1:L}) = p(x_1) p(x_2 \mid x_1) p(x_3 \mid x_1, x_2) \cdots p(x_L \mid x_{1:L-1}) = \prod_{i=1}^L p(x_i \mid x_{1:i-1}).
$$

特别地，我们需要理解  $p(x_{i}∣x_{1:i−1})$  是一个给定前面的记号  $x_{1:i−1}$ 后，下一个记号  $x_{i}$  的条件概率分布。在数学上，任何联合概率分布都可以通过这种方式表示。然而，自回归语言模型的特点是它可以利用例如前馈神经网络等方法有效计算出每个条件概率分布  $p(x_{i}∣x_{1:i−1})$  。在自回归语言模型  $p$  中生成整个序列 $x_{1:L}$ ，我们需要一次生成一个词元(token)，该词元基于之前以生成的词元进行计算获得：

$$
\begin{aligned}
\text { for } i & =1, \ldots, L: \\
x_i & \sim p\left(x_i \mid x_{1: i-1}\right)^{1 / T},
\end{aligned}
$$

其中  $T≥0$  是一个控制我们希望从语言模型中得到多少随机性的温度参数：

- $T=0$：确定性地在每个位置 i 选择最可能的词元 $x_{i}$
- $T=1$：从纯语言模型“正常（normally）”采样
- $T=∞$：从整个词汇表上的均匀分布中采样

这里的$T$指的是Temperature，我们把标准化后的这一坨成为退🔥条件概率。

### 北京人的工作

来康康牢资历捏

#### N-gram模型

N-gram模型。在一个n-gram模型中，关于$x_{i}$的预测只依赖于最后的 $n-1$ 个字符 $x_{i−(n−1):i−1}$ ，而不是整个历史：

$$
p(x_i \mid x_{1:i-1}) = p(x_i \mid x_{i-(n-1):i-1}).
$$

#### 神经语言模型

语言模型的一个重要进步是神经网络的引入。[Bengio等人](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)在2003年首次提出了神经语言模型，其中 $p(x_{i}∣x_{i−(n−1):i−1})$ 由神经网络给出：

$$
p(cheese∣ate,the)=some-neural-network(ate,the,cheese)。
$$

注意，上下文长度仍然受到n的限制，但现在对更大的n值估计神经语言模型在统计上是可行的。

典型的例子就是RNN和Transformer捏

## LLM的能力

<img src="https://raw.githubusercontent.com/SonicAged/uPic-public/master/2026/04/20/003233_9852D1795F390C80023FA1D0C2EA9DDD.jpg" style="width: 30%">
<center><small>图 1  已经知道语言模型、任务模型、adaptation、上下文学习的自信的鼠鼠</small></center>

### Language Modeling

困惑度（Perplexity）是一个重要的指标，用于衡量语言模型的性能。它可以解释为模型在预测下一个词时的平均不确定性。简单来说，如果一个模型的困惑度较低，那么它在预测下一个词的时候就会更加准确。对于给定的语言模型和一个测试数据集，困惑度被定义为：

$$
P(X) = P(x_1,x_2,...,x_N)^{(-1/N)}
$$

其中， $X=x_{1},x_{2},...,x_{N}$ 是测试集中的词序列， $N$ 是测试集中的总词数。困惑度与语言模型的质量紧密相关。一个优秀的语言模型应能准确预测测试数据中的词序列，因此它的困惑度应较低。相反，如果语言模型经常做出错误的预测，那么它的困惑度将较高。

具体来说，困惑度一般被计算为：

$$
\text{perplexity}_p\left(x_{1: L}\right)=\exp \left(\frac{1}{L} \sum_{i=1}^L \log \frac{1}{p\left(x_i \mid x_{1: i-1}\right)}\right) \text {. }
$$

困惑度可以被理解为每个标记（token）的平均"分支因子（branching factor）"。这里的“分支因子”可以理解为在每个位置，模型认为有多少种可能的词会出现。例如，若困惑度为10，那意味着每次模型在预测下一个词时，平均上会考虑10个词作为可能的选择。

## 模型架构

### 分词 (Tokenize)

之前已经比较详细的学习了捏

### 模型分类

#### Encoder-Only

由文本得到 Contextual Embedding。如BERT，RoBERTa。用于分类

#### Decoder-Only

绝大部分是鼠鼠们之前看到的自回归语言模型

#### Encoder-Decoder

Transformer

### 基础架构

鼠鼠已经长大了捏，这些还是会的捏