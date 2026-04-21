---
title: so-large-lm Day 2～3
categories:
  - LLMs and Agents
  - LLMs
  - so-large-lm
tags:
  - DIY
  - LLMs
date: 2026-04-22 00:18:02
---

# 大模型基础 ｜ Day 2～3

好像是一个LLM的入门讲解捏

<!--more-->

## 新的模型架构

### 混合专家

- 混合专家模型：我们创建一组专家。每个输入只激活一小部分专家。
    - 直觉：类似一个由专家组成的咨询委员会，每个人都有不同的背景（如历史、数学、科学等）。

$$
\text{input} \quad\quad\Rightarrow\quad\quad \text{expert}_1 \quad \text{expert}_2 \quad \text{expert}_3 \quad \text{expert}_4 \quad\quad\Rightarrow\quad\quad \text{output}.
$$

#### Sparsely-gated mixture of experts ([Lepikhin et al. 2021](https://arxiv.org/pdf/2006.16668.pdf))

- 因此，我们将混合专家的想法应用于：
    - 每个token
    - 每层Transformer block（或者隔层使用）
- 由于前馈层对于每个token是独立的，因此，我们将每个前馈网络转变为混合专家（MoE）前馈网络：

$$
\text{MoETransformerBlock}(x_{1:L}) = \text{AddNorm}(\text{MoEFeedForward}, \text{AddNorm}(\text{SelfAttention}, x_{1:L})).
$$

- 隔层使用MoE Transformer block。

![glam-architecture](https://raw.githubusercontent.com/SonicAged/uPic-public/master/2026/04/20/175302_glam-architecture.png)

我们将top-2专家的近似门控函数定义如下：

- 计算第一个专家： $e_1 = \arg\max_e g_e(x)$ 。

- 计算第二个专家： $e_2 = \arg\max_{e \neq e_1} g_e(x)$ 。

- 始终保留第一个专家，并随机保留第二个专家：
    - 设 $p = \min(2 g_{e_2}(x), 1)$ 
    - 在概率为 $p$ 的情况下， $\tilde g_{e_1}(x) = \frac{g_{e_1}(x)}{g_{e_1}(x) + g_{e_2}(x)}, \tilde g_{e_2}(x) = \frac{g_{e_2}(x)}{g_{e_1}(x) + g_{e_2}(x)}$ 。对于其他专家 $e \not\in \{ e_1, e_2 \}$，$\tilde g_e(x) = 0$ 。
    - 在概率为 $1 - p$ 的情况下， $\tilde g_{e_1}(x) = 1$。对于$e \neq e_1$，$\tilde g_e(x) = 0 $ 。

> 这里感觉需要多沉淀一下捏，后面应该会专门开一篇来学一下混合专家捏

### 基于检索的模型

就是RAG起手的吧，这又是一大坨了捏

## LLMs 的数据

就是介绍了一下常用的文本数据集捏

## 模型训练

从两个角度来看模型训练：**目标函数**和**优化算法**

### 目标函数

#### Decoder-only 模型 

*最大似然*就可以了捏

#### Encoder-only 模型

鼠鼠想下班了捏