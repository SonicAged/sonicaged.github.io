---
title: Reinforcement-Learning-L9
categories:
  - ReinForcement Learning
  - Mathmatical Foundation
date: 2025-07-28 11:55:11
tags:
---

# 策略梯度方法读书笔记  

<div style="display: flex; align-items: center;">
  <img src="/img/reinforcement-learning/09.png" style="width: 175px; margin-right: 20px;">
  <p>策略梯度直接优化参数化策略。本文对比表格与函数表示，推导策略梯度定理，分析平均状态值和平均奖励两种度量，给出REINFORCE算法的实现框架，讨论探索-利用平衡机制。</p>
</div>

<!-- more -->



## 一、策略梯度基本思想  
### 1. 策略表示方法演进  
#### (1) 传统表格表示  
- **存储方式**：二维表格存储每个状态-动作对的概率  
  $$
  \begin{array}{c|ccccc}
  s \backslash a & a_1 & a_2 & \cdots & a_m \\\\ \hline
  s_1 & \pi(a_1|s_1) & \pi(a_2|s_1) & \cdots & \pi(a_m|s_1) \\\\
  \vdots & \vdots & \vdots & \ddots & \vdots \\\\
  s_n & \pi(a_1|s_n) & \pi(a_2|s_n) & \cdots & \pi(a_m|s_n)
  \end{array}
  $$  
- **局限性**：  
  - 存储开销大（需存储 $|\mathcal{S}| \times |\mathcal{A}|$ 个条目）  
  - 缺乏泛化能力（状态间无信息共享）  

#### (2) 参数化函数表示  
- **数学形式**：$\pi(a|s,\theta)$，其中 $\theta \in \mathbb{R}^m$ 为参数向量  
- **实现方式**：  
  - 神经网络（输入状态s，输出动作概率分布）  
  - Softmax输出层确保 $\pi(a|s,\theta) > 0$：  
    $$
    \pi(a|s,\theta) = \frac{e^{h(s,a,\theta)}}{\sum_{a' \in \mathcal{A}} e^{h(s,a',\theta)}}
    $$  
- **优势**：  
  - 存储高效（仅需保存参数 $\theta$)  
  - 自动泛化（相似状态共享参数）  

### 2. 策略优化的核心思想  
- **目标函数**：定义标量度量 $J(\theta)$ 评估策略质量  
- **优化方法**：梯度上升算法  
  $$
  \theta_{t+1} = \theta_t + \alpha \nabla_\theta J(\theta_t)
  $$  
- **关键问题**：  
  1. 如何设计合适的度量 $J(\theta)$？  
  2. 如何高效计算梯度 $\nabla_\theta J(\theta)$？  

---

## 二、最优策略的度量指标  
### 1. 度量1：平均状态值（Average Value）  
$$
\bar{v}\_\pi = \sum\_{s \in \mathcal{S}} d(s) v\_\pi(s) = \mathbb{E}\_{S \sim d} [v\_\pi(S)]
$$  
- **权重分布 $d(s)$ 的选择**：  
  - **独立于策略**（如 $d_0$）：  
    - 均匀分布：$d_0(s) = 1/|\mathcal{S}|$  
    - 初始状态集中：$d_0(s_0)=1, d_0(s \neq s_0)=0$  
  - **依赖策略**（平稳分布 $d_\pi$）：  
    - 反映长期访问频率：高频状态权重更高  

### 2. 度量2：平均奖励（Average Reward）  
$$
\bar{r}\_\pi = \sum\_{s \in \mathcal{S}} d\_\pi(s) r\_\pi(s) = \mathbb{E}\_{S \sim d\_\pi} [r\_\pi(S)]
$$  
其中：  
- $r_\pi(s) = \sum_a \pi(a|s) r(s,a)$ 为状态s的期望即时奖励  
- $r(s,a) = \mathbb{E}[R|s,a]$ 为状态-动作对的期望奖励  

### 3. 度量等价性与关系  
| **度量**       | **等价形式**                                  | **适用场景**       |  
|----------------|---------------------------------------------|-------------------|  
| **平均状态值** | $\lim_{n\to\infty} \mathbb{E}[\sum_{t=0}^n \gamma^t R_{t+1}]$ | 折扣场景（$\gamma < 1$) |  
| **平均奖励**   | $\lim_{n\to\infty} \frac{1}{n} \mathbb{E}[\sum_{t=0}^{n-1} R_{t+1}]$ | 无折扣场景         |  

- **重要关系**（折扣场景下）：  
  $$
  \bar{r}\_\pi = (1-\gamma) \bar{v}\_\pi
  $$  
  表明二者可同时最大化  

---

## 三、度量梯度的计算  
### 1. 统一梯度表达式  
$$
\nabla_\theta J(\theta) = \sum_{s \in \mathcal{S}} \eta(s) \sum_{a \in \mathcal{A}} \nabla_\theta \pi(a|s,\theta) q_\pi(s,a)
$$  
- **关键变换**（对数导数技巧）：  
  $$
  \nabla_\theta \pi(a|s,\theta) = \pi(a|s,\theta) \nabla_\theta \ln \pi(a|s,\theta)
  $$  
- **期望形式**：  
  $$
  \nabla\_\theta J(\theta) = \mathbb{E}\_{S \sim \eta, A \sim \pi} \left[ \nabla\_\theta \ln \pi(A|S,\theta) q\_\pi(S,A) \right]
  $$  

### 2. 梯度估计的实践意义  
- **随机梯度近似**：  
  $$
  \nabla_\theta J(\theta) \approx \nabla_\theta \ln \pi(a|s,\theta) q_\pi(s,a)
  $$  
  其中 $(s,a)$ 为采样状态-动作对  
- **策略探索保证**：  
  - Softmax输出确保 $\pi(a|s,\theta) > 0$（避免动作概率归零）  

---

## 四、梯度上升算法  
### 1. 基础算法框架  
$$
\theta_{t+1} = \theta_t + \alpha \mathbb{E} \left[ \nabla_\theta \ln \pi(A|S,\theta_t) q_\pi(S,A) \right]
$$  
- **随机梯度版本**：  
  $$
  \theta_{t+1} = \theta_t + \alpha \nabla_\theta \ln \pi(a_t|s_t,\theta_t) q_t(s_t,a_t)
  $$  
- **REINFORCE算法**：当 $q_\pi$ 用蒙特卡洛估计时  

### 2. 算法特性分析  
#### (1) 采样机制  
- **状态分布**：$S \sim \eta$（通常为平稳分布 $d_\pi$)  
- **动作选择**：$A \sim \pi(\cdot|S,\theta)$（**同策略**要求）  

#### (2) 更新量分解  
令 $\beta_t = \frac{q_t(s_t,a_t)}{\pi(a_t|s_t,\theta_t)}$，则更新规则为：  
$$
\theta_{t+1} = \theta_t + \alpha \beta_t \nabla_\theta \pi(a_t|s_t,\theta_t)
$$  
- **行为解释**：  
  - $\beta_t > 0$：增加动作 $a_t$ 的选择概率  
  - $\beta_t < 0$：减少动作 $a_t$ 的选择概率  

#### (3) 探索-利用平衡  
- **利用机制**：$\beta_t \propto q_t(s_t,a_t)$ → 偏向高值动作  
- **探索机制**：$\beta_t \propto 1/\pi(a_t|s_t,\theta_t)$ → 提升低频动作概率  

---

## 五、总结与理论意义  
### 1. 策略梯度核心公式  
$$
\boxed{\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \ln \pi(A|S,\theta) q_\pi(S,A) \right]}
$$  

### 2. 算法特性对比  
| **特性**         | **表格方法**                     | **策略梯度方法**               |  
|------------------|----------------------------------|--------------------------------|  
| **策略表示**     | 查表                             | 参数化函数 $\pi(a|s,\theta)$ |  
| **更新对象**     | 动作概率直接修改                 | 参数 $\theta$ 迭代更新       |  
| **探索能力**     | 依赖 $\epsilon$-greedy         | 内置随机策略                   |  
| **收敛保证**     | 有限状态有理论保证               | 局部最优解                     |  

### 3. 理论扩展方向  
- **Actor-Critic架构**：用价值函数近似 $q_\pi$  
- **确定性策略梯度**（DPG）：处理连续动作空间  
- **自然策略梯度**：改进优化轨迹的收敛性  

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning" target="_blank">
  <span style="display: inline-block; vertical-align: middle;">
    <img src="/icon/github.svg" alt="github" style="height: 1.1em; vertical-align: text-bottom; margin-top: 16px;">
  </span>
  Book-Mathematical-Foundation-of-Reinforcement-Learning
</a>