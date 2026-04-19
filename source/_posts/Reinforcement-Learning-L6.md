---
title: Reinforcement-Learning-L6
categories:
  - ReinForcement Learning
  - Mathmatical Foundation
date: 2025-07-28 11:55:00
tags:
---

# 随机逼近与随机梯度下降读书笔记  

<div style="display: flex; align-items: center;">
  <img src="/img/reinforcement-learning/06.png" style="width: 175px; margin-right: 20px;">
  <p>随机逼近（RM算法）和随机梯度下降（SGD）是强化学习的优化基础。本文推导RM算法的收敛条件，对比BGD/SGD/MBGD的收敛特性，建立与TD学习的理论桥梁，解释优化中的两阶段行为。</p>
</div>

<!-- more -->



## 一、引言：核心概念与学习目标  
### 背景与意义  
- **知识衔接**：本讲是蒙特卡洛学习与时序差分(TD)学习的**桥梁**  
- **核心问题**：解决时序差分学习中的知识空白（为何有效、如何设计）  
- **主要内容**：  
  - 随机逼近(SA)基础：Robbins-Monro算法  
  - 随机梯度下降(SGD)：机器学习优化的核心工具  

### 关键问题驱动  
1. 如何**无模型求解方程根**？  
2. 如何高效处理**含随机性的优化问题**？  
3. 如何理解TD学习的**理论根基**？  

---

## 二、随机逼近算法：Robbins-Monro算法  
### 1. 问题描述  
求解方程根：  
$$
g(w) = 0, \quad w \in \mathbb{R}
$$  
- **挑战**：$g(w)$表达式未知，仅能通过**噪声观测**获取信息：  
$$
\tilde{g}(w, \eta_k) = g(w) + \eta_k
$$  
其中 $\eta_k$ 为观测噪声（如随机采样误差）  

### 2. 算法形式  
迭代更新：  
$$
w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)
$$  
- $w_k$：第k步的根估计值  
- $a_k$：步长序列，满足收敛条件  

### 3. 示例说明  
#### 示例1：解析函数求根  
- 目标方程：$g(w) = w - 10 = 0$  
- 参数：$w_1 = 20, \, a_k = 0.5, \, \eta_k = 0$  
- 迭代过程：  
  $$
  \begin{align*}
  w_2 &= 20 - 0.5 \times 10 = 15 \\\\
  w_3 &= 15 - 0.5 \times 5 = 12.5 \\\\
  &\vdots \\\\
  w_k &\to 10
  \end{align*}
  $$  
> **关键观察**：估计值从初始点快速收敛至真值  

#### 示例2：非线性函数收敛  
- 目标方程：$g(w) = \tanh(w-1) = 0$  
- 参数：$w_1=3, \, a_k=1/k$  
- 结果：$w_k \to 1$（真值）  

### 4. 收敛性分析  
**Robbins-Monro定理**（收敛条件）：  
1. **函数单调性**：  
   $$
   0 < c_1 \leq \nabla_w g(w) \leq c_2, \, \forall w
   $$  
   > 保证根存在唯一  
2. **步长衰减条件**：  
   $$
   \sum_{k=1}^\infty a_k = \infty \quad \text{且} \quad \sum_{k=1}^\infty a_k^2 < \infty
   $$  
   > 控制收敛速度与稳定性  
3. **噪声性质**：  
   $$
   \mathbb{E}[\eta_k | \mathcal{H}_k] = 0, \quad \mathbb{E}[\eta_k^2 | \mathcal{H}_k] < \infty
   $$  
   > 观测噪声零均值、有界方差  

### 5. 应用于均值估计  
#### 问题转换  
- 目标：估计 $\mathbb{E}[X]$  
- 转化为求根问题：  
  $$
  g(w) = w - \mathbb{E}[X] = 0
  $$  
- 噪声观测：$\tilde{g}(w, \eta_k) = w - x_k$  

#### RM算法实现  
$$
w_{k+1} = w_k - a_k (w_k - x_k)
$$  
- 当 $a_k = 1/k$ 时：  
  $$
  w_{k+1} = \frac{1}{k} \sum_{i=1}^k x_i \quad (\text{精确样本均值})
  $$  

---

## 三、随机梯度下降（SGD）  
### 1. 问题描述  
优化目标：  
$$
\min_w J(w) = \mathbb{E}[f(w, X)]
$$  
- **挑战**：目标函数期望无法直接计算  

### 2. 算法形式  
#### 三类梯度下降算法  
| **算法** | **更新公式** | **特点** |  
|----------|--------------|----------|  
| **批量梯度下降(BGD)** | $ w_{k+1} = w_k - \alpha_k \frac{1}{n} \sum_{i=1}^n \nabla_w f(w_k, x_i) $ | 高精度、低效率 |  
| **随机梯度下降(SGD)** | $ w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k) $ | 高效、带噪声 |  
| **小批量梯度下降(MBGD)** | $ w_{k+1} = w_k - \alpha_k \frac{1}{m} \sum_{j \in \mathcal{I}_k} \nabla_w f(w_k, x_j) $ | 平衡精度与效率 |  

> 注：$\mathcal{I}_k$ 为批量索引集，$m$ 为批量大小  

### 3. 示例与应用  
#### 均值估计的优化视角  
- 优化问题：  
  $$
  \min_w J(w) = \mathbb{E}\left[\frac{1}{2} \|w - X\|^2\right]
  $$  
- 解析解：$w^* = \mathbb{E}[X]$  
- SGD更新：  
  $$
  w_{k+1} = w_k - \alpha_k (w_k - x_k)
  $$  
  > 与RM算法的均值估计形式一致  

### 4. 收敛性分析  
#### SGD与RM算法的等价性  
- 优化目标等价于求根问题：  
  $$
  g(w) = \nabla_w J(w) = \mathbb{E}[\nabla_w f(w, X)] = 0
  $$  
- 观测噪声：  
  $$
  \tilde{g}(w, \eta_k) = \nabla_w f(w, x_k) = g(w) + \eta_k
  $$  
- **收敛定理**：若目标函数强凸（$\nabla_w^2 f \geq c > 0$）且步长满足衰减条件，则 $w_k \to w^*$ 几乎必然收敛  

### 5. 收敛模式  
#### 典型两阶段行为  
1. **快速收敛阶段**：  
   - 当 $w_k$ 远离 $w^*$ 时，随机梯度方向近似真实梯度  
   - 估计值快速向真值靠近  
2. **随机波动阶段**：  
   - 当 $w_k$ 接近 $w^*$ 时，随机噪声主导更新过程  
   - 估计值在真值附近波动，振幅随步长衰减而减小  

#### 理论解释  
相对误差上界：  
$$
\delta_k \leq \frac{|\nabla_w f(w_k, x_k) - \mathbb{E}[\nabla_w f(w_k, X)]|}{c |w_k - w^*|}
$$  
> $c$ 为函数凸性强度的下界，解释远离真值时收敛快、接近真值时波动大的现象  

### 6. BGD, MBGD, SGD对比  
#### 均值估计示例  
- **数据集**：100个均匀分布点（真值 $\mathbb{E}[X]=0$)  
- **收敛速度**：  
  [](@replace=1)  

| **算法** | 批量大小 | 收敛速度 | 稳定性 |  
|----------|----------|----------|--------|  
| BGD      | $n=100$ | 慢       | 高     |  
| MBGD     | $m=10$  | 中       | 中     |  
| SGD      | $m=1$   | 快       | 低     |  

---

## 四、总结  
### 1. 核心算法关系  
$$
\text{均值估计} \subset \text{RM算法} \subset \text{SGD} \subset \text{随机逼近框架}
$$  

### 2. 关键公式回顾  
| **问题**         | **核心公式**                     |  
|------------------|----------------------------------|  
| RM算法           | $ w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k) $ |  
| SGD              | $ w_{k+1} = w_k - \alpha_k \nabla_w f(w_k, x_k) $ |  
| 均值估计（SGD） | $ w_{k+1} = w_k - \alpha_k (w_k - x_k) $ |  

### 3. 理论意义  
- **统一框架**：为时序差分学习（如Q-learning、TD(λ)）提供收敛性分析工具  
- **优化基础**：SGD是深度学习与强化学习的共同优化引擎  
- **实践指导**：  
  - 远离最优解时采用大步长加速收敛  
  - 接近最优解时衰减步长提升稳定性  

### 4. 典型应用场景  
- **强化学习**：值函数估计、策略优化  
- **机器学习**：神经网络训练、在线学习  
- **统计估计**：大规模数据集的参数估计  

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning" target="_blank">
  <span style="display: inline-block; vertical-align: middle;">
    <img src="/icon/github.svg" alt="github" style="height: 1.1em; vertical-align: text-bottom; margin-top: 16px;">
  </span>
  Book-Mathematical-Foundation-of-Reinforcement-Learning
</a>