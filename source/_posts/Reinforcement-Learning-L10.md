---
title: Reinforcement-Learning-L10
categories:
  - Learning
  - ReinForcement Learning
  - Mathmatical Foundation
date: 2025-07-28 11:55:16
tags:
---

# 强化学习的数学基础

~~这是之前的遗产捏，还需要整理~~

<!-- more -->

# Actor-Critic方法读书笔记  

## 一、Actor-Critic方法核心思想  
### 1. 基本架构  
- **Actor（演员）**：  
  - 负责策略更新 $\pi(a|s,\theta)$  
  - 通过参数 $\theta$ 生成动作  
- **Critic（评论家）**：  
  - 负责策略评估，估计值函数 $q(s,a,w)$ 或 $v(s,w)$  
  - 通过参数 $w$ 提供反馈信号  

### 2. 策略梯度基础  
策略梯度统一表达式：  
$$
\nabla\_\theta J(\theta) = \mathbb{E}\_{S\sim\eta, A\sim\pi} \left[ \nabla\_\theta \ln \pi(A|S,\theta) q\_\pi(S,A) \right]
$$  

随机梯度更新：  

$$
\theta\_{t+1} = \theta\_t + \alpha \nabla\_\theta \ln \pi(a\_t|s\_t,\theta_t) q\_t(s\_t,a\_t)
$$  

---

## 二、基础算法：QAC（最简单的Actor-Critic）  
### 1. 算法流程  
```python  
初始化：策略函数 $\pi(a|s,\theta_0)$，值函数 $q(s,a,w_0)$  
for 每步 $t$：  
    # 交互  
    根据 $\pi(\cdot|s_t,\theta_t)$ 选择动作 $a_t$  
    执行 $a_t$，观测奖励 $r_{t+1}$ 和下一状态 $s_{t+1}$  
    根据 $\pi(\cdot|s_{t+1},\theta_t)$ 选择 $a_{t+1}$（仅用于Critic更新）  

    # Actor更新（策略改进）  
    $\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \ln \pi(a_t|s_t,\theta_t) q(s_t,a_t,w_t)$  

    # Critic更新（值估计）  
    $w_{t+1} = w_t + \alpha_w \left[ r_{t+1} + \gamma q(s_{t+1},a_{t+1},w_t) - q(s_t,a_t,w_t) \right] \nabla_w q(s_t,a_t,w_t)$
```  
> **注**：Critic使用SARSA风格更新，需额外采样 $a_{t+1}$

### 2. 算法特性  
- **同策略（On-Policy）**：行为策略与目标策略均为 $\pi(\theta_t)$  
- **双网络结构**：策略网络与值函数网络独立更新  
- **Critic本质**：带函数近似的SARSA算法  

---

## 三、优势Actor-Critic（A2C）  
### 1. 基线不变性原理  
策略梯度可引入基线函数 $b(S)$ 而不改变期望：  
$$
\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \ln \pi(A|S,\theta) (q_\pi(S,A) - b(S)) \right]
$$  
**最优基线**（最小化方差）：  
$$
b^\*(s) = \frac{\mathbb{E}_A \left[ \|\nabla_\theta \ln \pi(A|s,\theta)\|^2 q_\pi(s,A) \right]}{\mathbb{E}_A \left[ \|\nabla_\theta \ln \pi(A|s,\theta)\|^2 \right]}
$$  

### 2. 优势函数设计  
- **次优基线选择**：$b(s) = v_\pi(s)$  
- **优势函数定义**：  
  $$
  \delta_t = q_\pi(s_t,a_t) - v_\pi(s_t) \approx r_{t+1} + \gamma v(s_{t+1},w_t) - v(s_t,w_t)
  $$  
- **梯度简化**：  
  $$
  \nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \ln \pi(A|S,\theta) \delta(S,A) \right]
  $$  

### 3. A2C算法流程  
```python  
初始化：策略网络 $\pi(a|s,\theta_0)$，值网络 $v(s,w_0)$  
for 每步 $t$：  
    # 交互  
    根据 $\pi(\cdot|s_t,\theta_t)$ 选择 $a_t$，执行后观测 $r_{t+1}, s_{t+1}$  

    # 计算TD误差  
    $\delta_t = r_{t+1} + \gamma v(s_{t+1},w_t) - v(s_t,w_t)$  

    # Actor更新  
    $\theta_{t+1} = \theta_t + \alpha_\theta \delta_t \nabla_\theta \ln \pi(a_t|s_t,\theta_t)$  

    # Critic更新  
    $w_{t+1} = w_t + \alpha_w \delta_t \nabla_w v(s_t,w_t)$
```  
**核心优势**：只需一个值网络（无需动作值函数），显著降低方差  

---

## 四、离策略Actor-Critic  
### 1. 重要性采样原理  
- **问题**：行为策略 $\beta$ 生成样本，目标策略 $\pi$ 需离策略更新  
- **解决方案**：  
  
  $$
  \mathbb{E}\_{X\sim p\_0}[f(X)] = \mathbb{E}\_{X\sim p\_1} \left[ \frac{p\_0(X)}{p\_1(X)} f(X) \right]
  $$  

- **策略梯度修正**： 
   
  $$
  \nabla\_\theta J(\theta) = \mathbb{E}\_{S\sim\rho, A\sim\beta} \left[ \frac{\pi(A|S,\theta)}{\beta(A|S)} \nabla\_\theta \ln \pi(A|S,\theta) q\_\pi(S,A) \right]
  $$  

### 2. 离策略Actor-Critic算法  
```python  
初始化：行为策略 $\beta$，目标策略 $\pi(a|s,\theta_0)$，值函数 $v(s,w_0)$  
for 每步 $t$：  
    # 交互（使用行为策略）  
    根据 $\beta(\cdot|s_t)$ 选择 $a_t$，观测 $r_{t+1}, s_{t+1}$  

    # 计算TD误差  
    $\delta_t = r_{t+1} + \gamma v(s_{t+1},w_t) - v(s_t,w_t)$  

    # Actor更新（含重要性权重）  
    $\theta_{t+1} = \theta_t + \alpha_\theta \frac{\pi(a_t|s_t,\theta_t)}{\beta(a_t|s_t)} \delta_t \nabla_\theta \ln \pi(a_t|s_t,\theta_t)$  

    # Critic更新  
    $w_{t+1} = w_t + \alpha_w \frac{\pi(a_t|s_t,\theta_t)}{\beta(a_t|s_t)} \delta_t \nabla_w v(s_t,w_t)$
```  
**关键点**：重要性权重 $\frac{\pi}{\beta}$ 修正策略差异导致的分布偏移  

---

## 五、确定性Actor-Critic（DPG）  
### 1. 确定性策略梯度定理  
- **策略表示**：$a = \mu(s,\theta)$（确定性映射）  
- **梯度表达式**：  
  $$
  \nabla\_\theta J(\theta) = \mathbb{E}\_{S\sim\rho\_\mu} \left[ \nabla_\theta \mu(S,\theta) \left( \nabla_a q\_\mu(S,a) \right)|\_{a=\mu(S)} \right]
  $$  

### 2. DPG算法流程  
```python  
初始化：行为策略 $\beta$，确定性目标策略 $\mu(s,\theta_0)$，动作值函数 $q(s,a,w_0)$  
for 每步 $t$：  
    # 交互（使用行为策略）  
    根据 $\beta(\cdot|s_t)$ 选择 $a_t$，观测 $r_{t+1}, s_{t+1}$  

    # 计算TD误差  
    $\delta_t = r_{t+1} + \gamma q(s_{t+1},\mu(s_{t+1},\theta_t),w_t) - q(s_t,a_t,w_t)$  

    # Actor更新（沿Q梯度方向）  
    $\theta_{t+1} = \theta_t + \alpha_\theta \nabla_\theta \mu(s_t,\theta_t) \left( \nabla_a q(s_t,a,w_t) \right)|_{a=\mu(s_t)}$  

    # Critic更新  
    $w_{t+1} = w_t + \alpha_w \delta_t \nabla_w q(s_t,a_t,w_t)$
```  
**核心特性**：  
- 天然离策略（行为策略 $\beta$ 可独立设计）  
- 适用于连续动作空间（如DDPG算法）  

---

## 六、算法对比总结  
| **算法**               | **策略类型** | **策略更新特点**                     | **值估计方式**       |  
|------------------------|--------------|--------------------------------------|----------------------|  
| **QAC**                 | 随机         | 基于动作值 $q(s,a)$               | SARSA+函数近似       |  
| **A2C**                 | 随机         | 基于优势函数 $\delta = q - v$     | TD误差+状态值函数    |  
| **离策略Actor-Critic**  | 随机         | 重要性采样修正 $\pi/\beta$        | TD误差+状态值函数    |  
| **DPG**                 | 确定性       | 沿 $\nabla_a q$ 梯度更新策略       | Q-learning+函数近似 |  

> **关键演进脉络**：  
> - 方差减少（A2C引入优势函数）  
> - 样本效率提升（离策略机制）  
> - 连续控制扩展（确定性策略梯度）

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒