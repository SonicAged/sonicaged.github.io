---
title: Reinforcement-Learning-L2
categories:
  - Learning
  - ReinForcement Learning
  - Mathmatical Foundation
date: 2025-07-28 11:54:41
tags:
---

# 强化学习基础：状态值与贝尔曼方程读书笔记  

<div style="display: flex; align-items: center;">
  <img src="/img/reinforcement-learning/02.png" style="width: 175px; margin-right: 20px;">
  <p>状态值函数量化策略的长期收益，贝尔曼方程揭示其递归结构。本文通过网格世界示例展示状态值计算，推导贝尔曼方程的矩阵形式与求解方法，并分析状态值与动作值的关系，为策略评估奠定数学基础。</p>
</div>

<!-- more -->



## 一、核心概念与学习目标  
- **核心概念**：状态值（State Value）  
- **基本工具**：贝尔曼方程（Bellman Equation）  
- **学习内容**：  
  1. 状态值的定义与重要性  
  2. 贝尔曼方程的推导与求解  
  3. 状态值与动作值的关系  

---

## 二、动机示例：为什么回报（Return）重要？  
### 问题背景  
在网格世界中，从起点 $s_1$ 出发，评估三个策略的优劣：  
- **策略1**（左）：避开禁区直接走向目标  
- **策略2**（中）：经过禁区走向目标  
- **策略3**（右）：随机选择动作（50%概率进入禁区）  

### 回报计算（$\gamma = 0.9$）  
1. **策略1**（确定性）：  
   $$
   G_1 = 0 + \gamma \cdot 1 + \gamma^2 \cdot 1 + \cdots = \frac{\gamma}{1-\gamma} = 9
   $$
2. **策略2**（确定性）：  
   $$
   G_2 = -1 + \gamma \cdot 1 + \gamma^2 \cdot 1 + \cdots = -1 + \frac{\gamma}{1-\gamma} = 8
   $$
3. **策略3**（随机性）：  
   $$
   G_3 = 0.5 \times (-1 + 9) + 0.5 \times 9 = -0.5 + 9 = 8.5
   $$

### 关键结论  
- **策略优劣**：$G_1 > G_3 > G_2$（策略1最优，策略2最差）  
- **回报的作用**：  
  - 量化策略性能  
  - 反映长期收益（避免禁区惩罚）  
  - 验证直观决策的数学合理性  

---

## 三、状态值（State Value）的定义  
### 数学形式  
- **符号定义**：  
  - $S_t$：时刻 $t$ 的状态  
  - $A_t$：时刻 $t$ 的动作  
  - $R_{t+1}$：动作 $A_t$ 的即时奖励  
  - $G_t$：折扣回报 $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$  
- **状态值函数**：  
  $$
  v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]
  $$  
  > 表示从状态 $s$ 出发，遵循策略 $\pi$ 的期望累积回报  

### 核心性质  
1. **条件期望**：依赖初始状态 $s$ 和策略 $\pi$  
2. **模型依赖性**：由动态模型 $p(r \mid s,a)$ 和 $p(s' \mid s,a)$ 决定  
3. **与回报的关系**：  
   - 若策略、状态转移、奖励均为确定性 → $v_\pi(s) = G_t$  
   - 若存在随机性 → $v_\pi(s)$ 是所有可能轨迹回报的均值  

---

## 四、贝尔曼方程（Bellman Equation）的推导  
### 回报分解  
将 $G_t$ 拆分为即时奖励和未来折扣回报：  
$$
G_t = R_{t+1} + \gamma G_{t+1}
$$  

### 状态值分解  
代入状态值定义：  
$$
v_\pi(s) = \mathbb{E}[R_{t+1} \mid S_t = s] + \gamma \mathbb{E}[G_{t+1} \mid S_t = s]
$$  

#### 1. 即时奖励项（Immediate Reward）  
$$
\mathbb{E}[R_{t+1} \mid S_t = s] = \sum_a \pi(a \mid s) \sum_r p(r \mid s,a) r
$$  
> 策略 $\pi$ 下所有可能动作的奖励期望  

#### 2. 未来奖励项（Future Reward）  
$$
\mathbb{E}[G_{t+1} \mid S_t = s] = \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s,a) v_\pi(s')
$$  
> 马尔可夫性质保证：未来奖励仅依赖下一状态 $s'$  

#### 3. 最终形式  
$$
v_\pi(s) = \sum_a \pi(a \mid s) \left[ \sum_r p(r \mid s,a) r + \gamma \sum_{s'} p(s' \mid s,a) v_\pi(s') \right]
$$  

### 关键特性  
- **自举（Bootstrapping）**：状态值 $v_\pi(s)$ 依赖其他状态值 $v_\pi(s')$  
- **方程组结构**：每个状态对应一个方程（共 $|\mathcal{S}|$ 个方程）  
- **模型要求**：需已知动态模型 $p(r \mid s,a)$ 和 $p(s' \mid s,a)$  

---

## 五、贝尔曼方程的矩阵形式  
### 符号定义  
- **策略相关奖励**：  
  $$
  r_\pi(s) = \sum_a \pi(a \mid s) \sum_r p(r \mid s,a) r
  $$  
- **策略相关状态转移矩阵**：  
  $$
  p_\pi(s' \mid s) = \sum_a \pi(a \mid s) p(s' \mid s,a)
  $$

  $$
  P\_\pi \in \mathbb{R}^{n \times n}, \quad [P\_\pi]\_{ij} = p\_\pi(s\_j \mid s\_i)
  $$  

### 矩阵方程  
$$
v_\pi = r_\pi + \gamma P_\pi v_\pi
$$  
其中：  
- $v_\pi = [v_\pi(s_1), \dots, v_\pi(s_n)]^T$  
- $r_\pi = [r_\pi(s_1), \dots, r_\pi(s_n)]^T$  

#### 示例（四状态循环图） 

$$
\begin{bmatrix}
v_\pi(s_1) \\\\
v_\pi(s_2) \\\\
v_\pi(s_3) \\\\
v_\pi(s_4)
\end{bmatrix}=
\begin{bmatrix}
0 \\\\
1 \\\\
1 \\\\
1
\end{bmatrix}+
\gamma
\begin{bmatrix}
0 & 0 & 1 & 0 \\\\
0 & 0 & 0 & 1 \\\\
0 & 0 & 0 & 1 \\\\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
v_\pi(s_1) \\\\
v_\pi(s_2) \\\\
v_\pi(s_3) \\\\
v_\pi(s_4)
\end{bmatrix}
$$  

---

## 六、贝尔曼方程的求解方法  
### 1. 闭式解（Closed-Form Solution）  
$$
v_\pi = (I - \gamma P_\pi)^{-1} r_\pi
$$  
> 要求矩阵 $I - \gamma P_\pi$ 可逆（当 $\gamma < 1$ 时成立）  

### 2. 迭代解（Iterative Solution）  
迭代更新直至收敛：  
$$
v_{k+1} = r_\pi + \gamma P_\pi v_k
$$  
> **收敛性证明**：定义误差 $\delta_k = v_k - v_\pi$，则  
  $$
  \delta_{k+1} = \gamma P_\pi \delta_k \to 0 \quad (k \to \infty)
  $$  
  因 $\gamma < 1$ 且 $P_\pi$ 为概率矩阵（谱半径 $\leq 1$)  

#### 网格世界求解示例  
- **奖励设置**：禁区/边界 $r=-1$，目标 $r=+1$，$\gamma=0.9$  
- **策略评估**：计算各状态值（如 $s_1$ 在最优策略下 $v_{\pi}(s_1)=9$)  

---

## 七、动作值（Action Value）  
### 定义  
$$
q_\pi(s,a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]
$$  
> 表示在状态 $s$ 执行动作 $a$ 后，遵循策略 $\pi$ 的期望回报  

### 与状态值的关系  
1. **状态值分解为动作值**：  
   $$
   v_\pi(s) = \sum_a \pi(a \mid s) q_\pi(s,a)
   $$  
2. **动作值的贝尔曼方程**：  
   $$
   q_\pi(s,a) = \sum_r p(r \mid s,a) r + \gamma \sum_{s'} p(s' \mid s,a) v_\pi(s')
   $$  

### 重要性  
- **动作评估**：直接量化动作的长期收益（如 $q_\pi(s_1, a_3) = \gamma v_\pi(s_3)$）  
- **策略优化基础**：通过比较 $q_\pi(s,a)$ 改进策略（后续课程）  

---

## 八、总结  
### 核心概念  
| **概念**       | **定义**                                | **数学表示**                                                                 |
|----------------|----------------------------------------|-----------------------------------------------------------------------------|
| **状态值**     | 从状态 $s$ 出发的期望累积回报          | $v_\pi(s) = \mathbb{E}[G_t \mid S_t = s]$                               |
| **动作值**     | 执行动作 $a$ 后的期望累积回报          | $q_\pi(s,a) = \mathbb{E}[G_t \mid S_t = s, A_t = a]$                    |
| **贝尔曼方程** | 描述状态值递归关系的方程组               | $v_\pi(s) = \sum_a \pi(a \mid s) \left[ \sum_r p(r \mid s,a) r + \gamma \sum_{s'} p(s' \mid s,a) v_\pi(s') \right]$ |

### 关键结论  
1. **回报驱动策略评估**：回报 $G_t$ 是量化策略优劣的核心指标  
2. **状态值求解**：通过贝尔曼方程（闭式或迭代法）计算 $v_\pi$  
3. **动作值桥梁作用**：连接即时奖励与长期状态值（$q_\pi \leftrightarrow v_\pi$）  

### 学习意义  
- **策略评估（Policy Evaluation）**：计算给定策略 $\pi$ 的状态值 $v_\pi$  
- **后续基础**：为策略改进（Policy Improvement）和值迭代（Value Iteration）奠定基础  

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning" target="_blank">
  <span style="display: inline-block; vertical-align: middle;">
    <img src="/icon/github.svg" alt="github" style="height: 1.1em; vertical-align: text-bottom; margin-top: 16px;">
  </span>
  Book-Mathematical-Foundation-of-Reinforcement-Learning
</a>