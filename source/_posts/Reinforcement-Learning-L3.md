---
title: Bellman Optimality Equation
categories:
  - Learning
  - ReinForcement Learning
  - Mathmatical Foundation
date: 2025-07-28 11:54:49
tags:
---

# 最优策略与贝尔曼最优方程读书笔记  

<div style="display: flex; align-items: center;">
  <img src="/img/reinforcement-learning/03.png" style="width: 175px; margin-right: 20px;">
  <p>贝尔曼最优方程（BOE）是求解最优策略的核心工具。本文证明BOE解的存在唯一性，分析值迭代的收敛性，探讨奖励设计、折扣率对策略的影响，并给出确定性策略充分性的理论保证。</p>
</div>

<!-- more -->



## 动机示例：动作值驱动的策略改进  
### 场景描述  
在网格世界中，给定策略 $\pi$ 的状态值和动作值计算：  
- **状态值**（$\gamma=0.9$）：  
  $$
  v_\pi(s_1)=8,\quad v_\pi(s_2)=10,\quad v_\pi(s_3)=10,\quad v_\pi(s_4)=10
  $$
- **动作值**（状态 $s_1$）：  
  $$
  \begin{align*}
  q_\pi(s_1, a_1) &= -1 + \gamma v_\pi(s_1) = 6.2, \\\\
  q_\pi(s_1, a_2) &= -1 + \gamma v_\pi(s_2) = 8, \\\\
  q_\pi(s_1, a_3) &= 0 + \gamma v_\pi(s_3) = 9, \\\\
  q_\pi(s_1, a_4) &= -1 + \gamma v_\pi(s_1) = 6.2, \\\\
  q_\pi(s_1, a_5) &= 0 + \gamma v_\pi(s_1) = 7.2
  \end{align*}
  $$

<img src="/img/reinforcement-learning/Reinforcement-Learning-L3/grid.png" alt="grid" style="zoom:33%;" />

### 策略改进原理  
1. **当前策略**：在 $s_1$ 固定选择 $a_2$（向右移动）  
2. **改进依据**：选择最大化动作值的动作 $a_3$（向下移动）  
3. **新策略**：  
   $$
   \pi_{\text{new}}(a|s_1) = 
   \begin{cases} 
   1 & a = a_3 \\\\
   0 & a \neq a_3
   \end{cases}
   $$
> **关键思想**：动作值 $q_\pi(s,a)$ 量化动作的长期收益，最大化它可提升策略性能  

---

## 最优策略的数学定义  
### 核心概念  
- **状态值偏序关系**：若对所有状态 $s \in \mathcal{S}$ 满足  
  $$
  v_{\pi_1}(s) \geq v_{\pi_2}(s)
  $$
  则称策略 $\pi_1$ **优于** $\pi_2$  
- **最优策略** $\pi^*$ ：对所有状态 $s$ 和任意策略 $\pi$ 满足 

  $$
  v\_{\pi^*}(s) \geq v\_{\pi}(s)
  $$

### 理论问题  
1. **存在性**：最优策略是否存在？  
2. **唯一性**：最优策略是否唯一？  
3. **确定性**：最优策略是确定性还是随机性？  
4. **求解方法**：如何计算最优策略？  

> **贝尔曼最优方程（BOE）** 是回答这些问题的核心工具  

---

## 贝尔曼最优方程（BOE）的形式  
### 1. 元素形式（Elementwise Form）  
$$
v(s) = {\color{blue}{\max_{\pi}}} \sum_{a} \pi(a|s)  \underbrace{\left[\sum_{r} p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v(s')\right]}_{q(s,a)} 
$$
其中 $q(s,a)$ 是状态-动作值函数  

### 2. 矩阵形式（Matrix Form）  
$$
v = \max_{\pi} (r_{\pi} + \gamma P_{\pi} v)
$$
- $v = [v(s_1), \dots, v(s_n)]^T$：状态值向量  
- $r_\pi = [\sum_a \pi(a|s_i) \sum_r p(r|s_i,a) r]^T$：期望即时奖励向量  
- $P_\pi$：状态转移矩阵，$[P_\pi]_{ij} = \sum_a \pi(a|s_i) p(s_j|s_i, a)$  

### 3. 最大化简化形式  
**定理**：最优状态值等于最大动作值  
$$
v(s) = \max_{a \in \mathcal{A}(s)} q(s,a)
$$
- **最优策略**：确定性策略，选择使 $q(s,a)$ 最大的动作：  
  $$
  \pi^*(a|s) = 
  \begin{cases} 
  1 & a = \arg\max_a q(s,a) \\\\
  0 & \text{否则}
  \end{cases}
  $$

---

## BOE的求解：压缩映射理论  
### 关键步骤  
1. **定义映射函数**：  
   $$
   f(v) \triangleq \max_{\pi} (r_{\pi} + \gamma P_{\pi} v)
   $$
   BOE 转化为不动点方程 $v = f(v)$  
   
2. **压缩映射性质**：  
   $$
   \|f(v_1) - f(v_2)\| \leq \gamma \|v_1 - v_2\|
   $$
   > 证明依赖折扣率 $\gamma \in (0,1)$ 和概率矩阵性质  

### 解的存在性与唯一性  
**压缩映射定理**：  
1. **存在性**：BOE 存在解 $v^*$  
2. **唯一性**：解 $v^*$ 唯一  
3. **迭代求解**：对任意初始 $v_0$，序列  
   $$
   v_{k+1} = f(v_k) = \max_{\pi} (r_{\pi} + \gamma P_{\pi} v_k)
   $$
   指数收敛到 $v^*$（值迭代算法）  

---

## BOE与最优策略的关系  
### 最优性定理  
若 $v^*$ 是 BOE 的解，则：  
1. **对应策略**：贪心策略 $\pi^* = \arg\max_{\pi} (r_{\pi} + \gamma P_{\pi} v^*)$  
2. **策略最优性**：对所有策略 $\pi$ 和状态 $s$ 满足  
   $$
   v^*(s) \geq v_{\pi}(s)
   $$

### 确定性策略充分性  
**定理**：确定性策略足以达到最优性能  
- 随机策略不会产生比最优确定性策略更高的状态值  

---

## 最优策略的影响因素分析  
### 1. 奖励设计（Reward Design）  
- **绝对数值不影响策略**：奖励仿射变换 $r \to ar + b$ ($a > 0$) 不改变最优策略  
  $$
  v' = a v^* + \frac{b}{1-\gamma} \mathbf{1}
  $$
- **相对值决定策略**：禁区惩罚 $r_{\text{forbidden}}$ 与目标奖励 $r_{\text{target}}$ 的相对大小影响路径选择  
  - 示例：当 $r_{\text{forbidden}} = -10$（原为 $-1$），最优策略避开禁区  

### 2. 折扣率 $\gamma$  
- $\gamma \approx 1$：**长远规划**，愿承担风险穿越禁区  
- $\gamma \approx 0$：**短视行为**，回避所有风险（可能无法到达目标）  
  

### 3. 绕路惩罚机制  
- **隐含惩罚**：折扣率 $\gamma$ 天然惩罚长路径  
  - 最优路径回报：$\frac{1}{1-\gamma}$  
  - 绕路路径回报：$\frac{\gamma^k}{1-\gamma} < \frac{1}{1-\gamma}$（$k \geq 2$)  

---

## 总结：贝尔曼最优方程的意义  
| **问题**         | **BOE的解答**                                                                 |
|------------------|-----------------------------------------------------------------------------|
| **最优策略存在性** | BOE解存在 → 最优策略 $\pi^*$ 存在                                          |
| **策略形式**     | 确定性策略足够最优：$\pi^* = \arg\max_a q(s,a)$                            |
| **求解算法**     | 值迭代：$v_{k+1} = \max_{\pi} (r_{\pi} + \gamma P_{\pi} v_k)$             |
| **解的唯一性**   | 压缩映射保证唯一解 $v^*$                                                  |
| **最优性验证**   | $v^* \geq v_{\pi}$ 对所有策略 $\pi$ 成立                                |

### 关键公式回顾  
1. **BOE元素形式**：  
   $$
   v(s) = \max_{a} \left[ \sum_{r} p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v(s') \right]
   $$
2. **值迭代更新**：  
   $$
   v_{k+1}(s) = \max_{a} \left[ r(s,a) + \gamma \sum_{s'} p(s'|s,a) v_k(s') \right]
   $$
3. **最优策略**：  
   $$
   \pi^*(s) = \arg\max_{a} q(s,a)
   $$

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning" target="_blank">
  <span style="display: inline-block; vertical-align: middle;">
    <img src="/icon/github.svg" alt="github" style="height: 1.1em; vertical-align: text-bottom; margin-top: 16px;">
  </span>
  Book-Mathematical-Foundation-of-Reinforcement-Learning
</a>