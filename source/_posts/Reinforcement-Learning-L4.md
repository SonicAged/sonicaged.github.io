---
title: Value Iteration and Policy Iteration
categories:
  - Learning
  - ReinForcement Learning
  - Mathmatical Foundation
date: 2025-07-28 11:54:52
tags:
---

# 值迭代与策略迭代算法读书笔记  

<div style="display: flex; align-items: center;">
  <img src="/img/reinforcement-learning/04.png" style="width: 175px; margin-right: 20px;">
  <p>值迭代和策略迭代是求解MDP的经典动态规划方法。本文对比两种算法的流程、收敛性和计算效率，引入截断策略迭代作为平衡方案，并通过网格世界示例演示其应用，揭示广义策略迭代（GPI）的统一框架。</p>
</div>

<!-- more -->



## 值迭代算法（Value Iteration）  
### 核心思想  
通过迭代求解贝尔曼最优方程（BOE）:  
$$
v_{k+1} = \max_{\pi} (r_{\pi} + \gamma P_{\pi} v_k)
$$
逐步逼近最优状态值 $v^\*$，并提取最优策略 $\pi^\*$  

### 算法步骤（元素形式）  
1. **初始化**：  
   - 已知环境模型 $p(r \mid s,a)$ 和 $p(s' \mid s,a)$  
   - 任意初始化状态值函数 $v_0(s)$  

2. **迭代更新**：  
   **重复直到收敛**（$\|v_k - v_{k-1}\| < \varepsilon$）:  
   - **策略更新**（Policy Update）：  
     $$
     \pi_{k+1}(a|s) = 
     \begin{cases} 
     1 & a = \arg\max_a q_k(s,a) \\\\
     0 & \text{否则}
     \end{cases}
     $$
     其中动作值函数：  
     $$
     q_k(s,a) = \sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_k(s')
     $$
   - **值更新**（Value Update）：  
     $$
     v_{k+1}(s) = \max_a q_k(s,a)
     $$

### 伪代码流程  
<img src="/img/reinforcement-learning/Reinforcement-Learning-L4/image-20250804183547455.png" alt="Pseudocode: Value iteration algorithm" style="zoom:50%;" />

### 网格世界示例  
- **奖励设置**：边界/禁区 $r=-1$，目标 $r=+1$，$\gamma=0.9$  
- **初始值**：$v_0(s_1)=v_0(s_2)=v_0(s_3)=v_0(s_4)=0$  
- **迭代过程**：  
  - **k=0**：计算 $q_0$ → 策略 $\pi_1$ 选择 $a_5$（静止）→ $v_1=[0,1,1,1]$  
  - **k=1**：计算 $q_1$ → 策略 $\pi_2$ 选择 $a_3$（向下）→ $v_2=[\gamma,1+\gamma,1+\gamma,1+\gamma]$  
  - **收敛**：2次迭代即得最优策略（避开禁区直达目标）  

---

## 策略迭代算法（Policy Iteration）  
### 核心思想  
交替执行：  
1. **策略评估**（Policy Evaluation）：精确计算当前策略 $\pi_k$ 的状态值 $v_{\pi_k}$  
2. **策略改进**（Policy Improvement）：基于 $v_{\pi_k}$ 生成更优策略 $\pi_{k+1}$  

### 算法步骤  
1. **初始化**：任意初始策略 $\pi_0$  
2. **迭代直到收敛**：  
   - **策略评估**（求解贝尔曼方程）：  
     $$
     v_{\pi_k}^{(j+1)} = r_{\pi_k} + \gamma P_{\pi_k} v_{\pi_k}^{(j)}
     $$
     闭式解：$v_{\pi_k} = (I - \gamma P_{\pi_k})^{-1} r_{\pi_k}$  
   - **策略改进**（贪心策略）：  
     $$
     \pi_{k+1} = \arg\max_{\pi} (r_{\pi} + \gamma P_{\pi} v_{\pi_k})
     $$
     元素形式：  
     $$
     \pi_{k+1}(a|s) = 
     \begin{cases} 
     1 & a = \arg\max_a q_{\pi_k}(s,a) \\\\
     0 & \text{否则}
     \end{cases}
     $$

### 伪代码流程

<img src="/img/reinforcement-learning/Reinforcement-Learning-L4/image-20250804183711027.png" alt="Pseudocode: Policy iteration algorithm" style="zoom:50%;" />

### 关键性质

- **策略改进定理**：$v_{\pi_{k+1}} \geq v_{\pi_k}$（策略单调改进）  
- **收敛性**：序列 $\{v_{\pi_k}\}$ 收敛到最优值 $v^*$  

### 网格世界示例  
- **初始策略** $\pi_0$：在 $s_1$ 选择向右（$a_2$）  
- **策略评估**：  
  $$
  \begin{cases}
  v_{\pi_0}(s_1) = -1 + \gamma v_{\pi_0}(s_1) \\\\
  v_{\pi_0}(s_2) = \gamma v_{\pi_0}(s_1)
  \end{cases}
  $$
  解得 $v_{\pi_0}(s_1)=-10, v_{\pi_0}(s_2)=-9$  
- **策略改进**：计算 $q_{\pi_0}$ → 新策略 $\pi_1$ 在 $s_1$ 选择向右（$a_r$)  

---

## 三、截断策略迭代（Truncated Policy Iteration）  
### 核心思想  
在策略评估阶段**有限次迭代**（$j_{\text{truncate}}$ 次）近似计算 $v_{\pi_k}$，平衡计算效率与精度  

### 算法流程  
<img src="/img/reinforcement-learning/Reinforcement-Learning-L4/image-20250804183826039.png" alt="Pseudocode: Truncated policy iteration algorithm" style="zoom:50%;" />

### 理论保证  
- **值单调性**：若初始值 $v_k^{(0)} = v_{k-1}$，则 $v_k^{(j+1)} \geq v_k^{(j)}$  
- **收敛性**：仍收敛到最优策略（因策略改进步保证方向正确）  

---

## 四、算法对比与关系  
### 统一框架分析  
| **算法**          | **策略评估精度**       | **策略改进方式**       | **计算效率** |  
|-------------------|----------------------|----------------------|------------|  
| **值迭代**        | 单次迭代（近似）       | 最大化动作值           | ⭐⭐高       |  
| **策略迭代**      | 精确求解（无限次迭代） | 贪心策略               | ⭐低        |  
| **截断策略迭代**  | 有限次迭代（可调精度） | 贪心策略               | ⭐⭐中等     |  

### 数学等价性  
- **值迭代** = 策略迭代中策略评估仅进行 **1 次迭代** 的特例  
- **策略迭代** = 截断策略迭代中 $j_{\text{truncate}} \to \infty$ 的特例  

### 迭代过程对比  
- **策略迭代**：$\pi_0 \xrightarrow{\text{PE}} v_{\pi_0} \xrightarrow{\text{PI}} \pi_1 \xrightarrow{\text{PE}} v_{\pi_1} \cdots$  
- **值迭代**：$v_0 \xrightarrow{\text{PU}} \pi_1 \xrightarrow{\text{VU}} v_1 \xrightarrow{\text{PU}} \pi_2 \cdots$  
- **截断策略迭代**：$\pi\_k \xrightarrow{\text{PE (j次)}} \tilde{v}\_{\pi\_k} \xrightarrow{\text{PI}} \pi\_{k+1}$  

> **注**：PE=策略评估, PI=策略改进, PU=策略更新, VU=值更新  

具体来说就是：

$$
\begin{aligned}
\color{red}{v\_{\pi\_1}^{(0)}} &\color{red}{= v\_0  }\\\\ 
\text{值迭代}\leftarrow \color{red}{v\_1\leftarrow} v\_{\pi\_1}^{(1)}&=r\_{\pi\_1}+\gamma P\_{\pi\_1}v_{\pi\_1}^{(0)}\\\\
\vdots\\\\
v\_{\pi1}^{(2)}&=r\_{\pi\_1}+\gamma P_{\pi\_1}v\_{\pi\_1}^{(1)}\\\\
\text{截断策略迭代}\leftarrow\color{red}{\tilde{v}\_1\leftarrow} v\_{\pi\_1}^{(j)}&=r\_{\pi\_1}+\gamma P\_{\pi\_1}v_{\pi\_1}^{(j-1)}\\\\
\vdots\\\\
\text{策略迭代}\leftarrow \color{red}{v\_{\pi\_1}\leftarrow} v\_{\pi_1}^{(\infty)}&=r\_{\pi_1}+\gamma P\_{\pi\_1}v\_{\pi\_1}^{(\infty)}
\end{aligned}
$$

---

## 五、总结  
### 核心公式回顾  
| **算法**         | **关键更新方程**                                                                 |  
|------------------|---------------------------------------------------------------------------------|  
| 值迭代           | $ v_{k+1}(s) = \max_a \left[ \sum_r p(r\|s,a)r + \gamma \sum_{s'} p(s'\|s,a)v_k(s') \right] $ |  
| 策略迭代         | $ v_{\pi_k} = (I - \gamma P_{\pi_k})^{-1} r_{\pi_k} $                          |  
| 截断策略迭代     | $ v_k^{(j+1)}(s) = \sum_a \pi_k(a\|s) \left[ \sum_r p(r\|s,a)r + \gamma \sum_{s'} p(s'\|s,a)v_k^{(j)}(s') \right] $ |  

### 应用场景建议  
- **小规模问题**：策略迭代（精确解）  
- **大规模问题**：值迭代或截断策略迭代（计算高效）  
- **在线学习**：结合值迭代与实时更新（如Q-learning）  

### 理论意义  
三大算法构成了**广义策略迭代（GPI）框架**，统一了动态规划求解MDP的核心思想：**策略评估与改进的交替循环 → 收敛至最优策略**  

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning" target="_blank">
  <span style="display: inline-block; vertical-align: middle;">
    <img src="/icon/github.svg" alt="github" style="height: 1.1em; vertical-align: text-bottom; margin-top: 16px;">
  </span>
  Book-Mathematical-Foundation-of-Reinforcement-Learning
</a>