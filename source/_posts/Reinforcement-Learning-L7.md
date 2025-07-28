---
title: Reinforcement-Learning-L7
categories:
  - Learning
  - ReinForcement Learning
  - Mathmatical Foundation
date: 2025-07-28 11:55:04
tags:
---

# 强化学习的数学基础

~~这是之前的遗产捏，还需要整理~~

<!-- more -->

# 时序差分学习（TD Learning）读书笔记  

## 一、核心概念与学习目标  
### 1. TD学习的定位与意义  
- **模型无关性**：TD学习是继蒙特卡洛（MC）后的第二种**无模型强化学习方法**  
- **时序差分特性**：通过相邻状态间的值差异进行增量更新  
- **核心优势**：  
  - **在线学习**：即时更新值函数（无需等待幕结束）  
  - **适用性广**：支持连续任务（无终止状态）和幕任务  
  - **低方差**：比MC方法方差更低（因依赖更少随机变量）  

### 2. 理论基础：随机逼近（Stochastic Approximation）  
- **问题形式化**：求解方程根 $ g(w) = 0 $  
- **RM算法**：  
  $$
  w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)
  $$  
  其中 $\tilde{g}(w_k, \eta_k) = g(w_k) + \eta_k$ 为含噪声观测  
- **收敛条件**：  
  $$
  \sum_k a_k = \infty, \quad \sum_k a_k^2 < \infty
  $$

---

## 二、TD学习状态值函数  
### 1. 问题描述  
估计策略 $\pi$ 下的状态值函数 $ v_\pi(s) $：  
$$
v_\pi(s) = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s]
$$

### 2. TD(0) 算法  
- **更新规则**：  
  $$
  v_{t+1}(s_t) = v_t(s_t) - \alpha_t(s_t) \left[ v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1})) \right]
  $$  
- **关键概念**：  
  - **TD目标**：$\bar{v}\_t = r\_{t+1} + \gamma v\_t(s_{t+1})$  
  - **TD误差**：$\delta_t = v_t(s_t) - \bar{v}_t$  
- **数学本质**：求解贝尔曼期望方程的随机逼近算法  

### 3. 收敛性分析  
- **充分条件**：  
  $$
  \sum_t \alpha_t(s) = \infty, \quad \sum_t \alpha_t^2(s) < \infty \quad \forall s
  $$  
- **理论保证**：$ v_t(s) \to v_\pi(s) $（概率1收敛）  

### 4. 与MC学习的对比  
| **特性**       | TD学习                  | MC学习                  |  
|----------------|-------------------------|-------------------------|  
| **更新时机**   | 在线（即时更新）        | 离线（需幕结束）        |  
| **任务类型**   | 支持连续任务            | 仅幕任务                |  
| **方差**       | 低                      | 高                      |  
| **偏差**       | 初始值依赖（自举）      | 无偏                    |  

---

## 三、Sarsa算法：动作值函数估计  
### 1. 问题描述  
估计策略 $\pi$ 下的动作值函数 $ q_\pi(s,a) $：  
$$
q_\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma q_\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]
$$

### 2. Sarsa算法核心  
- **更新规则**：  
  $$
  q_{t+1}(s_t,a_t) = q_t(s_t,a_t) - \alpha_t(s_t,a_t) \left[ q_t(s_t,a_t) - (r_{t+1} + \gamma q_t(s_{t+1},a_{t+1})) \right]
  $$  
- **名称来源**：State-Action-Reward-State-Action (SARSA)  
- **策略改进**：结合 $\varepsilon$-贪心策略  
  $$
  \pi(a|s) = \begin{cases} 
  1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|} & a = \arg\max_a q(s,a) \\
  \frac{\varepsilon}{|\mathcal{A}(s)|} & \text{否则}
  \end{cases}
  $$

### 3. 收敛性分析  
同TD(0)：满足学习率条件下 $ q_t(s,a) \to q_\pi(s,a) $（概率1收敛）  

### 4. 示例分析  
- **环境**：网格世界（禁区/边界 $ r = -10 $，目标 $ r = 0 $)  
- **参数**：$\gamma = 0.9, \, \alpha = 0.1, \, \varepsilon = 0.1$  
- **结果**：Sarsa成功找到避开禁区的高效路径  

---

## 四、n步Sarsa算法  
### 1. 统一框架  
**动作值函数的n步回报**：  
$$
G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n q_t(s_{t+n}, a_{t+n})
$$  
- $ n=1 $：Sarsa  
- $ n \to \infty $：MC学习  

### 2. 算法更新  
$$
q_{t+n}(s_t, a_t) = q_{t+n-1}(s_t, a_t) - \alpha_{t+n-1} \left[ q_{t+n-1}(s_t, a_t) - G_t^{(n)} \right]
$$  

### 3. 偏差-方差权衡  
| **算法**     | **偏差** | **方差** |  
|--------------|----------|----------|  
| Sarsa ($n=1$) | 高       | 低       |  
| n步Sarsa     | 中       | 中       |  
| MC ($n=\infty$) | 低       | 高       |  

---

## 五、Q学习：离策略最优控制  
### 1. 核心思想  
直接估计**最优动作值函数** $ q^*(s,a) $：

$$
q^\*(s,a) = \mathbb{E} \left[ R\_{t+1} + \gamma \max\_{a'} q^*(S\_{t+1}, a') \mid S\_t = s, A\_t = a \right]
$$

### 2. Q学习算法  
- **更新规则**：  
  $$
  q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) \left[ q_t(s_t, a_t) - (r_{t+1} + \gamma \max_a q_t(s_{t+1}, a)) \right]
  $$  
- **离策略特性**：  
  - **行为策略** ($\pi_b$)：生成经验样本（如$\varepsilon$-贪心）  
  - **目标策略** ($\pi_T$)：贪心策略 $\pi_T(a|s) = \mathbf{1}_{a = \arg\max_a q(s,a)}$  

### 3. 探索的重要性  
- **示例对比**：  
  - $\varepsilon = 0.5$：充分探索 → 找到全局最优策略  
  - $\varepsilon = 0.1$：探索不足 → 陷入局部最优  
  [](@replace=1)  
  [](@replace=2)  

---

## 六、统一视角：TD算法框架  
### 1. 通用更新公式  
所有TD算法可统一表示为：  
$$
q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t(s_t, a_t) \left[ q_t(s_t, a_t) - \bar{q}_t \right]
$$  

### 2. 不同算法的TD目标  
| **算法**       | $\bar{q}_t$ 形式                     | **求解方程**         |  
|----------------|-----------------------------------------|----------------------|  
| TD(0)          | $ r_{t+1} + \gamma v_t(s_{t+1}) $     | 贝尔曼期望方程       |  
| Sarsa          | $ r_{t+1} + \gamma q_t(s_{t+1}, a_{t+1}) $ | 贝尔曼期望方程       |  
| n步Sarsa       | $ \sum_{i=1}^n \gamma^{i-1} r_{t+i} + \gamma^n q_t(s_{t+n}, a_{t+n}) $ | 贝尔曼期望方程       |  
| Q-learning     | $ r_{t+1} + \gamma \max_a q_t(s_{t+1}, a) $ | 贝尔曼最优方程       |  
| MC             | $ \sum_{i=1}^\infty \gamma^{i-1} r_{t+i} $ | 贝尔曼期望方程       |  

### 3. 数学本质  
所有算法均为**随机逼近方法**求解对应贝尔曼方程  

---

## 七、总结  
### 1. 关键算法对比  
| **特性**         | TD(0)       | Sarsa       | Q-learning  |  
|------------------|-------------|-------------|-------------|  
| **估计对象**     | 状态值 $v$ | 动作值 $q$ | 最优动作值 $q^*$ |  
| **策略依赖**     | 在策略       | 在策略       | 离策略       |  
| **更新复杂度**   | $O(1)$    | $O(1)$    | $O(1)$    |  
| **探索要求**     | 中           | 中           | 高           |  

### 2. 核心公式回顾  
- **TD(0)**：  
  $$
  v_{t+1}(s_t) = v_t(s_t) - \alpha_t \left( v_t(s_t) - (r_{t+1} + \gamma v_t(s_{t+1})) \right)
  $$  
- **Q-learning**：  
  $$
  q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - \alpha_t \left( q_t(s_t, a_t) - (r_{t+1} + \gamma \max_a q_t(s_{t+1}, a)) \right)
  $$  

### 3. 应用指导  
- **在线学习**：TD或Q-learning（数据即时可用）  
- **高方差环境**：优先TD而非MC  
- **探索关键场景**：Q-learning需配合高$\varepsilon$行为策略  
- **偏差-方差权衡**：n步Sarsa为折中方案  

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒