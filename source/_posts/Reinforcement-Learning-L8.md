---
title: Reinforcement-Learning-L8
categories:
  - Learning
  - ReinForcement Learning
  - Mathmatical Foundation
date: 2025-07-28 11:55:07
tags:
---

# 强化学习的数学基础

~~这是之前的遗产捏，还需要整理~~

<!-- more -->

# 值函数逼近方法读书笔记  

## 一、动机：从表格表示到函数逼近  
### 传统表格表示的局限性  
1. **存储问题**：  
   - 状态值表：需存储 $|\mathcal{S}|$ 个值  
     $$
     \begin{array}{c|c|c|c}
     \text{State} & s_1 & \cdots & s_n \\\\ \hline
     \text{Value} & v_\pi(s_1) & \cdots & v_\pi(s_n)
     \end{array}
     $$  
   - 动作值表：需存储 $|\mathcal{S}| \times |\mathcal{A}|$ 个值  
     $$
     \begin{array}{c|c|c|c}
     s \backslash a & a_1 & \cdots & a_m \\\\ \hline
     s_1 & q_\pi(s_1,a_1) & \cdots & q_\pi(s_1,a_m) \\\\
     \vdots & \vdots & \ddots & \vdots \\\\
     s_n & q_\pi(s_n,a_1) & \cdots & q_\pi(s_n,a_m)
     \end{array}
     $$  
2. **泛化能力缺失**：  
   - 无法处理未见过状态  
   - 相邻状态的值更新相互独立  

### 函数逼近的核心思想  
- **参数化函数**：$\hat{v}(s, w) \approx v_\pi(s)$ 或 $\hat{q}(s,a,w) \approx q_\pi(s,a)$  
- **参数向量**：$w \in \mathbb{R}^m$（$m \ll |\mathcal{S}|$)  
- **核心优势**：  
  1. **存储高效**：只需存储 $m$ 维参数  
  2. **泛化能力**：更新 $w$ 影响所有相关状态  

### 线性函数逼近示例  
- **特征向量**：$\phi(s) = [\phi_1(s), \dots, \phi_k(s)]^T$  
- **线性逼近**：  
  $$
  \hat{v}(s,w) = \phi^T(s)w = \sum_{i=1}^k \phi_i(s) w_i
  $$  
- **非线性扩展**：神经网络（如 Deep Q-Networks）  

---

## 二、状态值函数估计  
### 1. 目标函数设计  
基于**平稳分布** $d_\pi(s)$ 的加权平方误差：  
$$
J(w) = \mathbb{E}\_{S \sim d\_\pi} \left[ (v\_\pi(S) - \hat{v}(S,w))^2 \right] = \sum\_{s \in \mathcal{S}} d\_\pi(s) \left( v\_\pi(s) - \hat{v}(s,w) \right)^2
$$  

> **平稳分布 $d_\pi(s)$**：策略 $\pi$ 下状态的长期访问概率  
> - 满足 $d_\pi^T = d_\pi^T P_\pi$  
> - 高频访问状态具有更高权重  

### 2. 优化算法：随机梯度下降（SGD）  
#### 梯度计算  
$$
\nabla\_w J(w) = -2 \mathbb{E}\_{S \sim d_\pi} \left[ (v\_\pi(S) - \hat{v}(S,w)) \nabla\_w \hat{v}(S,w) \right]
$$  
#### SGD更新规则  
$$
w_{t+1} = w_t + \alpha_t (v_\pi(s_t) - \hat{v}(s_t,w_t)) \nabla_w \hat{v}(s_t,w_t)
$$  

### 3. 实现方法  
#### (1) 蒙特卡洛（MC）逼近  
用经验回报 $g_t$ 估计 $v_\pi(s_t)$：  
$$
w_{t+1} = w_t + \alpha_t (g_t - \hat{v}(s_t,w_t)) \nabla_w \hat{v}(s_t,w_t)
$$  
#### (2) 时序差分（TD）逼近  
用引导目标 $r_{t+1} + \gamma \hat{v}(s_{t+1},w_t)$ 估计 $v_\pi(s_t)$：  
$$
w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \hat{v}(s_{t+1},w_t) - \hat{v}(s_t,w_t) \right] \nabla_w \hat{v}(s_t,w_t)
$$  

#### 线性函数逼近特例  
若 $\hat{v}(s,w) = \phi^T(s)w$，则 $\nabla_w \hat{v} = \phi(s)$，更新简化为：  
$$
w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \phi^T(s_{t+1})w_t - \phi^T(s_t)w_t \right] \phi(s_t)
$$  

### 4. 算法流程  
```  
初始化：可微函数 $\hat{v}(s,w)$，参数 $w_0$  
for 每个由 $\pi$ 生成的轨迹 $(s_t, r_{t+1}, s_{t+1})$:  
    TD目标：$\bar{v}_t = r_{t+1} + \gamma \hat{v}(s_{t+1}, w_t)$  
    计算梯度：$\nabla_w \hat{v}(s_t, w_t)$  
    更新参数：$w_{t+1} = w_t + \alpha_t (\bar{v}_t - \hat{v}(s_t, w_t)) \nabla_w \hat{v}(s_t, w_t)$  
```  

---

## 三、Sarsa算法与函数逼近  
### 1. 动作值函数逼近  
- **逼近函数**：$\hat{q}(s,a,w) \approx q_\pi(s,a)$  
- **TD目标**：$r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w_t)$  
- **SGD更新**：  
  $$
  w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w_t) - \hat{q}(s_t, a_t, w_t) \right] \nabla_w \hat{q}(s_t, a_t, w_t)
  $$  

### 2. 策略搜索算法  
```  
初始化：$\hat{q}(s,a,w)$, $w$, $\epsilon$, $\alpha$  
for 每幕：  
    初始化状态 $s_0$，选择动作 $a_0 \sim \pi(\cdot|s_0)$  
    for 每步 $t = 0,1,\dots$:  
        执行 $a_t$，观测 $(r_{t+1}, s_{t+1})$  
        选择 $a_{t+1} \sim \pi(\cdot|s_{t+1})$  
        更新参数：  
            $w \leftarrow w + \alpha \left[ r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w) - \hat{q}(s_t, a_t, w) \right] \nabla_w \hat{q}(s_t, a_t, w)$  
        $\varepsilon$-贪心策略改进：  
            $\pi(a|s_t) = \begin{cases} 
            1-\varepsilon + \frac{\varepsilon}{|\mathcal{A}(s_t)|} & a = \arg\max_a \hat{q}(s_t,a,w) \\\\
            \frac{\varepsilon}{|\mathcal{A}(s_t)|} & \text{否则}
            \end{cases}$  
```  

---

## 四、Q-learning与函数逼近  
### 1. 最优动作值逼近  
- **TD目标**：$r_{t+1} + \gamma \max_{a'} \hat{q}(s_{t+1}, a', w_t)$  
- **SGD更新**：  
  $$
  w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \max_{a'} \hat{q}(s_{t+1}, a', w_t) - \hat{q}(s_t, a_t, w_t) \right] \nabla_w \hat{q}(s_t, a_t, w_t)
  $$  

### 2. 深度Q学习（DQN）  
- **关键技术**：  
  - **目标网络**：使用独立参数 $w^-$ 计算目标值  
    $$
    \text{目标} = r_{t+1} + \gamma \max_{a'} \hat{q}(s_{t+1}, a', w^-)
    $$  
  - **经验回放**：存储转移 $(s_t, a_t, r_{t+1}, s_{t+1})$ 到缓冲池，随机采样更新  
- **更新规则**：  
  $$
  w \leftarrow w + \alpha \left[ r_{t+1} + \gamma \max_{a'} \hat{q}(s_{t+1}, a', w^-) - \hat{q}(s_t, a_t, w) \right] \nabla_w \hat{q}(s_t, a_t, w)
  $$  

---

## 五、总结  
### 1. 关键公式对比  
| **算法**          | **更新规则**                                                                 |  
|-------------------|-----------------------------------------------------------------------------|  
| **TD-Linear**     | $ w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \phi^T(s_{t+1})w_t - \phi^T(s_t)w_t \right] \phi(s_t) $ |  
| **Sarsa 逼近**    | $ w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \hat{q}(s_{t+1}, a_{t+1}, w_t) - \hat{q}(s_t, a_t, w_t) \right] \nabla_w \hat{q}(s_t, a_t, w_t) $ |  
| **Q-learning 逼近** | $ w_{t+1} = w_t + \alpha_t \left[ r_{t+1} + \gamma \max_a \hat{q}(s_{t+1}, a, w_t) - \hat{q}(s_t, a_t, w_t) \right] \nabla_w \hat{q}(s_t, a_t, w_t) $ |  

### 2. 函数逼近器选择  
| **类型**         | **形式**                  | **特点**                     |  
|------------------|--------------------------|-----------------------------|  
| **线性逼近**     | $\hat{v}(s,w) = \phi^T(s)w$ | 理论收敛性好，特征工程关键     |  
| **神经网络**     | $\hat{q}(s,a,w) = \text{DNN}(s,a;w)$ | 表征能力强，需结合目标网络/经验回放 |  

### 3. 核心优势与挑战  
- **优势**：  
  - 处理连续/大规模状态空间  
  - 相似状态自动泛化  
- **挑战**：  
  - 收敛性理论保证减弱（尤其离策略和非线性逼近）  
  - 逼近误差可能导致策略退化  

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒