---
title: Reinforcement-Learning-L5
categories:
  - Learning
  - ReinForcement Learning
  - Mathmatical Foundation
date: 2025-07-28 11:54:56
tags:
---

# 强化学习的数学基础

~~这是之前的遗产捏，还需要整理~~

<!-- more -->

# 蒙特卡洛强化学习读书笔记   

## 一、蒙特卡洛估计基础  
### 核心思想  
蒙特卡洛（MC）方法通过**重复随机采样**估计期望值，无需已知概率模型：  
- **问题**：估计随机变量 $X$ 的期望 $E[X]$（概率分布 $p(x)$ 未知）  
- **解决方案**：  
  $$
  E[X] \approx \bar{x} = \frac{1}{N} \sum_{j=1}^{N} x_j
  $$  
  其中 $\{x_j\}_{j=1}^N$ 是独立同分布（i.i.d.）样本  

### 大数定律保证  
- **无偏性**：$E[\bar{x}] = E[X]$  
- **收敛性**：$\text{Var}(\bar{x}) = \frac{1}{N} \text{Var}(X) \to 0 \quad (N \to \infty)$  

### 在强化学习中的应用  
- **值函数本质**：状态值 $v_\pi(s)$ 和动作值 $q_\pi(s,a)$ 均为期望形式：  
  $$
  v_\pi(s) = E[G_t | S_t = s], \quad q_\pi(s,a) = E[G_t | S_t = s, A_t = a]
  $$  
- **无模型估计**：当环境动态 $p(r|s,a)$ 和 $p(s'|s,a)$ 未知时，MC方法可直接从经验样本估计值函数  

---

## 二、蒙特卡洛强化学习算法  
### 1. MC Basic算法  
#### 核心思想  
将**策略迭代算法**转化为无模型版本：  
1. **策略评估**：用MC方法估计 $q_{\pi_k}(s,a)$  
2. **策略改进**：生成贪心策略 $\pi_{k+1}(a|s) = \mathbf{1}_{a = \arg\max q_k(s,a)}$  

#### 算法流程  
```  
初始化：策略 $\pi_0$  
for k = 0, 1, 2, ... until convergence:  
    # 策略评估  
    for 每个状态-动作对 (s,a):  
        生成多个幕(episode)，起点为 (s,a)，遵循 $\pi_k$  
        q_k(s,a) = 所有回报 g(s,a) 的平均值  
    
    # 策略改进  
    for 每个状态 s:  
        a*_k(s) = argmax_a q_k(s,a)  
        π_{k+1}(a|s) = 1 if a=a*_k else 0  
```  

#### 优缺点  
- **优点**：理论直观，直接展示MC在RL中的核心思想  
- **缺点**：  
  - 需从**每个 (s,a)** 生成足够多幕 → 计算效率低  
  - 需满足**探索性出发(Exploring Starts)** 条件  

### 2. MC Exploring Starts算法  
#### 核心改进  
- **数据高效利用**：单幕更新所有访问过的 (s,a) 对  
- **在线更新**：逐幕更新策略，无需等待所有幕生成  

#### 算法流程  
```  
初始化：π_0, q(s,a), Returns(s,a)=0, Num(s,a)=0  
for 每个幕:  
    探索性出发：选择起始 (s_0, a_0)（覆盖所有(s,a)）  
    生成幕：s_0,a_0,r_1,s_1,a_1,r_2,...,s_{T-1},a_{T-1},r_T  
    g = 0  
    for t = T-1 to 0 (反向遍历):  
        g = γg + r_{t+1}  
        Returns(s_t,a_t) += g  
        Num(s_t,a_t) += 1  
        q(s_t,a_t) = Returns(s_t,a_t) / Num(s_t,a_t)  
        a* = argmax_a q(s_t,a)  
        π(a|s_t) = 1 if a=a* else 0  
```  

#### 关键特性  
- **高效数据利用**：一个幕更新所有访问的 (s,a)  
- **仍需探索性出发**：需保证每个 (s,a) 作为起点被无限次访问  

### 3. MC ε-Greedy算法  
#### 核心创新  
- **移除探索性出发**：通过 **ε-贪心策略** 保证探索  
- **软策略(Soft Policy)**：所有动作概率为正  

#### ε-贪心策略定义  
$$
\pi(a|s) = 
\begin{cases} 
1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|} & a = a^* \\
\frac{\varepsilon}{|\mathcal{A}(s)|} & a \neq a^*
\end{cases}
$$  
其中 $a^* = \arg\max_a q(s,a)$, $\varepsilon \in [0,1]$ 控制探索强度  

#### 算法流程  
```  
初始化：π_0, q(s,a), Returns(s,a)=0, Num(s,a)=0, ε∈(0,1]  
for 每个幕:  
    任意起始 (s_0,a_0)  # 无需探索性出发  
    生成幕（同MC Exploring Starts）  
    # 更新q和策略（同MC Exploring Starts，但策略改进不同）  
    ...  
    # 策略改进（ε-贪心）  
    a* = argmax_a q(s_t,a)  
    π_{k+1}(a|s_t) = ε-greedy(q(s_t,·), a*)  
```  

#### 平衡探索与利用  
| **ε值** | **探索性** | **策略特性**         | **最优性**       |  
|---------|------------|----------------------|------------------|  
| ≈ 0     | 弱         | 接近确定性策略       | 接近最优         |  
| ≈ 1     | 强         | 接近均匀随机策略     | 可能远离最优     |  
| 衰减ε   | 动态平衡   | 初期探索→后期利用     | 渐进收敛至最优   |  

---

## 三、算法对比与演进  
### 1. 算法关系  
$$
\text{MC Basic} \xrightarrow{\text{数据高效}} \text{MC Exploring Starts} \xrightarrow{\text{移除探索性出发}} \text{MC } \varepsilon\text{-Greedy}
$$  

### 2. 关键特性对比  
| **特性**         | MC Basic       | MC Exploring Starts | MC ε-Greedy    |  
|------------------|----------------|----------------------|----------------|  
| **需环境模型**   | ❌             | ❌                   | ❌             |  
| **需探索性出发** | ✔️             | ✔️                   | ❌             |  
| **策略类型**     | 确定性         | 确定性               | 随机性         |  
| **数据效率**     | 低             | 高                   | 高             |  
| **更新方式**     | 批量（所有幕） | 在线（逐幕）         | 在线（逐幕）   |  

### 3. 最优性分析  
- **ε-贪心策略的最优性局限**：  
  - 仅在策略空间 $\Pi_\varepsilon$（固定ε的ε-贪心策略集合）中最优  
  - 全局最优需 $\varepsilon \to 0$ 或使用衰减ε  
- **示例**（网格世界）：  
  | $\varepsilon$ | $v_\pi(s_{\text{target}})$ | 策略性能 |  
  |----------------|-------------------------------|----------|  
  | 0.0            | 10.0                          | 最优     |  
  | 0.1            | 8.5                           | 接近最优 |  
  | 0.5            | 2.0                           | 显著下降 |  

---

## 四、关键公式总结  
1. **蒙特卡洛估计**：  
   $$
   q_\pi(s,a) \approx \frac{1}{N} \sum_{i=1}^N g^{(i)}(s,a)
   $$  
2. **ε-贪心策略**：  
   $$
   \pi(a|s) = \begin{cases} 
   1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|} & a = a^* \\
   \frac{\varepsilon}{|\mathcal{A}(s)|} & a \neq a^*
   \end{cases}
   $$  
3. **在线值更新**：  
   $$
   q_{\text{new}}(s_t,a_t) = q_{\text{old}}(s_t,a_t) + \frac{1}{N} \left( g - q_{\text{old}}(s_t,a_t) \right)
   $$  

---

## 五、总结  
### 核心贡献  
1. **无模型估计框架**：  
   - 通过MC方法直接从经验样本估计值函数，突破环境动态未知的限制  
2. **算法演进三阶段**：  
   - **MC Basic** → **MC Exploring Starts** → **MC ε-Greedy**  
   - 数据效率↑，探索需求↓  
3. **探索-利用平衡**：  
   - ε-贪心策略提供可调节的探索机制  
   - 衰减ε方案实现从探索到利用的自然过渡  

### 实际应用建议  
- **小规模离散问题**：MC ε-Greedy（ε衰减）  
- **大规模/连续问题**：结合函数逼近（如Deep Q-Networks）  
- **探索策略设计**：  
  - 初期：较大ε（强探索）  
  - 后期：较小ε（强调用）  

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒