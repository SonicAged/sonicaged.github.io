---
title: Basic Concepts
categories:
  - Learning
  - ReinForcement Learning
  - Mathmatical Foundation
date: 2025-07-28 11:38:01
tags:
  - ReinForcement Learning
---

# 强化学习基础概念读书笔记  

<div style="display: flex; align-items: center;">
  <img src="/img/reinforcement-learning/01.png" style="width: 175px; margin-right: 20px;">
  <p>本文定义状态、动作、奖励、策略等核心概念，形式化MDP五元组框架。通过网格世界示例说明状态转移、回报计算和任务分类（分幕vs连续），为后续算法提供统一的数学建模基础。</p>
</div>




<!-- more -->



## 网格世界示例 (Grid-World Example)  
- **环境结构**：  
  - 网格由9个单元格组成（编号 $s_1$ 到 $s_9$）  
  - 单元格类型：可通行区域（如 $s_1, s_2$）、禁区（如 $s_6$）、目标区（如 $s_9$）、边界  
- **智能体任务**：  
  从任意起点找到通往目标区域 $s_9$ 的“优质”路径。  
  - “优质”定义：避免进入禁区、减少绕路、不触碰边界。  

<img src="/img/reinforcement-learning/Reinforcement-Learning-L1/grid%20world.png" alt="grid world" style="zoom: 30%;" />

---

## 核心概念与数学形式化  
### 状态 (State)  
- **定义**：智能体在环境中的**当前状态标识**（网格世界中为位置）  
- **状态空间 (State Space)**：  
  $$\mathcal{S} = \{s_i\}_{i=1}^{9} = \{s_1, s_2, \dots, s_9\}$$  

<img src="/img/reinforcement-learning/Reinforcement-Learning-L1/state.png" alt="state" style="zoom:23%;" />

### 动作 (Action)  
<img src="/img/reinforcement-learning/Reinforcement-Learning-L1/action.png" alt="image-20250729124818727" style="zoom:33%;" />

- **定义**：智能体在状态 $s_i$ 下可执行的操作  
- **动作空间 (Action Space)**：  
  $$\mathcal{A}(s_i) = \{a_k\}_{k=1}^{5} = \{a_1: \text{向上}, a_2: \text{向右}, a_3: \text{向下}, a_4: \text{向左}, a_5: \text{静止}\}$$  
  > **注意**：不同状态可能有不同的合法动作集（如边界处无法向外移动）  

### 状态转移 (State Transition)  
- **定义**：执行动作 $a$ 后，状态从 $s_i$ 迁移到 $s_j$ 的过程  
- **数学表示**：  
  - **确定性转移**（无随机因素）：  
    $$s_i \xrightarrow{a} s_j \quad \Rightarrow \quad p(s_j | s_i, a) = 1$$  
  - **随机性转移**（如受风力影响）：  
    $$p(s_j | s_i, a) \in [0, 1], \quad \sum_{s_j \in \mathcal{S}} p(s_j | s_i, a) = 1$$  
- **特殊规则**：  
  - **触碰边界**：$s_i \xrightarrow{a} s_i$（保持原位，奖励 $r_{\text{bound}}=-1$）  
  - **进入禁区**：  
    - 方案1（默认）：$s_i \xrightarrow{a} s_{\text{forbid}}$（允许进入但惩罚 $r_{\text{forbid}}=-1$)  
    - 方案2：$s_i \xrightarrow{a} s_i$（物理阻挡无法进入）  

#### 状态转移表示法  
| **状态\动作** | $a_1$ (上) | $a_2$ (右) | $a_3$ (下) | $a_4$ (左) | $a_5$ (静) |  
|--------------|-----------|-----------|-----------|-----------|-----------|  
| $s_1$        | $s_1$     | $s_2$     | $s_4$     | $s_1$     | $s_1$     |  
| $s_2$        | $s_2$     | $s_3$     | $s_5$     | $s_1$     | $s_2$     |  
| ...          | ...       | ...       | ...       | ...       | ...       |  
> **局限**：表格仅能表示确定性转移，随机转移需用概率分布 $p(s'|s,a)$。  

### 策略 (Policy)  
- **定义**：在状态 $s$ 下选择动作 $a$ 的规则  
- **数学表示**：  
  $$\pi(a|s) = \mathbb{P}(\text{选择动作 } a \mid \text{状态 } s), \quad \sum_{a \in \mathcal{A}(s)} \pi(a|s) = 1$$  
- **类型**：  
  - **确定性策略**：$\exists a' \text{ s.t. } \pi(a'|s)=1, \pi(a|s)=0 (\forall a \neq a')$  
  - **随机性策略**：$\exists a \text{ s.t. } \pi(a|s) \in (0,1)$（如探索需求）  

#### 策略表示法  
| **状态\动作** | $a_1$ | $a_2$ | $a_3$ | $a_4$ | $a_5$ |  
|--------------|-------|-------|-------|-------|-------|  
| $s_1$        | 0     | 0.5   | 0.5   | 0     | 0     |  
| $s_2$        | 0     | 0     | 1     | 0     | 0     |  
| ...          | ...   | ...   | ...   | ...   | ...   |  

### 奖励 (Reward)  
- **定义**：执行动作后环境反馈的标量值（鼓励/惩罚信号）  
- **设计原则**：  
  - $r_{\text{bound}} = -1$（触碰边界）  
  - $r_{\text{forbid}} = -1$（进入禁区）  
  - $r_{\text{target}} = +1$（到达目标）  
  - 其他情况 $r = 0$  
- **关键性质**：  
  - **相对性**：奖励的绝对数值不重要，相对大小决定策略优劣（例如 $\{+1,-1\}$ 等价于 $\{+2,0\}$）  
  - **随机性**：$p(r|s,a)$ 可建模不确定性（如学习效果波动）  

### 轨迹与回报 (Trajectory & Return)  
- **轨迹**：状态-动作-奖励序列  
  $$s_1 \xrightarrow[r=0]{a_2} s_2 \xrightarrow[r=0]{a_3} s_5 \xrightarrow[r=0]{a_3} s_8 \xrightarrow[r=1]{a_2} s_9$$  
- **回报 (Return)**：累积奖励之和  
  - **有限轨迹**：$\text{return} = \sum_{t=1}^{T} r_t = 0+0+0+1=1$  
  - **无限轨迹问题**：若在 $s_9$ 持续停留，回报 $\sum_{t=4}^{\infty} 1 \to \infty$（发散）  

#### 折扣回报 (Discounted Return)  
引入折扣因子 $\gamma \in (0,1)$ 保证收敛：  
$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots$$  
- **性质**：  
  - $\gamma \approx 0$：侧重近期奖励  
  - $\gamma \approx 1$：侧重远期奖励  
- **计算示例**（$s_1$ 到 $s_9$ 后停留）：  
  $$G = \gamma^3(1 + \gamma + \gamma^2 + \cdots) = \gamma^3 \frac{1}{1-\gamma}$$  

### 7. 任务分类  
| **类型**         | **特点**                     | **回报计算**       |  
|------------------|-----------------------------|-------------------|  
| **分幕任务** (Episodic) | 存在终止状态（如到达 $s_9$） | 有限步累积奖励      |  
| **连续任务** (Continuing) | 无终止状态                 | 折扣回报 $G_t$     |  

---

## 马尔可夫决策过程 (Markov Decision Process, MDP)  
### 关键五元组 $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$  
1. **状态集** $\mathcal{S}$  
2. **动作集** $\mathcal{A}(s)$（状态相关）  
3. **状态转移概率**：  
   $$P(s'|s,a) = \mathbb{P}(s_{t+1}=s' \mid s_t=s, a_t=a)$$  
4. **奖励函数**：  
   $$R(s,a) = \mathbb{E}[r_{t+1} \mid s_t=s, a_t=a] \quad \text{或} \quad p(r|s,a)$$  
5. **折扣因子** $\gamma \in (0,1)$  

### 马尔可夫性质 (Markov Property)  
未来状态/奖励仅依赖当前状态和动作：  
$$\begin{align*}  
p(s_{t+1}|s_t,a_t,\dots,s_0,a_0) &= p(s_{t+1}|s_t,a_t) \\\\  
p(r_{t+1}|s_t,a_t,\dots,s_0,a_0) &= p(r_{t+1}|s_t,a_t)  
\end{align*}$$  

### 抽象表示  

{% mermaid graph LR %} 
s1((s1)) -->|a1, r| s2((s2))  
s2 -->|a3, r| s5((s5))  
s5 -->|a3, r| s8((s8))  
s8 -->|a2, r=1| s9((s9))  
{% endmermaid %}

> 注：网格世界是MDP的具体实例，MDP是强化学习的通用数学模型。  

---

## 总结  
| **概念**          | **关键描述**                                  | **数学工具**          |  
|-------------------|---------------------------------------------|---------------------|  
| 状态与动作         | 环境描述与决策基础                            | 集合 $\mathcal{S}, \mathcal{A}$ |  
| 策略              | 状态到动作的映射规则                           | $\pi(a\|s)$          |  
| 状态转移与奖励      | 环境动态特性与反馈机制                         | $p(s'\|s,a), p(r\|s,a)$ |  
| 轨迹与回报         | 策略执行的序列与长期收益                       | $G_t=\sum \gamma^k r_{t+k+1}$ |  
| MDP框架           | 统一建模强化学习问题                           | $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$ |  

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning" target="_blank">
  <span style="display: inline-block; vertical-align: middle;">
    <img src="/icon/github.svg" alt="github" style="height: 1.1em; vertical-align: text-bottom; margin-top: 16px;">
  </span>
  Book-Mathematical-Foundation-of-Reinforcement-Learning
</a>