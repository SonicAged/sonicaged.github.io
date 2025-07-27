---
title: CDR model In 2024
date: 2025-07-26 20:11:14
categories:
  - CDR
  - models
tags:
  - CDR
  - model framework
  - practice
  - GNN
  - attention
  - contrast learning
---

# 2024年的CDR模型

想看2020到2022的可以前往{% post_link CDR-model-part1 %}

想看2023的可以前往{% post_link CDR-model-part2 %}

本文将介绍的是2024年的CDR模型

<!-- more -->

## 2024

### <a href="/paper/CDR/2024/Kim 等 - PANCDR precise medicine prediction using an adversarial network for cancer drug response.pdf" target="_blank">PANCDR</a>

将GAN引入到了CDR，目的是缩小TCGA和GDSC之间的数据差异，这也应该是鼠鼠读到的第一篇着眼于临床和临床前数据差异的论文

[Source Code](https://github.com/DMCB-GIST/PANCDR)

#### Framework

<img src="C:/Users/SonicAged/Desktop/Blog/source/img/CDR/Framework/CDR-model-part2/PANCDR.png" alt="PANCDR" style="zoom:50%;" />

（感觉这个可以用于重构代码时的练手）

但多的不说少的不唠，TCGA的数据集是不是不太能用捏

#### 数据集

TCGA和GDSC

#### 实验

<img src="C:/Users/SonicAged/Desktop/Blog/source/img/CDR/Framework/CDR-model-part2/PANCDR-result.png" alt="PANCDR-result" style="zoom:80%;" />

不得不承认这个实验很有想法捏

>CV结果见补充Table 1。除GDSC的召回率外，DeepCDR在AUC、ACC、精确率和F1值方面均优于PANCDR，其值分别为0.8361、0.7761、0.3095和0.4328。然而，在TCGA数据集中，PANCDR取得了最高的AUC、ACC、精确率和F1值，分别为0.7106、0.6686、0.6491和0.6704。结果表明，PANCDR并未过度拟合GDSC的临床前数据，可以推广到TCGA的临床数据中。

同时，还有改变TCGA比例的鲁棒性实验

### <a href="/paper/CDR/2024/Lao - 2024 - DeepAEG a model for predicting cancer drug response based on data enhancement and edge-collaborativ.pdf" target="_blank">DeepAEG</a>

第一次在图中引入边的特征的CDR，这里也开始提到了之后超图中反复提到的潜在高维信息，如原子特征和化学键信息等

#### Framework

<img src="C:/Users/SonicAged/Desktop/Blog/source/img/CDR/Framework/CDR-model-part2/DeepAEG.png" alt="DeepAEG" style="zoom: 33%;" />

下面来说明一下CGCN的具体原理

CGCN 的设计背景

传统图卷积网络（GCN）在处理分子图时主要关注**节点（原子）特征更新**，而忽视了**边（化学键）特征**的重要性。在药物分子中，化学键的类型（单键、双键、三键等）、键长和键能等边特征直接影响分子的化学性质和生物活性。DeepAEG 提出的 CGCN 通过**同步更新节点和边特征**，克服了这一局限。

##### 分子图表示形式

在 CGCN 中，药物分子被表示为图 $D_n = (V_n, A_n)$：
- $V_n \in \mathbb{R}^{N_n \times C_{\text{node}}}$：节点特征矩阵
  - $N_n$ = 分子中原子数量
  - $C_{\text{node}}$ = 原子特征通道数（如原子类型、电荷等）
- $A_n \in \mathbb{R}^{N_n \times N_n \times C_{\text{edge}}}$：边特征张量
  - $C_{\text{edge}}$ = 化学键特征通道数（如键类型、键级等）

邻接张量 $A_n$ 可分解为 $C_{\text{edge}}$ 个子矩阵：
$$
A_n = \{E_{n,k}\}_{k=1}^{C_{\text{edge}}}, \quad E_{n,k} \in \mathbb{R}^{N_n \times N_n}
$$

##### CGCN 的层式更新机制

CGCN 采用分层特征更新策略，每层包含两个核心操作：

1. 节点特征更新

    $$
    V^{(l+1)} = \text{Update}\left(V^{(l)}, \sigma\left( \sum_{k=1}^{C_{\text{edge}}} E_k^{(l)} V^{(l)} W_k^{(l+1)} + b_k^{(l+1)} \right)\right)
    $$
    - $\sigma(\cdot)$：非线性激活函数（如 ReLU）
    - $W_k^{(l+1)}, b_k^{(l+1)}$：可学习参数
    - **物理意义**：聚合通过不同类型化学边（$E_k$）传递的邻居信息，更新原子状态

2. 边特征更新（核心创新）
	
	1. 节点关系向量计算

   $$
    e_{ij}^{(l+1)} = \sigma\left( \left( V_i^{(l+1)} \parallel V_j^{(l+1)} \right) W_{ij}^{(l+1)} + b_{ij}^{(l+1)} \right)
   $$

     - $\parallel$：向量拼接操作
     - $V_i^{(l+1)}, V_j^{(l+1)}$：更新后的节点 $i,j$ 的特征向量
     - **物理意义**：基于更新后的原子状态，计算原子间的新关系

	2. 边特征融合更新

   $$
    E_{ij}^{(l+1)} = \operatorname{Update}\left( E_{ij}^{(l)}, e_{ij}^{(l+1)} \right)
   $$
   
     - **物理意义**：将原子间关系向量 $e_{ij}$ 与原始边特征 $E_{ij}$ 融合，更新化学键表示

##### CGCN 架构特点
1. **分层设计**：
   - 默认采用2层CGCN
   - 第1层：执行节点和边同步更新
   - 第2层：仅更新节点特征（减少计算开销）
   
2. **参数共享机制**：
   - 同一层的 $W_k$ 在不同边类型间共享
   - 不同层有独立的参数集

#### 数据集

GDSC和CCLE

### <a href="/paper/CDR/2024/Liu - 2024 - Improving anti-cancer drug response prediction using multi-task learning on graph convolutional netw.pdf" target="_blank">MTIGCN</a>

又一个拼好模捏

<img src="C:/Users/SonicAged/Desktop/Blog/source/img/CDR/Framework/CDR-model-part2/MTIGCN.png" alt="MTIGCN" style="zoom: 50%;" />

### <a href="/paper/CDR/2024/Peng - Hypergraph Representation Learning for Cancer Drug Response Prediction.pdf" target="_blank">HRLCDR</a>

一个严格意义上来讲不算超图学习的超图学习捏，但他确实通过信息整合的方式构造了很多个长得不一样的图，也算是提取了高维信息吧

<img src="C:/Users/SonicAged/Desktop/Blog/source/img/CDR/Framework/CDR-model-part2/HRLCDR.png" alt="HRLCDR" style="zoom: 50%;" />

还有就是这一篇投了两个刊物捏

### <a href="/paper/CDR/2024/Peng 等 - Predicting Anti-Cancer Drug Response Based on Hypergraph Representation Learning.pdf" target="_blank">DGCL</a>

来看一个真の超图捏，同时他说由于超图中的嵌入和超边都是由局部图的嵌入得到的，这么高高维信息很有可能会直接**过平滑**，所以在嵌入空间引入对比学习以增大局部和全局信息的差异

#### Framework

<img src="C:/Users/SonicAged/Desktop/Blog/source/img/CDR/Framework/CDR-model-part2/DGCL.png" alt="DGCL" style="zoom:50%;" />

接下来会给出详细的模型构造过程

##### 模型整体框架

DGCL 是一个**动态超图对比学习框架**，用于多关系药物-基因相互作用预测。其核心创新在于同时建模**局部拓扑关系**（通过二分图卷积）和**全局语义关系**（通过动态超图学习），并引入**自增强对比学习**约束超图结构学习。整体架构如图1所示：

##### 显式局部关系建模

1. 输入表示

   - 药物集合：$M$ 个药物，嵌入矩阵 $E^{(d)} \in \mathbb{R}^{M \times d}$

   - 基因集合：$N$ 个基因，嵌入矩阵 $E^{(g)} \in \mathbb{R}^{N \times d}$

   - 相互作用矩阵：$R \in \{0,1\}^{M \times N}$


2. 二分图构建

    构建增广邻接矩阵：$A = \begin{bmatrix} 
    0 & R \\
    R^\top & 0 
    \end{bmatrix} \in \mathbb{R}^{(M+N)\times(M+N)}$

    归一化处理：$\overline{A} = \overline{D}^{-\frac{1}{2}}(A + I_{M+N})\overline{D}^{-\frac{1}{2}}$

    其中 $\overline{D}_{ii} = \sum_j (A + I_{M+N})_{ij}$ 是度矩阵。

3. 图卷积操作

    采用简化的 GCN 层（无激活函数和线性变换）：

    $$
    E_{l-1} = [E_{l-1}^{(d)}, E_{l-1}^{(g)}]\\
    \tilde{E}_l = p(\overline{A}) E_{l-1}
    $$

    $p(\cdot)$ 表示边丢弃操作（防止过拟合），丢弃率 $\in \{0.25, 0.5, 0.75\}$


##### 动态超图结构学习

1. 超图表示
   
   - 超边集合：$K$ 个超边
   - 药物-超边矩阵：$H^{(d)} \in \mathbb{R}^{M \times K}$
   - 基因-超边矩阵：$H^{(g)} \in \mathbb{R}^{N \times K}$

2. 低秩分解（降低计算复杂度）
$$
H^{(d)} = E^{(d)} W^{(d)}, \quad H^{(g)} = E^{(g)} W^{(g)}
$$

其中 $W^{(d)}, W^{(g)} \in \mathbb{R}^{d \times K}$ 是可训练参数矩阵。

##### 隐式全局关系建模
1. 超图消息传递
    1. **超边嵌入生成**：$Z^{(d)} = H^{(d)\top} E_{l-1}^{(d)} \in \mathbb{R}^{K \times d}$

    2. **药物节点更新**：$\tilde{E}_l^{(d)} = H^{(d)} Z^{(d)} = H^{(d)} H^{(d)\top} E_{l-1}^{(d)}$

    3. **基因节点更新**（类似）：$\tilde{E}_l^{(g)} = H^{(g)} H^{(g)\top} E_{l-1}^{(g)}$

    > **关键优势**：超边连接多个节点，实现跨图消息传递，突破传统 GNN 的距离限制

##### 局部-全局自增强对比学习
1. 对比机制设计

   - **正样本对**：同一节点的局部嵌入 $\tilde{e}_{i,l}^{(d)}$ 和全局嵌入 $\tilde{e}_{i,l}^{(d)}$
   - **负样本对**：不同节点的局部和全局嵌入组合

###### 对比损失函数（基于 InfoNCE）

药物节点对比损失：
$$
\mathcal{L}_s^{(d)} = \sum_{i=1}^M \sum_{l=1}^L -\log \frac{
  \exp(\text{sim}(\tilde{e}_{i,l}^{(d)}, \tilde{e}_{i,l}^{(d)}) / \tau)
}{
  \sum_{i'=1}^M \exp(\text{sim}(\tilde{e}_{i,l}^{(d)}, \tilde{e}_{i',l}^{(d)}) / \tau)
}
$$

基因节点对比损失 $\mathcal{L}_s^{(g)}$ 形式类似。其中：
- $\text{sim}(u,v) = u^\top v / \|u\|\|v\|$（余弦相似度）
- $\tau$：温度超参数 $\in \{0.1,0.3,1,3,10\}$

> **创新点**：避免随机扰动导致的信息损失，直接用局部/全局视图作为对比对

##### 特征集成
1. 层内特征融合
$$\begin{align*}
e_{i,l}^{(d)} &= \tilde{e}_{i,l}^{(d)} + \tilde{e}_{i,l}^{(d)} \\
e_{j,l}^{(g)} &= \tilde{e}_{j,l}^{(g)} + \tilde{e}_{j,l}^{(g)}
\end{align*}$$
其中 $\tilde{e}$ 来自局部GCN，$\tilde{e}$ 来自全局超图

2. 跨层残差连接
$$\begin{align*}
\hat{e}_i^{(d)} &= \sum_{l=0}^L e_{i,l}^{(d)} \qquad
\hat{e}_j^{(g)} &= \sum_{l=0}^L e_{j,l}^{(g)}
\end{align*}$$
缓解过平滑问题，保留各层语义信息

##### 预测层

1. 相互作用类型预测
    $$
    \hat{y} = \text{softmax}\left( 
    W_2 \cdot \text{ReLU}\left( 
    W_1 [\hat{e}_i^{(d)} \| \hat{e}_j^{(g)}] + b_1 
    \right) + b_2 
    \right)
    $$

    其中：
    - $[\cdot \| \cdot]$ 表示向量拼接
    - $W_1 \in \mathbb{R}^{2d \times h}$, $W_2 \in \mathbb{R}^{h \times C}$ 为可训练参数
    - $C$ 是相互作用类型数

2. 预测损失函数
    $$\mathcal{L}_p = -\sum_{c=1}^C y_c \log \hat{y}_c$$

    $y_c \in \{0,1\}$ 表示真实标签


##### 多任务优化
1. 联合损失函数
    $$
    \mathcal{L} = \mathcal{L}_p + \lambda_1 (\mathcal{L}_s^{(d)} + \mathcal{L}_s^{(g)}) + \lambda_2 \|\Theta\|_2^2
    $$

    其中：
    - $\lambda_1 \in \{10^{-4},10^{-3},10^{-2},10^{-1}\}$ 控制对比学习权重
    - $\lambda_2 \in \{10^{-8},10^{-7},10^{-6},10^{-5}\}$ 控制 L2 正则化强度

#### 实验

去看原文捏，这篇论文的实验还是很精炼的捏

### <a href="/paper/CDR/2024/Peng 等 - Hierarchical graph representation learning with multi-granularity features for anti-cancer drug resp.pdf" target="_blank">HLMG</a>

这里用到的Hierarchical graph和之前 .{% post_link Feature-Related-Attention %}#Hierarchical Attention 类似捏，也会详细说明一下这一个模型的架构捏

#### Framework

<img src="C:/Users/SonicAged/Desktop/Blog/source/img/CDR/Framework/CDR-model-part2/HLMG.png" alt="HLMG" style="zoom:50%;" />

##### 整体框架
HLMG 是一个**分层图表示学习模型**，用于预测抗癌药物响应（IC50值）。其核心创新在于整合**多粒度特征**（细胞系/药物的全局+子结构特征）和**多级外部关联**（直接响应+相似性关联），通过异构图卷积实现信息融合。整体框架如图1所示：


##### 多粒度特征提取
1. 细胞系特征提取
   1. 全局基因表达特征
      - **输入**：基因表达矩阵 $C_g \in \mathbb{R}^{m \times d_{cg}}$ ($m$=细胞系数, $d_{cg}$=基因数)
      - **标准化**：$C_{g(i)} = \frac{(c_i - \mu_i)}{\sigma_i}$
      - **线性变换**：$H_g = C_g \cdot W_g \quad (W_g \in \mathbb{R}^{d_{cg} \times d_g})$

    2. 通路子结构特征
         - **输入**：34个癌症相关通路（KEGG数据库）
         - **通路图卷积**：
           $$
           \begin{aligned}
           c_p^i &= GCN^i(P_i, p_{gene}^i) W^i \\
           &= \left( (L_{P_i} \cdot p_{gene}^i)^T \cdot W_1^i \right) W_2^i \\
           L_{P_i} &= D_{pi}^{-\frac{1}{2}} P_i D_{pi}^{-\frac{1}{2}}
           \end{aligned}
           $$
         - **通路注意力机制**：
           $$
           V_{ap} = \delta(\sigma(C_p \cdot W_{ap1}) W_{ap2})
           $$
         - **特征融合**：$H_p = [V_{ap}^1 \odot c_p^1 \parallel \cdots \parallel V_{ap}^i \odot c_p^i]$

    3. 细胞系多粒度融合
        $$X_{cell} = [H_g \parallel H_p]$$

1. 药物特征提取
    1. 全局分子指纹特征
        $$H_f = D_f \cdot W_f \quad (D_f \in \mathbb{R}^{n \times d_d}, W_f \in \mathbb{R}^{d_d \times d_f})$$

    2. 分子子结构特征
         - **子结构分解**：BRICS算法分割SMILES序列
         - **子结构注意力**：
           $$
           V_{ad} = \delta(\sigma(D_s \cdot W_{ad1}) W_{ad2})
           $$
         - **特征聚合**：$H_s = sum(V_{ad} \odot D_s)$

    3. 药物多粒度融合
        $$X_{drug} = [H_f \parallel H_s]$$

##### 多级外部特征聚合
1. 异构图构建
   - **节点**：细胞系 + 药物
   - **边类型**：
     - 直接响应边：$A \in \{0,1\}^{m \times n}$ (IC50二值化)
     - 细胞系相似边：$C_{ij} = S_c(i,j)$ (基因表达+响应相似性)
     - 药物相似边：$B_{ij} = S_d(i,j)$ (分子指纹+响应相似性)

2. 细胞系特征聚合
    1. 自身特征更新
    $$
    \begin{aligned}
    Cell_s^{(k)} &= Cell_s w_{cs}^{(k-1)} + Cell_s \odot Cell_s^0 w_{cs}^{(k-1)} \\
    Cell_s &= (D_c^{-1} + I_c) Cell_s^{(k-1)}
    \end{aligned}
    $$

    2. 一级关联聚合（直接响应药物）
    $$
    \begin{aligned}
    Cell_1^{(k)} &= Cell_1 W_{c1}^{(k-1)} + Cell_1 \odot Cell_s^0 W_{c1}^{(k-1)} \\
    Cell_1 &= (D_c^{\frac{1}{2}} A D_d^{\frac{1}{2}}) Drug_s^{(k-1)}
    \end{aligned}
    $$

   3. 二级关联聚合（相似药物）
    $$
    \begin{aligned}
    Cell_2^{(k)} &= (D_c^{\frac{1}{2}} A D_d^{\frac{1}{2}}) Drug_{sim}^{(k-1)} W_{c2}^{(k-1)} + 
    \\\\ 
    &(D_c^{\frac{1}{2}} A D_d^{\frac{1}{2}}) Drug_{sim}^{(k-1)} W_{c2}^{(k-1)}\odot Cell_s^0W_{c2}^{k-1} \\\\
    Drug_{sim}^{(k-1)} &= \sigma((D_B^{\frac{1}{2}} B D_B^{\frac{1}{2}}) Drug_s^{(k-1)} W_{ds}^{(k)})
    \end{aligned}
    $$

    4. 最终融合
    $$Cell_{agg}^{(k)} = \sigma(Cell_s^{(k)} + Cell_1^{(k)} + Cell_2^{(k)})$$

3. 药物特征聚合（对称操作）
- 自身更新：类似公式(18)
- 一级关联：$Drug_1^{(k)} = f(D_d^{\frac{1}{2}} A^T D_c^{\frac{1}{2}} Cell_s^{(k-1)})$
- 二级关联：$Drug_2^{(k)} = f(D_d^{\frac{1}{2}} A^T D_c^{\frac{1}{2}} Cell_{sim}^{(k-1)})$
- 最终融合：$Drug_{agg}^{(k)} = \sigma(Drug_s^{(k)} + Drug_1^{(k)} + Drug_2^{(k)})$

##### 药物响应预测
1. 线性相关系数计算
$$
\operatorname{Corr}(h_i, h_j) = \frac{(h_i - \mu_i)(h_j - \mu_j)^T}{\sqrt{(h_i - \mu_i)(h_i - \mu_i)^T} \sqrt{(h_j - \mu_j)(h_j - \mu_j)^T}}
$$

2. 响应矩阵重构
$$\hat{A} = \varphi(\operatorname{Corr}(Cell_{agg}^{(k)}, Drug_{agg}^{(k)}))$$
其中 $\varphi(h) = \frac{1}{1+e^{-\gamma h}}$ 为Sigmoid函数

3. 损失函数
$$
\begin{aligned}
L(A,\hat{A}) = -\frac{1}{m \times n} \sum_{i,j} & M_{ij} [A_{ij} \ln(\hat{A}_{ij}) \\
& + (1-A_{ij}) \ln(1-\hat{A}_{ij})]
\end{aligned}
$$
$M_{ij}$ 表示训练掩码矩阵

##### 模型创新点
1. **多粒度特征融合**：
   - 细胞系：基因表达（全局）+ 通路子结构
   - 药物：分子指纹（全局）+ BRICS子结构
   - 注意力机制加权关键组分

2. **三级特征聚合**：

3. **相似性增强传播**：
   - 细胞系相似度：$S_c(i,j)= \frac{1}{2}(e^{-\frac{\|x_i-x_j\|_2}{2\epsilon^2}} + e^{-\frac{\|A_i-A_j\|^2}{2\epsilon^2}})$
   - 药物相似度：$S_d(i,j)= \frac{1}{2}(\frac{|d_i \cap d_j|}{|d_i \cup d_j|} + e^{-\frac{\|A_i^T-A_j^T\|^2}{2\epsilon^2}})$



### <a href="/paper/CDR/2024/Wang - 2024 - A hierarchical attention network integrating multi-scale relationship for drug response prediction.pdf" target="_blank">MultiDRP</a>



### <a href="/paper/CDR/2024/Shi - MCMVDRP a multi-channel multi-view deep learning framework for cancer drug response prediction.pdf" target="_blank">MCMVDRP</a>



### <a href="/paper/CDR/2024/Sun - 2024 - DAPM-CDR A domain adaptation prompting model for drug response prediction.pdf" target="_blank">DAPM CDR</a>



#### <a href="/paper/CDR/2024/Wang - 2024 - XGraphCDS An explainable deep learning model for predicting drug sensitivity from gene pathways and.pdf" target="_blank">XGraphCDS</a>

将模型可解释性的，但这不是鼠鼠想关注的捏

### <a href="/paper/CDR/2024/Xu 等 - RedCDR Dual Relation Distillation for Cancer Drug Response Prediction.pdf" target="_blank">RedCDR</a>



### <a href="/paper/CDR/2024/Zhou - 2024 - Multi-omics fusion based on attention mechanism for survival and drug response prediction in Digesti.pdf" target="_blank">MFGAN</a>

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/CDR/2024/Cui 等 - 2024 - Heterogeneous graph contrastive learning with gradient balance for drug repositioning.pdf" target="_blank">📄 Cui 等 - 2024 - Heterogeneous graph contrastive learning with gradient balance for drug repositioning</a>

<a href="/paper/CDR/2024/Kim 等 - PANCDR precise medicine prediction using an adversarial network for cancer drug response.pdf" target="_blank">📄 Kim 等 - PANCDR precise medicine prediction using an adversarial network for cancer drug response</a>

<a href="/paper/CDR/2024/Lao - 2024 - DeepAEG a model for predicting cancer drug response based on data enhancement and edge-collaborativ.pdf" target="_blank">📄 Lao - 2024 - DeepAEG a model for predicting cancer drug response based on data enhancement and edge-collaborativ</a>

<a href="/paper/CDR/2024/Liu - 2024 - Improving anti-cancer drug response prediction using multi-task learning on graph convolutional netw.pdf" target="_blank">📄 Liu - 2024 - Improving anti-cancer drug response prediction using multi-task learning on graph convolutional netw</a>

<a href="/paper/CDR/2024/Peng - Hypergraph Representation Learning for Cancer Drug Response Prediction.pdf" target="_blank">📄 Peng - Hypergraph Representation Learning for Cancer Drug Response Prediction</a>

<a href="/paper/CDR/2024/Peng 等 - Hierarchical graph representation learning with multi-granularity features for anti-cancer drug resp.pdf" target="_blank">📄 Peng 等 - Hierarchical graph representation learning with multi-granularity features for anti-cancer drug resp</a>

<a href="/paper/CDR/2023/Tao 等 - Prediction of multi-relational drug–gene interaction via Dynamic hyperGraph Contrastive Learning.pdf" target="_blank">📄 Tao 等 - Prediction of multi-relational drug–gene interaction via Dynamic hyperGraph Contrastive Learning</a>

<a href="/paper/CDR/2024/Peng 等 - Predicting Anti-Cancer Drug Response Based on Hypergraph Representation Learning.pdf" target="_blank">📄 Peng 等 - Predicting Anti-Cancer Drug Response Based on Hypergraph Representation Learning</a>

<a href="/paper/CDR/2024/Shi - MCMVDRP a multi-channel multi-view deep learning framework for cancer drug response prediction.pdf" target="_blank">📄 Shi - MCMVDRP a multi-channel multi-view deep learning framework for cancer drug response prediction</a>

<a href="/paper/CDR/2024/Sun - 2024 - DAPM-CDR A domain adaptation prompting model for drug response prediction.pdf" target="_blank">📄 Sun - 2024 - DAPM-CDR A domain adaptation prompting model for drug response prediction</a>

<a href="/paper/CDR/2024/Wang - 2024 - A hierarchical attention network integrating multi-scale relationship for drug response prediction.pdf" target="_blank">📄 Wang - 2024 - A hierarchical attention network integrating multi-scale relationship for drug response prediction</a>

<a href="/paper/CDR/2024/Wang - 2024 - XGraphCDS An explainable deep learning model for predicting drug sensitivity from gene pathways and.pdf" target="_blank">📄 Wang - 2024 - XGraphCDS An explainable deep learning model for predicting drug sensitivity from gene pathways and</a>

<a href="/paper/CDR/2024/Xu 等 - RedCDR Dual Relation Distillation for Cancer Drug Response Prediction.pdf" target="_blank">📄 Xu 等 - RedCDR Dual Relation Distillation for Cancer Drug Response Prediction</a>

<a href="/paper/CDR/2024/Zhou - 2024 - Multi-omics fusion based on attention mechanism for survival and drug response prediction in Digesti.pdf" target="_blank">📄 Zhou - 2024 - Multi-omics fusion based on attention mechanism for survival and drug response prediction in Digesti</a>