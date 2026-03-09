---
title: The Overview of Drug-Drug Interaction
categories:
  - DDI
  - Review
tags:
  - DDI
  - model framework
  - practice
  - GNN
  - attention
  - Knowledge Graph
date: 2026-03-29 23:12:02
---

# DDI 总览

这是搞完（实际并非）CDR之后的另一个东西捏，大致有以下内容：

1. DDI 的任务主要是什么
2. DDI 的难点（创新点出现（希望）的位置）
3. 药物处理方面与 CDR 有什么不同

<!-- more -->

## 什么是 DDI

目前，DDI 讨论还是**两**种药物之间的相互作用（$G_x, G_y, r_{xy}$），即一种药物的效用因为另一种药物的存在而发生改变的现象。

### 常用数据集

在构建 DDI 预测模型时，数据往往分布在异构的网络中。最核心的数据集通常分为机制类和表型类：

- **DrugBank**：最权威且使用最为频繁的数据集。主要包含药物之间的代谢层面影响（例如药物 A 抑制了代谢药物 B 的酶）。通常包含高度确定的相互作用类型。
- **TWOSIDES**：侧重于表型层面（Phenotypic）的信息。它记录了同时服用两种药物产生的、且在单药服用时不会出现的副作用（如恶心、心痛等）。
- **DRKG (Drug Repurposing Knowledge Graph) / Hetionet**：大型的综合生物医学知识图谱。除了包含只属于药物间的关系外，还整合了药物与基因、蛋白质、疾病之间的复杂关系。

### 不同的实验

由于 DDI 所引入的药物数量实际上远多于 CDR，这也就引出了其与 CDR（正常）的 *第一个* 不同：*DDI 经常讨论测试集中药物不存在与训练集中的情况*，具体来说，我们可以将其分为：

- **直推式 (Transductive / Warm-start)**：测试集中的药物分子在训练集中出现过，我们只是预测已知药物之间的新边（关系）。
- **归纳式 (Inductive / Cold-start)**：模型在部署时遇到的是结构、特征全新的、从未见过的药物分子（未见分布 OOD），需要仅凭其化学特征来预测交互。

---

## 药物和关系的表示方法

要把化学物质喂给机器学习模型，我们首先得把它们变成计算机理解的代数对象。与CDR不同的是，有很多 paper 都倾向于使用多种表示方法进行多模态的学习，主要有以下四个维度的表征：

1. 二进制/指纹表示法 (Binary Fingerprints)
   最传统的表示方式（如 MACCS, PubChem, Morgan Fingerprints）。通过检测药物内部是否包含特定的化学子结构，将其转化为定长的独热（One-hot）/布尔向量。
   - **数学表示**：设药物数量为 $N$，特征维度为 $D$（如 MACCS 为 166 维，PubChem 为 881 维）。药物分子可以直接表示为一个特征向量：
   $$
   \mathbf{x} \in \\{0, 1\\}^D 
   $$
    其中 $x_i = 1$ 表示存在第 $i$ 个子结构。

2. 序列表示法 (Sequence / SMILES Representation)
   使用 SMILES 字符串，把 2D 分子结构转化为一维的 ASCII 字符序列（把它当成一段本文来处理），常用 NLP 中的 Transformer / LSTM 提取特征。
   - **数学表示**：药物图被展平为由词表 $\mathcal{V}\_{vocab}$ 中标记组成的序列：
    $$ S =[s\_1, s\_2, \dots, s\_L], \quad s\_i \in \mathcal{V}\_{vocab} $$
     其中 $L$ 是字符串的长度。

3. 二维分子图表示 (2D Graph Representation) (当前最热门)
   用**图结构数据**直接描述分子，原子是节点，化学键是边。利用 GNN (Graph Neural Networks) 进行消息传递。
   - **数学表示**：一个药物分子定义为一个图 $G = (\mathcal{V}, \mathcal{E})$。
    *   节点特征矩阵：$\mathbf{X} \in \mathbb{R}^{|\mathcal{V}| \times F_v}$（每一行 $\mathbf{x}_i$ 是第 $i$ 个原子的特征向量，如电荷数、杂化状态）。
    *   边类型/特征矩阵：$\mathbf{E} \in \mathbb{R}^{|\mathcal{E}| \times F_e}$（$\mathbf{e}_{ij}$ 为键的类型，如单键、双键）。
    *   邻接矩阵：$\mathbf{A} \in \\{0, 1\\}^{|\mathcal{V}| \times |\mathcal{V}|}$。

4. 三维构象表示 (3D Structure)
5. 除了分子拓扑连接外，还包含空间原子的相对位置（坐标）。
   - **数学表示**：除了 2D 的 $\mathbf{X}$ 以外，还引入空间坐标矩阵模型：
    $$ \mathbf{R} \in \mathbb{R}^{|\mathcal{V}| \times 3} $$

---

### 关联关系与知识图谱的表示方法 (Relation & KG Representation)

为了预测交互作用，两两药物的关系不仅需要依据自身结构，还依赖于外界丰富知识网络的嵌入（Knowledge Graph Embedding, KGE）。

我们要构建一个包含了实体（药物、蛋白、疾病等）和它们之间关系的异构图谱 $\mathcal{G}_{KG}$：
*   **知识库的基础单元**：由成千上万个**三元组 (Triples)** 构成：
    $$ \mathcal{G}_{KG} = \{(h, r, t) \mid h \in \mathcal{E}, t \in \mathcal{E}, r \in \mathcal{R}\} $$
    其中 $\mathcal{E}$ 是所有实体（Entities）的集合，$\mathcal{R}$ 是所有相互作用关系（Relations）的集合。$h$ 是头实体，$t$ 是尾实体。

有了图谱结构后，使用 KGE 算法将离散的网络空间映射到连续的低维向量空间 $\mathbb{R}^d$中。论文中总结了几大流派（附经典打分函数 $f(h, r, t)$）：

#### A. 翻译距离模型 (Translational Models, 如 TransE, TransH, TransR)
思想：两个实体之间的关系被视为向量空间中的一种几何**平移 (Translation)**。
*   **数学表示 (以 TransE 为例)**：
    令投影后的连续向量为 $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{R}^d$。
    目标是使得 $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$。
    其打分函数为（得分越高表示越不可能是真实的 DDI 边，需加负号）：
    $$ f(h, r, t) = - \|\mathbf{h} + \mathbf{r} - \mathbf{t}\|_{1/2} $$

#### B. 张量分解/语义匹配模型 (Tensor Factorization, 如 RESCAL, DistMult, ComplEx)
思想：不找平移距离，而是利用矩阵/张量乘法，衡量头尾实体在某关系下的相似度及相互匹配性。这是预测 DDI 最经典的 KGE 手段。
*   **数学表示 (以 RESCAL 或其简化版 DistMult 为例)**：
    实体编码为 $\mathbf{h}, \mathbf{t} \in \mathbb{R}^d$。关系被表示为对角矩阵 $\mathbf{M}_r = \text{diag}(\mathbf{r})$（其中 $\mathbf{r} \in \mathbb{R}^d$）。
    打分函数定义为双线性内积：
    $$ f(h, r, t) = \mathbf{h}^\top \mathbf{M}\_r \mathbf{t} = \sum\_{i=1}^{d} h\_i \cdot r\_i \cdot t\_i $$
    （在更复杂的 ComplEx 模型中，向量会被映射到复数域 $\mathbf{h}, \mathbf{r}, \mathbf{t} \in \mathbb{C}^d$，以解决 DDI 关系中的非对称性问题）。

#### C. 数据分析中的联合打分 (Fusion & Prediction Layer)
最终判断药物 A 和药物 B 会发生关系 $r$ 的概率时，我们通常会将两种模式的表征拼接，送入多层感知机 (MLP)：
$$ P(r \mid A, B) = \sigma \Big( \mathbf{W} \cdot \big[\text{GNN}\_{structure}(G\_A) \oplus \text{GNN}\_{structure}(G\_B) \oplus \mathbf{h}\_{KG} \oplus \mathbf{t}\_{KG} \big] + \mathbf{b} \Big) $$
其中 $\sigma(\cdot)$ 是 Sigmoid 函数返回 $[0, 1]$ 之间的概率，$\oplus$ 代表向量拼接。