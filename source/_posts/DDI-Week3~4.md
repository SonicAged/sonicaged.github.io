---
title: Half-Monthly DDI Record
categories:
  - DDI
  - Daily
tags:
  - DDI
  - model framework
  - practice
  - GNN
  - attention
  - Knowledge Graph
date: 2026-03-24 23:12:02
---

# 3~4周里入到🧠的DDI

DDI 真神奇捏。阿巴阿巴阿巴

<!-- more -->

## 这周看到的蛮不赖的几篇捏

| Title | Algorithm | Journal/Conference | Motivation (challenges) | DMG | KG | Fusion strategy | etc. | Year | Objection | Exp | Drugs | Interactions | Others | Questions | LeZi |
| ---| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RaHDCL: Relation-aware hypergraph diffusion contrastive learning for drug-drug interaction event prediction | RaHDCL | Knowledge-Based Systems | 阐述了similarity-based & graph-based model的缺陷; 1. 关于特征信息的化学一致性  2. 对比学习 | | 超图, 扩散模型, 多时间步noising denoising超图 | | 1. 扩散模型, 每一时间步产生noising和denoising两幅图; 2. 相同event连接多个节点构成超图, event超边是有边特征的 | 2026 | multiclass classification | case study; Robustness analysis | SMILES (2 fingerprint) | Deng’s dataset; Ryu’s dataset |  | | 照理来说把两两关系揉在一起不是会丢失部分信息吗，可以有方法在构建超图的时候将两两关系也考虑进去吗 | |
| Dual-channel heterogeneous graph framework with multi-view contrastive learning for drug–drug interaction prediction | GEMCL | Engineering Applications of Artificial Intelligence | 1. 目前工作在实验中缺乏对 new or unseen drugs 的interaction预测; 2. 对于同时使用药物molecular graph和KG的方法, 融合策略欠缺 | 图扰动+预训练GCN | 多生物节点KG+图扰动+GCN | gated feature Fusion + CL | 1. 设计了图扰动策略和对比学习; 2. 使用预训练的GNN模型图学习. 预训练的调参方法也很新颖 in 2.4 | 2026 | multiclass & multilabel classification        | 1. 实验涵设计充分, transductive & Inductive; 2. 在3.9中有模块可解释性分析. 3. case study | SMILES | DB; TWOSIDES (multilabel数据集) | 多生物实体KG来自数据集external DRKG | | 何意味？！BKG的关系有108种？？？ |
| DMCL-DDI: A dual-view molecular contrastive learning framework for drug-drug interaction prediction | DMCL-DDI | Expert Systems with Applications| 一些方法仅考虑 single-view, 多数 dual-view 的方法则没考虑 atomic level 的特征 | DMG 通过 GAT 提取特征, inter (药物 pair 的特征, 通过乘积构建) + intra (单个药物) | | CL | 新颖的 drug pair 之间子结构关联的构建方式, 构建 inter grpah 是两个药物 molecular graph 的乘积 | 2026 | binary classification| inductive exp; Case study 有 substructure 的交互图|SMILES | DB;TWOSIDES | | | |
| SynCaps-DDI: A hybrid neural model combining capsule networks and syntactic graph convolution for drug–drug interaction extraction | SynCaps-DDI  | Biomedical Signal Processing and Control | | | 利用句法构建 KG(非 DDI 关系) | concat | 利用 BioBERT 处理 token,送入并行的双分支结构(RGCN 提取句法图连接信息,CapsNetwork 提取长程依赖信息) | 2026 | multi-class classification | 1.声明了不平衡数据无下采样等操作.<br/>2.外部验证.<br/>3.定量误差分析,分析模型模块与出错模式之间的关联.<br/>4.Robustness analysis | 无;模型的输入是一句说明性的话,xx 与药物 xx 可以联用之类的 | DDI Extraction 2013 dataset ;TAC 2018 DDI extraction dataset (外部验证) | 输入的是一句话,交给分词模型 | | |
| Devil in the Tail: A Multi-Modal Framework for Drug-Drug Interaction Prediction in Long Tail Distinction | **TFMD** | CIKM '24 | DDI 类型分布呈极端长尾特性；现有模型对头部类存在高度偏见；忽视了药物的多面属性及其协作互补。 | 基于图的特征（聚合邻居节点信息） | 未直接使用外部 KG ，但融合了 Target 和 Enzyme 特征。 | **模态增强融合 (Modality Enhancement Fusion)**：使用 SMILES 语义增强化学图空间特征，使用 Enzyme 增强 Target 表达。 | 引入 **Tailed Focal Loss (TFL)** 解决长尾分布下的梯度消失问题；使用 Transformer 提取 SMILES 序列特征。 | 2024 | multi-class (110 & 171类) | 宏平均指标 (Accuracy, F1, AUPR 等)；长尾数据集验证。 | SMILES, 化学图, Target, Enzyme | DDI-DB110, DDI-DB171, DDIMDL, MUFFIN | 源码已开源 (GitHub) | | |
| DDIPrompt: Drug-Drug Interaction Event Prediction based on Graph Prompt Learning | **DDIPrompt** | CIKM '24 | DDI 事件分布高度不平衡；罕见事件的标签数据极度稀缺；现有 GNN 模型依赖大量监督。 | 双视图图注意力网络 (Capture intra-molecular structures) | 结合预训练的生物医学知识图谱嵌入 (Pre-trained KG embedding) | **分层预训练 + Prompt 调优**：先通过相似度预测和链路预测进行预训练，再冻结模型使用类原型(Class Prototypes)作为 Prompt 。 | 提出**原型增强提示机制 (Prototype-enhanced prompting)**，适配少样本 (Few-shot) 场景。 | 2024 | multi-class edge prediction | 弱监督 (20%训练比例)；消融实验；超参数敏感性分析。 | 分子图, 知识图谱嵌入 | Deng's dataset, DrugBank dataset | 首次在药物领域尝试图提示学习 (Graph Prompt Learning) | | |
| IIB-DDI: Invariant Information Bottleneck Theory for Out-of-Distribution Drug-Drug Interaction Prediction | **IIB-DDI** | IEEE TCBB (2025) | 药物子结构分布不均导致模型依赖虚假关系；分布偏移 (OOD) 下泛化能力差。 | 使用 MPNN (消息传递神经网络) 编码分子结构 | **环境码本 (Environment Codebook)**：通过矢量量化 (VQ) 学习潜在环境分布。 | **不变信息瓶颈 (Invariant IB)**：利用IB理论提取核心子图，结合VQ聚类环境因素。 | 引入**不变学习 (Invariant Learning)** 寻找在不同环境下性能稳定的“理性”子结构；通过噪声注入优化互信息。 | 2025 | multiclass & binary | 换能 (Transductive) 与 归纳 (Inductive, 包括Type 1/2) 实验；OOD 场景验证。 | 分子图 (SMILES转换) | ZhangDDI, ChChMiner, DeepDDI, DrugBank | 侧重于解决分布偏移和泛化性能 (OOD) | | |

### RaHDCL 超图扩散 + 对比学习

1. 用关系构造超图，具有同一 DDI event 的药物之间连一条超边
2. 超图扩散以增强鲁棒性，之后也有对应的鲁棒性实验，但这个不是应该对比下降幅度吗？难道是结果不太好？
3. 层次对比学习。通过对原始视图和扩散增强视图进行层次化对比，模型能够学习到更具判别力的表征，充分挖掘隐藏在稀疏交互数据后的语义一致性

#### 具体来说

关系感知超图扩散（Relation-aware Hypergraph Diffusion）

模型通过扩散过程对药物-事件关系进行增强。输入为药物 $i$ 在超图中的关系向量 $h_i \in \{0, 1\}^{|\mathcal{E}|}$。

1. 前向扩散过程（Forward Process）
    模型在 $T$ 个时间步内向原始结构 $x_0 = h_i$ 逐步添加高斯噪声。
    给定噪声表（Noise Schedule） $\beta_t \in (0, 1)$，第 $t$ 步的状态分布为：
    $$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})$$

    利用重参数化技巧，可以直接从 $x_0$ 推导出 $x_t$ 的闭式解：
    $$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$$
    其中 $\alpha_t = 1 - \beta_t$，且 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$。

2. 关系感知逆向去噪（Reverse Denoising）
    去噪网络 $\theta$ 的目标是根据噪声状态 $x_t$ 和**条件信息 $x_a$**（药物成对关系）还原 $x_0$。
    RaHDCL 引入了**交叉注意力机制（Cross-Attention）**来融合超图噪声状态和药物关系条件：

    假设 $z_h = \text{Linear}(x_t)$，$z_a = \text{Linear}(x_a)$，则预测的 $x_0$ 为：
    $$x_{\theta}(x_t, x_a, t) = \sigma \left( z_h + \text{softmax}\left( \frac{(z_h \mathbf{W}^Q)(z_a \mathbf{W}^K)^T}{\sqrt{d}} \right) (z_h \mathbf{W}^V) \right)$$
    其中 $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ 是可学习的投影矩阵，$d$ 是特征维度。

    **扩散损失函数**：
    $$\mathcal{L}\_d = \mathbb{E}\_{t, x\_0, \epsilon} \left[ \frac{\alpha\_{t-1} - \alpha\_t}{2(1-\bar{\alpha}\_{t-1})(1-\bar{\alpha}\_t)} \| x\_{\theta}(x\_t, x\_a, t) - x\_0 \|^2\_2 \right]$$

层次对比学习（Hierarchical Contrastive Learning）

扩散增强后的超图 $\mathcal{G}'$ 与原始超图 $\mathcal{G}$ 构成对比视图。模型在两个层级应用 InfoNCE 损失。

1. 超边级对比学习（Hyperedge-level Contrast）
    该层级旨在确保相同的 DDI 事件（超边）在不同视图下具有一致的表征。
    设 $h_e$ 为原始视图中事件 $e$ 的嵌入，$h'_e$ 为增强视图中的嵌入，其对比损失为：
    $$L\_h = \frac{1}{|\mathcal{E}|} \sum\_{e \in \mathcal{E}} -\log \frac{\exp(\text{sim}(h\_e, h'\_e) / \tau)}{\sum\_{j \in \mathcal{E}} \exp(\text{sim}(h\_e, h'\_j) / \tau)}$$
    其中 $\text{sim}(\cdot)$ 为余弦相似度，$\tau$ 为温度参数。

2. 药物节点级对比学习（Drug-level Contrast）
    该层级旨在增强药物在 sparse（稀疏）DDI 图中的鲁棒性。
    设 $v_n$ 为原始视图中药物 $n$ 的嵌入，$v'_n$ 为增强视图中（由增强后的边特征生成的）药物嵌入：
    $$L\_n = \frac{1}{|\mathcal{V}|} \sum\_{n \in \mathcal{V}} -\log \frac{\exp(\text{sim}(v\_n, v'\_n) / \tau)}{\sum\_{k \in \mathcal{V}} \exp(\text{sim}(v\_n, v'\_k) / \tau)}$$

3. 联合优化目标
    模型最终通过加权组合监督预测损失（交叉熵 $\mathcal{L}\_{sup}$）和层次对比损失进行训练：
    $$\mathcal{L}\_{total} = \mathcal{L}\_{sup} + \epsilon\_1 L_h + (1 - \epsilon\_1) L_n$$
    其中 $\epsilon\_1$ 是用于调节超边级和节点级对比强度权重的超参数。

### GEMCL 自适应门控特征融合 + 多视图对比学习

1. 自适应门控特征融合 (Gated Feature Fusion)：设计了一个动态门控机制，能够根据不同预测场景（如已知药物 vs. 新药）自动调节结构特征与语义特征的权重。
2. 多视图对比学习 (Multi-view Contrastive Learning)：在训练阶段引入四个维度的对比损失（视图内一致性与跨通道对齐），解决了数据偏移问题并增强了表征的判别力。
3. 预训练机制：针对 DMG 采用节点遮蔽（Node Masking），针对 BKG 采用链路预测（Link Prediction），为下游任务提供了优秀的特征初始化。

#### 关于可解释性实验

1. **在 S1 场景（已知药物预测）中**：的中位数约为 0.42。这说明当药物在 BKG 中有丰富的关联数据（如已知的靶点、疾病关系）时，模型认为知识图谱特征和结构特征同等重要，甚至稍微偏向利用外部知识。
2. **在 S3 场景（新药-新药预测）中**：的中位数显著上升至 0.87。
   解释：对于完全没见过的新药，它们在 BKG 中通常是孤立节点，缺乏可靠的外部链接语义。此时，模型通过门控机制“自发地”意识到外部知识是不可信的（Noise），因此自动将注意力转移到药物的**内在化学结构（DMG）**上。

### DMCL-DDI

1. 引入了药物和药物的乘积的概念
2. 子结构可视化的实验

### IIB-DDI OOD + 不变学习

1.  **聚焦 OOD 泛化问题（Out-of-Distribution）**：
    传统的 DDI（药物-药物相互作用）预测模型在面对未见过的新药或分布改变的数据集时，性能往往大幅下降。本文首次系统地结合了**信息瓶颈（IB）理论**和**不变学习（Invariant Learning）**，旨在解决药物子结构分布不均导致的“伪相关”问题。

2.  **引入环境码本（Environment Codebook）**：
    由于药物数据的“环境”（如不同的临床场景、不同的分子骨架背景）是未知且难以标注的，论文引入了**向量量化（Vector Quantization, VQ）**技术。通过构建一个可学习的“环境码本”，将数据中的潜环境聚类，模拟出多样化的数据分布。

3.  **不变性合理化提取（Invariant Rationales）**：
    模型不仅是寻找对预测最重要的子图，更强调寻找在**各种环境下都能保持预测一致性**的核心子结构（即 Rationale）。这确保了模型学到的是生物化学层面的本质交互，而非数据集里的统计噪音。

4.  **数据集依赖的噪声注入（Dataset-dependent Noise Injection）**：
    相比于传统的随机高斯噪声或实例依赖噪声，IIB-DDI 利用从环境码本中提取的分布作为噪声注入。这种方法能更有效地平衡互信息优化过程，使训练曲线更平滑、损失更低，同时保留了数据集相关的语义信息。

### TFMD 长尾学习