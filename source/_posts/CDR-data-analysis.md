---
title: CDR Input Data Analysis
date: 2025-07-09 21:24:58
tags:
  - CDR
  - Data Analysis
  - 可能有点用
  - graph theory
categories:
  - CDR
  - Data Analysis
---

# CDR 数据源分析

本文主要是介绍一下 **深度学习** 在 _药物反应预测_ 中运用到的数据源。~~但由于本人比较捞~~ 本文主要从 **深度学习** 角度来看待这些数据源，对其在医学方面的意义~~（主要是鼠鼠也不会捏）~~不会有太多的描述

## CDR = Cancer Drug Response

我们的数据源有三种：

- _Cancer Representations_（癌症特征的表示）
- _Representations of Drug Compounds_（药物特征的表示）
- _Representations of Treatment Response_（治疗响应的表示）

接下来会按顺序进行说明

---

### Cancer Representations

癌症的特征是多组学的 ~~这不是理所应当吗~~

#### 多组学类型

通常基于以下四类组学数据：

- 基因组（Genomic）

  - 突变（Mutation）：体细胞突变（如单核苷酸变异 SNVs）可能驱动癌症进展，并影响药物靶点。
  - 拷贝数变异（CNV）：基因拷贝数的增加或缺失可能影响药物敏感性（如 HER2 扩增与曲妥珠单抗疗效相关）。

- 转录组（Transcriptomic）

  - 基因表达（Gene Expression）：通过微阵列或 RNA 测序（RNA-Seq）量化基因的 mRNA 水平。例如，高表达的耐药基因可能预示治疗失败。

- 表观组（Epigenomic）

  - DNA 甲基化（Methylation）：启动子区域的甲基化可能沉默抑癌基因，影响药物反应。

- 蛋白质组（Proteomic）
  - 蛋白质表达（RPPA 等）：直接测量蛋白质丰度（如激酶活性），更接近功能表型。

对于同一种组学数据，他们被表示成一组 **维数相同的向量**

#### 预处理与整合

1. 数据预处理

- 包括标准化（normalization）、批次效应校正（batch effect correction）和质量控制（QC）。例如，RNA-Seq 数据需通过 RPKM 或 TPM 标准化。

2. 多组学整合方法 ：
   - 早期整合（Early Integration）：直接拼接不同组学特征为单一向量，但可能因维度灾难（curse of dimensionality）导致过拟合。
   - 晚期整合（Late Integration）：通过独立子网络处理每组学数据（如 CNN 处理突变，GNN 处理表达数据），再融合特征。例如，MOLI 模型通过三重损失函数整合多组学数据，显著提升跨癌症模型的泛化能力。

#### 基因特征具有优势及新兴趋势

> 2014 年 NCI-DREAM 挑战赛表明， 基因表达数据在预测乳腺癌细胞系药物敏感性时最具预测力（优于突变或 CNV）。因此，约 90%的 DRP 模型使用基因表达（单独或联合其他组学）
> <img src="/img/CDR-data-analysis/gene.png" alt="gene" width="50%">

##### 新兴趋势

1. **结构生物学整合**：如利用蛋白质-蛋白质相互作用（PPI）网络（STRING 数据库）或通路信息（GSEA）构建生物网络，增强模型可解释性。
2. **图神经网络（GNN）**：将基因视为节点、相互作用为边，学习拓扑特征（如 GraOmicDRP 模型）。

---

### Representations of Drug Compounds

对药物的表示主要分为三种，一般只选取其中的一种 ~~虽然也有选用几种的 **创新** 方式~~。值得一提的是，在选定药物的表示方式后，之后的特征工程的方式目前来看非常的统一。接下来一一说明每一种表示方式。

#### SMILES（简化分子输入行条目系统）

1. _定义_：SMILES 是一种**线性字符串**表示法，通过符号编码分子结构（如`CCO`表示乙醇）。
2. _优势_：
   - 易于存储和处理，广泛用于化学信息学工具（如 RDKit）。
   - 可直接用于序列模型（如 RNN、Transformer）或通过预处理转换为其他表示（如图结构）。

#### 分子指纹（Fingerprints, FPs）和描述符（Descriptors）

1. 分子指纹

   - _定义_：**二进制向量**，表示分子中是否存在特定子结构（如药效团或官能团）。
   - _常用类型_：
     - **Morgan 指纹（ECFP）**：基于原子邻域的圆形拓扑指纹，长度通常为 512 或 1024 位。
     - **RDKit 指纹**：开源工具生成的二进制指纹。
   - _优势_：固定长度，适合传统机器学习模型（如随机森林）。

2. 分子描述符

   - _定义_：**数值向量**，编码物理化学性质（如分子量、疏水性、极性表面积等）。
   - _工具_：PaDEL、Mordred、Dragon 等软件可自动计算数百至数千个描述符。

#### 图结构表示（Graph-based Representations）

1. _定义_ ：将分子表示为**图**，其中原子为**节点**，化学键为**边**，节点和边可附加属性（如原子类型、键类型）。
2. _优势_ ：
   - 更自然地表征分子拓扑结构，适合图神经网络（GNN）。
   - 可捕捉局部和全局分子特征（如官能团相互作用）。

---

### Representations of Treatment Response

从构造模型的角度出发，这是 DRP 的核心数据源

- 它决定了模型最后完成的**任务类型**：训练连续值的**回归任务**和训练离散值的**分类任务**
- 他的数据质量很大程度上决定了模型的结果的优劣，即对该数据源对模型的好坏影响很大

此外，很少有从数据分析的角度出发分析这个数据源的文献，于是在这里给出简要的说明

#### 2.1 连续值表示（Continuous Measures）

1. **IC50**

   - 半数抑制浓度，即抑制 50%细胞活力所需的药物浓度。
   - _优势_：直观反映药物效力，广泛用于回归模型（如预测 IC50 的数值）。
   - _局限性_：仅反映单一浓度点的效果，可能忽略剂量-反应曲线的整体形状。

2. **AUC/AAC**
   - 剂量-反应曲线下面积（Area Under the Curve）或曲线上面积（Activity Area）。
   - _优势_：全局度量，综合所有浓度点的效果，对噪声更鲁棒。
   - _应用_：如 DeepCDR 等模型使用 AUC 作为回归目标，实证表明其泛化性优于 IC50。

#### 分类表示（Categorical Measures）

1. **二分类（敏感/耐药）**

   - 通过阈值（如瀑布算法、LOBICO）将连续反应（如 IC50）转化为离散标签。
   - _优势_：更贴近临床决策需求（如选择敏感药物）。
   - _示例_：Sharifi-Noghabi et al. (2021) 使用二分类训练深度神经网络，预测患者肿瘤的敏感性。

2. **多分类**
   - 如低/中/高反应性，适用于更细粒度的临床分级。

#### 排序表示（Ranking）

1. _目标_

   - 为个性化治疗推荐药物排序（如 Top-k 最有效药物）。

2. _方法_

   - Prasse et al. (2022)：将 IC50 转化为相关性分数，设计可微排序损失函数。
   - PPORank：利用强化学习动态优化排序，适应新增数据。

3. _优势_
   - 直接支持临床优先级排序，优于传统回归或分类。

#### 数据分析

由于本人大概率会做个分类模型，所以会将主要分析的是**分类表示**的数据在**图神经网络**中比较重视的几个指标，这里分析 _CCLE_ 和 _GDSC_ 两个数据集在选用主流阈值选取方法之后的表示。

直接先看结果捏（这里画了两个小图）

- CCLE

<img src="/img/CDR-data-analysis/comprehensive_bipartite_analysis_ccle.png" alt="CCLE" style="max-width: 100%; height: auto;">

- GDSC

<img src="/img/CDR-data-analysis/comprehensive_bipartite_analysis_gdsc.png" alt="GDSC" style="max-width: 100%; height: auto;">

<p>
  👉 <a href="/code/data_analysis/visualize_graph_analysis.py" target="_blank">查看用于生成上述图表的本地 Python 脚本：visualize_graph_analysis.py</a>
</p>

##### 🔍 关键数据对比

| 特征         | CCLE   | GDSC    | 倍数差异    |
| ------------ | ------ | ------- | ----------- |
| **数据规模** |
| 总节点数     | 341    | 783     | 2.3×        |
| 第一类节点   | 317    | 561     | 1.8×        |
| 第二类节点   | 24     | 222     | 9.3×        |
| 总边数       | 7,307  | 100,572 | 13.8×       |
| **图结构**   |
| 密度         | 0.9604 | 0.8075  | 0.84×       |
| 稀疏性       | 0.0396 | 0.1925  | 4.9×        |
| 平均度       | 42.86  | 256.89  | 6.0×        |
| 图直径       | 3      | 4       | 1.3×        |
| **边分布**   |
| 正边数量     | 1,375  | 11,591  | 8.4×        |
| 负边数量     | 5,932  | 88,981  | 15.0×       |
| 正边比例     | 18.8%  | 11.5%   | 0.61×       |
| 正负边比例   | 1:4.3  | 1:7.7   | 1.8× 不平衡 |

##### 📊 GNN 训练挑战分析

###### 过平滑风险评估

- **CCLE**: ⚠️ 高风险 (平均度 42.86)
- **GDSC**: 🚨 极高风险 (平均度 256.89)

###### 样本不平衡程度

- **CCLE**: 正负边比例 1:4.3 (中等不平衡)
- **GDSC**: 正负边比例 1:7.7 (严重不平衡)

###### 邻居相似度分析

```python
# 邻居重叠度对比
CCLE_similarity = {
    "第一类节点": 0.9374,  # 高度相似
    "第二类节点": 0.9274   # 高度相似
}

GDSC_similarity = {
    "第一类节点": 0.7659,  # 中等相似
    "第二类节点": 0.7143   # 中等相似
}
```

**结论**: CCLE 结构更均匀但多样性不足，GDSC 结构更复杂但多样性更好

##### 🎯 GNN 架构建议对比

###### 推荐架构优先级

- CCLE 推荐架构

  1. **Bipartite GNN** + Signed GCN
  2. **简单异构图 GNN** (HetGNN)
  3. **标准 GCN** + 强正则化

- GDSC 推荐架构

  1. **采样型 GNN** (GraphSAINT, FastGCN) + SGCN
  2. **大规模异构图 GNN** (HGT, RGCN)
  3. **图 Transformer** (处理复杂结构)

###### 具体参数建议

| 参数           | CCLE         | GDSC         | 原因                |
| -------------- | ------------ | ------------ | ------------------- |
| **网络深度**   | 2-3 层       | 严格 2 层    | GDSC 过平滑风险更高 |
| **隐藏维度**   | 64-128       | 128-256      | GDSC 需要更大容量   |
| **Dropout 率** | 0.3-0.5      | 0.5-0.7      | GDSC 需要更强正则化 |
| **学习率**     | 0.001-0.01   | 0.0001-0.001 | GDSC 需要更保守训练 |
| **批次大小**   | 32-64 个子图 | 16-32 个子图 | GDSC 内存限制       |
| **采样策略**   | 可选         | 必须         | GDSC 无法全图训练   |

# 📚 𝒥𝑒𝒻𝑒𝓇𝑒𝓃𝒸𝑒

<a href="/paper/Partin - Deep learning methods for drug response prediction in cancer Predominant and emerging trends.pdf" target="_blank">📄 Partin - Deep learning methods for drug response prediction in cancer Predominant and emerging trends</a>
