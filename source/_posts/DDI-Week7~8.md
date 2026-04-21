---
title: Half-Monthly DDI Record | Series 4
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
date: 2026-04-22 23:12:02
---

# 7~8周里入到🧠的DDI

DDI 真神奇捏。阿巴阿巴阿巴

<!-- more -->

## 这周看到的蛮不赖的几篇捏

Emmm，类似的表格传到Google Docs上了捏，，，，懒得动了捏

### CAREDDI

![image-20260421150321849](https://raw.githubusercontent.com/SonicAged/uPic-public/master/2026/04/21/150321_image-20260421150321849.png)

这篇论文中提到，现有的数据集中存在虚假的药物关联关系（由于实验偏差、知识更新滞后或标注不一致）

所以他把异常检测引入了DDI模型，从而增强模型在遇到spurious DDIs时的鲁棒性

#### 架构

(a) 部分就是一个非常常规的cross-attention捏。

重点是(b)部分捏

##### 总体

分解为两个DDI元任务：一个用于学习有价值的标注型DDI特征分布的标注特征分布学习任务，以及一个用于确保整体预测准确性的DDI预测器任务。

MCD（最小协方差矩阵） 的目标是**从这 $n$ 个样本中寻找一个子集（包含 $h$ 个样本），使得这个子集的协方差矩阵的行列式（Determinant）最小。**

提出了一种基于密度的新型特征分布学习层，用于动态适应标注DDI特征分布中的鲁棒多变量位置与离散度。

- **位置感知聚合 (Location-aware aggregation)：** 学习 DDI 特征分布的中心位置 dlocationdlocation。
- **散射感知聚合 (Scatter-aware aggregation)：** 学习特征的离散程度 PscatterPscatter。
- **损失函数设计：**
    $L_{CDSL}$ (Chi-square Distribution-based Soft Loss)： 强制让标注的 DDI 特征分布逼近卡方分布 (χ2χ2)。如果一个样本偏离该分布过远，则更有可能被判定为“伪 DDI”。

### RISE-DDI

![image-20260421193905214](https://raw.githubusercontent.com/SonicAged/uPic-public/master/2026/04/21/193905_image-20260421193905214.png)

1. 总体架构组件

模型由两个核心模块组成，子图采样器（RL Agent）和交互预测器（结构感知奖励模型）

他将子图采样建模为马尔可夫决策过程，定义了对应的环境

### PC-DDI

![image-20260421203816565](https://raw.githubusercontent.com/SonicAged/uPic-public/master/2026/04/21/203816_image-20260421203816565.png)