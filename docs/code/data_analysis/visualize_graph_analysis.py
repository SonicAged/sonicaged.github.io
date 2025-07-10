#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二分图可视化分析脚本
生成详细的图表来展示二分图的性质
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.text import Text
import seaborn as sns
import pandas as pd
from collections import Counter
import networkx as nx
from networkx.algorithms import bipartite
from typing import List, Tuple, Dict, Any, cast

# 设置统一配色方案
NATURE_COLORS = {
    'turquoise': '#8ECFC9',  # RGB(142,207,201) 青绿色
    'orange': '#FFBE7A',     # RGB(255,190,122) 橙色
    'coral': '#FA7F6F',      # RGB(250,127,111) 珊瑚红
    'blue': '#82B0D2',       # RGB(130,176,210) 蓝色
    'light_turquoise': '#A8DAD5',  # 青绿色的浅色版
    'light_orange': '#FFD4A5',     # 橙色的浅色版
    'light_coral': '#FBB3A9',      # 珊瑚红的浅色版
    'light_blue': '#ADC9E2'        # 蓝色的浅色版
}

# 设置全局绘图样式
sns.set_style("whitegrid")  # 使用seaborn的whitegrid样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True

def load_and_analyze_data():
    """加载并分析数据"""
    file_path = "data/CCLE/allpairs.npy"
    data = np.load(file_path)
    
    # 基本统计
    node1_ids = data[:, 0]
    node2_ids = data[:, 1]
    edge_weights = data[:, 2]
    
    return data, node1_ids, node2_ids, edge_weights

def create_comprehensive_visualization():
    """创建综合可视化"""
    data, node1_ids, node2_ids, edge_weights = load_and_analyze_data()
    
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_alpha(0.0)  # 设置透明背景
    
    # 1. 边权重分布
    ax1 = plt.subplot(3, 4, 1)
    ax1.patch.set_alpha(0.0)  # 设置透明背景
    edge_counter = Counter(edge_weights)
    weights = list(edge_counter.keys())
    counts = list(edge_counter.values())
    bars = ax1.bar([str(w) for w in weights], counts, 
                   color=[NATURE_COLORS['coral'] if w < 0 else NATURE_COLORS['turquoise'] for w in weights])
    ax1.set_title('边权重分布')
    ax1.set_xlabel('权重值')
    ax1.set_ylabel('边数量')
    for i, (w, c) in enumerate(zip(weights, counts)):
        ax1.text(i, c + 50, str(c), ha='center', fontweight='bold')
    
    # 2. 节点度分布 - 第一类节点
    ax2 = plt.subplot(3, 4, 2)
    ax2.patch.set_alpha(0.0)
    node1_degrees = Counter(node1_ids)
    degrees1 = list(node1_degrees.values())
    ax2.hist(degrees1, bins=20, alpha=0.7, color=NATURE_COLORS['blue'], edgecolor='black')
    ax2.set_title('第一类节点度分布')
    ax2.set_xlabel('度')
    ax2.set_ylabel('节点数量')
    ax2.axvline(float(np.mean(degrees1)), color=NATURE_COLORS['coral'], linestyle='--', 
                label=f'平均值: {float(np.mean(degrees1)):.1f}')
    ax2.legend()
    
    # 3. 节点度分布 - 第二类节点
    ax3 = plt.subplot(3, 4, 3)
    ax3.patch.set_alpha(0.0)
    node2_degrees = Counter(node2_ids)
    degrees2 = list(node2_degrees.values())
    ax3.hist(degrees2, bins=20, alpha=0.7, color=NATURE_COLORS['orange'], edgecolor='black')
    ax3.set_title('第二类节点度分布')
    ax3.set_xlabel('度')
    ax3.set_ylabel('节点数量')
    ax3.axvline(float(np.mean(degrees2)), color=NATURE_COLORS['coral'], linestyle='--', 
                label=f'平均值: {float(np.mean(degrees2)):.1f}')
    ax3.legend()
    
    # 4. 正负边在节点中的分布
    ax4 = plt.subplot(3, 4, 4)
    ax4.patch.set_alpha(0.0)
    pos_edges = data[data[:, 2] > 0]
    neg_edges = data[data[:, 2] < 0]
    
    node1_pos_count = len(np.unique(pos_edges[:, 0]))
    node1_neg_count = len(np.unique(neg_edges[:, 0]))
    node2_pos_count = len(np.unique(pos_edges[:, 1]))
    node2_neg_count = len(np.unique(neg_edges[:, 1]))
    
    categories = ['第一类\n正边', '第一类\n负边', '第二类\n正边', '第二类\n负边']
    values = [node1_pos_count, node1_neg_count, node2_pos_count, node2_neg_count]
    colors = [NATURE_COLORS['turquoise'], NATURE_COLORS['coral'], 
             NATURE_COLORS['blue'], NATURE_COLORS['orange']]
    
    bars = ax4.bar(categories, values, color=colors)
    ax4.set_title('节点参与正负边的数量')
    ax4.set_ylabel('节点数量')
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(value), ha='center', fontweight='bold')
    
    # 5. 节点ID分布热图
    ax5 = plt.subplot(3, 4, 5)
    ax5.patch.set_alpha(0.0)
    # 创建连接矩阵（采样）
    unique_node1 = sorted(np.unique(node1_ids))
    unique_node2 = sorted(np.unique(node2_ids))
    
    # 创建小的采样矩阵进行可视化
    sample_size1 = min(50, len(unique_node1))
    sample_size2 = min(24, len(unique_node2))
    sample_node1 = unique_node1[::len(unique_node1)//sample_size1][:sample_size1]
    sample_node2 = unique_node2[:sample_size2]
    
    matrix = np.zeros((len(sample_node1), len(sample_node2)))
    for edge in data:
        if edge[0] in sample_node1 and edge[1] in sample_node2:
            i = sample_node1.index(edge[0])
            j = sample_node2.index(edge[1])
            matrix[i, j] = edge[2]
    
    im = ax5.imshow(matrix, cmap='RdBu_r', aspect='auto')  # 使用反转的RdBu配色
    ax5.set_title('连接模式热图（采样）')
    ax5.set_xlabel('第二类节点')
    ax5.set_ylabel('第一类节点')
    plt.colorbar(im, ax=ax5, label='边权重')
    
    # 6. 累积度分布
    ax6 = plt.subplot(3, 4, 6)
    ax6.patch.set_alpha(0.0)
    degrees1_sorted = sorted(degrees1, reverse=True)
    degrees2_sorted = sorted(degrees2, reverse=True)
    
    ax6.plot(range(len(degrees1_sorted)), degrees1_sorted, '-', 
            color=NATURE_COLORS['blue'], label='第一类节点', linewidth=2)
    ax6.plot(range(len(degrees2_sorted)), degrees2_sorted, '-', 
            color=NATURE_COLORS['orange'], label='第二类节点', linewidth=2)
    ax6.set_title('累积度分布')
    ax6.set_xlabel('节点排名')
    ax6.set_ylabel('度')
    ax6.legend()
    ax6.set_yscale('log')
    
    # 7. 边权重随节点ID的变化
    ax7 = plt.subplot(3, 4, 7)
    ax7.patch.set_alpha(0.0)
    pos_data = data[data[:, 2] > 0]
    neg_data = data[data[:, 2] < 0]
    
    ax7.scatter(pos_data[:, 0], pos_data[:, 1], c=NATURE_COLORS['turquoise'], 
               alpha=0.5, s=1, label='正边')
    ax7.scatter(neg_data[:, 0], neg_data[:, 1], c=NATURE_COLORS['coral'], 
               alpha=0.5, s=1, label='负边')
    ax7.set_title('边的空间分布')
    ax7.set_xlabel('第一类节点ID')
    ax7.set_ylabel('第二类节点ID')
    ax7.legend()
    
    # 8. 度分布的Box Plot
    ax8 = plt.subplot(3, 4, 8)
    ax8.patch.set_alpha(0.0)
    box_data = [degrees1, degrees2]
    box_labels = ['第一类节点', '第二类节点']
    
    # 使用正确的参数调用boxplot
    bp = ax8.boxplot(box_data, patch_artist=True,
                     medianprops=dict(color=NATURE_COLORS['coral']),
                     flierprops=dict(marker='o', markerfacecolor=NATURE_COLORS['light_blue']))
    
    # 设置x轴标签
    ax8.set_xticklabels(box_labels)
    
    # 设置箱体颜色
    bp['boxes'][0].set_facecolor(NATURE_COLORS['blue'])
    bp['boxes'][1].set_facecolor(NATURE_COLORS['orange'])
    
    ax8.set_title('度分布箱线图')
    ax8.set_ylabel('度')
    
    # 9. 邻接度相关性分析
    ax9 = plt.subplot(3, 4, 9)
    ax9.patch.set_alpha(0.0)
    # 计算每个第一类节点连接的第二类节点的度
    node1_neighbor_degrees = []
    node2_degree_dict = dict(node2_degrees)
    
    for node1 in unique_node1[:100]:  # 采样
        connected_node2s = data[data[:, 0] == node1][:, 1]
        if len(connected_node2s) > 0:
            avg_neighbor_degree = np.mean([node2_degree_dict[n2] for n2 in connected_node2s])
            node1_neighbor_degrees.append((node1_degrees[node1], avg_neighbor_degree))
    
    if node1_neighbor_degrees:
        x_vals, y_vals = zip(*node1_neighbor_degrees)
        ax9.scatter(x_vals, y_vals, alpha=0.6, color=NATURE_COLORS['turquoise'])
        ax9.set_title('度-度相关性')
        ax9.set_xlabel('第一类节点度')
        ax9.set_ylabel('邻居平均度')
        
        # 添加趋势线
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        ax9.plot(x_vals, p(x_vals), "--", color=NATURE_COLORS['coral'], alpha=0.8)
    
    # 10. 正负边比例饼图
    ax10 = plt.subplot(3, 4, 10)
    ax10.patch.set_alpha(0.0)
    edge_counts = [len(pos_edges), len(neg_edges)]
    edge_labels = [f'正边\n{len(pos_edges)}', f'负边\n{len(neg_edges)}']
    pie_result = ax10.pie(edge_counts, labels=edge_labels, 
                         colors=[NATURE_COLORS['turquoise'], NATURE_COLORS['coral']], 
                         autopct='%1.1f%%', startangle=90)
    wedges, texts = pie_result[:2]
    if len(pie_result) > 2:
        autotexts = pie_result[2]
    ax10.set_title('正负边比例')
    
    # 11. 连接模式分析
    ax11 = plt.subplot(3, 4, 11)
    ax11.patch.set_alpha(0.0)
    # 分析每个第二类节点的正负边比例
    node2_pos_neg_ratio = []
    for node2 in unique_node2:
        pos_count = len(pos_edges[pos_edges[:, 1] == node2])
        neg_count = len(neg_edges[neg_edges[:, 1] == node2])
        total = pos_count + neg_count
        if total > 0:
            pos_ratio = pos_count / total
            node2_pos_neg_ratio.append(pos_ratio)
    
    ax11.hist(node2_pos_neg_ratio, bins=20, alpha=0.7, 
             color=NATURE_COLORS['orange'], edgecolor='black')
    ax11.set_title('第二类节点正边比例分布')
    ax11.set_xlabel('正边比例')
    ax11.set_ylabel('节点数量')
    ax11.axvline(float(np.mean(node2_pos_neg_ratio)), color=NATURE_COLORS['coral'], 
                 linestyle='--', label=f'平均值: {float(np.mean(node2_pos_neg_ratio)):.3f}')
    ax11.legend()
    
    # 12. 网络统计汇总
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # 计算关键统计量
    total_nodes = len(unique_node1) + len(unique_node2)
    total_edges = len(data)
    density = total_edges / (len(unique_node1) * len(unique_node2))
    pos_ratio = len(pos_edges) / total_edges
    
    stats_text = f"""
    网络统计汇总
    
    总节点数: {total_nodes}
    第一类节点: {len(unique_node1)}
    第二类节点: {len(unique_node2)}
    
    总边数: {total_edges}
    正边数: {len(pos_edges)}
    负边数: {len(neg_edges)}
    
    密度: {density:.4f}
    正边比例: {pos_ratio:.3f}
    
    平均度:
    第一类: {np.mean(degrees1):.1f}
    第二类: {np.mean(degrees2):.1f}
    """
    
    ax12.text(0.1, 0.9, stats_text, transform=ax12.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('comprehensive_bipartite_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return data

if __name__ == "__main__":
    print("生成二分图综合分析可视化...")
    data = create_comprehensive_visualization()
    print("可视化完成！图表已保存为 'comprehensive_bipartite_analysis.png'") 