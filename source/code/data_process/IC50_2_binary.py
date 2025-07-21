"""
将IC50数据转换为二进制数据
"""
import csv
import numpy as np
import pandas as pd
import math

# 文件路径
ic50_file = 'data/Celline/GDSC_IC50.csv'
thred_file = 'data/Drug/drug_threshold.txt'
drug_info_file = 'data/Drug/1.Drug_listMon Jun 24 09_00_55 2019.csv'
cellline_info_file = 'data/Celline/Cell_lines_annotations.txt'
mutation_file = 'data/Celline/genomic_mutation_34673_demap_features.csv'
gexpr_file = 'data/Celline/genomic_expression_561celllines_697genes_demap_features.csv'

# 读取阈值表
drug2thred = {}
with open(thred_file, 'r', encoding='utf-8') as f:
    for line in f.readlines()[1:]:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        drug2thred[str(parts[0])] = float(parts[1])

# 读取pubchem药物id映射
drugid2pubchemid = {}
with open(drug_info_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) > 5 and row[5].isdigit():
            drugid2pubchemid[row[0]] = row[5]

# 读取cellline2cancertype
cellline2cancertype = {}
with open(cellline_info_file, 'r', encoding='utf-8') as f:
    for line in f.readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        cellline2cancertype[cellline_id] = TCGA_label

# 读取mutation和gexpr的index
mutation_feature = pd.read_csv(mutation_file, sep=',', header=0, index_col=0)
gexpr_feature = pd.read_csv(gexpr_file, sep=',', header=0, index_col=0)
# 只保留在gexpr中的细胞系
gexpr_cellline_set = set([str(idx) for idx in gexpr_feature.index])
mutation_cellline_set = set([str(idx) for idx in mutation_feature.index])
final_cellline_set = gexpr_cellline_set & mutation_cellline_set

# 读取IC50表
with open(ic50_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    rows = list(reader)
cellline_ids = rows[0][1:]
all_drug_ids = [row[0] for row in rows[1:]]
ic50_mat = np.array([row[1:] for row in rows[1:]], dtype=object)

# 先整体过滤
keep_drug_idx = []
filtered_drug_ids = []
for i, d in enumerate(all_drug_ids):
    drug_id = d.split(':')[-1]
    pubchem_id = drugid2pubchemid.get(drug_id, None)
    if pubchem_id is not None and pubchem_id in drug2thred:
        keep_drug_idx.append(i)
        filtered_drug_ids.append(d)
# 只保留cellline_id在cellline2cancertype和gexpr和mutation交集中的细胞系
keep_cell_idx = [i for i, c in enumerate(cellline_ids) if c in cellline2cancertype and c in final_cellline_set]
filtered_cellline_ids = [cellline_ids[i] for i in keep_cell_idx]
filtered_ic50 = ic50_mat[np.ix_(keep_drug_idx, keep_cell_idx)]

# 生成所有(pubchem_id, cellline_id, binary, mask)元组
records = []
for i, drug in enumerate(filtered_drug_ids):
    drug_id = drug.split(':')[-1]
    pubchem_id = drugid2pubchemid[drug_id]
    thred = drug2thred.get(pubchem_id, None)
    for j, cell in enumerate(filtered_cellline_ids):
        val = filtered_ic50[i, j]
        v = float('nan')
        try:
            v = float(val)
        except:
            pass
        if math.isnan(v):
            binary = 0.0
            mask = 1
        else:
            if thred is not None:
                binary = 1.0 if v < thred else 0.0
            else:
                binary = 0.0
            mask = 0
        records.append((cell, pubchem_id, binary, mask))

# 去重，仿照data_load.py消除重复
records_sorted = sorted(records, key=lambda x: [x[0], x[1], x[2]], reverse=True)
data_tmp = set()
data_new = []
data_idx1 = [(i[0], i[1]) for i in records_sorted]
for idx, k in zip(data_idx1, records_sorted):
    if idx not in data_tmp:
        data_tmp.add(idx)
        data_new.append(k)

# 重新组织为DataFrame
final_cellline_ids = sorted(list(set([item[0] for item in data_new])))
final_drug_ids = sorted(list(set([item[1] for item in data_new])))
binary_matrix = np.zeros((len(final_drug_ids), len(final_cellline_ids)), dtype=float)
mask_matrix = np.zeros((len(final_drug_ids), len(final_cellline_ids)), dtype=int)
cellline_id_to_idx = {cid: i for i, cid in enumerate(final_cellline_ids)}
drug_id_to_idx = {did: i for i, did in enumerate(final_drug_ids)}
for cell, pubchem_id, binary, mask in data_new:
    i = drug_id_to_idx[pubchem_id]
    j = cellline_id_to_idx[cell]
    binary_matrix[i, j] = binary
    mask_matrix[i, j] = mask

binary_df = pd.DataFrame(binary_matrix, index=final_drug_ids, columns=final_cellline_ids)
mask_df = pd.DataFrame(mask_matrix, index=final_drug_ids, columns=final_cellline_ids)

# 打印统计
print(f'原始药物数: {len(all_drug_ids)}')
print(f'原始细胞系数: {len(cellline_ids)}')
print(f'被筛掉的药物数: {len(all_drug_ids) - len(final_drug_ids)}')
print(f'被筛掉的细胞系数: {len(cellline_ids) - len(final_cellline_ids)}')
print(f'药物数（行）: {binary_df.shape[0]}')
print(f'细胞系数（列）: {binary_df.shape[1]}')
print(f'NA总个数: {int(mask_df.values.sum())}')

total = mask_df.size  # 总单元格数 = 药物数 × 细胞系数
na_count = int(mask_df.values.sum())  # NA的个数
valid_count = total - na_count        # 有效的个数
print(f'有效的个数: {valid_count}')

binary_df.to_csv('data/Celline/GDSC_IC50_binary.csv')
mask_df.to_csv('data/Celline/null_mask.csv')
print('已生成 GDSC_IC50_binary.csv 和 null_mask.csv')
