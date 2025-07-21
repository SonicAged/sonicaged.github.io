"""
多组学按照id升序排列(MOFGCN)里是要升序排列才能对应
"""
import pandas as pd

# 组学文件路径
omics_files = [
    'data/Celline/genomic_expression_561celllines_697genes_demap_features.csv',
    'data/Celline/genomic_mutation_34673_demap_features.csv',
    'data/Celline/genomic_methylation_561celllines_808genes_demap_features.csv'
]

# 读取IC50binary的列名（细胞系id），去除引号
ic50_binary_file = 'data/Celline/GDSC_IC50_binary.csv'
with open(ic50_binary_file, 'r', encoding='utf-8') as f:
    header = f.readline().strip().split(',')
final_cellline_ids = [x.strip('"') for x in header[1:]]  # 跳过第一个""

for omics_file in omics_files:
    # 读取组学数据为DataFrame
    df = pd.read_csv(omics_file, header=0)
    df = df.set_index(df.columns[0])
    # 只保留并按顺序排列final_cellline_ids
    df_reordered = df.loc[final_cellline_ids]
    # 保存
    out_file = omics_file.replace('.csv', '_sorted.csv')
    df_reordered.to_csv(out_file)
    print(f'已保存: {out_file}')
