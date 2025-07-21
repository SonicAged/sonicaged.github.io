"""
将数据集的行和列互换
"""
import pandas as pd

files = [
    'data/Celline/GDSC_IC50_binary.csv',
    'data/Celline/null_mask.csv'
]

for file in files:
    df = pd.read_csv(file, index_col=0)
    df_T = df.T
    out_file = file.replace('.csv', '_row_celline.csv')
    df_T.to_csv(out_file)
    print(f'已保存: {out_file}')
