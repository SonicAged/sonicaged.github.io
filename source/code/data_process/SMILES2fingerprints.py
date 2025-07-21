"""
将SMILES转换为药物指纹
"""
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# 输入输出路径
smiles_file = 'data/Drug/222drugs_pubchem_smiles.txt'
output_csv = 'data/Drug/drug_fingerprints.csv'

# 读取SMILES信息
drug_ids = []
smiles_list = []
with open(smiles_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        drug_ids.append(parts[0])
        smiles_list.append(parts[1])

# 生成指纹
fp_size = 2048
fingerprints = []
for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        # 若SMILES非法，填充全0
        fingerprints.append([0]*fp_size)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
        arr = list(fp)
        fingerprints.append(arr)

# 构建DataFrame并保存
fp_df = pd.DataFrame(fingerprints)
fp_df.index = drug_ids  # 行索引为药物id
fp_df.to_csv(output_csv, index=True, header=[str(i) for i in range(fp_size)])
print(f'已保存到 {output_csv}')

# 药物个数: 222
# 指纹特征维度: 2048