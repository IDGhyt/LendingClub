import os
import pandas as pd

# 创建输出目录
os.makedirs('outputs/data', exist_ok=True)

# 仅加载核心字段和前100万行
cols = ['loan_amnt', 'int_rate', 'grade', 'annual_inc', 'dti', 
        'fico_range_low', 'loan_status', 'total_pymnt', 'term', 'issue_d']

print("正在加载数据...")
df = pd.read_csv('outputs/data/accepted_2007_to_2018Q4.csv.gz',
                 compression='gzip', 
                 usecols=cols, 
                 nrows=1_000_000,
                 parse_dates=['issue_d'])

# 保存精简文件
df.to_csv('outputs/data/loan_quick.csv', index=False)
print(f"精简数据保存成功，维度: {df.shape}")
print("文件路径: outputs/data/loan_quick.csv")