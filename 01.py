import pandas as pd

# 读取CSV文件
df = pd.read_csv('svmguide1.csv')

# 移除最后一列
df = df.iloc[:, :-1]

# 保存到新的CSV文件（可选）
df.to_csv('svmguide1_processed.csv', index=False)

# 显示处理后的数据（前几行）
print(df.head())