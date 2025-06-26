import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 提取指定数据集
selected_data = {
    'Datasets': ['Ecoli', 'fourclass', 'svmguide1', 'Pen', 'Cod-RNA', 'Skin-Seg', 'Mean'],
     'CMQGB-TWDC': ['0.8971 ±0.000', '1.000 ±0.000', '0.9661 ±0.000', '0.9717 ±0.00', '0 ±0.000', '0.997 ±0.000',
                '0.885 ±0.029'],
    'GBkNN++': ['0.863 ±0.049', '0.992 ±0.009', '0.960 ±0.008', '0.989 ±0.003', '0.895 ±0.004', '0.995 ±0.002',
                '0.885 ±0.029'],
    'ORI-GBkNN': ['0.854 ±0.043', '0.993 ±0.011', '0.960 ±0.010', '0.988 ±0.002', '0.894 ±0.005', '0.998 ±0.001',
                  '0.836 ±0.031'],
    'ACC-GBkNN': ['0.836 ±0.038', '0.986 ±0.011', '0.955 ±0.008', '0.986 ±0.003', '0.861 ±0.004', '0.996 ±0.001',
                  '0.823 ±0.029'],
    'kNN': ['0.850 ±0.021', '0.998 ±0.001', '0.963 ±0.004', '0.991 ±0.002', '0.918 ±0.021', '0.998 ±0.000',
            '0.864 ±0.070'],
    'CART': ['0.777 ±0.082', '0.987 ±0.021', '0.959 ±0.010', '0.961 ±0.006', '0.949 ±0.004', '0.999 ±0.001',
             '0.836 ±0.033'],
    'SVM': ['0.768 ±0.026', '0.797 ±0.029', '0.954 ±0.007', '0.978 ±0.005', '0.943 ±0.004', '0.970 ±0.002',
            '0.807 ±0.021']
}

# 创建DataFrame
df = pd.DataFrame(selected_data)
df.set_index('Datasets', inplace=True)

# 创建图形
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis('off')

# 创建表格
table = ax.table(
    cellText=df.values,
    rowLabels=df.index,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(11)

# 设置边框样式（三线表格式）
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('white')  # 默认隐藏所有边框

    # 顶部粗线
    if row == 0:
        cell.set_edgecolor('black')
        cell.set_linewidth(2.0)
        cell.get_text().set_fontweight('bold')

    # 底部粗线
    elif row == len(df):
        cell.set_edgecolor('black')
        cell.set_linewidth(2.0)

    # 数据集名称下面的线
    elif row == 1 and col == -1:
        cell.set_edgecolor('black')
        cell.set_linewidth(1.0)

    # Mean行上面的线
    elif row == len(df) - 1 and col >= 0:
        cell.set_edgecolor('black')
        cell.set_linewidth(1.0)

    # 行标题加粗
    if col == -1:
        cell.get_text().set_fontweight('bold')

# 调整列宽
table.auto_set_column_width([i for i in range(len(df.columns) + 1)])

# 添加表格标题
plt.title('Model Performance Comparison (Selected Datasets)', pad=20, fontsize=12, fontweight='bold')

# 保存图像
plt.savefig('selected_datasets_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()