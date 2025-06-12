import pandas as pd
from pathlib import Path

# 1. 下载数据
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
df = pd.read_csv(url, header=None)

# 2. 添加列名（根据UCI的names文件）
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises',
    'odor', 'gill-attachment', 'gill-spacing', 'gill-size',
    'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-type', 'veil-color',
    'ring-number', 'ring-type', 'spore-print-color',
    'population', 'habitat'
]
df.columns = columns

# 3. 创建保存目录（如果不存在）
save_dir = Path('./datasets')
save_dir.mkdir(exist_ok=True)

# 4. 保存到指定路径
save_path = save_dir / 'mushroom_processed.csv'
df.to_csv(save_path, index=False, encoding='utf-8')

print(f"数据集已保存到：{save_path.absolute()}")