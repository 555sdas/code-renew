import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class DataLoaderPen:
    """数据加载与预处理类（支持Pen-Based手写数字数据集）"""

    def __init__(self, config_path=None, data_root=None, dataset_dir=None,
                 file_name=None, normalize=None, test_size=None, random_state=None):
        """
        初始化数据加载器

        参数:
            config_path: 配置文件路径
            data_root: 数据根目录
            dataset_dir: 数据集子目录
            file_name: 数据文件名
            normalize: 是否标准化
            test_size: 测试集比例（Pen数据集使用固定测试集，此参数无效）
            random_state: 随机种子
        """
        self.config_path = config_path
        self.data_root = data_root
        self.dataset_dir = dataset_dir
        self.file_name = file_name
        self.normalize = normalize
        self.test_size = test_size
        self.random_state = random_state
        self.config = None
        self.scaler = None
        self.label_mapping = {i: str(i) for i in range(10)}  # 数字0-9的标签映射
        self.feature_names = [f"{coord}_{i // 2 + 1}" for i in range(16)
                              for coord in ('x', 'y')]  # x1,y1,x2,y2,...,x8,y8

    def _get_config_path(self):
        """获取配置文件路径"""
        if self.config_path:
            return Path(self.config_path)
        project_root = Path("E:/徐兵鑫毕业论文/code-renew")
        config_path = project_root / "configs/v1_params.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        return config_path

    def _load_pen_data(self, file_path):
        """加载Pen数据集文件"""
        data = np.loadtxt(file_path, delimiter=',')
        X = data[:, :-1]  # 特征（16维坐标）
        y = data[:, -1]  # 标签（0-9）
        return X, y

    def load_pen(self):
        """加载Pen-Based手写数字数据集"""
        # 加载配置
        with open(self._get_config_path(), encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 使用参数或配置
        data_root = Path(self.data_root) if self.data_root else Path(self.config["paths"]["data_root"])
        dataset_dir = self.dataset_dir if self.dataset_dir else "pen-based"
        normalize = self.normalize if self.normalize is not None else self.config["data_loading"]["normalize"]

        # 构建数据路径
        data_dir = data_root
        train_path = data_root / "pendigits.tra"
        test_path = data_root / "pendigits.tes"

        # 加载数据
        X_train, y_train = self._load_pen_data(train_path)
        X_test, y_test = self._load_pen_data(test_path)

        # 特征标准化
        self.scaler = StandardScaler() if normalize else None
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        # 打印统计信息
        print("\n=== Pen-Based数据集统计 ===")
        print(f"训练集样本数: {len(X_train)} (应为7,494)")
        print(f"测试集样本数: {len(X_test)} (应为3,498)")
        print(f"特征维度: {X_train.shape[1]} (应为16)")
        print("\n=== 训练集标签分布 ===")
        print(pd.Series(y_train).value_counts().sort_index())
        print("\n=== 测试集标签分布 ===")
        print(pd.Series(y_test).value_counts().sort_index())

        return {
            "data": {
                "train": (X_train, y_train),
                "test": (X_test, y_test),
                "full": (np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]))
            },
            "metadata": {
                "feature_names": self.feature_names,
                "label_mapping": self.label_mapping,
                "scaler": self.scaler,
                "file_path": str(data_dir),
                "stats": {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "train_ratio": len(X_train) / (len(X_train) + len(X_test)),
                    "test_ratio": len(X_test) / (len(X_train) + len(X_test)),
                    "feature_stats": pd.DataFrame(X_train).describe().to_dict(),
                    "class_distribution": {
                        "train": dict(zip(*np.unique(y_train, return_counts=True))),
                        "test": dict(zip(*np.unique(y_test, return_counts=True)))
                    }
                }
            },
            "config": self.config
        }

    def plot_pen_sample(self, X, y, index=0):
        """绘制单个手写数字的轨迹"""
        x_coords = X[index][::2]  # 奇数位是x坐标
        y_coords = X[index][1::2]  # 偶数位是y坐标

        plt.figure(figsize=(6, 6))
        plt.plot(x_coords, y_coords, 'bo-')
        plt.title(f"Label: {int(y[index])}")
        plt.gca().invert_yaxis()  # 反转y轴（数字板坐标系）
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.grid(True)
        plt.show()


# 测试代码
if __name__ == "__main__":
    loader = DataLoaderPen(data_root="E:\徐兵鑫毕业论文\数据集\pen", normalize=True)
    result = loader.load_pen()

    # 打印标准化后的统计信息
    X_train, y_train = result['data']['train']
    X_test, y_test = result['data']['test']

    print("\n=== 标准化后训练集统计 ===")
    print(pd.DataFrame(X_train).describe().round(2))

    print("\n=== 标准化后测试集统计 ===")
    print(pd.DataFrame(X_test).describe().round(2))