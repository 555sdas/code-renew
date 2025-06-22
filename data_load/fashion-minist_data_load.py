import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import gzip
import struct


class DataLoaderminist:
    """数据加载与预处理类（支持Fashion-MNIST）"""

    def __init__(self, config_path=None, data_root=None, dataset_dir=None,
                 file_name=None, normalize=None, test_size=None, random_state=None):
        """
        初始化数据加载器

        参数:
            config_path: 配置文件路径
            data_root: 数据根目录
            dataset_dir: 数据集子目录（如'fashion-mnist-master'）
            file_name: 数据文件名
            normalize: 是否标准化
            test_size: 测试集比例（Fashion-MNIST使用固定测试集，此参数无效）
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
        self.label_mapping = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }
        self.feature_names = [f"pixel_{i}" for i in range(784)]

    def _get_config_path(self):
        """获取配置文件路径"""
        if self.config_path:
            return Path(self.config_path)
        project_root = Path("E:/徐兵鑫毕业论文/code-renew")
        config_path = project_root / "configs/v1_params.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        return config_path

    def _load_mnist_images(self, file_path):
        """加载MNIST格式的图像文件"""
        with gzip.open(file_path, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num, rows * cols)
        return images

    def _load_mnist_labels(self, file_path):
        """加载MNIST格式的标签文件"""
        with gzip.open(file_path, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def load_fashion_mnist(self):
        """加载Fashion-MNIST数据集"""
        # 加载配置
        with open(self._get_config_path(), encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 使用参数或配置
        data_root = Path(self.data_root) if self.data_root else Path(self.config["paths"]["data_root"])
        dataset_dir = self.dataset_dir if self.dataset_dir else "fashion-mnist-master"
        normalize = self.normalize if self.normalize is not None else self.config["data_loading"]["normalize"]

        # 构建数据路径
        data_dir = data_root / dataset_dir / "data" / "fashion"

        # 加载训练集
        X_train = self._load_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
        y_train = self._load_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")

        # 加载测试集
        X_test = self._load_mnist_images(data_dir / "t10k-images-idx3-ubyte.gz")
        y_test = self._load_mnist_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

        # 特征标准化
        self.scaler = StandardScaler() if normalize else None
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        else:
            # 即使不标准化，也至少缩放到[0,1]范围
            X_train = X_train / 255.0
            X_test = X_test / 255.0

        # 打印统计信息
        print("\n=== Fashion-MNIST数据集统计 ===")
        print(f"训练集样本数: {len(X_train)}")
        print(f"测试集样本数: {len(X_test)}")
        print(f"特征维度: {X_train.shape[1]}")
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

    # 保留原有方法以兼容其他数据集
    def load_fourclass(self):
        """原有方法保持不变"""
        # ... (原有代码不变)


# 测试代码
if __name__ == "__main__":
    loader = DataLoaderminist(data_root="E:/徐兵鑫毕业论文/数据集")
    result = loader.load_fashion_mnist()

    # 可视化样本示例
    X_train, y_train = result['data']['train']
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
        plt.title(result['metadata']['label_mapping'][y_train[i]])  # 修改这里
        plt.axis('off')
    plt.show()