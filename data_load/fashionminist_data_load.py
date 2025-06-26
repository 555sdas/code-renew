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
            dataset_dir: 数据集子目录
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
        try:
            with gzip.open(file_path, 'rb') as f:
                magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
                if magic != 2051:
                    raise ValueError(f"无效的图像文件魔数: {magic} (应为2051)")

                images = np.frombuffer(f.read(), dtype=np.uint8)
                if len(images) != num * rows * cols:
                    raise ValueError("图像数据长度与头信息不匹配")

                images = images.reshape(num, rows * cols)
                return images.astype(np.float32)  # 明确指定数据类型
        except Exception as e:
            raise RuntimeError(f"加载图像文件失败: {file_path}") from e

    def _load_mnist_labels(self, file_path):
        """加载MNIST格式的标签文件"""
        try:
            with gzip.open(file_path, 'rb') as f:
                magic, num = struct.unpack(">II", f.read(8))
                if magic != 2049:
                    raise ValueError(f"无效的标签文件魔数: {magic} (应为2049)")

                labels = np.frombuffer(f.read(), dtype=np.uint8)
                if len(labels) != num:
                    raise ValueError("标签数据长度与头信息不匹配")

                return labels
        except Exception as e:
            raise RuntimeError(f"加载标签文件失败: {file_path}") from e

    def _validate_data(self, X, y, dataset_name):
        """验证数据集完整性"""
        if len(X) != len(y):
            raise ValueError(f"{dataset_name}数据长度不匹配: 图像{len(X)} != 标签{len(y)}")

        if X.shape[1] != 784:
            raise ValueError(f"{dataset_name}特征维度应为784，实际为{X.shape[1]}")

        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, np.arange(10)):
            raise ValueError(f"{dataset_name}标签应包含0-9，实际为{unique_labels}")

    def _validate_data_range(self, data, expected_min=0, expected_max=255):
        """验证数据范围"""
        actual_min, actual_max = np.min(data), np.max(data)
        if actual_min < expected_min or actual_max > expected_max:
            raise ValueError(
                f"数据范围异常: 应为[{expected_min}, {expected_max}]，实际[{actual_min:.2f}, {actual_max:.2f}]"
            )

        # 检查异常值比例
        outlier_ratio = np.mean((data < expected_min) | (data > expected_max))
        if outlier_ratio > 0:
            print(f"警告: 发现{outlier_ratio:.2%}的异常值超出[{expected_min}, {expected_max}]范围")

    def _preprocess_data(self, X_train, X_test, normalize=True):
        """数据预处理流程"""
        print("\n=== 数据预处理 ===")

        # 1. 验证原始数据范围
        self._validate_data_range(X_train)
        self._validate_data_range(X_test)

        # 2. 缩放到[0,1]范围
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        print(f"缩放后范围 - 训练集: [{X_train.min():.2f}, {X_train.max():.2f}]")
        print(f"缩放后范围 - 测试集: [{X_test.min():.2f}, {X_test.max():.2f}]")

        # 3. 标准化处理
        if normalize:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            print("\n=== 标准化验证 ===")
            print(f"训练集均值: {X_train.mean():.4f} (应接近0)")
            print(f"训练集标准差: {X_train.std():.4f} (应接近1)")
            print(f"测试集均值: {X_test.mean():.4f}")
            print(f"测试集标准差: {X_test.std():.4f}")

            # 验证标准化后范围
            self._validate_standardized_data(X_train, "训练集")
            self._validate_standardized_data(X_test, "测试集")

        return X_train, X_test

    def _validate_standardized_data(self, data, dataset_name):
        """验证标准化后数据分布"""
        threshold = 5  # 通常标准化数据应在[-5,5]范围内
        outlier_ratio = np.mean((data < -threshold) | (data > threshold))

        if outlier_ratio > 0.01:  # 超过1%的异常值
            print(f"警告: {dataset_name}有{outlier_ratio:.2%}的值超出[{-threshold}, {threshold}]范围")

        # 打印极端值统计
        extreme_values = data[(data < -10) | (data > 10)]
        if len(extreme_values) > 0:
            print(f"极端值统计({dataset_name}):")
            print(f"  最小值: {np.min(data):.2f}, 最大值: {np.max(data):.2f}")
            print(f"  极端值数量: {len(extreme_values)} (占总数的{len(extreme_values) / len(data):.2%})")

    def load_fashion_mnist(self):
        """加载Fashion-MNIST数据集"""
        # 加载配置
        with open(self._get_config_path(), encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 使用参数或配置
        data_root = Path(self.data_root) if self.data_root else Path(self.config["paths"]["data_root"])
        dataset_dir = self.dataset_dir if self.dataset_dir else "fashion-mnist-master"
        normalize = self.normalize if self.normalize is not None else self.config["data_loading"]["normalize"]

        # 查找数据目录
        possible_paths = [
            data_root / dataset_dir / "data" / "fashion",
            data_root / "fashion-mnist",
            data_root / "FashionMNIST",
            data_root / "data"
        ]

        data_dir = None
        for path in possible_paths:
            if (path / "train-images-idx3-ubyte.gz").exists():
                data_dir = path
                break

        if data_dir is None:
            raise FileNotFoundError(f"未找到Fashion-MNIST数据文件，已尝试以下路径:\n{possible_paths}")

        # 加载原始数据
        print("\n=== 正在加载数据 ===")
        X_train = self._load_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
        y_train = self._load_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")
        X_test = self._load_mnist_images(data_dir / "t10k-images-idx3-ubyte.gz")
        y_test = self._load_mnist_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

        # 验证数据完整性
        self._validate_data(X_train, y_train, "训练集")
        self._validate_data(X_test, y_test, "测试集")

        # 数据预处理
        X_train, X_test = self._preprocess_data(X_train, X_test, normalize)

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


# 测试代码
if __name__ == "__main__":
    # 测试两种预处理模式
    for normalize in [True, False]:
        print(f"\n{'=' * 40}")
        print(f"=== 测试模式: normalize={normalize} ===")
        print(f"{'=' * 40}")

        try:
            loader = DataLoaderminist(
                data_root="E:/徐兵鑫毕业论文/数据集",
                normalize=normalize
            )
            result = loader.load_fashion_mnist()

            # 可视化样本
            X_train, y_train = result['data']['train']
            plt.figure(figsize=(10, 10))
            for i in range(25):
                plt.subplot(5, 5, i + 1)
                plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
                plt.title(result['metadata']['label_mapping'][y_train[i]])
                plt.axis('off')
            plt.suptitle(f"Normalize={normalize}", y=0.92)
            plt.show()

        except Exception as e:
            print(f"测试失败: {str(e)}")
            raise