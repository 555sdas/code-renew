import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import yaml
import os


class DataLoaderEcoli:
    """数据加载与预处理类（支持E.coli数据集）"""

    def __init__(self, config_path=None, data_root=None, dataset_dir=None,
                 file_name=None, normalize=None, test_size=0.2, random_state=42):
        """
        初始化数据加载器

        参数:
            config_path: 配置文件路径
            data_root: 数据根目录
            dataset_dir: 数据集子目录
            file_name: 数据文件名
            normalize: 是否标准化
            test_size: 测试集比例（默认20%）
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
            'cp': 'cytoplasmic membrane',
            'im': 'inner membrane',
            'pp': 'periplasmic',
            'imU': 'inner membrane, uncharacterized',
            'om': 'outer membrane',
            'omL': 'outer membrane lipoprotein',
            'imL': 'inner membrane lipoprotein',
            'imS': 'inner membrane, signal peptide'
        }  # 更详细的标签映射
        self.feature_names = [f"Feature_{i}" for i in range(1, 8)]  # 7个特征

    def _get_config_path(self):
        """获取配置文件路径"""
        if self.config_path:
            return Path(self.config_path)
        project_root = Path("E:/徐兵鑫毕业论文/code-renew")
        config_path = project_root / "configs/v1_params.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        return config_path

    def _load_ecoli_data(self, file_path):
        """加载E.coli数据集文件"""
        df = pd.read_csv(file_path, header=None)

        # 第一列为蛋白质ID（丢弃），第2-8列为特征，最后一列为类别标签
        X = df.iloc[:, 1:-1].values.astype(float)  # 特征 (n_samples, 7)
        y = df.iloc[:, -1].values  # 原始标签 (如 'cp', 'im')

        # 将原始标签映射到更详细的描述
        y_mapped = np.array([self.label_mapping.get(label, label) for label in y])

        return X, y_mapped

    def loadecoli_data(self):
        """加载并预处理数据"""
        # 加载配置
        if self._get_config_path().exists():
            with open(self._get_config_path(), encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

        # 使用参数或配置
        data_root = Path(self.data_root) if self.data_root else Path(self.config["paths"]["data_root"])
        dataset_dir = self.dataset_dir if self.dataset_dir else "ecoli_data"
        file_name = self.file_name if self.file_name else "ecoli_data.csv"
        normalize = self.normalize if self.normalize is not None else self.config["data_loading"]["normalize"]

        # 构建数据路径
        data_path = data_root / dataset_dir / file_name

        # 加载数据
        X, y = self._load_ecoli_data(data_path)

        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # 特征标准化
        self.scaler = StandardScaler() if normalize else None
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        # 打印统计信息
        print("\n=== E.coli数据集统计 ===")
        print(f"总样本数: {len(X)}")
        print(f"训练集样本数: {len(X_train)}")
        print(f"测试集样本数: {len(X_test)}")
        print(f"特征维度: {X_train.shape[1]} (7个特征)")
        print("\n=== 训练集标签分布 ===")
        print(pd.Series(y_train).value_counts())
        print("\n=== 测试集标签分布 ===")
        print(pd.Series(y_test).value_counts())

        return {
            "data": {
                "train": (X_train, y_train),
                "test": (X_test, y_test),
                "full": (X, y)
            },
            "metadata": {
                "feature_names": self.feature_names,
                "label_mapping": self.label_mapping,
                "scaler": self.scaler,
                "file_path": str(data_path),
                "stats": {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "train_ratio": len(X_train) / len(X),
                    "test_ratio": len(X_test) / len(X),
                    "feature_stats": pd.DataFrame(X_train, columns=self.feature_names).describe().to_dict(),
                    "class_distribution": {
                        "train": dict(zip(*np.unique(y_train, return_counts=True))),
                        "test": dict(zip(*np.unique(y_test, return_counts=True)))
                    }
                }
            },
            "config": self.config
        }

    def plot_feature_distribution(self, X, y, feature_idx=0):
        """绘制指定特征的分布直方图"""
        feature_name = self.feature_names[feature_idx]
        df = pd.DataFrame(X, columns=self.feature_names)
        df['Label'] = y

        plt.figure(figsize=(10, 6))
        for label in df['Label'].unique():
            plt.hist(df[df['Label'] == label][feature_name], bins=30, alpha=0.5, label=label)
        plt.title(f"Feature {feature_idx + 1} ({feature_name}) Distribution")
        plt.xlabel(feature_name)
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()


# 测试代码
if __name__ == "__main__":
    loader = DataLoaderEcoli(
        config_path=None,
        data_root="E:/徐兵鑫毕业论文/数据集",
        dataset_dir="ecoli_data",
        file_name="ecoli_data.csv",
        normalize=True
    )
    result = loader.loadecoli_data()

    # 打印标准化后的统计信息
    X_train, y_train = result['data']['train']
    X_test, y_test = result['data']['test']

    print("\n=== 标准化后训练集统计 ===")
    print(pd.DataFrame(X_train, columns=result['metadata']['feature_names']).describe().round(2))

    print("\n=== 标准化后测试集统计 ===")
    print(pd.DataFrame(X_test, columns=result['metadata']['feature_names']).describe().round(2))

    # 可视化第一个特征的分布
    loader.plot_feature_distribution(result['data']['full'][0], result['data']['full'][1], feature_idx=0)