import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt


class DataLoadersvmguide1:
    """数据加载与预处理类"""

    def __init__(self, config_path=None, data_root=None, svmguide1_dir=None,
                 file_svmguide1_name=None, normalize=None, test_size=None, random_state=None):
        """
        初始化数据加载器

        参数:
            config_path: 配置文件路径
            data_root: 数据根目录
            svmguide1_dir: 数据子目录
            file_svmguide1_name: 数据文件名
            normalize: 是否标准化
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.config_path = config_path
        self.data_root = data_root
        self.svmguide1_dir = svmguide1_dir
        self.file_svmguide1_name = file_svmguide1_name
        self.normalize = normalize
        self.test_size = test_size
        self.random_state = random_state
        self.config = None
        self.scaler = None
        self.label_mapping = None
        self.feature_names = None

    def _get_config_path(self):
        """获取配置文件路径"""
        if self.config_path:
            return Path(self.config_path)
        project_root = Path("E:/徐兵鑫毕业论文/code-renew")
        config_path = project_root / "configs/v1_params.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        return config_path

    def _detect_data_file(self, data_dir: Path, base_name: str) -> Path:
        """自动检测文件类型"""
        # 直接返回指定路径（严格替换数据集路径）
        fixed_path = Path("E:/徐兵鑫毕业论文/数据集/svmguide1/svmguide1.csv")
        if fixed_path.exists():
            print(f"找到数据文件: {fixed_path}")
            return fixed_path
        raise FileNotFoundError(f"未找到数据文件: {fixed_path}")

    def _load_data_file(self, file_path: Path) -> pd.DataFrame:
        """根据扩展名加载数据文件"""
        ext = file_path.suffix.lower()
        try:
            if ext == '.csv':
                return pd.read_csv(file_path)
            elif ext == '.arff':
                data, _ = loadarff(file_path)
                df = pd.DataFrame(data)
                str_cols = df.select_dtypes([object]).columns
                df[str_cols] = df[str_cols].apply(lambda x: x.str.decode('utf-8'))
                return df
            elif ext in ('.txt', '.data'):
                return pd.read_csv(file_path, sep=None, engine='python')
            elif ext == '.xlsx':
                return pd.read_excel(file_path)
            else:
                raise ValueError(f"不支持的扩展名: {ext}")
        except Exception as e:
            raise ValueError(f"文件加载失败（{ext}格式可能不匹配）: {str(e)}")

    def _print_data_stats(self, df: pd.DataFrame, label_col: str = None):
        """打印数据集统计信息"""
        print("\n=== 数据集详细信息 ===")
        print(f"总样本数: {len(df)}")
        print(f"特征数: {df.shape[1] - 1 if label_col else df.shape[1]}")

        if label_col:
            print("\n=== 标签分布 ===")
            print(df[label_col].value_counts())

        print("\n=== 特征统计 ===")
        features = df.drop(columns=[label_col]) if label_col else df
        print(features.describe())

    def load_svmguide1(self):
        """加载并预处理数据集"""
        # 加载配置
        with open(self._get_config_path(), encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 使用参数或配置
        data_root = Path(self.data_root) if self.data_root else Path(self.config["paths"]["data_root"])
        svmguide1_dir = self.svmguide1_dir if self.svmguide1_dir else self.config["paths"]["svmguide1_dir"]
        file_svmguide1_name = self.file_svmguide1_name if self.file_svmguide1_name else self.config["data_loading"]["file_svmguide1_name"]
        normalize = self.normalize if self.normalize is not None else self.config["data_loading"]["normalize"]
        test_size = self.test_size if self.test_size is not None else self.config["data_loading"]["test_size"]
        random_state = self.random_state if self.random_state is not None else self.config["data_loading"][
            "random_state"]

        # 构建数据路径（会被_detect_data_file覆盖）
        data_dir = data_root / svmguide1_dir
        file_path = self._detect_data_file(data_dir, file_svmguide1_name)

        # 加载数据
        df = self._load_data_file(file_path)

        # 提取特征和标签
        label_col = next((col for col in ['class', 'label', 'target', 'y'] if col in df.columns), None)

        # 打印统计信息
        self._print_data_stats(df, label_col)

        X = df.drop(columns=label_col).values if label_col else df.iloc[:, :-1].values
        y = df[label_col].values if label_col else df.iloc[:, -1].values

        # 标签编码
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
        self.feature_names = df.columns[:-1].tolist() if label_col is None else df.drop(label_col,
                                                                                        axis=1).columns.tolist()

        # 特征标准化
        self.scaler = StandardScaler() if normalize else None
        X = self.scaler.fit_transform(X) if self.scaler else X

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        # 计算比例
        train_ratio = len(X_train) / (len(X_train) + len(X_test))
        test_ratio = len(X_test) / (len(X_train) + len(X_test))

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
                "file_path": str(file_path),
                "stats": {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "train_ratio": train_ratio,
                    "test_ratio": test_ratio,
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
    loader = DataLoadersvmguide1()
    result = loader.load_svmguide1()
    print("测试加载完成，样本数:", len(result['data']['train'][0]))