import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


class DataLoadermushroom:
    """数据加载与预处理类"""

    def __init__(self, config_path=None, data_root=None, mushroom_dir=None,
                 file_mushroom_name=None, normalize=None, test_size=None, random_state=None):
        """
        初始化数据加载器

        参数:
            config_path: 配置文件路径
            data_root: 数据根目录
            mushroom_dir: 数据子目录
            file_mushroom_name: 数据文件名
            normalize: 是否标准化
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.config_path = config_path
        self.data_root = data_root
        self.mushroom_dir = mushroom_dir
        self.file_mushroom_name = file_mushroom_name
        self.normalize = normalize
        self.test_size = test_size
        self.random_state = random_state
        self.config = None
        self.scaler = None
        self.label_mapping = None
        self.feature_names = None
        self.preprocessor = None

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
        extensions = ['.csv', '.arff', '.data', '.txt', '.xlsx']
        for ext in extensions:
            file_path = data_dir / f"{base_name}{ext}"
            if file_path.exists():
                print(f"找到数据文件: {file_path}")
                return file_path
        raise FileNotFoundError(f"未找到{base_name}开头的数据文件，尝试了以下扩展名: {extensions}")

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
        print(features.describe(include='all'))

    def load_mushroom(self):
        """加载并预处理mushroom数据集"""
        # 加载配置
        with open(self._get_config_path(), encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 使用参数或配置
        data_root = Path(self.data_root) if self.data_root else Path(self.config["paths"]["data_root"])
        mushroom_dir = self.mushroom_dir if self.mushroom_dir else self.config["paths"]["mushroom_dir"]
        file_mushroom_name = self.file_mushroom_name if self.file_mushroom_name else self.config["data_loading"]["file_mushroom_name"]
        normalize = self.normalize if self.normalize is not None else self.config["data_loading"]["normalize"]
        test_size = self.test_size if self.test_size is not None else self.config["data_loading"]["test_size"]
        random_state = self.random_state if self.random_state is not None else self.config["data_loading"][
            "random_state"]

        # 构建数据路径
        data_dir = data_root / mushroom_dir
        file_path = self._detect_data_file(data_dir, file_mushroom_name)

        # 加载数据
        df = self._load_data_file(file_path)

        # 提取特征和标签
        label_col = next((col for col in ['class', 'label', 'target', 'y'] if col in df.columns), None)

        # 打印统计信息
        self._print_data_stats(df, label_col)

        # 分离特征和标签
        X = df.drop(columns=label_col) if label_col else df.iloc[:, :-1]
        y = df[label_col] if label_col else df.iloc[:, -1]

        # 标签编码
        le = LabelEncoder()
        y = le.fit_transform(y)
        self.label_mapping = dict(zip(le.classes_, range(len(le.classes_))))
        self.feature_names = X.columns.tolist()

        # 预处理管道 - 对所有特征进行one-hot编码
        categorical_features = X.columns.tolist()  # 所有特征都是分类的
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)
            ])

        # 应用预处理
        X_processed = self.preprocessor.fit_transform(X)

        # 如果需要标准化
        if normalize:
            self.scaler = StandardScaler(with_mean=False)  # 稀疏矩阵不能中心化
            X_processed = self.scaler.fit_transform(X_processed)

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        # 计算比例 - 使用shape[0]而不是len()来处理稀疏矩阵
        total_samples = X_train.shape[0] + X_test.shape[0]
        train_ratio = X_train.shape[0] / total_samples
        test_ratio = X_test.shape[0] / total_samples

        return {
            "data": {
                "train": (X_train, y_train),
                "test": (X_test, y_test),
                "full": (X_processed, y)
            },
            "metadata": {
                "feature_names": self.feature_names,
                "label_mapping": self.label_mapping,
                "preprocessor": self.preprocessor,
                "scaler": self.scaler,
                "file_path": str(file_path),
                "stats": {
                    "train_samples": X_train.shape[0],
                    "test_samples": X_test.shape[0],
                    "train_ratio": train_ratio,
                    "test_ratio": test_ratio,
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
    loader = DataLoadermushroom()
    result = loader.load_mushroom()
    print("测试加载完成，训练样本数:", result['metadata']['stats']['train_samples'])
    print("测试样本数:", result['metadata']['stats']['test_samples'])