import numpy as np
from typing import Dict, Tuple, List
from granular_ball.v4_granular_ball_信息熵优化 import GranularBallV4
from three_way_decision.v1_three_way_decision_固定阈值 import ThreeWayDecisionV1
from utils.evaluater import ThreeWayEvaluator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from data_load.fourclass_data_load import DataLoader
from data_load.mushroom_data_load import DataLoadermushroom
from data_load.svmguide1_data_load import DataLoadersvmguide1
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.environ['LOKY_MAX_CPU_COUNT'] = '8'  # 设置并行线程数

class GranularThreeWayClassifierV4:
    """
    基于信息熵优化的粒球三支分类器V4
    改进点：
    1. 集成GBkNN和GBSVM作为基分类器
    2. 动态三支决策阈值调整
    3. 粒球密度加权决策
    """

    def __init__(self,
                 max_entropy: float = 0.8,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 radius_shrink_factor: float = 0.9):
        self.gb_model = GranularBallV4(
            max_entropy=max_entropy,
            radius_shrink_factor=radius_shrink_factor
        )
        self.tw_model = ThreeWayDecisionV1(alpha=alpha, beta=beta)
        self.ensemble_model = None
        self.ball_stats_ = []
        self.feature_importances_ = None

        # [DEBUG] 初始化信息
        print(f"[Classifier] 初始化: alpha={alpha}, beta={beta}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """训练粒球模型和集成分类器"""
        self.gb_model.fit(X_train, y_train)

        # 收集粒球统计信息
        self.ball_stats_ = []
        for ball_info in self.gb_model.balls_:
            center, radius, indices, _, density = ball_info
            labels = y_train[indices]
            entropy_val = self.gb_model._calculate_entropy(labels)
            majority_label = self.gb_model._get_majority_label(labels)

            self.ball_stats_.append({
                'center': center,
                'radius': radius,
                'indices': indices,
                'density': density,
                'entropy': entropy_val,
                'majority_label': majority_label
            })

        # 构建集成分类器
        self._build_ensemble(X_train, y_train)
        self.feature_importances_ = self._calculate_feature_importance(X_train)
        return self._get_training_report()

    def _build_ensemble(self, X: np.ndarray, y: np.ndarray):
        """构建GBkNN和GBSVM集成模型"""
        # 提取粒球特征
        ball_features = []
        ball_labels = []

        # 统计每个粒球的类别分布
        for stats in self.ball_stats_:
            # 获取粒球内样本的真实标签
            ball_labels_in_ball = y[stats['indices']]
            unique_classes = np.unique(ball_labels_in_ball)

            # 记录所有粒球，不筛选
            ball_features.append(stats['center'])
            ball_labels.append(stats['majority_label'])

        ball_features = np.array(ball_features)
        ball_labels = np.array(ball_labels)
        if len(ball_features) < 5:  # 当粒球过少时
            # 使用原始数据训练备用分类器
            self.fallback_model = SVC().fit(X, y)

            # 定义基分类器
        gbknn = KNeighborsClassifier(n_neighbors=min(3, len(ball_features)),
                                     weights='distance')

        # 检查是否有多个类别
        if len(np.unique(ball_labels)) > 1:
            gbsvm = SVC(kernel='rbf', probability=True, class_weight='balanced')

            # 使用粒球中心训练集成分类器
            self.ensemble_model = VotingClassifier(
                estimators=[
                    ('gbknn', gbknn),
                    ('gbsvm', gbsvm)
                ],
                voting='soft',
                weights=[1, 1.5]
            )

            try:
                self.ensemble_model.fit(ball_features, ball_labels)
                print("Ensemble model trained successfully")
            except Exception as e:
                print(f"Warning: Ensemble training failed - {str(e)}")
                print("Using simple KNN as fallback")
                self.ensemble_model = gbknn
                self.ensemble_model.fit(ball_features, ball_labels)
        else:
            print("Only one class in granular balls, using simple KNN")
            self.ensemble_model = gbknn
            self.ensemble_model.fit(ball_features, ball_labels)

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """集成分类与三支决策"""
        decisions = []
        similarities = []
        proba_matrix = np.zeros((len(X_test), 2))  # 二分类概率矩阵

        for i, x in enumerate(X_test):
            # 找到最近粒球
            ball_idx = self.gb_model.predict_ball(x)
            if ball_idx == -1:
                decisions.append(0)
                similarities.append(0)
                continue

            ball_info = self.ball_stats_[ball_idx]

            # 计算相似度 (密度加权)
            distance = np.linalg.norm(x - ball_info['center'])
            radius = ball_info['radius']
            # 修改后（添加密度正则化）：
            density_factor = np.tanh(ball_info['density'] / 100)  # 压缩密度影响范围到[0,1]
            similarity = max(0, 1 - distance / (radius + 1e-6)) * density_factor

            # 动态调整阈值（基于粒球纯度）
            effective_alpha = max(self.tw_model.alpha, 1 - ball_info['entropy'])
            effective_beta = min(self.tw_model.beta, ball_info['entropy'])

            # 三支决策
            decision = self.tw_model.predict(similarity)

            # 处理延迟决策
            if decision == "delay":
                if self.ensemble_model is not None:
                    try:
                        proba = self.ensemble_model.predict_proba([x])[0]
                        pred_label = np.argmax(proba)
                        proba_matrix[i] = proba
                    except:
                        # 集成模型预测失败时使用粒球主要标签
                        pred_label = ball_info['majority_label']
                        proba_matrix[i] = [1.0, 0.0] if pred_label == 0 else [0.0, 1.0]
                else:
                    pred_label = ball_info['majority_label']
                    proba_matrix[i] = [1.0, 0.0] if pred_label == 0 else [0.0, 1.0]
            else:
                pred_label = 1 if decision == "accept" else 0
                proba_matrix[i] = [1 - similarity, similarity] if decision == "accept" else [similarity, 1 - similarity]

            decisions.append(pred_label)
            similarities.append(similarity)

        return np.array(decisions), np.array(similarities), proba_matrix

    def _calculate_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """基于信息熵的特征重要性"""
        if not self.ball_stats_:
            return np.ones(X.shape[1]) / X.shape[1]

        importances = np.zeros(X.shape[1])
        for ball in self.ball_stats_:
            center = ball['center']
            radius = ball['radius']
            entropy_val = ball['entropy']

            # 熵越低(纯度越高)的粒球贡献更大
            weights = (1 - entropy_val) * np.exp(-np.abs(X - center) / (radius + 1e-6))
            importances += np.sum(weights, axis=0)

        return importances / (importances.sum() + 1e-6)

    def _get_training_report(self) -> Dict:
        """生成训练报告"""
        entropies = [b['entropy'] for b in self.ball_stats_]
        densities = [b['density'] for b in self.ball_stats_]

        return {
            'n_balls': self.gb_model.n_balls,
            'avg_entropy': np.mean(entropies),
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies),
            'avg_density': np.mean(densities),
            'feature_importances': self.feature_importances_.tolist()
        }

    def __str__(self):
        """返回模型信息"""
        info = [
            "GranularThreeWayClassifier V4 (信息熵优化) 模型结构:",
            f"- 粒球模型: GranularBallV4",
            f"  * 最大熵阈值: {self.gb_model.max_entropy}",
            f"  * 半径收缩因子: {self.gb_model.radius_shrink_factor}",
            f"  * 粒球数量: {getattr(self.gb_model, 'n_balls', '未训练')}",
            f"- 三支决策模型: {type(self.tw_model).__name__}",
            f"  * alpha(接受阈值): {self.tw_model.alpha}",
            f"  * beta(拒绝阈值): {self.tw_model.beta}",
            f"- 集成分类器: {type(self.ensemble_model).__name__ if self.ensemble_model else '未训练'}"
        ]
        if hasattr(self, 'ball_stats_'):
            info.append(f"- 粒球统计:")
            info.append(f"  * 平均熵: {np.mean([b['entropy'] for b in self.ball_stats_]):.2f}")
            info.append(f"  * 平均密度: {np.mean([b['density'] for b in self.ball_stats_]):.2f}")
        return "\n".join(info)
# 在v4_model_信息熵优化.py文件末尾添加以下代码

if __name__ == "__main__":
    # 1. fourclass数据集实验
    print("=== 数据集fourclass ===")
    loader = DataLoader()
    result = loader.load_fourclass()
    train_data = result['data']['train']
    X_train, y_train = train_data
    test_data = result['data']['test']
    X_test, Y_test = test_data

    # 检查标签类别
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError("Training data must contain at least 2 classes")

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. 训练模型
    print("=== 开始训练 ===")
    try:
        model = GranularThreeWayClassifierV4(
            max_entropy=0.1,  # 放宽熵阈值（原0.000001）
            alpha=0.9,  # 放宽接受阈值
            beta=0.001,  # 提高拒绝阈值
            radius_shrink_factor=0.95
        )
        model.fit(X_train, y_train)

        print("\n=== 训练后模型结构 ===")
        print(model)

        # 3. 特征分析
        print("\n=== 特征重要性 ===")
        feature_importance = model.feature_importances_
        print(feature_importance)

        # 4. 预测测试集
        print("\n=== 开始预测 ===")
        y_pred, similarities, _ = model.predict(X_test)

        # 5. 评估结果
        print("\n=== 评估结果 ===")
        eval_results = ThreeWayEvaluator.evaluate(
            y_true=Y_test,
            y_pred=y_pred,
            similarities=similarities,
            alpha=0.9,
            beta=0.001
        )
        ThreeWayEvaluator.print_report(eval_results)

    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Please check your data or adjust parameters")

    # # 2. mushroom数据集实验
    # print("\n=== 数据集mushroom ===")
    # loader = DataLoadermushroom()
    # result = loader.load_mushroom()
    # X_train, y_train = result['data']['train']
    # X_test, Y_test = result['data']['test']
    #
    # # 数据标准化
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    #
    # # 训练模型
    # print("=== 开始训练 ===")
    # try:
    #     model = GranularThreeWayClassifierV4(
    #         max_entropy=0.6,    # 更严格的熵阈值
    #         alpha=0.9,          # 更高接受阈值
    #         beta=0.02,          # 更高拒绝阈值
    #         radius_shrink_factor=0.9  # 更温和的收缩
    #     )
    #     model.fit(X_train, y_train)
    #
    #     print("\n=== 训练后模型结构 ===")
    #     print(model)
    #
    #     print("\n=== 特征重要性 ===")
    #     feature_importance = model.feature_importances_
    #     print(feature_importance)
    #
    #     print("\n=== 开始预测 ===")
    #     y_pred, similarities, _ = model.predict(X_test)
    #
    #     print("\n=== 评估结果 ===")
    #     eval_results = ThreeWayEvaluator.evaluate(
    #         y_true=Y_test,
    #         y_pred=y_pred,
    #         similarities=similarities,
    #         alpha=0.9,
    #         beta=0.02
    #     )
    #     ThreeWayEvaluator.print_report(eval_results)
    #
    #     print("\n混淆矩阵:")
    #     print(confusion_matrix(Y_test, y_pred))
    #
    # except Exception as e:
    #     print(f"Error during training: {str(e)}")
    #     print("Please check your data or adjust parameters")
    #
    # # 3. svmguide1数据集实验
    # print("\n=== 数据集svmguide1 ===")
    # loader = DataLoadersvmguide1()
    # result = loader.load_svmguide1()
    # X_train, y_train = result['data']['train']
    # X_test, Y_test = result['data']['test']
    #
    # # 数据标准化
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    #
    # # 训练模型
    # print("=== 开始训练 ===")
    # try:
    #     model = GranularThreeWayClassifierV4(
    #         max_entropy=0.4,    # 较宽松的熵阈值
    #         alpha=0.8,          # 中等接受阈值
    #         beta=0.00001,       # 中等拒绝阈值
    #         radius_shrink_factor=0.8  # 较强收缩
    #     )
    #     model.fit(X_train, y_train)
    #
    #     print("\n=== 训练后模型结构 ===")
    #     print(model)
    #
    #     print("\n=== 特征重要性 ===")
    #     feature_importance = model.feature_importances_
    #     print(feature_importance)
    #
    #     print("\n=== 开始预测 ===")
    #     y_pred, similarities, _ = model.predict(X_test)
    #
    #     print("\n=== 评估结果 ===")
    #     eval_results = ThreeWayEvaluator.evaluate(
    #         y_true=Y_test,
    #         y_pred=y_pred,
    #         similarities=similarities,
    #         alpha=0.8,
    #         beta=0.00001
    #     )
    #     ThreeWayEvaluator.print_report(eval_results)
    #
    # except Exception as e:
    #     print(f"Error during training: {str(e)}")
    #     print("Please check your data or adjust parameters")
    #
