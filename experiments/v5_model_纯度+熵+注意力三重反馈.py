import numpy as np
from typing import Dict, Tuple, List
from granular_ball.v5_granular_ball_三重反馈 import GranularBallV5
from three_way_decision.v1_three_way_decision_固定阈值 import ThreeWayDecisionV1
from utils.evaluater import ThreeWayEvaluator
from data_load.fourclass_data_load import DataLoader
from data_load.mushroom_data_load import DataLoadermushroom
from data_load.svmguide1_data_load import DataLoadersvmguide1
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['LOKY_MAX_CPU_COUNT'] = '8'  # 设置并行线程数


class GranularThreeWayClassifierV5:
    """
    基于纯度-熵-注意力三重反馈的粒球三支分类器V5
    工作流程:
    1. 生成粒球（基于纯度、熵和注意力）
    2. 计算样本与粒球的相似度（注意力加权）
    3. 应用三支决策规则分类
    """

    def __init__(self,
                 min_purity: float = 0.85,
                 max_entropy: float = 0.8,
                 base_radius: float = 1.0,
                 alpha: float = 0.9,
                 beta: float = 0.1):
        self.gb_model = GranularBallV5(
            min_purity=min_purity,
            max_entropy=max_entropy,
            base_radius=base_radius,
            attention_dim=2
        )
        self.tw_model = ThreeWayDecisionV1(alpha=alpha, beta=beta)
        self.ball_stats_ = []
        self.feature_importances_ = None
        self.attention_matrix_ = None

        # 打印初始化信息
        print(f"[ClassifierV5] 初始化: min_purity={min_purity}, max_entropy={max_entropy}")
        print(f"base_radius={base_radius}, alpha={alpha}, beta={beta}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """训练模型"""
        self.gb_model.fit(X_train, y_train)

        # 计算粒球统计
        self.ball_stats_ = []
        for ball_info in self.gb_model.balls_:
            center, radius, indices, level, density = ball_info
            labels = y_train[indices]

            purity = self.gb_model._calculate_purity(labels)
            entropy_val = self.gb_model._calculate_entropy(labels)
            majority_label = self._get_majority_label(labels)

            self.ball_stats_.append({
                'center': center,
                'radius': radius,
                'density': density,
                'purity': purity,
                'entropy': entropy_val,
                'majority_label': majority_label
            })

        # 计算特征重要性和注意力矩阵
        self.feature_importances_ = self.gb_model._calculate_feature_importance()
        self.attention_matrix_ = self._calculate_attention_matrix(X_train)

        return self._get_training_report()

    def _get_majority_label(self, y: np.ndarray) -> int:
        """获取多数类标签"""
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测测试样本"""
        decisions = []
        similarities = []

        for x in X_test:
            # 找到最近粒球
            ball_idx = self.gb_model.predict_ball(x)
            if ball_idx == -1 or ball_idx >= len(self.ball_stats_):
                decisions.append(0)
                similarities.append(0)
                continue

            ball_info = self.ball_stats_[ball_idx]

            # 计算注意力加权相似度
            distance = np.linalg.norm(x - ball_info['center'])
            similarity = self._calculate_similarity(x, ball_info)

            # 三支决策
            if similarity >= self.tw_model.alpha:
                decisions.append(1)  # 接受
            elif similarity < self.tw_model.beta:
                decisions.append(0)  # 拒绝
            else:
                decisions.append(ball_info['majority_label'])  # 延迟决策

            similarities.append(similarity)

        return np.array(decisions), np.array(similarities)

    def _calculate_similarity(self, x: np.ndarray, ball_info: Dict) -> float:
        """计算注意力加权相似度"""
        # 计算基础距离
        distance = np.linalg.norm(x - ball_info['center'])

        # 注意力因子
        attn_weights = self.gb_model._attention_weights(np.array([x]))
        avg_attn = np.mean(attn_weights)

        # 三重联合相似度
        purity_factor = ball_info['purity'] ** 2  # 纯度越高，相似度越可信
        entropy_factor = 1 - min(1.0, ball_info['entropy'] * 2)  # 熵越高，相似度越低
        similarity = (1 - distance / ball_info['radius']) * purity_factor * entropy_factor * avg_attn

        return max(0, min(1.0, similarity))

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        return self.feature_importances_

    def visualize_attention(self, plot: bool = True) -> np.ndarray:
        """
        获取并可视化注意力矩阵
        :param plot: 是否绘制热力图
        :return: 注意力矩阵
        """
        if plot and self.attention_matrix_.shape[0] > 1:
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.attention_matrix_, annot=True, cmap='viridis')
            plt.title("Feature Attention Matrix")
            plt.show()

        return self.attention_matrix_

    def _calculate_attention_matrix(self, X: np.ndarray) -> np.ndarray:
        """计算特征间注意力矩阵"""
        if X.shape[1] <= 1:
            return np.eye(1)

        # 基于特征协方差和重要性
        cov_matrix = np.cov(X.T)
        importance_matrix = np.outer(self.feature_importances_, self.feature_importances_)
        return cov_matrix * importance_matrix

    def __str__(self):
        """返回模型结构信息"""
        info = [
            "GranularThreeWayClassifier V5 (纯度+熵+注意力三重反馈) 模型结构:",
            f"- 粒球模型: GranularBallV5",
            f"  * 最小纯度: {self.gb_model.min_purity}",
            f"  * 最大熵阈值: {self.gb_model.max_entropy}",
            f"  * 基础半径: {self.gb_model.base_radius}",
            f"  * 粒球数量: {self.gb_model.n_balls}",
            f"- 三支决策模型: {type(self.tw_model).__name__}",
            f"  * alpha(接受阈值): {self.tw_model.alpha}",
            f"  * beta(拒绝阈值): {self.tw_model.beta}"
        ]

        if hasattr(self, 'ball_stats_') and self.ball_stats_:
            info.append(f"- 粒球统计:")
            purities = [b['purity'] for b in self.ball_stats_]
            entropies = [b['entropy'] for b in self.ball_stats_]
            radii = [b['radius'] for b in self.ball_stats_]

            info.append(f"  * 平均纯度: {np.mean(purities):.3f}")
            info.append(f"  * 平均熵: {np.mean(entropies):.3f}")
            info.append(f"  * 平均半径: {np.mean(radii):.3f}")

        return "\n".join(info)

    def _get_training_report(self) -> Dict:
        """生成训练报告"""
        if not self.ball_stats_:
            return {}

        purities = [b['purity'] for b in self.ball_stats_]
        entropies = [b['entropy'] for b in self.ball_stats_]
        densities = [b['density'] for b in self.ball_stats_]
        radii = [b['radius'] for b in self.ball_stats_]

        return {
            'n_balls': len(self.ball_stats_),
            'avg_purity': np.mean(purities),
            'min_purity': np.min(purities),
            'max_purity': np.max(purities),
            'avg_entropy': np.mean(entropies),
            'min_entropy': np.min(entropies),
            'max_entropy': np.max(entropies),
            'avg_radius': np.mean(radii),
            'avg_density': np.mean(densities),
            'feature_importances': self.feature_importances_.tolist(),
            'attention_matrix': self.attention_matrix_.tolist() if self.attention_matrix_ is not None else []
        }


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
    model = GranularThreeWayClassifierV5(
        min_purity=0.9,
        max_entropy=0.7,
        base_radius=0.5,
        alpha=1,
        beta=0.001
    )
    model.fit(X_train, y_train)

    print("\n=== 训练后模型结构 ===")
    print(model)

    # 查看特征重要性
    print("\n=== 特征重要性 ===")
    feature_importance = model.get_feature_importance()
    print(feature_importance)

    # 获取并可视化注意力矩阵
    print("\n=== 注意力矩阵 ===")
    attention_matrix = model.visualize_attention()
    print(attention_matrix)

    # 4. 预测测试集
    print("\n=== 开始预测 ===")
    y_pred, similarities = model.predict(X_test)

    # 5. 评估结果
    print("\n=== 评估结果 ===")
    eval_results = ThreeWayEvaluator.evaluate(
        y_true=Y_test,
        y_pred=y_pred,
        similarities=similarities,
        alpha=1,
        beta=0.001
    )
    ThreeWayEvaluator.print_report(eval_results)

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
    # # 2. 训练模型
    # print("=== 开始训练 ===")
    # model = GranularThreeWayClassifierV5(
    #     min_purity=0.95,
    #     max_entropy=0.8,
    #     base_radius=1.5,
    #     alpha=0.9,
    #     beta=0.02
    # )
    # model.fit(X_train, y_train)
    #
    # print("\n=== 训练后模型结构 ===")
    # print(model)
    #
    # # 查看特征重要性
    # print("\n=== 特征重要性 ===")
    # feature_importance = model.get_feature_importance()
    # print(feature_importance)
    #
    # # 获取并可视化注意力矩阵
    # print("\n=== 注意力矩阵 ===")
    # attention_matrix = model.visualize_attention()
    # print(attention_matrix)
    #
    # # 4. 预测测试集
    # print("\n=== 开始预测 ===")
    # y_pred, similarities = model.predict(X_test)
    #
    # # 5. 评估结果
    # print("\n=== 评估结果 ===")
    # eval_results = ThreeWayEvaluator.evaluate(
    #     y_true=Y_test,
    #     y_pred=y_pred,
    #     similarities=similarities,
    #     alpha=0.9,
    #     beta=0.02
    # )
    # ThreeWayEvaluator.print_report(eval_results)
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
    # # 2. 训练模型
    # print("=== 开始训练 ===")
    # model = GranularThreeWayClassifierV5(
    #     min_purity=0.75,
    #     max_entropy=0.8,
    #     base_radius=0.8,
    #     alpha=0.85,
    #     beta=0.001
    # )
    # model.fit(X_train, y_train)
    #
    # print("\n=== 训练后模型结构 ===")
    # print(model)
    #
    # # 查看特征重要性
    # print("\n=== 特征重要性 ===")
    # feature_importance = model.get_feature_importance()
    # print(feature_importance)
    #
    # # 获取并可视化注意力矩阵
    # print("\n=== 注意力矩阵 ===")
    # attention_matrix = model.visualize_attention()
    # print(attention_matrix)
    #
    # # 4. 预测测试集
    # print("\n=== 开始预测 ===")
    # y_pred, similarities = model.predict(X_test)
    #
    # # 5. 评估结果
    # print("\n=== 评估结果 ===")
    # eval_results = ThreeWayEvaluator.evaluate(
    #     y_true=Y_test,
    #     y_pred=y_pred,
    #     similarities=similarities,
    #     alpha=0.85,
    #     beta=0.001
    # )
    # ThreeWayEvaluator.print_report(eval_results)