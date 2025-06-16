import numpy as np
from typing import Dict, Tuple, List
from granular_ball.v6_granular_ball_维度分裂 import GranularBallClassCentric
from three_way_decision.v1_three_way_decision_固定阈值 import ThreeWayDecisionV1
from utils.evaluater import ThreeWayEvaluator
from data_load.fourclass_data_load import DataLoader
from data_load.mushroom_data_load import DataLoadermushroom
from data_load.svmguide1_data_load import DataLoadersvmguide1
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['LOKY_MAX_CPU_COUNT'] = '8'  # 设置为您的CPU逻辑核心数


class GranularThreeWayClassifierV3:
    """
    基于维度进行分裂V6
    工作流程：
    1. 使用论文方法生成粒球
    2. 对测试样本计算与粒球的相似度
    3. 应用三支决策规则分类
    """

    def __init__(self,
                 min_purity: float = 0.85,
                 alpha: float = 0.7,
                 beta: float = 0.3):
        self.gb_model = GranularBallClassCentric(min_purity=min_purity)
        self.tw_model = ThreeWayDecisionV1(alpha=alpha, beta=beta)
        self.ball_stats_ = []
        self.feature_importances_ = None
        self.attention_matrix_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """训练模型"""
        self.gb_model.fit(X_train, y_train)

        # 计算每个粒球的统计信息
        self.ball_stats_ = []
        # 新的粒球信息结构: (center, radius, sample_indices, class_label, level)
        for ball_info in self.gb_model.balls_:
            center, radius, indices, class_label, level = ball_info

            labels = y_train[indices]
            unique, counts = np.unique(labels, return_counts=True)
            purity = counts.max() / len(labels)
            self.ball_stats_.append({
                'center': center,
                'radius': radius,
                'label_dist': dict(zip(unique, counts)),
                'majority_label': unique[counts.argmax()],
                'purity': purity
            })

        # 计算特征重要性
        self.feature_importances_ = self._calculate_feature_importance(X_train)

        # 计算可视化矩阵
        self.attention_matrix_ = self._calculate_attention_matrix(X_train)

        return self._get_training_report()

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测测试样本"""
        decisions = []
        similarities = []

        for x in X_test:
            # 找到最近粒球
            ball_idx = self.gb_model.predict_ball(x)
            if ball_idx == -1 or ball_idx >= len(self.ball_stats_):  # 处理无效索引
                decisions.append(0)
                similarities.append(0)
                continue

            ball_info = self.ball_stats_[ball_idx]

            # 计算相似度 (1 - 归一化距离)
            distance = np.linalg.norm(x - ball_info['center'])
            radius = ball_info['radius']
            if radius < 1e-6:  # 处理极小半径情况
                similarity = 1.0 if distance < 1e-6 else 0.0
            else:
                similarity = max(0, 1 - distance / radius)

            # 三支决策
            decision = self.tw_model.predict(similarity)

            # 如果是延迟决策，使用粒球的主要标签
            if decision == "delay":
                pred_label = ball_info['majority_label']
            else:
                pred_label = 1 if decision == "accept" else 0

            decisions.append(pred_label)
            similarities.append(similarity)

        return np.array(decisions), np.array(similarities)

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if self.feature_importances_ is None:
            return np.zeros(self.gb_model.X_.shape[1]) if hasattr(self.gb_model, 'X_') else np.array([])
        return self.feature_importances_

    def visualize_attention(self, plot: bool = True) -> np.ndarray:
        """
        获取可视化矩阵
        :param plot: 是否自动绘制热力图
        :return: 注意力矩阵
        """
        if self.attention_matrix_ is None:
            return np.eye(1)  # 返回单位矩阵作为占位符

        if plot and self.attention_matrix_.shape[0] > 1:
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.attention_matrix_, annot=True, cmap='viridis')
            plt.title("Feature Attention Matrix")
            #plt.show()

        return self.attention_matrix_

    def _calculate_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """基于粒球覆盖范围计算特征重要性"""
        if not self.ball_stats_:
            return np.ones(X.shape[1]) / X.shape[1]

        importances = np.zeros(X.shape[1])
        for ball in self.ball_stats_:
            center = ball['center']
            radius = ball['radius']
            # 计算特征在粒球内的区分度
            distances = np.abs(X - center)
            weights = np.exp(-distances / (radius + 1e-6))
            importances += np.sum(weights, axis=0)

        # 归一化
        return importances / (importances.sum() + 1e-6)

    def _calculate_attention_matrix(self, X: np.ndarray) -> np.ndarray:
        """计算特征间注意力矩阵"""
        if not hasattr(self.gb_model, 'X_') or self.gb_model.X_.shape[1] <= 1:
            return np.eye(1)

        # 基于特征协方差和重要性
        cov_matrix = np.cov(X.T)
        importance_matrix = np.outer(self.feature_importances_, self.feature_importances_)
        return cov_matrix * importance_matrix

    def __str__(self):
        """返回模型结构信息"""
        info = [
            "GranularThreeWayClassifier V3 (论文方法) 模型结构:",
            f"- 粒球模型: GranularBallV3",
            f"  * 最小纯度: {self.gb_model.min_purity}",
            f"  * 粒球数量: {getattr(self.gb_model, 'n_balls', '未训练')}",
            f"- 三支决策模型: {type(self.tw_model).__name__}",
            f"  * alpha(接受阈值): {self.tw_model.alpha}",
            f"  * beta(拒绝阈值): {self.tw_model.beta}",
            f"- 训练状态: {'已训练' if hasattr(self, 'ball_stats_') else '未训练'}"
        ]
        if hasattr(self, 'ball_stats_'):
            info.append(f"- 粒球统计:")
            info.append(f"  * 平均纯度: {np.mean([b['purity'] for b in self.ball_stats_]):.2f}")
            info.append(f"  * 最小纯度: {np.min([b['purity'] for b in self.ball_stats_]):.2f}")
            info.append(f"  * 最大纯度: {np.max([b['purity'] for b in self.ball_stats_]):.2f}")
        return "\n".join(info)

    def _get_training_report(self) -> Dict:
        """生成训练报告"""
        purities = [b['purity'] for b in self.ball_stats_]
        return {
            'n_balls': self.gb_model.n_balls,
            'avg_purity': np.mean(purities),
            'min_purity': np.min(purities),
            'max_purity': np.max(purities),
            'ball_distribution': [b['label_dist'] for b in self.ball_stats_],
            'feature_importances': self.feature_importances_.tolist(),
            'attention_matrix': self.attention_matrix_.tolist() if self.attention_matrix_ is not None else []
        }


if __name__ == "__main__":
    # 1. 创建 DataLoader 实例
    print("=== 数据集fourclass ===")
    loader = DataLoader()
    result = loader.load_fourclass()
    train_data = result['data']['train']
    X_train, y_train = train_data
    test_data = result['data']['test']
    X_test, Y_test = test_data
    #min_radius: float = 0.51参数设置

    # 2. 训练模型
    print("=== 开始训练 ===")
    model = GranularThreeWayClassifierV3(
        min_purity=0.9,  # 降低纯度阈值，允许更多分裂
        alpha=0.95,  # 降低接受阈值
        beta=0.01,  # 降低拒绝阈值
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练模型
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
        alpha=0.95,  # 与模型初始化时相同的alpha
        beta=0.001  # 与模型初始化时相同的beta
    )
    ThreeWayEvaluator.print_report(eval_results)

    # #2. mushroom数据集
    # print("\n=== 数据集mushroom ===")
    # loader = DataLoadermushroom()
    # result = loader.load_mushroom()
    # X_train, y_train = result['data']['train']
    # X_test, Y_test = result['data']['test']
    #
    # # 2. 训练模型
    # print("=== 开始训练 ===")
    # model = GranularThreeWayClassifierV3(
    #     min_purity=0.9,  # 降低纯度阈值，允许更多分裂
    #     alpha=1,  # 降低接受阈值
    #     beta=0.02,  # 降低拒绝阈值
    # )
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    #
    # # 训练模型
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
    #     alpha=1,  # 与模型初始化时相同的alpha
    #     beta=0.02  # 与模型初始化时相同的beta
    # )
    # ThreeWayEvaluator.print_report(eval_results)

    # 3. svmguide1数据集
    print("\n=== 数据集svmguide1 ===")
    loader = DataLoadersvmguide1()
    result = loader.load_svmguide1()
    X_train, y_train = result['data']['train']
    X_test, Y_test = result['data']['test']

    # 2. 训练模型
    print("=== 开始训练 ===")
    model = GranularThreeWayClassifierV3(
        min_purity=0.95,  # 降低纯度阈值，允许更多分裂
        alpha=1,  # 降低接受阈值
        beta=0.01,  # 降低拒绝阈值
    )
    #min_radius: float = 0.59
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 训练模型
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
        alpha=1,  # 与模型初始化时相同的alpha
        beta=0.01  # 与模型初始化时相同的beta
    )
    ThreeWayEvaluator.print_report(eval_results)