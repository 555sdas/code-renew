import numpy as np
from typing import Dict, Tuple
# 修改导入语句，引入V2版本
from granular_ball.v2_granular_ball_超球体覆盖 import GranularBallV2
from three_way_decision.v1_three_way_decision_固定阈值 import ThreeWayDecisionV1
from data_load.fourclass_data_load import DataLoader
from utils.evaluater import ThreeWayEvaluator
from data_load.mushroom_data_load import DataLoadermushroom
from data_load.svmguide1_data_load import DataLoadersvmguide1


class GranularThreeWayClassifier:
    """
    基于粒球计算的三支分类器V2（使用注意力机制）
    工作流程：
    1. 使用带注意力机制的粒球生成覆盖训练数据
    2. 对测试样本计算与粒球的相似度
    3. 应用三支决策规则分类
    """

    def __init__(self,
                 gb_radius: float = 0.5,
                 attention_dims: int = 8,  # 新增：注意力维度参数
                 attention_lr: float = 0.01,  # 新增：注意力学习率
                 alpha: float = 0.7,
                 beta: float = 0.3):
        # 使用V2版本替代V1
        self.gb_model = GranularBallV2(
            radius=gb_radius,
            attention_dims=attention_dims,
            attention_lr=attention_lr
        )
        self.tw_model = ThreeWayDecisionV1(alpha=alpha, beta=beta)
        self.attention_dims = attention_dims
        self.attention_lr = attention_lr

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        训练模型：
        1. 生成带注意力的粒球
        2. 计算每个粒球的纯度
        """
        self.gb_model.fit(X_train)

        # 计算每个粒球的标签分布
        self.ball_stats_ = []
        for center, radius, indices in self.gb_model.balls_:
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

        # 获取特征重要性
        self.feature_importances_ = self.gb_model.get_feature_importance()

        return self._get_training_report()

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测测试样本：
        1. 找到最近粒球（使用加权特征空间）
        2. 计算相似度
        3. 应用三支决策
        """
        decisions = []
        similarities = []

        # 获取加权特征空间
        W = self.gb_model.feature_weights
        weighted_X_test = X_test.dot(W.T) if len(X_test) > 0 else np.array([])

        for i, x in enumerate(X_test):
            # 找到最近粒球（使用注意力机制）
            ball_idx, sim = self._find_nearest_ball(x, weighted_X_test[i] if len(weighted_X_test) > i else x)

            # 获取粒球信息
            ball_info = self.ball_stats_[ball_idx]

            # 三支决策
            decision = self.tw_model.predict(sim)

            # 如果是延迟决策，使用粒球的主要标签
            if decision == "delay":
                pred_label = ball_info['majority_label']
            else:
                pred_label = 1 if decision == "accept" else 0

            decisions.append(pred_label)
            similarities.append(sim)

        return np.array(decisions), np.array(similarities)

    def __str__(self):
        """返回模型结构信息"""
        info = [
            "GranularThreeWayClassifier V2 (带注意力机制) 模型结构:",
            f"- 粒球模型: GranularBallV2",
            f"  * 粒球半径: {self.gb_model.radius}",
            f"  * 粒球数量: {getattr(self.gb_model, 'n_balls', '未训练')}",
            f"  * 注意力维度: {self.attention_dims}",
            f"  * 学习率: {self.attention_lr}",
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
        if hasattr(self, 'feature_importances_'):
            info.append(f"- 特征重要性:")
            info.append(f"  * 最重要特征: {np.argmax(self.feature_importances_)}")
            info.append(f"  * 最不重要特征: {np.argmin(self.feature_importances_)}")
        return "\n".join(info)

    def _find_nearest_ball(self, x: np.ndarray, weighted_x: np.ndarray) -> Tuple[int, float]:
        """使用注意力机制计算样本与粒球的相似度"""
        min_dist = float('inf')
        nearest_idx = -1

        # 获取特征权重矩阵
        W = self.gb_model.feature_weights

        for i, (center, radius, _) in enumerate(self.gb_model.balls_):
            # 使用加权特征计算距离
            weighted_center = center.dot(W.T)
            dist = np.linalg.norm(weighted_x - weighted_center)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # 相似度 = 1 - 归一化距离
        normalized_dist = min_dist / self.gb_model.balls_[nearest_idx][1]
        similarity = max(0, 1 - normalized_dist)

        return nearest_idx, similarity

    def _get_training_report(self) -> Dict:
        """生成训练报告"""
        purities = [b['purity'] for b in self.ball_stats_]
        return {
            'n_balls': self.gb_model.n_balls,
            'avg_purity': np.mean(purities),
            'min_purity': np.min(purities),
            'max_purity': np.max(purities),
            'ball_distribution': [b['label_dist'] for b in self.ball_stats_],
            'feature_importances': self.feature_importances_.tolist()
        }

    def visualize_attention(self):
        """可视化注意力矩阵"""
        return self.gb_model.visualize_attention()


if __name__ == "__main__":
    # 1. 创建 DataLoader 实例
    print("=== 数据集fourclass ===")
    loader = DataLoader()
    result = loader.load_fourclass()
    train_data = result['data']['train']
    X_train, y_train = train_data
    test_data = result['data']['test']
    X_test, Y_test = test_data

    # 2. 训练模型（使用注意力机制）
    print("=== 开始训练 ===")
    model = GranularThreeWayClassifier(
        gb_radius=0.2,  # 粒球半径
        attention_dims=4,  # 注意力维度
        attention_lr=0.01,  # 学习率
        alpha=1,  # 接受阈值
        beta=0.2  # 拒绝阈值
    )

    # 打印初始模型结构
    print("\n=== 初始模型结构 ===")
    print(model)

    train_report = model.fit(X_train, y_train)

    print("\n=== 训练后模型结构 ===")
    print(model)

    print(f"\n训练完成！生成 {train_report['n_balls']} 个粒球")
    print(f"平均纯度: {train_report['avg_purity']:.2f}")
    print(f"特征重要性: {train_report['feature_importances']}")

    # 3. 可视化注意力
    attention_matrix = model.visualize_attention()
    print("\n注意力矩阵:")
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
        beta=0.2  # 与模型初始化时相同的beta
    )
    ThreeWayEvaluator.print_report(eval_results)

    # 2. mushroom数据集
    print("\n=== 数据集mushroom ===")
    loader = DataLoadermushroom()
    result = loader.load_mushroom()
    X_train, y_train = result['data']['train']
    X_test, Y_test = result['data']['test']

    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")

    print("=== 开始训练 ===")
    model = GranularThreeWayClassifier(
        gb_radius=4,
        attention_dims=10,  # 约1/7特征维度
        attention_lr=0.001,  # 更小学习率
        alpha=1,
        beta=0.2
    )
    # 打印初始模型结构
    print("\n=== 初始模型结构 ===")
    print(model)

    train_report = model.fit(X_train, y_train)

    print("\n=== 训练后模型结构 ===")
    print(model)

    print(f"\n训练完成！生成 {train_report['n_balls']} 个粒球")
    print(f"平均纯度: {train_report['avg_purity']:.2f}")
    print(f"特征重要性: {train_report['feature_importances']}")

    # 3. 可视化注意力
    attention_matrix = model.visualize_attention()
    print("\n注意力矩阵:")
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
        beta=0.2  # 与模型初始化时相同的beta
    )
    ThreeWayEvaluator.print_report(eval_results)

    # 3. svmguide1数据集
    print("\n=== 数据集svmguide1 ===")
    loader = DataLoadersvmguide1()
    result = loader.load_svmguide1()
    X_train, y_train = result['data']['train']
    X_test, Y_test = result['data']['test']

    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")

    print("=== 开始训练 ===")
    model = GranularThreeWayClassifier(
        gb_radius=1,
        attention_dims=6,  # 2倍特征维度
        attention_lr=0.01,  # 稍大学习率
        alpha=1,
        beta=0.1
    )
    # 打印初始模型结构
    print("\n=== 初始模型结构 ===")
    print(model)

    train_report = model.fit(X_train, y_train)

    print("\n=== 训练后模型结构 ===")
    print(model)

    print(f"\n训练完成！生成 {train_report['n_balls']} 个粒球")
    print(f"平均纯度: {train_report['avg_purity']:.2f}")
    print(f"特征重要性: {train_report['feature_importances']}")

    # 3. 可视化注意力
    attention_matrix = model.visualize_attention()
    print("\n注意力矩阵:")
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
        beta=0.1  # 与模型初始化时相同的beta
    )
    ThreeWayEvaluator.print_report(eval_results)