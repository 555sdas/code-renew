import numpy as np
from typing import Dict, Tuple
from granular_ball.v1_granular_ball_超球体覆盖 import GranularBallV1
from three_way_decision.v1_three_way_decision_固定阈值 import ThreeWayDecisionV1
from data_load.fourclass_data_load import DataLoader
from utils.evaluater import ThreeWayEvaluator
from data_load.mushroom_data_load import DataLoadermushroom

class GranularThreeWayClassifier:
    """
    基于粒球计算的三支分类器V1
    工作流程：
    1. 生成粒球覆盖训练数据
    2. 对测试样本计算与粒球的相似度
    3. 应用三支决策规则分类
    """

    def __init__(self,
                 gb_radius: float = 0.5,
                 alpha: float = 0.7,
                 beta: float = 0.3):
        self.gb_model = GranularBallV1(radius=gb_radius)
        self.tw_model = ThreeWayDecisionV1(alpha=alpha, beta=beta)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        训练模型：
        1. 生成粒球
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

        return self._get_training_report()

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测测试样本：
        1. 找到最近粒球
        2. 计算相似度
        3. 应用三支决策
        """
        decisions = []
        similarities = []

        for x in X_test:
            # 找到最近粒球
            ball_idx, sim = self._find_nearest_ball(x)

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
            "GranularThreeWayClassifier 模型结构:",
            f"- 粒球模型: {type(self.gb_model).__name__}",
            f"  * 粒球半径: {self.gb_model.radius}",
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

    def _find_nearest_ball(self, x: np.ndarray) -> Tuple[int, float]:
        """计算样本与所有粒球的相似度（归一化距离的补）"""
        min_dist = float('inf')
        nearest_idx = -1

        for i, (center, radius, _) in enumerate(self.gb_model.balls_):
            dist = np.linalg.norm(x - center)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        # 相似度 = 1 - 归一化距离（距离越近相似度越高）
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
            'ball_distribution': [b['label_dist'] for b in self.ball_stats_]
        }


if __name__ == "__main__":
    # 1. 模拟生成数据（替换为您的实际数据加载逻辑）
    # 1. 创建 DataLoader 实例
    print("=== 数据集fourclass ===")
    loader = DataLoader()

    # 2. 调用 load_fourclass() 方法获取结果字典
    result = loader.load_fourclass()  # 这里返回的是字典，不是 DataLoader 实例

    train_data = result['data']['train']  # 正确：对字典使用 []
    X_train, y_train = train_data
    test_data = result['data']['test']
    X_test, Y_test = test_data
    print(f"训练集特征形状: {X_train.shape}")
    print(f"训练集标签形状: {y_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    print(f"训练集标签形状: {Y_test.shape}")

    # 2. 训练模型
    print("=== 开始训练 ===")
    model = GranularThreeWayClassifier(
        gb_radius=0.2,  # 粒球半径
        alpha=1,  # 接受阈值
        beta=0.3  # 拒绝阈值
    )

    # 打印初始模型结构
    print("\n=== 初始模型结构 ===")
    print(model)

    train_report = model.fit(X_train, y_train)

    print("\n=== 训练后模型结构 ===")
    print(model)

    print(f"\n训练完成！生成 {train_report['n_balls']} 个粒球")
    print(f"平均纯度: {train_report['avg_purity']:.2f}")
    print(f"最小纯度: {train_report['min_purity']:.2f}")
    print(f"最大纯度: {train_report['max_purity']:.2f}")

    # 3. 预测测试集
    print("\n=== 开始预测 ===")
    y_pred, similarities = model.predict(X_test)

    # 4. 评估结果
    print("\n=== 评估结果 ===")
    eval_results = ThreeWayEvaluator.evaluate(
        y_true=Y_test,
        y_pred=y_pred,
        similarities=similarities,
        alpha=1,  # 与模型初始化时相同的alpha
        beta=0.3  # 与模型初始化时相同的beta
    )
    ThreeWayEvaluator.print_report(eval_results)

    print("=== 数据集mushroom ===")
    loader = DataLoadermushroom()
    result = loader.load_mushroom()
    train_data = result['data']['train']  # 正确：对字典使用 []
    X_train, y_train = train_data
    test_data = result['data']['test']
    X_test, Y_test = test_data
    print(f"训练集特征形状: {X_train.shape}")
    print(f"训练集标签形状: {y_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    print(f"训练集标签形状: {Y_test.shape}")
    # 2. 训练模型
    print("=== 开始训练 ===")
    model = GranularThreeWayClassifier(
        gb_radius=10,  # 粒球半径
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
    print(f"最小纯度: {train_report['min_purity']:.2f}")
    print(f"最大纯度: {train_report['max_purity']:.2f}")

    # 3. 预测测试集
    print("\n=== 开始预测 ===")
    y_pred, similarities = model.predict(X_test)

    # 4. 评估结果
    print("\n=== 评估结果 ===")
    eval_results = ThreeWayEvaluator.evaluate(
        y_true=Y_test,
        y_pred=y_pred,
        similarities=similarities,
        alpha=1,  # 与模型初始化时相同的alpha
        beta=0.2  # 与模型初始化时相同的beta
    )
    ThreeWayEvaluator.print_report(eval_results)