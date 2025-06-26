import numpy as np
from typing import Dict, Tuple, List
from granular_ball.v6_granular_ball_维度分裂v202 import GranularBallClassCentric
from three_way_decision.v1_three_way_decision_固定阈值 import ThreeWayDecisionV1
from utils.evaluater多任务十字验证 import SmartEvaluator
from data_load.fourclass_data_load import DataLoader
from data_load.mushroom_data_load import DataLoadermushroom
from data_load.svmguide1_data_load import DataLoadersvmguide1
from data_load.pen_data_load import DataLoaderPen
from data_load.fashionminist_data_load import DataLoaderminist
from data_load.pen_data_load import DataLoaderPen
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
        # 训练粒球模型(内部会自动调用后处理)
        self.gb_model.fit(X_train, y_train)

        # 打印后处理前后的粒球数量对比
        if hasattr(self.gb_model, 'generated_balls'):
            print(f"粒球数量变化: 生成 {len(self.gb_model.generated_balls)} → 后处理 {self.gb_model.n_balls}")

        # 计算每个粒球的统计信息(使用后处理后的粒球)
        self.ball_stats_ = []
        for ball_info in self.gb_model.balls_:
            center, radius, indices, class_label, level = ball_info

            # 使用全局数据集计算纯度(而不仅仅是训练集中的纯度)
            global_labels = self.gb_model.y_full[indices]
            unique, counts = np.unique(global_labels, return_counts=True)
            purity = counts.max() / len(global_labels)

            self.ball_stats_.append({
                'center': center,
                'radius': radius,
                'label_dist': dict(zip(unique, counts)),
                'majority_label': unique[counts.argmax()],
                'purity': purity,
                'level': level  # 添加粒球层级信息
            })

        # 计算特征重要性
        self.feature_importances_ = self._calculate_feature_importance(X_train)

        # 计算可视化矩阵
        self.attention_matrix_ = self._calculate_attention_matrix(X_train)

        return self._get_training_report()

    def visualize_granular_balls(self, X: np.ndarray, y: np.ndarray,
                                 title: str = "Granular Balls Visualization",
                                 save_path: str = None):
        """
        可视化粒球与样本分布（仅适用于2D数据）
        :param X: 样本特征 (n_samples, 2)
        :param y: 样本标签 (n_samples,)
        :param title: 图表标题
        :param save_path: 如果提供路径，则保存图像到本地（例如：'./granular_balls.png'）
        """
        plt.figure(figsize=(10, 8))

        # 绘制样本点
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6,
                              edgecolors='w', s=40, label='Samples')

        # 绘制粒球
        for ball in self.ball_stats_:
            center = ball['center']
            radius = ball['radius']
            label = ball['majority_label']

            color = 'red' if label == 1 else 'blue'  # 按类别设置颜色
            circle = plt.Circle(center, radius, color=color, fill=False,
                                linestyle='--', linewidth=1.5, alpha=0.7)
            plt.gca().add_patch(circle)
            plt.scatter(center[0], center[1], color=color, marker='*', s=100,
                        edgecolors='k', label=f'Class {label} Center')

        # 添加图例和标签
        handles, labels = scatter.legend_elements()
        plt.legend(handles, ['Class 0', 'Class 1'], title="True Labels")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # 保存或显示图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存到: {save_path}")
            plt.close()  # 关闭图形避免内存泄漏
        else:
            plt.show()

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测测试样本（按新三支决策规则）"""
        decisions = []
        similarities = []

        for x in X_test:
            # 找出所有包含该样本的粒球
            containing_balls = []
            for i, ball in enumerate(self.ball_stats_):
                distance = np.linalg.norm(x - ball['center'])
                if distance <= ball['radius']:
                    containing_balls.append((i, distance))

            if not containing_balls:
                # 样本不在任何粒球内，使用最近粒球的延迟决策
                ball_idx = self.gb_model.predict_ball(x)
                if ball_idx == -1:
                    decisions.append(0)
                    similarities.append(0)
                    continue

                ball_info = self.ball_stats_[ball_idx]
                decisions.append(ball_info['majority_label'])
                similarities.append(0)  # 相似度设为0表示不确定
                continue

            if len(containing_balls) == 1:
                # 情况1：只在一个粒球内，直接接受
                ball_idx, distance = containing_balls[0]
                ball_info = self.ball_stats_[ball_idx]
                decisions.append(ball_info['majority_label'])
                similarities.append(1.0)  # 相似度设为1表示完全接受
            else:
                # 情况2：在多个粒球内，选择最近的
                containing_balls.sort(key=lambda x: x[1])  # 按距离排序
                nearest_ball_idx, nearest_distance = containing_balls[0]
                ball_info = self.ball_stats_[nearest_ball_idx]

                # 计算归一化相似度
                similarity = 1 - (nearest_distance / (ball_info['radius'] + 1e-6))
                similarity = max(0, min(1, similarity))  # 裁剪到[0,1]

                # 总是延迟决策（因为存在多个粒球重叠）
                decisions.append(ball_info['majority_label'])
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
            plt.show()

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
        ]

        # 添加后处理信息
        if hasattr(self.gb_model, 'generated_balls'):
            info.append(f"  * 原始生成粒球数: {len(self.gb_model.generated_balls)}")
            info.append(f"  * 后处理移除粒球数: {len(self.gb_model.generated_balls) - self.gb_model.n_balls}")

        info.extend([
            f"- 三支决策模型: {type(self.tw_model).__name__}",
            f"  * alpha(接受阈值): {self.tw_model.alpha}",
            f"  * beta(拒绝阈值): {self.tw_model.beta}",
            f"- 训练状态: {'已训练' if hasattr(self, 'ball_stats_') else '未训练'}"
        ])

        if hasattr(self, 'ball_stats_'):
            info.append(f"- 粒球统计:")
            info.append(f"  * 平均纯度: {np.mean([b['purity'] for b in self.ball_stats_]):.2f}")
            info.append(f"  * 最小纯度: {np.min([b['purity'] for b in self.ball_stats_]):.2f}")
            info.append(f"  * 最大纯度: {np.max([b['purity'] for b in self.ball_stats_]):.2f}")
            info.append(f"  * 最大层级: {np.max([b.get('level', 0) for b in self.ball_stats_])}")

        return "\n".join(info)

    def _get_training_report(self) -> Dict:
        """生成训练报告"""
        purities = [b['purity'] for b in self.ball_stats_]
        report = {
            'n_balls': self.gb_model.n_balls,
            'avg_purity': np.mean(purities),
            'min_purity': np.min(purities),
            'max_purity': np.max(purities),
            'ball_distribution': [b['label_dist'] for b in self.ball_stats_],
            'feature_importances': self.feature_importances_.tolist(),
            'attention_matrix': self.attention_matrix_.tolist() if self.attention_matrix_ is not None else []
        }

        # 添加后处理信息
        if hasattr(self.gb_model, 'generated_balls'):
            report['initial_balls'] = len(self.gb_model.generated_balls)
            report['removed_balls'] = len(self.gb_model.generated_balls) - self.gb_model.n_balls

        return report


if __name__ == "__main__":
    # 1. 加载完整数据集
    print("=== 数据集svmguide1 ===")
    loader = DataLoadersvmguide1()
    result = loader.load_svmguide1()
    X_full = np.concatenate([result['data']['train'][0], result['data']['test'][0]])
    y_full = np.concatenate([result['data']['train'][1], result['data']['test'][1]])

    # 2. 准备10折交叉验证
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results_list = []

    for fold, (train_index, test_index) in enumerate(kf.split(X_full)):
        print(f"\n=== 第 {fold + 1}/10 折交叉验证 ===")

        # 划分训练集和测试集
        X_train, X_test = X_full[train_index], X_full[test_index]
        y_train, y_test = y_full[train_index], y_full[test_index]

        # 标准化数据
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 2. 训练模型
        print("=== 开始训练 ===")
        model = GranularThreeWayClassifierV3(
            min_purity=0.98,
            alpha=1,
            beta=-0.01,
        )
        model.fit(X_train, y_train)

        # 4. 预测测试集
        print("\n=== 开始预测 ===")
        y_pred, similarities = model.predict(X_test)

        # 5. 评估结果
        print("\n=== 评估结果 ===")
        eval_results = SmartEvaluator.evaluate(
            y_true=y_test,
            y_pred=y_pred,
            similarities=similarities,
            alpha=1,
            beta=-0.01
        )
        results_list.append(eval_results)
        SmartEvaluator.print_report(eval_results)

    # 聚合并打印多次实验结果
    aggregated_results = SmartEvaluator.aggregate_results(results_list)
    SmartEvaluator.print_aggregated_report(aggregated_results)