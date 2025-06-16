import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.neighbors import KDTree
import math

from three_way_decision.v1_three_way_decision_固定阈值 import ThreeWayDecisionV1


class GranularBallClassCentric:
    """
    按类别独立构建粒球系统：
    1. 每个类别从包含所有该类样本的粒球开始
    2. 仅使用该类样本分布进行分裂决策
    3. 纯度不满足继续分裂，直至达到最小半径
    """

    def __init__(self,
                 min_purity: float = 0.85,
                 max_iter: int = 100,
                 min_radius: float = 0.51):
        self.min_purity = min_purity
        self.max_iter = max_iter
        self.min_radius = min_radius  # 最小半径限制
        self.balls_ = []  # [(center, radius, sample_indices, class_label, level)]
        self.purities_ = []
        self.X_full = None  # 存储完整数据集
        self.y_full = None  # 存储完整标签集

    def _get_label_and_purity(self, y: np.ndarray, target_class: int) -> float:
        """计算针对目标类别的纯度"""
        n_class_samples = np.sum(y == target_class)
        return n_class_samples / len(y)  # 目标类样本比例

    def _calculate_center_and_radius(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """计算类别中心点和半径（确保最小半径）"""
        center = np.median(X, axis=0)  # 中位数作为中心点
        distances = np.linalg.norm(X - center, axis=1)
        radius = np.percentile(distances, 80)  # 80%百分位数
        return center, max(radius, self.min_radius)  # 确保不小于最小半径

    def _select_important_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """使用注意力机制选择最重要的特征（改进的分类型特征处理）"""
        n_features = X.shape[1]

        # 如果特征数较少，直接使用所有特征
        if n_features <= 4:
            return np.arange(n_features)

        # 计算特征重要性分数（基于方差和互信息）
        feature_scores = np.zeros(n_features)

        # 1. 基于互信息的重要性（最适合分类数据）
        if len(np.unique(y)) > 1:  # 确保y有多个类别
            try:
                from sklearn.feature_selection import mutual_info_classif

                # 计算互信息
                mi_scores = mutual_info_classif(X, y, random_state=42)
                if np.max(mi_scores) > 0:
                    mi_normalized = mi_scores / np.max(mi_scores)
                    feature_scores += mi_normalized
            except Exception as e:
                print(f"互信息计算错误: {e}")

        # 2. 如果互信息不可用，使用方差作为备份
        variances = np.var(X, axis=0)
        non_zero_variance_mask = variances > 1e-6
        if np.any(non_zero_variance_mask):
            var_normalized = variances[non_zero_variance_mask] / np.max(variances[non_zero_variance_mask])
            feature_scores[non_zero_variance_mask] += var_normalized

        # 3. 处理可能存在的NaN值
        feature_scores = np.nan_to_num(feature_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # 如果所有特征得分都是零，返回前4个特征
        if np.all(feature_scores == 0):
            return np.arange(min(4, n_features))

        # 选择得分最高的2-4个特征
        n_select = min(max(2, int(n_features * 0.2)), 4)
        top_features = np.argsort(feature_scores)[-n_select:]

        print(f"选择了 {len(top_features)} 个重要特征: {top_features}")
        return top_features

    def _quadrant_split(self, X: np.ndarray, center: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """基于注意力机制选择的重要维度进行分裂"""
        # 选择重要特征
        selected_features = self._select_important_features(X, y)  # 使用传入的 y 而不是 self.y
        n_selected = len(selected_features)

        if n_selected == 0:
            # 如果没有选择特征，使用最多4个维度
            n_features = min(4, X.shape[1])
            selected_features = np.arange(n_features)
            n_selected = n_features

        max_splits = 2 ** n_selected

        split_indices = []

        # 创建每个象限的掩码
        for i in range(max_splits):
            mask = np.ones(len(X), dtype=bool)  # 初始化为全True

            # 使用二进制表示确定每个特征的象限
            for j, feature_idx in enumerate(selected_features):
                bit = (i >> j) & 1  # 当前特征的象限位

                # 设置象限条件：1表示特征值 >= 中心点值，0表示 <
                if bit == 1:
                    mask &= (X[:, feature_idx] >= center[feature_idx])
                else:
                    mask &= (X[:, feature_idx] < center[feature_idx])

            if np.any(mask):
                split_indices.append(mask)

        return split_indices

    def _get_global_purity(self, center: np.ndarray, radius: float, class_label: int) -> float:
        """计算粒球在整个数据集上的纯度"""
        if self.X_full is None:
            return 0.0

        # 计算整个数据集中样本到中心的距离
        distances = np.linalg.norm(self.X_full - center, axis=1)

        # 找到粒球内的所有样本
        ball_indices = np.where(distances <= radius)[0]

        if len(ball_indices) == 0:
            return 0.0

        # 计算目标类别的样本数
        n_target = np.sum(self.y_full[ball_indices] == class_label)

        return n_target / len(ball_indices)

    def _build_class_granular_ball(self, X: np.ndarray, y: np.ndarray,
                                   class_label: int, indices: np.ndarray, level: int = 0):
        """递归构建针对特定类别的粒球（使用注意力机制优化）"""
        # 计算当前粒球的中心点和半径
        center, radius = self._calculate_center_and_radius(X)

        # 在整个数据集上计算纯度（考虑了其他类别的样本）
        purity = self._get_global_purity(center, radius, class_label)

        # 打印调试信息
        class_only_purity = len(y) / (len(y) + len(self.y_full) - len(np.unique(self.y_full)))
        print(
            f"类别 {class_label} 粒球: 级别={level}, 类别样本数={len(X)}, 球体总样本={len(np.where(np.linalg.norm(self.X_full - center, axis=1) <= radius)[0])}, 纯度={purity:.3f} (仅类内纯度={class_only_purity:.3f})")

        # 停止条件判断
        stop_condition = (
                purity >= self.min_purity or  # 纯度满足要求
                level >= self.max_iter or  # 达到最大迭代次数
                radius <= self.min_radius or  # 达到最小半径
                len(X) <= 2  # 样本数太少
        )

        if stop_condition:
            global_indices = np.where(np.linalg.norm(self.X_full - center, axis=1) <= radius)[0]
            self.balls_.append((center, radius, global_indices, class_label, level))
            self.purities_.append(purity)
            return

        print(
            f"分裂类别 {class_label} 的粒球: 层级={level}, 球体总样本={len(np.where(np.linalg.norm(self.X_full - center, axis=1) <= radius)[0])}, 纯度={purity:.3f}")

        # 使用注意力机制选择重要特征进行象限分裂
        quadrant_masks = self._quadrant_split(X, center, y)  # 添加 y 参数

        if len(quadrant_masks) <= 1:
            # 无法有效分裂，直接保存当前粒球
            global_indices = np.where(np.linalg.norm(self.X_full - center, axis=1) <= radius)[0]
            self.balls_.append((center, radius, global_indices, class_label, level))
            self.purities_.append(purity)
            return

        # 递归处理每个象限
        for mask in quadrant_masks:
            if np.sum(mask) == 0:  # 跳过空象限
                continue

            X_quad = X[mask]
            y_quad = y[mask]
            quad_indices = indices[mask]

            self._build_class_granular_ball(
                X_quad, y_quad, class_label, quad_indices, level + 1
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GranularBallClassCentric':
        self.X_full = X  # 保存完整数据集
        self.y_full = y  # 保存完整标签集
        self.X_ = X
        self.y_ = y
        self.balls_ = []
        self.purities_ = []
        self.ball_tree_ = None

        print(f"\n初始数据统计: 样本数={len(X)}, 类别分布={dict(zip(*np.unique(y, return_counts=True)))}")

        # 获取所有类别
        unique_classes = np.unique(y)

        # 为每个类别单独构建粒球系统
        for cls in unique_classes:
            print(f"\n=== 为类别 {cls} 构建粒球系统 ===")

            # 获取该类所有样本
            class_mask = (y == cls)
            X_cls = X[class_mask]
            y_cls = y[class_mask]
            indices_cls = np.arange(len(X))[class_mask]

            # 为该类别创建初始粒球（包含所有该类样本）
            self._build_class_granular_ball(X_cls, y_cls, cls, indices_cls, 0)

        print(f"\n生成粒球统计: 总数={len(self.balls_)}, 平均纯度={np.mean(self.purities_):.4f}")

        # 构建粒球空间索引
        if self.balls_:
            centers = np.array([ball[0] for ball in self.balls_])
            self.ball_tree_ = KDTree(centers)

        return self

    def predict_ball(self, x: np.ndarray) -> int:
        """找到最近粒球"""
        if self.ball_tree_ is None or not self.balls_:
            return -1

        _, idx = self.ball_tree_.query([x], k=1)
        return idx[0][0]

    @property
    def n_balls(self) -> int:
        return len(self.balls_)

    def __str__(self):
        info = [
            f"GranularBallClassCentric(类别中心粒球系统)",
            f"- 粒球总数: {self.n_balls}",
            f"- 最小纯度: {self.min_purity}",
            f"- 最小半径: {self.min_radius}",
            f"- 平均纯度: {np.mean(self.purities_):.4f}",
            f"- 最小纯度: {np.min(self.purities_):.4f}",
            f"- 最大纯度: {np.max(self.purities_):.4f}"
        ]
        return "\n".join(info)


class GranularThreeWayClassifierClassCentric:
    """
    基于类别中心粒球的三支分类器
    """

    def __init__(self,
                 min_purity: float = 0.85,
                 alpha: float = 0.7,
                 beta: float = 0.3,
                 min_radius: float = 0.001):
        self.gb_model = GranularBallClassCentric(min_purity=min_purity, min_radius=min_radius)
        self.tw_model = ThreeWayDecisionV1(alpha=alpha, beta=beta)
        self.ball_stats_ = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """训练模型"""
        self.gb_model.fit(X_train, y_train)

        # 计算每个粒球的统计信息
        self.ball_stats_ = []
        for ball_info in self.gb_model.balls_:
            center, radius, indices, class_label, level = ball_info
            labels = y_train[indices]
            n_class_samples = np.sum(labels == class_label)
            purity = n_class_samples / len(labels)

            self.ball_stats_.append({
                'center': center,
                'radius': radius,
                'class_label': class_label,
                'purity': purity,
                'level': level
            })

        return self._get_training_report()

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """预测测试样本"""
        decisions = []
        similarities = []

        for x in X_test:
            # 找到最近粒球
            ball_idx = self.gb_model.predict_ball(x)
            if ball_idx == -1 or ball_idx >= len(self.ball_stats_):
                # 无法找到合适粒球，直接延迟决策
                decisions.append(2)  # 使用2表示延迟决策
                similarities.append(0)
                continue

            ball_info = self.ball_stats_[ball_idx]
            center = ball_info['center']
            radius = ball_info['radius']
            target_class = ball_info['class_label']

            # 计算相似度 (1 - 归一化距离)
            distance = np.linalg.norm(x - center)
            similarity = max(0, 1 - distance / radius) if radius > 0 else 1.0

            # 三支决策
            decision = self.tw_model.predict(similarity)

            # 决策映射：accept->1, reject->0, delay->2
            if decision == "accept":
                decisions.append(1)
            elif decision == "reject":
                decisions.append(0)
            else:  # delay
                decisions.append(2)

            similarities.append(similarity)

        return np.array(decisions), np.array(similarities)

    def _get_training_report(self) -> Dict:
        """生成训练报告"""
        purities = [b['purity'] for b in self.ball_stats_]
        return {
            'n_balls': self.gb_model.n_balls,
            'avg_purity': np.mean(purities),
            'min_purity': np.min(purities),
            'max_purity': np.max(purities),
            'min_radius': self.gb_model.min_radius
        }

    def __str__(self):
        """返回模型结构信息"""
        info = [
            "GranularThreeWayClassifierClassCentric 模型结构:",
            f"- 粒球模型: GranularBallClassCentric",
            f"  * 最小纯度: {self.gb_model.min_purity}",
            f"  * 最小半径: {self.gb_model.min_radius}",
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