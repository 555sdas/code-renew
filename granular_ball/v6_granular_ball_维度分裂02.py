import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.neighbors import KDTree
import math
import copy  # 添加copy模块

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
                 min_radius: float = 0.01):
        self.min_purity = min_purity
        self.max_iter = max_iter
        self.min_radius = min_radius  # 最小半径限制
        self.balls_ = []  # [(center, radius, sample_indices, class_label, level)]
        self.purities_ = []
        self.X_full = None  # 存储完整数据集
        self.y_full = None  # 存储完整标签集
        self.generated_balls = []  # 存储最初生成的粒球（未处理后处理）

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

    def _split_ball(self, ball: Tuple) -> List[Tuple]:
        """分裂给定的粒球（纯度不满足继续分裂）"""
        center, radius, indices, class_label, level = ball

        # 获取粒球的样本
        X = self.X_full[indices]
        y = self.y_full[indices]

        # 使用象限分裂
        quadrant_masks = self._quadrant_split(X, center, y)

        if len(quadrant_masks) <= 1:
            # 无法分裂，返回原始球
            return [ball]

        # 收集分裂后的子球
        sub_balls = []

        for mask in quadrant_masks:
            if np.sum(mask) == 0:  # 跳过空象限
                continue

            X_quad = X[mask]
            y_quad = y[mask]
            quad_indices = indices[mask]

            # 计算新球的中心和半径
            new_center, new_radius = self._calculate_center_and_radius(X_quad)

            # 计算新球的全局纯度
            new_purity = self._get_global_purity(new_center, new_radius, class_label)

            # 创建新的球信息
            new_ball = (new_center, new_radius, quad_indices, class_label, level + 1)

            # 如果纯度满足要求或达到最小半径，直接加入结果
            if new_purity >= self.min_purity or new_radius <= self.min_radius:
                sub_balls.append(new_ball)
            else:
                # 否则递归分裂
                sub_sub_balls = self._split_ball(new_ball)
                sub_balls.extend(sub_sub_balls)

        return sub_balls

    def _postprocess_balls(self):
        print("\n=== 进入_postprocess_balls方法 ===")  # 调试输出
        if not self.balls_:
            print("警告：粒球列表为空，跳过处理")  # 调试输出
        """粒球后处理：按照要求处理相互重叠的情况"""
        if not self.balls_:
            return

        print("\n=== 开始粒球后处理 ===")
        print(f"初始粒球数量: {len(self.balls_)}")

        # 保存最初生成的粒球（用于比较）
        self.generated_balls = copy.deepcopy(self.balls_)

        max_iter = 5  # 最多迭代5次防止无限循环
        changed = True
        iteration = 0
        total_balls_removed = 0

        while changed and iteration < max_iter:
            iteration += 1
            changed = False
            print(f"\n后处理: 第 {iteration} 次迭代")

            # 按半径降序排序（大球在前）
            self.balls_.sort(key=lambda x: x[1], reverse=True)

            # 复制当前球列表
            current_balls = copy.deepcopy(self.balls_)
            new_balls = []
            balls_to_remove = set()

            for i, ball_a in enumerate(current_balls):
                center_a, radius_a, indices_a, class_a, level_a = ball_a

                # 检查是否已被移除
                if i in balls_to_remove:
                    continue

                # 标记此球是否包含其他小球
                contains_balls = False
                for j, ball_b in enumerate(current_balls):
                    if j <= i or j in balls_to_remove:
                        continue

                    center_b, radius_b, indices_b, class_b, level_b = ball_b

                    # 计算球心距离
                    dist = np.linalg.norm(center_a - center_b)

                    # 检查是否完全包含: dist(center_A, center_B) + radius_B <= radius_A
                    if dist + radius_b <= radius_a:
                        contains_balls = True
                        print(f"球 {j} (r={radius_b:.3f}) 被球 {i} (r={radius_a:.3f}) 完全包含")
                        balls_to_remove.add(j)
                        total_balls_removed += 1

                if contains_balls:
                    # 分裂球A
                    print(f"分裂包含其他球的粒球 {i} (r={radius_a:.3f})")
                    sub_balls = self._split_ball(ball_a)
                    new_balls.extend(sub_balls)
                    balls_to_remove.add(i)
                    changed = True
                else:
                    # 如果没有包含其他球，保留此球
                    new_balls.append(ball_a)

            # 构建最终球列表
            self.balls_ = [
                ball for idx, ball in enumerate(current_balls)
                if idx not in balls_to_remove and (idx >= len(current_balls) or ball not in current_balls[idx + 1:])
            ]

            print(f"本次迭代分裂粒球数量: {len(balls_to_remove)}")
            print(f"当前粒球总数: {len(self.balls_)}")

        print(
            f"\n后处理完成: 原始粒球数={len(self.generated_balls)}, 处理后粒球数={len(self.balls_)}, 移除的粒球数={total_balls_removed}")

    def _recursive_build(self, X: np.ndarray, y: np.ndarray,
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

            self._recursive_build(
                X_quad, y_quad, class_label, quad_indices, level + 1
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GranularBallClassCentric':
        print("\n=== 进入GranularBallClassCentric.fit方法 ===")
        self.X_full = X
        self.y_full = y
        self.X_ = X
        self.y_ = y
        self.balls_ = []
        self.purities_ = []
        self.ball_tree_ = None

        print(f"初始数据统计: 样本数={len(X)}, 类别分布={dict(zip(*np.unique(y, return_counts=True)))}")

        # 获取所有类别
        unique_classes = np.unique(y)

        # 为每个类别单独构建粒球系统
        for cls in unique_classes:
            print(f"\n=== 为类别 {cls} 构建粒球系统 ===")
            class_mask = (y == cls)
            X_cls = X[class_mask]
            y_cls = y[class_mask]
            indices_cls = np.arange(len(X))[class_mask]
            self._recursive_build(X_cls, y_cls, cls, indices_cls, 0)

        print(f"\n生成粒球统计: 总数={len(self.balls_)}, 平均纯度={np.mean(self.purities_):.4f}")

        # 确保调用粒球后处理
        print("\n=== 开始粒球后处理 ===")
        self._postprocess_balls()
        print("=== 粒球后处理完成 ===")

        # 更新纯度列表
        self.purities_ = [self._get_global_purity(ball[0], ball[1], ball[3]) for ball in self.balls_]
        print(f"最终粒球统计: 总数={len(self.balls_)}, 平均纯度={np.mean(self.purities_):.4f}")

        # 构建粒球空间索引
        if self.balls_:
            centers = np.array([ball[0] for ball in self.balls_])
            self.ball_tree_ = KDTree(centers)
        else:
            print("警告：最终粒球列表为空")

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
            f"- 原始粒球数: {len(self.generated_balls)}",
            f"- 最小纯度: {self.min_purity}",
            f"- 最小半径: {self.min_radius}",
            f"- 平均纯度: {np.mean(self.purities_):.4f}",
            f"- 最小纯度: {np.min(self.purities_):.4f}",
            f"- 最大纯度: {np.max(self.purities_):.4f}"
        ]
        return "\n".join(info)