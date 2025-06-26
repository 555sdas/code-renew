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
    4. 添加完全包裹处理策略
    """

    def __init__(self,
                 min_purity: float = 0.85,
                 max_iter: int = 100,
                 min_radius: float = 1.2,
                 max_contain_process_cycles: int =4):  # 新增参数
        self.min_purity = min_purity
        self.max_iter = max_iter
        self.min_radius = min_radius  # 最小半径限制
        self.max_contain_process_cycles = max_contain_process_cycles  # 最大处理循环次数
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
        center = np.median(X, axis=0)  # 改为平均值（原为 np.median中位数）
        distances = np.linalg.norm(X - center, axis=1)
        radius = np.percentile(distances, 100)  # 80%百分位数
        return center, max(radius, self.min_radius)  # 确保不小于最小半径

    def _select_important_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """使用注意力机制选择最重要的特征（改进的分类型特征处理）"""
        n_features = X.shape[1]

        # 如果特征数较少，直接使用所有特征
        if n_features <= 2:
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
            return np.arange(min(16, n_features))

        # 选择得分最高的2-4个特征
        n_select = min(max(2, int(n_features)), 16)
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
        center, radius = self._calculate_center_and_radius(X)

        # 计算粒球覆盖的所有样本
        ball_indices = np.where(np.linalg.norm(self.X_full - center, axis=1) <= radius)[0]

        # 计算各类纯度
        global_purity = len(X) / len(ball_indices)  # 当前类别样本在粒球中的比例
        purity = np.sum(self.y_full[ball_indices] == class_label) / len(ball_indices)
        class_coverage = len(X) / np.sum(self.y_full == class_label)

        # 调试信息
        print(f"           --- 级别 {level} --- | "
              f"中心点坐标范围: [{np.min(center):.2f}, {np.max(center):.2f}] | "
              f"半径: {radius:.2f} | "
              f"当前类别样本: {len(X)} | "
              f"粒球覆盖总样本: {len(ball_indices)} | "
              f"在当前分裂使用的纯度: {global_purity:.3f} | "
              f"真实纯度: {purity:.3f} | "
              f"类内覆盖率: {class_coverage:.3f}")
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

    def _process_fully_contained_balls(self):
        """处理完全被包裹的粒球"""
        if len(self.balls_) <= 1:
            return

        print("\n=== 处理完全被包裹的粒球 ===")
        current_cycle = 0

        while current_cycle < self.max_contain_process_cycles:
            current_cycle += 1
            print(f"\n处理循环 {current_cycle}/{self.max_contain_process_cycles}")

            # 按半径降序排列粒球
            sorted_balls = sorted(self.balls_, key=lambda x: x[1], reverse=True)
            new_balls = []
            processed_indices = set()
            total_contained = 0

            for i, (center_A, radius_A, indices_A, label_A, level_A) in enumerate(sorted_balls):
                if i in processed_indices:
                    continue

                # 检查是否有被A完全包含的粒球
                contained_balls = []
                for j, (center_B, radius_B, indices_B, label_B, level_B) in enumerate(sorted_balls[i + 1:],
                                                                                      start=i + 1):
                    if j in processed_indices:
                        continue

                    # 检查B是否被A完全包含
                    dist = np.linalg.norm(center_A - center_B)

                    # 添加对最小半径的特殊处理
                    if radius_A <= self.min_radius + 1e-6 and radius_B <= self.min_radius + 1e-6:
                        continue

                    if dist + radius_B <= radius_A + 1e-6:  # 添加小的容差
                        contained_balls.append(j)

                if not contained_balls:
                    new_balls.append((center_A, radius_A, indices_A, label_A, level_A))
                    processed_indices.add(i)
                    continue

                # 记录包含情况
                total_contained += len(contained_balls)
                print(f"粒球 {i} (半径={radius_A:.3f}) 完全包含了 {len(contained_balls)} 个粒球，需要分裂")

                # 获取A的原始样本
                X_A = self.X_full[indices_A]
                y_A = self.y_full[indices_A]

                # 分裂A
                temp_balls = []
                self._split_ball_recursive(X_A, y_A, indices_A, label_A, level_A, temp_balls)

                # 将分裂后的新粒球加入待处理列表
                for ball in temp_balls:
                    new_balls.append(ball)
                    processed_indices.discard(i)  # 确保原A被移除

                # 标记被包含的粒球为已处理
                for j in contained_balls:
                    processed_indices.add(j)

            # 更新粒球列表
            self.balls_ = new_balls
            self.purities_ = [self._get_global_purity(b[0], b[1], b[3]) for b in self.balls_]

            print(f"本轮处理完成: 共处理了 {total_contained} 个被包含关系")

            # 如果没有被包含的粒球了，提前退出循环
            if total_contained == 0:
                print("没有检测到新的完全包含关系，提前终止处理")
                break

        print(f"\n完成完全包含处理，共进行了 {current_cycle} 轮处理")

    def _split_ball_recursive(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                              class_label: int, level: int, temp_balls: list):
        """递归分裂粒球直到满足纯度要求"""
        center, radius = self._calculate_center_and_radius(X)
        purity = self._get_global_purity(center, radius, class_label)

        # 停止条件
        if (purity >= self.min_purity or
                level >= self.max_iter or
                radius <= self.min_radius or
                len(X) <= 2):
            global_indices = np.where(np.linalg.norm(self.X_full - center, axis=1) <= radius)[0]
            temp_balls.append((center, radius, global_indices, class_label, level))
            return

        # 分裂粒球
        quadrant_masks = self._quadrant_split(X, center, y)

        if len(quadrant_masks) <= 1:
            global_indices = np.where(np.linalg.norm(self.X_full - center, axis=1) <= radius)[0]
            temp_balls.append((center, radius, global_indices, class_label, level))
            return

        for mask in quadrant_masks:
            if np.sum(mask) == 0:
                continue

            X_quad = X[mask]
            y_quad = y[mask]
            quad_indices = indices[mask]

            self._split_ball_recursive(X_quad, y_quad, quad_indices, class_label, level + 1, temp_balls)

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

        # 处理完全被包裹的粒球
        print("\n=== 处理完全被包裹的粒球 ===")
        self._process_fully_contained_balls()

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