import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree


class GranularBallV3:
    """
    基于论文的改进版粒球生成方法，主要增强点：
    1. 自适应纯度阈值调整
    2. 层次化粒球生成
    3. 动态重叠处理机制
    4. 边界敏感优化
    """

    def __init__(self,
                 min_purity: float = 0.85,
                 max_iter: int = 100,
                 overlap_threshold: float = 0.8,
                 adaptive_factor: float = 0.1):
        """
        :param min_purity: 初始最小纯度阈值
        :param max_iter: 最大迭代次数
        :param overlap_threshold: 重叠判定阈值(0-1)
        :param adaptive_factor: 自适应调整因子
        """
        self.min_purity = min_purity
        self.max_iter = max_iter
        self.overlap_threshold = overlap_threshold
        self.adaptive_factor = adaptive_factor
        self.balls_ = []  # [(center, radius, sample_indices, level)]
        self.purities_ = []
        self.feature_importances_ = None
        self.ball_tree_ = None  # 用于快速查询的KD树

    def _get_label_and_purity(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 1:
            return unique[0], 1.0

        # 直接使用比例计算纯度，不再使用加权
        proportions = counts / len(y)
        majority_label = unique[proportions.argmax()]
        purity = proportions.max()
        return majority_label, purity

    def _calculate_center_and_radius(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """改进的中心和半径计算，防止零半径"""
        center = np.median(X, axis=0)  # 使用中位数更鲁棒
        distances = np.linalg.norm(X - center, axis=1)
        radius = np.percentile(distances, 80)  # 使用80百分位数避免异常值影响
        return center, max(radius, 1e-6)  # 确保最小半径

    def _adaptive_purity_threshold(self, n_samples: int, n_features: int) -> float:
        """自适应纯度阈值，确保在min_purity附近波动"""
        size_factor = 1 - np.log1p(n_samples) / np.log1p(10000)  # 样本越多，阈值可适当降低
        dim_factor = 1 - np.log1p(n_features) / 20  # 特征越多，阈值可适当降低
        return max(0.5, self.min_purity * size_factor * dim_factor)  # 确保最小阈值为0.5

    def _split_ball(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """改进的分裂策略"""
        n_classes = len(np.unique(y))
        if len(X) <= 2:  # 样本太少不分裂
            return [(X, y)]

        # 根据样本数量和类别决定聚类数
        n_clusters = min(3, n_classes, int(np.sqrt(len(X) / 2)))

        # 使用常规k-means
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=5)
        labels = kmeans.fit_predict(X)
        return [(X[labels == i], y[labels == i]) for i in range(n_clusters)]

    def _get_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """根据类别分布生成样本权重"""
        unique, counts = np.unique(y, return_counts=True)
        weights = np.ones_like(y, dtype=float)
        for label, count in zip(unique, counts):
            weights[y == label] = 1.0 / count
        return weights

    def _is_overlap(self, ball1: Tuple, ball2: Tuple) -> bool:
        """改进的重叠检测，考虑密度分布"""
        center1, radius1, indices1, _ = ball1
        center2, radius2, indices2, _ = ball2
        distance = np.linalg.norm(center1 - center2)

        # 考虑重叠区域的样本密度
        overlap_ratio = distance / (radius1 + radius2 + 1e-6)
        return overlap_ratio < self.overlap_threshold

    def _merge_overlapping_balls(self):
        """层次化合并重叠粒球"""
        if len(self.balls_) <= 1:
            return

        # 按层次和半径排序(大粒球优先处理)
        sorted_balls = sorted(self.balls_, key=lambda x: (-x[3], -x[1]))
        merged = []
        used = set()

        for i in range(len(sorted_balls)):
            if i in used:
                continue

            current = sorted_balls[i]
            to_merge = [current]

            for j in range(i + 1, len(sorted_balls)):
                if j in used:
                    continue

                if self._is_overlap(current, sorted_balls[j]):
                    to_merge.append(sorted_balls[j])
                    used.add(j)

            if len(to_merge) > 1:
                # 合并重叠粒球
                all_indices = np.concatenate([ball[2] for ball in to_merge])
                X_merged = self.X_[all_indices]
                y_merged = self.y_[all_indices]

                center, radius = self._calculate_center_and_radius(X_merged)
                label, purity = self._get_label_and_purity(X_merged, y_merged)
                level = min(ball[3] for ball in to_merge) - 1  # 合并后层次加深

                merged.append((center, radius, all_indices, level))
            else:
                merged.append(current)

        self.balls_ = merged

    def _build_ball_tree(self):
        """构建粒球空间索引"""
        if self.balls_:
            centers = np.array([ball[0] for ball in self.balls_])
            self.ball_tree_ = KDTree(centers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GranularBallV3':
        self.X_ = X
        self.y_ = y
        self.balls_ = []

        # 添加调试日志
        print(f"\n初始数据统计: 样本数={len(X)}, 类别分布={dict(zip(*np.unique(y, return_counts=True)))}")

        queue = [(X.copy(), y.copy(), np.arange(len(X)), 0)]

        while queue:
            X_ball, y_ball, indices, level = queue.pop(0)
            label, purity = self._get_label_and_purity(X_ball, y_ball)

            # 打印当前粒球信息
            print(f"处理粒球: 层级={level}, 样本数={len(X_ball)}, 纯度={purity:.2f}", end="")

            # 修正阈值计算（确保min_purity是下限）
            adaptive_thresh = max(self.min_purity,
                                  self._adaptive_purity_threshold(len(X_ball), X.shape[1]))
            print(f", 实际阈值={adaptive_thresh:.2f}")

            if purity < adaptive_thresh and len(X_ball) > 1:
                print("--> 触发分裂")
                try:
                    split_results = self._split_ball(X_ball, y_ball)
                    for X_split, y_split in split_results:
                        # 保持索引对应关系
                        mask = np.array([np.any(np.all(X_split == x, axis=1)) for x in X_ball])
                        split_indices = indices[mask]
                        if len(split_indices) > 0:
                            queue.append((X_split, y_split, split_indices, level + 1))
                except Exception as e:
                    print(f"分裂粒球时警告: {e}")
                    center, radius = self._calculate_center_and_radius(X_ball)
                    self.balls_.append((center, radius, indices, level))
                    self.purities_.append(purity)
            else:
                center, radius = self._calculate_center_and_radius(X_ball)
                self.balls_.append((center, radius, indices, level))
                self.purities_.append(purity)

        # 合并重叠粒球
        self._merge_overlapping_balls()
        self._build_ball_tree()

        # 特征重要性(基于粒球覆盖度)
        self.feature_importances_ = self._calculate_feature_importance()

        return self

    def _calculate_feature_importance(self) -> np.ndarray:
        """基于粒球覆盖范围的特征重要性"""
        if not self.balls_:
            return np.ones(self.X_.shape[1]) / self.X_.shape[1]

        importances = np.zeros(self.X_.shape[1])
        for center, radius, indices, _ in self.balls_:
            distances = np.abs(self.X_[indices] - center)
            importances += np.sum(1.0 / (distances + 1e-6), axis=0)

        return importances / (importances.sum() + 1e-6)

    def predict_ball(self, x: np.ndarray) -> int:
        """使用KD树加速最近粒球查询"""
        if self.ball_tree_ is None:
            return -1

        _, idx = self.ball_tree_.query([x], k=1)
        return idx[0][0]

    @property
    def n_balls(self) -> int:
        return len(self.balls_)

    def get_feature_importance(self) -> np.ndarray:
        return self.feature_importances_

    def visualize_attention(self) -> np.ndarray:
        """改进的可视化矩阵，反映特征间关系"""
        if not hasattr(self, 'X_') or self.X_.shape[1] <= 1:
            return np.eye(1)

        # 基于特征重要性的协方差矩阵
        cov_matrix = np.cov(self.X_.T)
        importance_matrix = np.outer(self.feature_importances_, self.feature_importances_)
        return cov_matrix * importance_matrix