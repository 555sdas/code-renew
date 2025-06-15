# v4_granular_ball_信息熵优化.py
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy.stats import entropy


class GranularBallV4:
    """
    基于信息熵优化的粒球生成方法V4，主要改进：
    1. 信息熵作为分裂准则替代纯度阈值
    2. 动态半径收缩策略
    3. 粒球密度感知的覆盖优化
    """
    """
           :param max_entropy: 最大允许熵值(超过则分裂)
           :param min_samples: 粒球最小样本数
           :param max_iter: 最大迭代次数
           :param radius_shrink_factor: 半径收缩因子(0-1)
           :param adaptive_factor: 熵调整因子
           """

    def __init__(self,
                 max_entropy: float = 0.8,
                 min_samples: int = 3,
                 max_iter: int = 100,
                 radius_shrink_factor: float = 0.9,
                 adaptive_factor: float = 0.1):
        self.max_entropy = max_entropy
        self.min_samples = min_samples
        self.max_iter = max_iter
        self.radius_shrink_factor = radius_shrink_factor
        self.adaptive_factor = adaptive_factor
        self.balls_ = []  # [(center, radius, sample_indices, level, density)]
        self.entropies_ = []
        self.feature_importances_ = None
        self.ball_tree_ = None
        self.n_classes = None  # [MODIFIED] 记录类别数
        self.avg_radius = 0.0  # [MODIFIED] 跟踪平均半径

        # [DEBUG] 参数检查
        print(f"[GranularBall] 初始化参数: "
              f"max_entropy={max_entropy}, min_samples={min_samples}, "
              f"shrink_factor={radius_shrink_factor}")

    def _calculate_entropy(self, y: np.ndarray) -> float:
        """[MODIFIED] 归一化熵计算"""
        # 关键修复：处理空样本情况
        if len(y) == 0:
            print(f"[WARNING] 尝试计算空样本的熵，返回高熵值1.0")
            return 1.0

        _, counts = np.unique(y, return_counts=True)
        if len(counts) == 1:
            return 0.0
        prob = counts / counts.sum()
        raw_entropy = entropy(prob, base=2)
        max_entropy = np.log2(len(counts))  # 计算最大可能熵
        norm_entropy = raw_entropy / (max_entropy + 1e-10)  # 避免除零错误

        # [DEBUG] 熵值输出
        print(f"[Entropy] 计算: 原始熵={raw_entropy:.3f}, "
              f"归一化熵={norm_entropy:.3f}, 类别数={len(counts)}")
        return norm_entropy

    def _get_majority_label(self, y: np.ndarray) -> int:
        """获取多数类标签"""
        # 关键修复：处理空样本情况
        if len(y) == 0:
            print("[WARNING] 尝试获取空样本的多数标签，返回-1")
            return -1

        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]

    def _calculate_center_and_radius(self, X: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """[MODIFIED] 返回密度信息"""
        center = np.median(X, axis=0)
        distances = np.linalg.norm(X - center, axis=1)


        radius = np.mean(distances) + np.std(distances)
        density = len(X) / (radius ** X.shape[1] + 1e-6)

        # [DEBUG] 中心点计算
        print(f"[Center] 计算: 中心点={center}, 半径={radius:.3f}, 密度={density:.3f}")
        return center, max(radius, 1e-6), density

    def _calc_interclass_distance(self, X: np.ndarray, y: np.ndarray) -> float:
        """[MODIFIED] 计算类间距离"""
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return np.inf

        class_centers = [np.median(X[y == cls], axis=0) for cls in unique_classes]
        pairwise_dist = [np.linalg.norm(c1 - c2)
                         for i, c1 in enumerate(class_centers)
                         for c2 in class_centers[i + 1:]]
        return np.min(pairwise_dist)

    # 修改_should_split方法
    def _should_split(self, X: np.ndarray, y: np.ndarray) -> bool:
        entropy_val = self._calculate_entropy(y)
        # 关键修复：初始化avg_radius以避免未定义错误
        if not hasattr(self, 'avg_radius') or self.avg_radius <= 1e-6:
            self.avg_radius = 1.0

        # [DEBUG] 分裂条件检查
        print(f"[SplitCheck] 熵={entropy_val:.3f}, 阈值={self.max_entropy}, 平均半径={self.avg_radius:.3f}")
        return entropy_val > self.max_entropy

    def _adaptive_entropy_threshold(self, n_samples: int, n_features: int) -> float:
        """自适应熵阈值"""
        size_factor = 1 + np.log1p(n_samples) / np.log1p(1000)
        dim_factor = 1 + np.log1p(n_features) / 10
        return min(2.0, self.max_entropy * size_factor * dim_factor)  # 限制最大熵为2

    def _shrink_radius(self, X: np.ndarray, y: np.ndarray,
                       center: np.ndarray, radius: float) -> Tuple[float, np.ndarray]:
        """[MODIFIED] 密度感知半径收缩"""
        distances = np.linalg.norm(X - center, axis=1)
        density = len(X) / (np.pi * (radius ** 2 + 1e-6))
        shrink_factor = max(0.7, 1 - self.adaptive_factor * density)

        # [DEBUG] 收缩过程
        print(f"[Shrink] 收缩前: 半径={radius:.3f}, 密度={density:.3f}, 因子={shrink_factor:.3f}")
        new_radius = radius * shrink_factor
        mask = distances <= new_radius

        # [DEBUG] 收缩结果
        print(f"[Shrink] 收缩后: 新半径={new_radius:.3f}, 保留样本={np.sum(mask)}/{len(X)}")
        return new_radius, mask

    def _split_ball(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """[MODIFIED] 熵驱动的分裂策略"""
        if len(X) <= self.min_samples:
            return [(X, y)]

        current_entropy = self._calculate_entropy(y)
        n_clusters = min(3, int(current_entropy / 0.5) + 2)  # 保证至少分成2簇

        # [DEBUG] 分裂信息
        print(f"[Split] 正在分裂: 样本数={len(X)}, 熵={current_entropy:.3f}, 簇数={n_clusters}")

        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=5)
        labels = kmeans.fit_predict(X)
        return [(X[labels == i], y[labels == i]) for i in range(n_clusters)]

    def _is_overlap(self, ball1: Tuple, ball2: Tuple) -> bool:
        """密度加权的重叠检测"""
        center1, radius1, _, _, density1 = ball1
        center2, radius2, _, _, density2 = ball2
        distance = np.linalg.norm(center1 - center2)

        # 密度越高，重叠阈值越小
        density_factor = min(density1, density2) / (max(density1, density2) + 1e-6)
        overlap_threshold = 0.7 * density_factor  # 基础阈值0.7乘以密度因子
        return distance < (radius1 + radius2) * overlap_threshold

    def _merge_overlapping_balls(self):
        """密度感知的粒球合并"""
        if len(self.balls_) <= 1:
            return

        sorted_balls = sorted(self.balls_, key=lambda x: (-x[3], -x[1]))  # 按层次和半径排序
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

                center, radius, density = self._calculate_center_and_radius(X_merged)
                entropy_val = self._calculate_entropy(y_merged)
                level = min(ball[3] for ball in to_merge) - 1

                merged.append((center, radius, all_indices, level, density))
                self.entropies_.append(entropy_val)
            else:
                merged.append(current)

        self.balls_ = merged


    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GranularBallV4':
        """[MODIFIED] 增强训练过程 - 保留所有样本不丢弃"""
        self.X_ = X
        self.y_ = y
        self.n_classes = len(np.unique(y))  # 初始化类别数
        self.balls_ = []
        self.entropies_ = []
        self.avg_radius = 0.0  # 明确初始化平均半径

        # [DEBUG] 数据检查
        print(f"\n=== 数据检查 ===")
        print(f"样本数: {len(X)}, 特征数: {X.shape[1]}")
        print(f"类别分布: {np.unique(y, return_counts=True)}")
        print(f"全局熵: {self._calculate_entropy(y):.3f}\n")

        queue = [(X.copy(), y.copy(), np.arange(len(X)), 0)]
        iter_count = 0

        while queue and iter_count < self.max_iter:
            X_ball, y_ball, indices, level = queue.pop(0)
            iter_count += 1

            # [DEBUG] 迭代信息
            print(f"\n=== 迭代 {iter_count} ===")
            print(f"当前粒球: 样本数={len(X_ball)}, 层级={level}")

            # 计算中心和半径
            center, radius, density = self._calculate_center_and_radius(X_ball)

            # 计算所有样本到中心的距离
            distances = np.linalg.norm(X_ball - center, axis=1)

            # 标识内部和外部样本
            mask = distances <= radius
            outer_mask = ~mask

            # 内部样本处理
            X_inner, y_inner = X_ball[mask], y_ball[mask]
            inner_indices = indices[mask]

            if len(X_inner) >= self.min_samples:
                # 收缩处理内部样本
                new_radius, inner_mask = self._shrink_radius(X_inner, y_inner, center, radius)
                X_shrunk, y_shrunk = X_inner[inner_mask], y_inner[inner_mask]
                shrunk_indices = inner_indices[inner_mask]

                # 关键修复：只添加有样本的粒球
                if len(X_shrunk) == 0:
                    print(f"[WARNING] 收缩后粒球样本数为0，跳过添加")
                else:
                    # 分裂决策
                    if len(X_shrunk) >= self.min_samples and self._should_split(X_shrunk, y_shrunk):
                        split_results = self._split_ball(X_shrunk, y_shrunk)
                        print(f"[Split] 分裂为{len(split_results)}个子粒球")
                        for X_split, y_split in split_results:
                            split_mask = np.array([np.any(np.all(X_split == x, axis=1)) for x in X_shrunk])
                            split_indices = shrunk_indices[split_mask]
                            if len(split_indices) > 0:
                                queue.append((X_split, y_split, split_indices, level + 1))
                    else:
                        entropy_val = self._calculate_entropy(y_shrunk)
                        self.balls_.append((center, new_radius, shrunk_indices, level, density))
                        self.entropies_.append(entropy_val)
                        # 更新平均半径，避免初始为0
                        if len(self.balls_) > 0:
                            self.avg_radius = np.mean([b[1] for b in self.balls_])

                        # [DEBUG] 粒球添加
                        print(f"[Ball] 新增粒球: 中心={center}, 半径={new_radius:.3f}, "
                              f"样本数={len(shrunk_indices)}, 熵={entropy_val:.3f}")

            # 外部样本处理 (创建新粒球继续处理)
            if np.any(outer_mask):
                X_outer, y_outer = X_ball[outer_mask], y_ball[outer_mask]
                outer_indices = indices[outer_mask]

                # [DEBUG] 外部样本信息
                print(f"[Outer] 发现外部样本: {len(X_outer)}个")

                # 关键修复：只添加有样本的粒球
                if len(X_outer) == 0:
                    print("[WARNING] 外部样本为空，跳过")
                # 如果外部样本数量超过最小值，创建新粒球处理
                elif len(X_outer) >= self.min_samples:
                    queue.append((X_outer, y_outer, outer_indices, level + 1))
                else:
                    # 小样本直接添加到粒球，确保有样本
                    entropy_val = self._calculate_entropy(y_outer)
                    center_outer, radius_outer, density_outer = self._calculate_center_and_radius(X_outer)
                    # 只添加有样本的粒球
                    if len(outer_indices) > 0:
                        self.balls_.append((center_outer, radius_outer, outer_indices, level, density_outer))
                        self.entropies_.append(entropy_val)
                        # 更新平均半径
                        if len(self.balls_) > 0:
                            self.avg_radius = np.mean([b[1] for b in self.balls_])

                        # [DEBUG] 小样本粒球添加
                        print(f"[SmallBall] 新增小粒球: 样本数={len(X_outer)}, 熵={entropy_val:.3f}")
                    else:
                        print("[WARNING] 外部样本为空，跳过添加")

        # 后处理
        self._merge_overlapping_balls()
        self._build_ball_tree()
        self.feature_importances_ = self._calculate_feature_importance()

        # [DEBUG] 最终统计
        print(f"\n=== 训练完成 ===")
        print(f"生成粒球数: {len(self.balls_)}")
        print(f"平均半径: {np.mean([b[1] for b in self.balls_]):.3f}" if self.balls_ else "平均半径: 0.000")
        print(f"平均熵: {np.mean(self.entropies_):.3f}" if self.entropies_ else "平均熵: 0.000")
        print(f"覆盖样本数: {sum([len(ball[2]) for ball in self.balls_])}/{len(X)}")

        return self

    def _build_ball_tree(self):
        """构建粒球空间索引"""
        if self.balls_:
            centers = np.array([ball[0] for ball in self.balls_])
            self.ball_tree_ = KDTree(centers)

    def _calculate_feature_importance(self) -> np.ndarray:
        """基于信息熵的特征重要性"""
        if not self.balls_:
            return np.ones(self.X_.shape[1]) / self.X_.shape[1]

        importances = np.zeros(self.X_.shape[1])
        for ball, entropy_val in zip(self.balls_, self.entropies_):
            center, radius, indices, _, _ = ball
            # 关键修复：跳过空粒球
            if len(indices) == 0:
                continue

            distances = np.abs(self.X_[indices] - center)

            # 计算当前粒球的类间距离
            inter_dist = self._calc_interclass_distance(self.X_[indices], self.y_[indices])

            # 修改权重计算
            weights = (1 - entropy_val) * np.exp(-distances / (radius + 1e-6)) * (1 - inter_dist)
            importances += np.sum(weights, axis=0)

        # 避免除以0错误
        total_importance = importances.sum()
        return importances / (total_importance + 1e-6) if total_importance > 0 else importances

    def predict_ball(self, x: np.ndarray) -> int:
        """使用KD树加速最近粒球查询"""
        if self.ball_tree_ is None:
            return -1
        _, idx = self.ball_tree_.query([x], k=1)
        return idx[0][0]

    @property
    def n_balls(self) -> int:
        return len(self.balls_)