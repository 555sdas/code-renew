import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from sklearn.neighbors import KDTree
import math
import copy
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from sklearn.feature_selection import mutual_info_classif


class GranularBallClassCentric:
    """
    优化存储的多线程版本：类别串行处理，类内粒球分裂并行化
    1. 每个类别顺序处理
    2. 类内粒球分裂使用线程池并行
    3. 公用线程安全队列存储最终粒球
    4. 优化存储结构减少内存占用
    """

    def __init__(self,
                 min_purity: float = 0.85,
                 max_iter: int = 100,
                 min_radius: float = 0.000001,
                 max_contain_process_cycles: int = 1,
                 n_workers: int = 2):
        # 参数设置
        self.min_purity = min_purity
        self.max_iter = max_iter
        self.min_radius = min_radius
        self.max_contain_process_cycles = max_contain_process_cycles
        self.n_workers = n_workers

        # 存储结构优化
        self.balls_ = []  # 存储粒球元组(center, radius, indices, label, level)
        self.purities_ = np.array([], dtype=np.float32)  # 使用numpy数组更节省空间
        self.ball_tree_ = None

        # 线程安全数据结构
        self.ball_queue = queue.Queue()  # 线程安全队列
        self.lock = threading.Lock()  # 用于保护共享资源

        # 公共数据(避免多线程复制)
        self.X_full = None  # 原始数据集(只读)
        self.y_full = None  # 原始标签(只读)
        self.global_indices = None  # 全局索引(避免重复计算)

    def _get_label_and_purity(self, y: np.ndarray, target_class: int) -> float:
        """计算纯度(使用更高效的numpy操作)"""
        return np.mean(y == target_class)

    def _calculate_center_and_radius(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """计算中心和半径(优化计算效率)"""
        center = np.mean(X, axis=0)
        distances = np.linalg.norm(X - center, axis=1)
        radius = np.percentile(distances, 80)
        return center, max(radius, self.min_radius)

    def _select_important_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """特征选择(优化内存使用)"""
        n_features = X.shape[1]
        if n_features <= 2:
            return np.arange(n_features)

        feature_scores = np.zeros(n_features, dtype=np.float32)

        # 1. 基于互信息的重要性
        if len(np.unique(y)) > 1:
            try:
                mi_scores = mutual_info_classif(X, y, random_state=42)
                if np.max(mi_scores) > 0:
                    feature_scores += mi_scores / np.max(mi_scores)
            except Exception as e:
                print(f"Mutual info error: {e}")

        # 2. 方差作为备份
        variances = np.var(X, axis=0)
        non_zero_variance_mask = variances > 1e-6
        if np.any(non_zero_variance_mask):
            var_normalized = variances[non_zero_variance_mask] / np.max(variances[non_zero_variance_mask])
            feature_scores[non_zero_variance_mask] += var_normalized

        feature_scores = np.nan_to_num(feature_scores, nan=0.0, posinf=0.0, neginf=0.0)

        if np.all(feature_scores == 0):
            return np.arange(min(4, n_features))

        n_select = min(max(2, int(n_features * 0.2)), 4)
        top_features = np.argpartition(feature_scores, -n_select)[-n_select:]
        return top_features

    def _quadrant_split(self, X: np.ndarray, center: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
        """象限分裂(优化内存使用)"""
        selected_features = self._select_important_features(X, y)
        n_selected = len(selected_features)
        if n_selected == 0:
            n_features = min(4, X.shape[1])
            selected_features = np.arange(n_features)
            n_selected = n_features

        max_splits = 2 ** n_selected
        split_indices = []

        for i in range(max_splits):
            mask = np.ones(len(X), dtype=bool)
            for j, feature_idx in enumerate(selected_features):
                bit = (i >> j) & 1
                mask &= (X[:, feature_idx] >= center[feature_idx]) if bit else (X[:, feature_idx] < center[feature_idx])
            if np.any(mask):
                split_indices.append(mask)
        return split_indices

    def _get_global_purity(self, center: np.ndarray, radius: float, class_label: int) -> float:
        """计算全局纯度(使用预计算的全局数据)"""
        if self.X_full is None:
            return 0.0

        distances = np.linalg.norm(self.X_full - center, axis=1)
        ball_indices = distances <= radius
        if not np.any(ball_indices):
            return 0.0
        return np.mean(self.y_full[ball_indices] == class_label)

    def _process_ball(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                      class_label: int, level: int = 0):
        """处理单个粒球(线程任务) - 优化内存使用"""
        center, radius = self._calculate_center_and_radius(X)
        purity = self._get_global_purity(center, radius, class_label)

        # 停止条件
        if (purity >= self.min_purity or
                level >= self.max_iter or
                radius <= self.min_radius or
                len(X) <= 2):
            distances = np.linalg.norm(self.X_full - center, axis=1)
            global_indices = np.where(distances <= radius)[0]
            with self.lock:
                self.ball_queue.put((center, radius, global_indices, class_label, level))
            return

        # 分裂粒球
        quadrant_masks = self._quadrant_split(X, center, y)
        if len(quadrant_masks) <= 1:
            distances = np.linalg.norm(self.X_full - center, axis=1)
            global_indices = np.where(distances <= radius)[0]
            with self.lock:
                self.ball_queue.put((center, radius, global_indices, class_label, level))
            return

        # 创建子任务 - 避免存储不必要的中间变量
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for mask in quadrant_masks:
                if not np.any(mask):
                    continue
                futures.append(executor.submit(
                    self._process_ball,
                    X[mask], y[mask], indices[mask],
                    class_label, level + 1
                ))
            for future in futures:
                future.result()  # 等待所有子任务完成

    def _build_class_granular_ball(self, X_cls: np.ndarray, y_cls: np.ndarray,
                                   class_label: int, indices_cls: np.ndarray):
        """构建单个类别的粒球系统(优化内存使用)"""
        print(f"\n=== Processing class {class_label} with {len(X_cls)} samples ===")

        # 使用线程池处理
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            executor.submit(
                self._process_ball,
                X_cls, y_cls, indices_cls,
                class_label, 0
            ).result()

        # 收集结果 - 批量处理减少锁竞争
        temp_balls = []
        while not self.ball_queue.empty():
            temp_balls.append(self.ball_queue.get())

        with self.lock:
            self.balls_.extend(temp_balls)
            self.purities_ = np.append(self.purities_, [self._get_global_purity(b[0], b[1], b[3]) for b in temp_balls])

    def _process_fully_contained_balls(self):
        """处理完全被包裹的粒球(优化内存使用)"""
        if len(self.balls_) <= 1:
            return

        print("\n=== Processing fully contained balls ===")
        current_cycle = 0

        while current_cycle < self.max_contain_process_cycles:
            current_cycle += 1
            print(f"\nProcessing cycle {current_cycle}/{self.max_contain_process_cycles}")

            # 使用生成器表达式减少内存
            sorted_balls = sorted(
                ((b[0], b[1], b[2], b[3], b[4]) for b in self.balls_),
                key=lambda x: x[1],
                reverse=True
            )

            new_balls = []
            processed_indices = set()
            total_contained = 0

            for i, (center_A, radius_A, indices_A, label_A, level_A) in enumerate(sorted_balls):
                if i in processed_indices:
                    continue

                contained_balls = []
                for j, (center_B, radius_B, indices_B, label_B, level_B) in enumerate(
                        sorted_balls[i + 1:], start=i + 1
                ):
                    if j in processed_indices:
                        continue

                    dist = np.linalg.norm(center_A - center_B)
                    if radius_A <= self.min_radius + 1e-6 and radius_B <= self.min_radius + 1e-6:
                        continue

                    if dist + radius_B <= radius_A + 1e-6:
                        contained_balls.append(j)

                if not contained_balls:
                    new_balls.append((center_A, radius_A, indices_A, label_A, level_A))
                    processed_indices.add(i)
                    continue

                total_contained += len(contained_balls)
                print(f"Ball {i} (radius={radius_A:.3f}) contains {len(contained_balls)} balls, splitting...")

                # 临时处理(串行)
                temp_balls = []
                self._split_ball_recursive(
                    self.X_full[indices_A],
                    self.y_full[indices_A],
                    indices_A,
                    label_A,
                    level_A,
                    temp_balls
                )

                new_balls.extend(temp_balls)
                processed_indices.add(i)
                processed_indices.update(contained_balls)

            self.balls_ = new_balls
            self.purities_ = np.array([self._get_global_purity(b[0], b[1], b[3]) for b in self.balls_])

            if total_contained == 0:
                print("No new containment relations found, early stopping")
                break

    def _split_ball_recursive(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                              class_label: int, level: int, temp_balls: list):
        """递归分裂粒球(优化内存使用)"""
        center, radius = self._calculate_center_and_radius(X)
        purity = self._get_global_purity(center, radius, class_label)

        if (purity >= self.min_purity or
                level >= self.max_iter or
                radius <= self.min_radius or
                len(X) <= 2):
            distances = np.linalg.norm(self.X_full - center, axis=1)
            global_indices = np.where(distances <= radius)[0]
            temp_balls.append((center, radius, global_indices, class_label, level))
            return

        quadrant_masks = self._quadrant_split(X, center, y)
        if len(quadrant_masks) <= 1:
            distances = np.linalg.norm(self.X_full - center, axis=1)
            global_indices = np.where(distances <= radius)[0]
            temp_balls.append((center, radius, global_indices, class_label, level))
            return

        for mask in quadrant_masks:
            if not np.any(mask):
                continue
            self._split_ball_recursive(
                X[mask], y[mask], indices[mask],
                class_label, level + 1, temp_balls
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GranularBallClassCentric':
        """拟合模型(优化初始化过程)"""
        # 初始化公共数据(只读)
        self.X_full = X
        self.y_full = y
        self.global_indices = np.arange(len(X))

        # 重置存储
        self.balls_ = []
        self.purities_ = np.array([], dtype=np.float32)
        self.ball_tree_ = None

        print(
            f"\nInitial data stats: samples={len(X)}, class distribution={dict(zip(*np.unique(y, return_counts=True)))}")

        # 按类别串行处理
        unique_classes = np.unique(y)
        for cls in unique_classes:
            class_mask = (y == cls)
            self._build_class_granular_ball(
                X[class_mask], y[class_mask],
                cls, self.global_indices[class_mask]
            )

        # 串行处理完全包含关系
        self._process_fully_contained_balls()

        print(f"\nFinal ball stats: total={len(self.balls_)}, avg purity={np.mean(self.purities_):.4f}")

        # 构建KDTree用于预测
        if self.balls_:
            centers = np.array([ball[0] for ball in self.balls_])
            self.ball_tree_ = KDTree(centers)

        return self

    def predict_ball(self, x: np.ndarray) -> int:
        """预测样本所属粒球"""
        if self.ball_tree_ is None or not self.balls_:
            return -1
        _, idx = self.ball_tree_.query([x], k=1)
        return idx[0][0]

    @property
    def n_balls(self) -> int:
        """获取粒球数量"""
        return len(self.balls_)

    def __str__(self):
        """对象字符串表示"""
        info = [
            f"GranularBallClassCentric(Optimized Storage Version)",
            f"- Total balls: {self.n_balls}",
            f"- Min purity: {self.min_purity}",
            f"- Min radius: {self.min_radius}",
            f"- Avg purity: {np.mean(self.purities_):.4f}",
            f"- Min purity: {np.min(self.purities_):.4f}",
            f"- Max purity: {np.max(self.purities_):.4f}",
            f"- Workers: {self.n_workers}"
        ]
        return "\n".join(info)