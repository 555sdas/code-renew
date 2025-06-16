import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances
import torch
import torch.nn as nn
from scipy.stats import entropy


class GranularBallV5:
    """
    基于纯度-熵-注意力三重反馈的粒球生成算法V5
    核心创新：
    1. 动态半径生成（基于纯度、熵和注意力）
    2. 类别驱动的分裂决策
    3. 注意力引导的特征加权
    """

    def __init__(self,
                 min_purity: float = 0.85,
                 max_entropy: float = 0.8,
                 base_radius: float = 1.0,
                 attention_dim: int = 4,
                 learning_rate: float = 0.01,
                 radius_buffer: float = 1.1):  # 添加半径缓冲系数
        self.min_purity = min_purity
        self.max_entropy = max_entropy
        self.base_radius = base_radius
        self.attention_dim = attention_dim
        self.learning_rate = learning_rate
        self.radius_buffer = radius_buffer  # 用于确保覆盖所有样本的缓冲系数
        self.balls_ = []  # [(center, radius, sample_indices, level, density)]
        self.entropies_ = []
        self.purities_ = []
        self.feature_importances_ = None
        self.ball_tree_ = None

        # 注意力网络
        self.attention_net = self._build_attention_net()

    def _build_attention_net(self):
        """构建简易注意力网络"""
        return nn.Sequential(
            nn.Linear(self.attention_dim, 8),
            nn.ReLU(),
            nn.Linear(8, self.attention_dim)
        )

    def _calculate_purity(self, y: np.ndarray) -> float:
        """计算粒球纯度"""
        unique, counts = np.unique(y, return_counts=True)
        return counts.max() / len(y)

    def _calculate_entropy(self, y: np.ndarray) -> float:
        """计算归一化熵"""
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) == 1:
            return 0.0
        return entropy(counts, base=2) / np.log2(len(unique))

    def _attention_weights(self, X: np.ndarray) -> np.ndarray:
        """获取注意力权重"""
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            return self.attention_net(X_tensor).mean(0).numpy()

    def _calculate_dynamic_radius(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """计算动态半径和注意力加权中心"""
        # 计算基本统计量
        purity = self._calculate_purity(y)
        entropy_val = self._calculate_entropy(y)

        # 获取注意力权重
        attn_weights = self._attention_weights(X)

        # 计算注意力加权中心
        attn_X = X * attn_weights
        center = np.median(attn_X, axis=0)

        # 动态半径公式
        distances = np.linalg.norm(attn_X - center, axis=1)
        radius_base = np.percentile(distances, 80)

        # 三重联合调节
        purity_factor = np.sqrt(1 - purity)  # 纯度越低，半径越大
        entropy_factor = np.tanh(entropy_val * 3)  # 熵越高，半径越大
        radius = self.base_radius * radius_base * purity_factor * entropy_factor

        return max(radius, 1e-6), attn_weights

    def _update_attention_net(self, X: np.ndarray, y: np.ndarray):
        """更新注意力网络参数"""
        # 最大化类间距离，最小化类内距离
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return

        # 准备训练数据
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)

        # 前向传播
        attn_weights = self.attention_net(X_tensor)

        # 计算类间距离损失
        class_centers = []
        for cls in unique_classes:
            cls_mask = (y == cls)
            if cls_mask.sum() == 0:
                continue
            center = attn_weights[cls_mask].mean(dim=0)
            class_centers.append(center)

        # 计算类间距离损失
        inter_class_dist = sum([torch.norm(c1 - c2)
                                for i, c1 in enumerate(class_centers)
                                for c2 in class_centers[i + 1:]])

        # 反向传播
        optimizer = torch.optim.SGD(self.attention_net.parameters(), lr=self.learning_rate)
        loss = -inter_class_dist  # 最大化类间距离
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _should_split(self, y: np.ndarray) -> bool:
        """基于纯度决定是否分裂"""
        purity = self._calculate_purity(y)
        return purity < self.min_purity

    def _get_split_count(self, y: np.ndarray) -> int:
        """根据样本种类和熵确定分裂数量"""
        n_classes = len(np.unique(y))
        entropy_val = self._calculate_entropy(y)

        # 计算基础分裂数量
        base_count = max(2, min(n_classes, int(entropy_val * n_classes * 2)))

        # 确保不超过样本种类总数
        return min(base_count, n_classes)

    def _split_ball(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """按类别分裂粒球，允许样本重复分配到多个粒球"""
        n_split = self._get_split_count(y)

        if n_split <= 1:
            return []

        # 获取所有类别
        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)

        # 按样本数量降序排序类别
        sorted_indices = np.argsort(class_counts)[::-1]
        sorted_classes = unique_classes[sorted_indices]
        sorted_counts = class_counts[sorted_indices]

        # 仅选择前n_split个主要类别
        selected_classes = sorted_classes[:n_split]

        split_results = []

        # 第一步：为每个选中的类别创建一个初始粒球
        for cls in selected_classes:
            # 获取当前类别的所有样本
            cls_mask = (y == cls)
            X_cls = X[cls_mask]

            # 计算类别中心
            center = np.mean(X_cls, axis=0)

            # 计算该类所有样本到中心的距离
            distances = np.linalg.norm(X_cls - center, axis=1)
            radius = np.max(distances) * self.radius_buffer

            # 创建一个空的粒球容器
            ball_indices = np.where(cls_mask)[0].tolist()

            # 存储粒球信息：(中心，半径，临时索引)
            split_results.append({
                'center': center,
                'radius': radius,
                'indices': ball_indices,  # 初始只包含本类样本
                'class': cls
            })

        # 第二步：找出所有类别（包括未选中的）的样本
        all_class_indices = np.arange(len(y))

        # 第三步：将样本分配到所有覆盖它的粒球
        for idx in all_class_indices:
            point = X[idx]

            # 检查哪些粒球覆盖此样本
            for ball in split_results:
                distance = np.linalg.norm(point - ball['center'])
                if distance <= ball['radius']:
                    if idx not in ball['indices']:
                        ball['indices'].append(idx)

                    # 更新半径以确保包含新样本（如果必要）
                    if distance > ball['radius']:
                        ball['radius'] = distance * self.radius_buffer

        # 第四步：组织结果
        final_results = []
        for ball in split_results:
            indices_arr = np.array(ball['indices'])
            # 确保去重和排序
            unique_indices = np.unique(indices_arr)
            X_ball = X[unique_indices]
            y_ball = y[unique_indices]

            final_results.append((X_ball, y_ball))

        return final_results

    def _calculate_loss(self, purity: float, entropy_val: float, feature_std: float) -> float:
        """三重联合损失函数"""
        return (1 - purity) * entropy_val * feature_std

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 100) -> 'GranularBallV5':
        self.X_ = X
        self.y_ = y
        self.balls_ = []

        print(f"\n[GranularBallV5] 初始化参数: min_purity={self.min_purity}, max_entropy={self.max_entropy}")
        print(f"初始数据统计: 样本数={len(X)}, 特征数={X.shape[1]}")
        print("类别分布: " + str(dict(zip(*np.unique(y, return_counts=True)))))
        print(f"全局纯度: {self._calculate_purity(y):.2f}, 全局熵: {self._calculate_entropy(y):.2f}")

        # 初始注意力网络训练
        for _ in range(5):
            self._update_attention_net(X, y)

        queue = [(X.copy(), y.copy(), np.arange(len(X)), 0)]
        iter_count = 0

        while queue and iter_count < max_iter:
            X_ball, y_ball, indices, level = queue.pop(0)
            iter_count += 1

            # 计算关键指标
            purity = self._calculate_purity(y_ball)
            entropy_val = self._calculate_entropy(y_ball)

            # 打印迭代信息
            print(f"\n=== 迭代 {iter_count} ===")
            print(f"当前粒球: 层级={level}, 样本数={len(X_ball)}")
            weights_info = "[" + ", ".join([f"{w:.3f}" for w in self._attention_weights(X_ball[:3])]) + "]" if len(
                X_ball) > 0 else "[]"
            print(f"纯度={purity:.3f}, 熵={entropy_val:.3f}, 注意力权重={weights_info}")

            # 分裂决策
            if len(X_ball) > 0 and self._should_split(y_ball):
                print("触发分裂")
                split_results = self._split_ball(X_ball, y_ball)
                print(f"分裂为{len(split_results)}个子粒球")

                # 更新注意力网络
                if len(X_ball) > 0:
                    self._update_attention_net(X_ball, y_ball)

                for X_split, y_split in split_results:
                    if len(X_split) == 0:
                        continue

                    # 正确的索引处理
                    split_indices = np.arange(len(X))[np.isin(X, X_split).all(axis=1)]

                    queue.append((X_split, y_split, split_indices, level + 1))
            else:
                if len(X_ball) > 0:
                    radius, _ = self._calculate_dynamic_radius(X_ball, y_ball)
                    center = np.median(X_ball, axis=0)
                    density = len(X_ball) / (radius ** X_ball.shape[1] + 1e-6)

                    self.balls_.append((center, radius, indices, level, density))
                    self.purities_.append(purity)
                    self.entropies_.append(entropy_val)
                    print(f"新增粒球: 中心={center}, 半径={radius:.3f}, 密度={density:.3f}")

        # 后处理
        self._merge_overlapping_balls()
        self._build_ball_tree()
        self.feature_importances_ = self._calculate_feature_importance()

        # 最终统计
        print(f"\n=== 训练完成 ===")
        print(f"生成粒球数: {len(self.balls_)}")
        if self.balls_:
            print(f"平均半径: {np.mean([b[1] for b in self.balls_]):.3f}")
            print(f"平均纯度: {np.mean(self.purities_):.3f}")
            print(f"平均熵: {np.mean(self.entropies_):.3f}")
            covered_samples = len(set(idx for ball in self.balls_ for idx in ball[2]))
            print(f"覆盖样本数: {covered_samples}/{len(X)}")

        return self

    def _merge_overlapping_balls(self):
        """合并重叠粒球（简化版）"""
        if len(self.balls_) <= 1:
            return

        merged_balls = []
        sorted_balls = sorted(self.balls_, key=lambda x: -x[1])  # 按半径降序

        for ball in sorted_balls:
            center, radius, indices, level, density = ball
            merged = False

            for i, mball in enumerate(merged_balls):
                mcenter, mradius, _, _, _ = mball
                distance = np.linalg.norm(center - mcenter)

                # 检查是否重叠
                if distance < (radius + mradius) * 0.7:
                    # 合并球
                    mcenter = (mcenter * len(mball[2]) + center * len(indices)) / (len(mball[2]) + len(indices))
                    mradius = max(mradius, radius + distance)
                    merged_balls[i] = (mcenter, mradius, np.concatenate([mball[2], indices]), level, density)
                    merged = True
                    break

            if not merged:
                merged_balls.append(ball)

        self.balls_ = merged_balls

    def _build_ball_tree(self):
        """构建粒球空间索引"""
        if self.balls_:
            centers = np.array([ball[0] for ball in self.balls_])
            self.ball_tree_ = KDTree(centers)

    def _calculate_feature_importance(self) -> np.ndarray:
        """基于注意力计算特征重要性"""
        if not self.balls_:
            return np.ones(self.X_.shape[1]) / self.X_.shape[1]

        importances = np.zeros(self.X_.shape[1])
        for ball in self.balls_:
            center, _, indices, _, _ = ball
            X_sub = self.X_[indices]
            attn_weights = self._attention_weights(X_sub)
            distances = np.linalg.norm(X_sub - center, axis=1)
            weights = np.exp(-distances) * attn_weights.mean(axis=0)
            importances += np.sum(weights, axis=0)

        return importances / (importances.sum() + 1e-6)

    def predict_ball(self, x: np.ndarray) -> int:
        """找到最近的粒球"""
        if self.ball_tree_ is None:
            return -1

        _, idx = self.ball_tree_.query([x], k=1)
        return idx[0][0]

    @property
    def n_balls(self) -> int:
        return len(self.balls_)