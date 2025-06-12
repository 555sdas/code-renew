import numpy as np
from typing import List, Tuple, Optional


class GranularBallV2:
    """
    引入特征级注意力的粒球生成方法V2（基于超球体覆盖）
    核心改进：在距离计算中加入可学习的特征权重矩阵
    """

    def __init__(self,
                 radius: float = 0.3,
                 attention_dims: int = 8,
                 attention_lr: float = 0.01,
                 init_weights: Optional[np.ndarray] = None):
        """
        :param radius: 粒球半径（所有粒球半径相同）
        :param attention_dims: 注意力空间的维度
        :param attention_lr: 注意力权重更新的学习率
        :param init_weights: 初始特征权重矩阵（可选）
        """
        self.radius = radius
        self.attention_dims = attention_dims
        self.attention_lr = attention_lr
        self.balls_ = []  # 存储粒球信息
        self.W = None  # 延迟初始化

        # 用于注意力权重更新过程
        self.center_grads_ = []
        self.weights_history_ = []

    def _init_weights(self, n_features: int):
        """根据输入特征维度初始化权重矩阵"""
        # Xavier/Glorot初始化
        limit = np.sqrt(6 / (n_features + self.attention_dims))
        return np.random.uniform(-limit, limit, (self.attention_dims, n_features))

    def fit(self, X: np.ndarray, max_iters: int = 100) -> 'GranularBallV2':
        """
        生成覆盖数据集的粒球（加入特征级注意力机制）
        :param X: 输入数据 (n_samples, n_features)
        :param max_iters: 最大迭代次数（用于权重更新）
        :return: self
        """
        n_samples, n_features = X.shape

        # 延迟初始化权重矩阵
        if self.W is None:
            self.W = self._init_weights(n_features)

        self.center_grads_ = np.zeros((n_samples, n_features))
        self.weights_history_.append(self.W.copy())

        unassigned = np.ones(n_samples, dtype=bool)

        while np.any(unassigned):
            available_indices = np.where(unassigned)[0]
            center_idx = np.random.choice(available_indices)
            center = X[center_idx]

            # 迭代更新注意力权重和粒球中心
            for _ in range(max_iters):
                # 计算加权特征空间中的距离
                weighted_X = X.dot(self.W.T)
                weighted_center = center.dot(self.W.T)
                distances = np.linalg.norm(weighted_X - weighted_center, axis=1)

                # 识别粒球成员
                in_ball = distances <= self.radius
                members = np.where(in_ball)[0]

                # 确保当前中心在成员内
                if not in_ball[center_idx]:
                    center_idx = members[np.argmin(distances[members])]
                    center = X[center_idx]
                    continue

                # 更新中心位置（靠近高质量样本）
                high_quality = distances[members] < (self.radius / 2)
                if np.any(high_quality):
                    new_center = X[members[high_quality]].mean(axis=0)
                    center_grad = new_center - center
                    center += center_grad * self.attention_lr
                    self.center_grads_[center_idx] = center_grad

                    # 修正权重更新方式
                    grad_norm = np.linalg.norm(center_grad)
                    if grad_norm > 1e-6:  # 避免除以零
                        # 将梯度信息扩展到注意力维度
                        expanded_grad = np.tile(center_grad, (self.attention_dims, 1))
                        weight_update = expanded_grad * (center_grad / grad_norm).reshape(1, -1)
                        self.W += weight_update * self.attention_lr
                break

            # 最终确认粒球成员
            weighted_X = X.dot(self.W.T)
            weighted_center = center.dot(self.W.T)
            distances = np.linalg.norm(weighted_X - weighted_center, axis=1)
            in_ball = distances <= self.radius
            sample_indices = np.where(in_ball)[0].tolist()

            self.balls_.append((center, self.radius, sample_indices))
            unassigned[in_ball] = False

            # 记录权重变化
            self.weights_history_.append(self.W.copy())

        return self

    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性（基于注意力权重的范数）"""
        return np.linalg.norm(self.W, axis=0)  # 改为按列计算

    def visualize_attention(self) -> np.ndarray:
        """生成注意力可视化矩阵（便于绘图）"""
        return self.W.T.dot(self.W)  # 改为特征×特征的矩阵

    def predict_ball(self, x: np.ndarray) -> int:
        """
        预测样本所属粒球索引
        :param x: 单个样本 (n_features,)
        :return: 粒球索引（未找到返回-1）
        """
        weighted_x = x.dot(self.W.T)
        for i, (center, radius, _) in enumerate(self.balls_):
            weighted_center = center.dot(self.W.T)
            if np.linalg.norm(weighted_x - weighted_center) <= radius:
                return i
        return -1

    @property
    def n_balls(self) -> int:
        """返回生成的粒球数量"""
        return len(self.balls_)

    @property
    def feature_weights(self) -> np.ndarray:
        """返回最终的特征权重矩阵"""
        return self.W