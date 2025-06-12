import numpy as np
from typing import List, Tuple

class GranularBallV1:
    """
    最基础粒球生成方法V1（基于超球体覆盖）
    核心思想：以样本点为球心，固定半径生成粒球
    """
    def __init__(self, radius: float = 0.3):
        """
        :param radius: 粒球半径（所有粒球半径相同）
        """
        self.radius = radius
        self.balls_ = []  # 存储所有粒球：[ (center, radius, sample_indices), ... ]

    def fit(self, X: np.ndarray) -> 'GranularBallV1':
        """
        生成覆盖数据集的粒球
        :param X: 输入数据 (n_samples, n_features)
        :return: self
        """
        n_samples = X.shape[0]
        unassigned = np.ones(n_samples, dtype=bool)  # 标记未分配样本

        while np.any(unassigned):
            # 随机选择一个未分配样本作为球心
            available_indices = np.where(unassigned)[0]
            center_idx = np.random.choice(available_indices)
            center = X[center_idx]

            # 计算该样本与所有样本的距离
            distances = np.linalg.norm(X - center, axis=1)

            # 找出半径内的样本（即该粒球的成员）
            in_ball = distances <= self.radius
            sample_indices = np.where(in_ball)[0].tolist()

            # 记录粒球信息（球心、半径、成员索引）
            self.balls_.append((center, self.radius, sample_indices))

            # 标记这些样本为已分配
            unassigned[in_ball] = False

        return self

    def predict_ball(self, x: np.ndarray) -> int:
        """
        预测样本所属粒球索引
        :param x: 单个样本 (n_features,)
        :return: 粒球索引（未找到返回-1）
        """
        for i, (center, radius, _) in enumerate(self.balls_):
            if np.linalg.norm(x - center) <= radius:
                return i
        return -1

    @property
    def n_balls(self) -> int:
        """返回生成的粒球数量"""
        return len(self.balls_)