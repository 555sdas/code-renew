import numpy as np
from typing import Union, Literal

class ThreeWayDecisionV1:
    """
    基础三支决策方法V1（基于固定阈值）
    核心逻辑：
    - 若相似度 >= α: 接受
    - 若相似度 < β: 拒绝
    - 否则: 延迟决策
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        """
        :param alpha: 接受阈值（≥α时判定为正类）
        :param beta: 拒绝阈值（<β时判定为负类）
        """
        assert alpha > beta, "α必须大于β"
        self.alpha = alpha
        self.beta = beta

    def predict(self, similarity: float) -> Literal["accept", "reject", "delay"]:
        """
        三支决策预测
        :param similarity: 样本与粒球的相似度（范围[0,1]）
        :return: 决策结果
        """
        if similarity >= self.alpha:
            return "accept"
        elif similarity < self.beta:
            return "reject"
        else:
            return "delay"

    def batch_predict(self, similarities: np.ndarray) -> np.ndarray:
        """
        批量预测
        :param similarities: 相似度数组 (n_samples,)
        :return: 决策结果数组 (n_samples,)
        """
        return np.array([self.predict(s) for s in similarities])

    def set_thresholds(self, alpha: float, beta: float) -> None:
        """动态更新阈值"""
        assert alpha > beta, "α必须大于β"
        self.alpha = alpha
        self.beta = beta