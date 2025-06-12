from typing import Dict

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


class ThreeWayEvaluator:
    """
    三支决策模型评估器
    功能：
    1. 计算传统分类指标
    2. 统计三支决策结果分布
    3. 可视化评估结果
    """

    @staticmethod
    def evaluate(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 similarities: np.ndarray,
                 alpha: float,
                 beta: float) -> Dict:
        """
        综合评估模型性能
        """
        # 计算传统指标
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        # 计算三支决策分布
        decisions = np.where(
            similarities >= alpha, "accept",
            np.where(similarities < beta, "reject", "delay")
        )

        decision_stats = {
            'accept_ratio': np.mean(decisions == "accept"),
            'reject_ratio': np.mean(decisions == "reject"),
            'delay_ratio': np.mean(decisions == "delay"),
            'decision_distribution': {
                'accept': int(sum(decisions == "accept")),
                'reject': int(sum(decisions == "reject")),
                'delay': int(sum(decisions == "delay"))
            }
        }

        # 延迟决策的准确率
        delay_mask = (decisions == "delay")
        if sum(delay_mask) > 0:
            delay_acc = accuracy_score(y_true[delay_mask], y_pred[delay_mask])
            decision_stats['delay_accuracy'] = float(delay_acc)

        return {**metrics, **decision_stats}

    @staticmethod
    def print_report(eval_results: Dict):
        """打印评估报告"""
        print("\n=== 分类性能 ===")
        print(f"准确率: {eval_results['accuracy']:.4f}")
        print(f"精确率: {eval_results['precision']:.4f}")
        print(f"召回率: {eval_results['recall']:.4f}")
        print(f"F1分数: {eval_results['f1']:.4f}")

        print("\n=== 三支决策分布 ===")
        print(f"接受区域: {eval_results['accept_ratio']:.1%}")
        print(f"拒绝区域: {eval_results['reject_ratio']:.1%}")
        print(f"延迟决策: {eval_results['delay_ratio']:.1%}")

        if 'delay_accuracy' in eval_results:
            print(f"延迟决策准确率: {eval_results['delay_accuracy']:.4f}")

        print("\n混淆矩阵:")
        print(np.array(eval_results['confusion_matrix']))