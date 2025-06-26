from typing import Dict, Optional, Union, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)


class SmartEvaluator:
    """
    智能评估器（自动检测二分类/多分类）
    功能：
    1. 自动检测问题类型（二分类/多分类）
    2. 动态切换评估指标
    3. 支持三支决策统计
    """

    @staticmethod
    def _detect_problem_type(y: np.ndarray) -> str:
        """自动检测问题类型"""
        unique_classes = np.unique(y)
        return "binary" if len(unique_classes) <= 2 else "multiclass"

    @staticmethod
    def evaluate(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            similarities: Optional[np.ndarray] = None,
            alpha: Optional[float] = None,
            beta: Optional[float] = None,
            class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        智能评估主函数

        参数:
            y_true: 真实标签
            y_pred: 预测标签
            similarities: 相似度分数（三支决策用）
            alpha: 接受阈值
            beta: 拒绝阈值
            class_names: 类别名称列表

        返回:
            评估结果字典（自动适配二分类/多分类）
        """
        # 自动检测问题类型
        problem_type = SmartEvaluator._detect_problem_type(y_true)

        # 基础指标计算
        metrics = {
            'problem_type': problem_type,
            'accuracy': accuracy_score(y_true, y_pred)
        }

        # 动态选择评估策略
        if problem_type == "binary":
            # 二分类专用指标
            metrics.update({
                'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
                'auc_roc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else None
            })
        else:
            # 多分类指标（计算所有平均方式）
            avg_methods = ['micro', 'macro', 'weighted']
            for avg in avg_methods:
                metrics.update({
                    f'precision_{avg}': precision_score(y_true, y_pred, average=avg, zero_division=0),
                    f'recall_{avg}': recall_score(y_true, y_pred, average=avg, zero_division=0),
                    f'f1_{avg}': f1_score(y_true, y_pred, average=avg, zero_division=0)
                })

            # 添加分类报告
            metrics['classification_report'] = classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
            )

        # 三支决策统计（如果提供了相关参数）
        if similarities is not None and alpha is not None and beta is not None:
            decisions = np.where(
                similarities >= alpha, "accept",
                np.where(similarities < beta, "reject", "delay")
            )

            decision_stats = {
                'accept_ratio': float(np.mean(decisions == "accept")),
                'reject_ratio': float(np.mean(decisions == "reject")),
                'delay_ratio': float(np.mean(decisions == "delay")),
                'decision_distribution': {
                    'accept': int(sum(decisions == "accept")),
                    'reject': int(sum(decisions == "reject")),
                    'delay': int(sum(decisions == "delay"))
                }
            }

            # 延迟决策的准确率
            delay_mask = (decisions == "delay")
            if sum(delay_mask) > 0:
                decision_stats['delay_accuracy'] = float(accuracy_score(y_true[delay_mask], y_pred[delay_mask]))

            metrics.update(decision_stats)

        return metrics

    @staticmethod
    def print_report(eval_results: Dict):
        """智能打印评估报告"""
        print(f"\n=== 评估报告（{eval_results['problem_type']}）===")

        # 通用指标
        print(f"\n[基础指标]")
        print(f"准确率: {eval_results['accuracy']:.4f}")

        # 按问题类型打印
        if eval_results['problem_type'] == "binary":
            print(f"\n[二分类指标]")
            print(f"精确率: {eval_results['precision']:.4f}")
            print(f"召回率: {eval_results['recall']:.4f}")
            print(f"F1分数: {eval_results['f1']:.4f}")
            if 'auc_roc' in eval_results:
                print(f"AUC-ROC: {eval_results.get('auc_roc', 'N/A'):.4f}")
        else:
            print(f"\n[多分类指标]")
            for avg in ['micro', 'macro', 'weighted']:
                print(f"{avg}平均:")
                print(f"  精确率: {eval_results[f'precision_{avg}']:.4f}")
                print(f"  召回率: {eval_results[f'recall_{avg}']:.4f}")
                print(f"  F1分数: {eval_results[f'f1_{avg}']:.4f}")

        # 三支决策结果
        if 'accept_ratio' in eval_results:
            print(f"\n[三支决策]")
            print(f"接受区域: {eval_results['accept_ratio']:.1%}")
            print(f"拒绝区域: {eval_results['reject_ratio']:.1%}")
            print(f"延迟决策: {eval_results['delay_ratio']:.1%}")
            if 'delay_accuracy' in eval_results:
                print(f"延迟决策准确率: {eval_results['delay_accuracy']:.4f}")

        # 混淆矩阵
        if 'confusion_matrix' in eval_results:
            print("\n[混淆矩阵]")
            print(np.array(eval_results['confusion_matrix']))

        # # 分类报告（多分类）
        # if 'classification_report' in eval_results:
        #     print("\n[分类报告]")
        #     print(classification_report(
        #         None, None,
        #         target_names=eval_results['classification_report'].get('target_names'),
        #         output_dict=False,
        #         digits=4
        #     ))