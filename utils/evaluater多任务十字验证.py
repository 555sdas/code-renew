from typing import Dict, Optional, Union, List, Tuple
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
    4. 支持多次实验的均值±标准差统计
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
    def aggregate_results(results_list: List[Dict]) -> Dict:
        """
        聚合多次实验的结果，计算均值和标准差

        参数:
            results_list: 多次实验的结果列表

        返回:
            包含均值和标准差的聚合结果字典
        """
        if not results_list:
            return {}

        # 获取所有指标的键
        all_keys = set()
        for result in results_list:
            all_keys.update(result.keys())

        aggregated = {}

        for key in all_keys:
            # 跳过非数值类型的键
            if key in ['problem_type', 'classification_report', 'decision_distribution']:
                continue

            # 收集所有实验中的该指标值
            values = []
            for result in results_list:
                if key in result and result[key] is not None:
                    values.append(result[key])

            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)

        return aggregated

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

    @staticmethod
    def print_aggregated_report(aggregated_results: Dict):
        """
        打印聚合后的评估报告（均值±标准差格式）

        参数:
            aggregated_results: 由aggregate_results方法生成的聚合结果
        """
        print("\n=== 聚合评估报告（Mean ± Std）===")

        # 基础指标
        print("\n[基础指标]")
        print(
            f"准确率: {aggregated_results.get('accuracy_mean', 0):.4f} ± {aggregated_results.get('accuracy_std', 0):.4f}")

        # 二分类/多分类指标
        if 'precision_mean' in aggregated_results:
            print("\n[二分类指标]")
            print(f"精确率: {aggregated_results['precision_mean']:.4f} ± {aggregated_results['precision_std']:.4f}")
            print(f"召回率: {aggregated_results['recall_mean']:.4f} ± {aggregated_results['recall_std']:.4f}")
            print(f"F1分数: {aggregated_results['f1_mean']:.4f} ± {aggregated_results['f1_std']:.4f}")
            if 'auc_roc_mean' in aggregated_results:
                print(f"AUC-ROC: {aggregated_results['auc_roc_mean']:.4f} ± {aggregated_results['auc_roc_std']:.4f}")
        else:
            # 多分类指标
            print("\n[多分类指标]")
            for avg in ['micro', 'macro', 'weighted']:
                print(f"{avg}平均:")
                print(
                    f"  精确率: {aggregated_results[f'precision_{avg}_mean']:.4f} ± {aggregated_results[f'precision_{avg}_std']:.4f}")
                print(
                    f"  召回率: {aggregated_results[f'recall_{avg}_mean']:.4f} ± {aggregated_results[f'recall_{avg}_std']:.4f}")
                print(
                    f"  F1分数: {aggregated_results[f'f1_{avg}_mean']:.4f} ± {aggregated_results[f'f1_{avg}_std']:.4f}")

        # 三支决策结果
        if 'accept_ratio_mean' in aggregated_results:
            print("\n[三支决策]")
            print(
                f"接受区域: {aggregated_results['accept_ratio_mean']:.4f} ± {aggregated_results['accept_ratio_std']:.4f}")
            print(
                f"拒绝区域: {aggregated_results['reject_ratio_mean']:.4f} ± {aggregated_results['reject_ratio_std']:.4f}")
            print(
                f"延迟决策: {aggregated_results['delay_ratio_mean']:.4f} ± {aggregated_results['delay_ratio_std']:.4f}")
            if 'delay_accuracy_mean' in aggregated_results:
                print(
                    f"延迟决策准确率: {aggregated_results['delay_accuracy_mean']:.4f} ± {aggregated_results['delay_accuracy_std']:.4f}")