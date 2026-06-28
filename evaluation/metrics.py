"""评估指标计算模块"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef,
    classification_report,
)
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """计算全面的分类评估指标"""

    def compute_all(self, y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: Optional[np.ndarray] = None,
                    class_names: Optional[List[str]] = None,
                    train_time: Optional[float] = None,
                    inference_time: Optional[float] = None) -> Dict:
        """计算全部指标"""
        metrics = {}

        # 基本指标
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
        metrics['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall_weighted'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

        # Cohen's Kappa 和 MCC
        metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
        metrics['matthews_corrcoef'] = float(matthews_corrcoef(y_true, y_pred))

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # 逐类指标
        if class_names is None:
            class_names = [str(i) for i in range(cm.shape[0])]

        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics['per_class'] = {}
        for i, name in enumerate(class_names):
            metrics['per_class'][name] = {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1': float(per_class_f1[i]),
                'support': int(np.sum(y_true == i)),
            }

        # 时间指标
        if train_time is not None:
            metrics['train_time_seconds'] = float(train_time)
        if inference_time is not None:
            metrics['inference_time_seconds'] = float(inference_time)

        return metrics

    def confusion_matrix_to_df(self, cm: np.ndarray,
                               class_names: List[str]) -> pd.DataFrame:
        """混淆矩阵转 DataFrame"""
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
        return df

    def classification_report_df(self, y_true, y_pred, class_names) -> pd.DataFrame:
        """分类报告转 DataFrame"""
        report = classification_report(
            y_true, y_pred, target_names=class_names,
            output_dict=True, zero_division=0
        )
        return pd.DataFrame(report).T
