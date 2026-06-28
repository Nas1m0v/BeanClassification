"""算法对比器 — 统调所有模型的训练、评估和对比"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import time
import logging

from models.base import BaseClassifier
from evaluation.metrics import MetricsCalculator
from config.settings import CLASS_NAMES

logger = logging.getLogger(__name__)


class AlgorithmComparator:
    """运行全部算法并收集对比结果"""

    def __init__(self, class_names: List[str] = None):
        self.metrics_calc = MetricsCalculator()
        self.class_names = class_names or CLASS_NAMES
        self.results: Dict = {}

    def run_single(self, model: BaseClassifier,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   X_val: Optional[np.ndarray] = None,
                   y_val: Optional[np.ndarray] = None) -> Dict:
        """训练并评估单个模型"""
        logger.info(f"--- {model.name} ---")

        # 训练
        model.fit(X_train, y_train, X_val, y_val)

        # 推理
        t0 = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - t0

        # 预测概率
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

        # 训练集预测（用于过拟合分析）
        y_pred_train = model.predict(X_train)

        # 计算指标
        test_metrics = self.metrics_calc.compute_all(
            y_test, y_pred, y_proba, self.class_names,
            train_time=model.train_time,
            inference_time=inference_time,
        )

        train_accuracy = float(np.mean(y_pred_train == y_train))

        result = {
            'model_name': model.name,
            'metrics': test_metrics,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_metrics['accuracy'],
            'predictions': y_pred,
            'probabilities': y_proba,
            'predictions_train': y_pred_train,
            'training_history': model.training_history,
            'feature_importance': model.get_feature_importance(),
            'train_time': model.train_time,
            'inference_time': inference_time,
            'best_params': model.get_params(),
        }

        logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Balanced Acc:  {test_metrics['balanced_accuracy']:.4f}")
        logger.info(f"  Macro F1:      {test_metrics['f1_macro']:.4f}")
        logger.info(f"  Inference:     {inference_time:.4f}s")

        self.results[model.name] = result
        result['_model'] = model  # 保存模型引用
        return result

    def run_all(self, model_builders: Dict[str, callable],
                data: Dict[str, np.ndarray]) -> Dict:
        """
        运行全部模型

        Args:
            model_builders: {name: builder_function(**) -> BaseClassifier}
            data: {'X_train': ..., 'y_train': ..., 'X_val': ..., 'y_val': ...,
                   'X_test': ..., 'y_test': ...}
        Returns:
            {model_name: result_dict}
        """
        logger.info("=" * 60)
        logger.info("开始算法对比实验")
        logger.info("=" * 60)

        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        X_val = data.get('X_val')
        y_val = data.get('y_val')

        for name, build_fn in model_builders.items():
            model = build_fn()
            self.run_single(model, X_train, y_train, X_test, y_test, X_val, y_val)

        logger.info("=" * 60)
        logger.info("算法对比实验完成")
        return self.results

    def create_comparison_table(self) -> pd.DataFrame:
        """创建算法对比汇总表"""
        rows = []
        for name, result in self.results.items():
            m = result['metrics']
            rows.append({
                '算法': name,
                'Accuracy': f"{m['accuracy']:.4f}",
                'Balanced Acc': f"{m['balanced_accuracy']:.4f}",
                'Macro F1': f"{m['f1_macro']:.4f}",
                'Weighted F1': f"{m['f1_weighted']:.4f}",
                'Cohen Kappa': f"{m['cohen_kappa']:.4f}",
                'Training Acc': f"{result['train_accuracy']:.4f}",
                'Train Time (s)': f"{result['train_time']:.2f}",
                'Inference (ms)': f"{result['inference_time'] * 1000:.2f}",
            })
        return pd.DataFrame(rows)

    def create_per_class_table(self) -> pd.DataFrame:
        """创建每类 F1 对比表"""
        rows = []
        for name, result in self.results.items():
            per_class = result['metrics']['per_class']
            for cls_name, cls_metrics in per_class.items():
                rows.append({
                    '算法': name,
                    '类别': cls_name,
                    'Precision': f"{cls_metrics['precision']:.4f}",
                    'Recall': f"{cls_metrics['recall']:.4f}",
                    'F1': f"{cls_metrics['f1']:.4f}",
                    'Support': cls_metrics['support'],
                })
        return pd.DataFrame(rows)

    def statistical_test(self) -> Dict:
        """Friedman 检验 + 逐对比较"""
        from scipy.stats import friedmanchisquare

        # 收集各模型在各 fold 的分数（用 CV 做近似）
        # 这里用各模型的 per-class accuracy 作为观测
        model_names = list(self.results.keys())
        per_class_data = {}
        for name in model_names:
            r = self.results[name]
            pc = r['metrics']['per_class']
            per_class_data[name] = [pc[cls]['recall'] for cls in self.class_names]

        # 如果至少有两个模型
        if len(model_names) >= 2:
            arrays = [per_class_data[n] for n in model_names]
            try:
                stat, p_value = friedmanchisquare(*arrays)
                friedman_result = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'significant_at_0.05': p_value < 0.05,
                }
            except Exception:
                friedman_result = {'error': 'Friedman test failed'}
        else:
            friedman_result = {'error': 'Need >= 2 models'}

        return {
            'friedman': friedman_result,
            'model_names': model_names,
        }
