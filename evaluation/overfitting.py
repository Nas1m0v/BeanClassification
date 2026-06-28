"""过拟合分析模块"""
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class OverfittingAnalyzer:
    """分析模型的过拟合程度"""

    def assess_overfitting(self, train_acc: float, test_acc: float) -> str:
        """根据训练/测试精度差评定过拟合等级"""
        gap = train_acc - test_acc
        if gap < 0.02:
            return '低 (Low)'
        elif gap < 0.05:
            return '中等 (Moderate)'
        elif gap < 0.10:
            return '高 (High)'
        else:
            return '严重 (Severe)'

    def analyze_single(self, train_acc: float, test_acc: float) -> Dict:
        """分析单个模型的过拟合"""
        gap = train_acc - test_acc
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'gap': gap,
            'gap_ratio': gap / (test_acc + 1e-8),
            'assessment': self.assess_overfitting(train_acc, test_acc),
        }

    def analyze_all(self, results: Dict) -> Dict:
        """分析全部模型的过拟合情况"""
        logger.info("=" * 50)
        logger.info("过拟合分析")
        logger.info("=" * 50)

        overfitting_results = {}
        for model_name, result in results.items():
            train_acc = result.get('train_accuracy', 0)
            test_acc = result.get('test_accuracy', 0)
            analysis = self.analyze_single(train_acc, test_acc)
            overfitting_results[model_name] = analysis
            logger.info(f"  {model_name}: Train={train_acc:.4f}, Test={test_acc:.4f}, "
                       f"Gap={analysis['gap']:.4f} → {analysis['assessment']}")

        return overfitting_results
