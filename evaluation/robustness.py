"""鲁棒性测试模块"""
import numpy as np
from typing import Dict, List
import time
import logging

from models.base import BaseClassifier
from data.augmenter import NoiseAugmenter

logger = logging.getLogger(__name__)


class RobustnessTester:
    """测试模型在各种噪声条件下的鲁棒性"""

    def __init__(self, random_state: int = 42):
        self.augmenter = NoiseAugmenter(random_state=random_state)
        self.results: Dict = {}
        self.noise_configs = self.augmenter.get_noise_configs()

    def test_single_noise(self, model_builder: callable,
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          noise_config: Dict, n_classes: int = 7) -> float:
        """
        在带噪声训练数据上训练，在干净测试集上评估
        Returns: accuracy
        """
        # 应用噪声
        X_noisy, y_noisy = self.augmenter.apply_noise(
            X_train, y_train, noise_config, n_classes
        )

        # 训练模型
        model = model_builder()
        model.build()
        model.fit(X_noisy, y_noisy)

        # 评估
        y_pred = model.predict(X_test)
        return float(np.mean(y_pred == y_test))

    def test_all_noises(self, model_builders: Dict[str, callable],
                        data: Dict[str, np.ndarray]) -> Dict:
        """
        对所有模型和噪声配置进行测试
        Returns: {model_name: {noise_label: accuracy}}
        """
        logger.info("=" * 60)
        logger.info("开始鲁棒性测试")
        logger.info("=" * 60)

        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        max_y = int(max(y_train.max(), y_test.max())) + 1

        all_results = {}

        for model_name, build_fn in model_builders.items():
            logger.info(f"--- {model_name} ---")
            model_results = {}

            # 干净基线
            logger.info(f"  [0/{len(self.noise_configs)}] 干净数据基线")
            base_model = build_fn()
            base_model.build()
            base_model.fit(X_train, y_train)
            clean_acc = float(np.mean(base_model.predict(X_test) == y_test))
            model_results['干净数据'] = clean_acc
            logger.info(f"    干净: {clean_acc:.4f}")

            for i, config in enumerate(self.noise_configs):
                label = config['label']
                logger.info(f"  [{i+1}/{len(self.noise_configs)}] {label}")
                try:
                    acc = self.test_single_noise(
                        build_fn, X_train, y_train, X_test, y_test,
                        config, n_classes=max_y
                    )
                    model_results[label] = acc
                    logger.info(f"    {label}: {acc:.4f} (下降 {clean_acc - acc:.4f})")
                except Exception as e:
                    logger.error(f"    {label}: 失败 - {e}")
                    model_results[label] = None

            all_results[model_name] = model_results

        self.results = all_results
        logger.info("鲁棒性测试完成")
        return all_results

    def compute_degradation(self) -> Dict:
        """计算各条件下准确率下降"""
        degradation = {}
        for model_name, results in self.results.items():
            clean = results.get('干净数据', 0)
            degradation[model_name] = {}
            for condition, acc in results.items():
                if condition == '干净数据' or acc is None:
                    continue
                degradation[model_name][condition] = clean - acc
        return degradation

    def create_robustness_table(self) -> dict:
        """创建鲁棒性汇总表"""
        degradation = self.compute_degradation()
        return {
            'absolute': self.results,
            'degradation': degradation,
        }
