"""推理速度基准测试模块"""
import numpy as np
import time
from typing import Dict, List, Optional
import logging

from models.base import BaseClassifier

logger = logging.getLogger(__name__)


class SpeedBenchmark:
    """测量模型的训练和推理速度"""

    def __init__(self, n_warmup: int = 10, n_repeat: int = 100):
        self.n_warmup = n_warmup
        self.n_repeat = n_repeat
        self.results: Dict = {}

    def benchmark_inference(self, model: BaseClassifier,
                            X_test: np.ndarray) -> Dict:
        """
        测量推理速度
        Returns: {single_sample_ms, batch_100_ms, batch_1000_ms, throughput_per_sec}
        """
        results = {}
        n_samples = X_test.shape[0]

        # Warmup
        for _ in range(self.n_warmup):
            model.predict(X_test[:10])

        # 单样本
        times = []
        for _ in range(self.n_repeat):
            idx = np.random.randint(0, n_samples)
            t0 = time.perf_counter()
            model.predict(X_test[idx:idx+1])
            times.append(time.perf_counter() - t0)
        results['single_sample_ms'] = float(np.median(times) * 1000)

        # 批量 100
        times = []
        for _ in range(self.n_repeat):
            start_idx = np.random.randint(0, max(n_samples - 100, 1))
            batch = X_test[start_idx:start_idx + 100]
            t0 = time.perf_counter()
            model.predict(batch)
            times.append(time.perf_counter() - t0)
        results['batch_100_ms'] = float(np.median(times) * 1000)

        # 批量 1000
        if n_samples >= 1000:
            times = []
            for _ in range(self.n_repeat):
                start_idx = np.random.randint(0, max(n_samples - 1000, 1))
                batch = X_test[start_idx:start_idx + 1000]
                t0 = time.perf_counter()
                model.predict(batch)
                times.append(time.perf_counter() - t0)
            results['batch_1000_ms'] = float(np.median(times) * 1000)
            results['throughput_per_sec'] = float(1000 / results['batch_1000_ms'] * 1000)

        return results

    def benchmark_all(self, models: Dict[str, BaseClassifier],
                      X_test: np.ndarray) -> Dict:
        """对所有训练好的模型进行速度基准测试"""
        logger.info("=" * 50)
        logger.info("推理速度基准测试")
        logger.info("=" * 50)

        for name, model in models.items():
            logger.info(f"  {name}...")
            result = self.benchmark_inference(model, X_test)
            self.results[name] = result
            logger.info(f"    单样本: {result['single_sample_ms']:.4f} ms")
            logger.info(f"    批量100: {result['batch_100_ms']:.4f} ms")
            if 'batch_1000_ms' in result:
                logger.info(f"    批量1000: {result['batch_1000_ms']:.4f} ms")
                logger.info(f"    吞吐量: {result['throughput_per_sec']:.0f} 样本/秒")

        return self.results
