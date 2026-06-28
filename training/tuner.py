"""超参数调优器"""
import numpy as np
from typing import Dict, Type, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import logging

from models.base import BaseClassifier
from config.hyperparameters import PARAM_GRIDS, TUNE_METHOD, N_ITER_RANDOM
from config.settings import RANDOM_SEED, CV_FOLDS

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """超参数调优 — GridSearchCV 或 RandomizedSearchCV"""

    def __init__(self, cv_folds: int = CV_FOLDS, random_state: int = RANDOM_SEED):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        self.results: Dict[str, Dict] = {}

    def tune(self, model_class, param_grid: Dict, X: np.ndarray, y: np.ndarray,
             model_key: str = '') -> Dict:
        """调优单个模型"""
        method = TUNE_METHOD.get(model_key, 'grid')
        logger.info(f"  调优 {model_key} ({method} search)...")

        # 创建临时模型实例
        temp_model = model_class()
        temp_model.build()

        scoring = 'balanced_accuracy'

        if method == 'random':
            search = RandomizedSearchCV(
                temp_model.model, param_grid,
                n_iter=min(N_ITER_RANDOM, len(param_grid)),
                cv=self.cv, scoring=scoring,
                random_state=self.random_state, n_jobs=-1, verbose=0,
            )
        else:
            search = GridSearchCV(
                temp_model.model, param_grid,
                cv=self.cv, scoring=scoring,
                n_jobs=-1, verbose=0,
            )

        search.fit(X, y)

        result = {
            'best_params': search.best_params_,
            'best_score': float(search.best_score_),
            'cv_results_mean': float(search.cv_results_['mean_test_score'].max()),
            'cv_results_std': float(search.cv_results_['std_test_score'][search.best_index_]),
        }
        self.results[model_key] = result
        logger.info(f"    最佳参数: {search.best_params_}")
        logger.info(f"    最佳 {scoring}: {search.best_score_:.4f}")
        return result

    def tune_all(self, model_classes: Dict[str, type], X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """调优所有模型"""
        logger.info("=" * 50)
        logger.info("开始超参数调优（所有模型）")
        logger.info("=" * 50)

        for key, cls in model_classes.items():
            param_grid = PARAM_GRIDS.get(key, {})
            if param_grid:
                self.tune(cls, param_grid, X, y, model_key=key)

        logger.info("超参数调优完成")
        return self.results
