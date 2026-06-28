"""XGBoost 分类器"""
import numpy as np
from typing import Dict, Optional
import xgboost as xgb
import time

from .base import BaseClassifier


class XGBoostClassifier(BaseClassifier):
    """XGBoost — 梯度提升树（已讲授）"""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(name='XGBoost', params=params or {})

    def build(self, **kwargs):
        p = {**self.params, **kwargs}
        p.setdefault('n_estimators', 300)
        p.setdefault('max_depth', 6)
        p.setdefault('learning_rate', 0.1)
        p.setdefault('subsample', 0.8)
        p.setdefault('colsample_bytree', 0.8)
        p.setdefault('gamma', 0)
        p.setdefault('reg_alpha', 0)
        p.setdefault('reg_lambda', 1)
        p.setdefault('objective', 'multi:softprob')
        p.setdefault('eval_metric', 'mlogloss')
        p.setdefault('random_state', 42)
        p.setdefault('n_jobs', -1)
        p.setdefault('verbosity', 0)
        p.pop('class_weight', None)

        self.model = xgb.XGBClassifier(**p)
        self.params = p

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict:
        if self.model is None:
            self.build()

        history = {'loss_curve': None, 'train_time': 0.0}

        # 准备 eval_set 以记录 loss 曲线
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        t0 = time.time()
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        history['train_time'] = time.time() - t0
        self.train_time = history['train_time']

        # 提取 loss 曲线
        results = self.model.evals_result()
        history['loss_curve'] = {
            'iterations': list(range(1, len(results['validation_0']['mlogloss']) + 1)),
            'train_loss': results['validation_0']['mlogloss'],
            'val_loss': results['validation_1']['mlogloss'] if 'validation_1' in results else None,
            'type': 'mlogloss',
        }

        self.is_trained = True
        self.training_history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self.is_trained:
            return self.model.feature_importances_
        return None
