"""Random Forest 分类器"""
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier as SklearnRF
import time

from .base import BaseClassifier


class RandomForestClassifier(BaseClassifier):
    """随机森林 — Bagging 集成方法"""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(name='Random Forest', params=params or {})

    def build(self, **kwargs):
        p = {**self.params, **kwargs}
        p.setdefault('n_estimators', 200)
        p.setdefault('max_depth', 20)
        p.setdefault('min_samples_split', 2)
        p.setdefault('min_samples_leaf', 1)
        p.setdefault('class_weight', 'balanced')
        p.setdefault('bootstrap', True)
        p.setdefault('oob_score', True)
        p.setdefault('random_state', 42)
        p.setdefault('n_jobs', -1)

        self.model = SklearnRF(**p)
        self.params = p

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict:
        if self.model is None:
            self.build()

        history = {'loss_curve': None, 'train_time': 0.0}

        t0 = time.time()
        self.model.fit(X_train, y_train)
        history['train_time'] = time.time() - t0
        self.train_time = history['train_time']

        # RF 没有原生 loss 曲线；使用 OOB error 和不同 n_estimators 下的验证准确率
        n_est_range = [10, 20, 50, 100, 150, 200, 300, 500]
        val_accs = []
        oob_errors = []
        n_est_actual = self.params.get('n_estimators', 200)

        for n in n_est_range:
            if n > n_est_actual:
                continue
            rf_temp = SklearnRF(
                n_estimators=n, max_depth=self.params.get('max_depth', 20),
                class_weight='balanced', oob_score=(n == max(n_est_range) or n >= 50),
                random_state=42, n_jobs=-1
            )
            rf_temp.fit(X_train, y_train)
            if X_val is not None and y_val is not None:
                val_accs.append(rf_temp.score(X_val, y_val))
            if rf_temp.oob_score:
                oob_errors.append(1 - rf_temp.oob_score_)
            else:
                oob_errors.append(None)

        history['loss_curve'] = {
            'n_estimators': [n for n in n_est_range if n <= n_est_actual],
            'val_accuracy': val_accs,
            'oob_error': oob_errors,
            'type': 'oob_error_and_val_acc',
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
