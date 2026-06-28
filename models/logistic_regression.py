"""Logistic Regression 分类器（基线模型）"""
import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
import time

from .base import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """多分类 Logistic Regression — 线性基线方法"""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(name='Logistic Regression', params=params or {})

    def build(self, **kwargs):
        p = {**self.params, **kwargs}
        p.setdefault('C', 1.0)
        p.setdefault('penalty', 'l2')
        p.setdefault('solver', 'lbfgs')
        p.setdefault('max_iter', 3000)
        p.setdefault('class_weight', 'balanced')
        p.setdefault('random_state', 42)
        # 不设置 multi_class (新版 sklearn 已弃用)

        self.model = LogisticRegression(**p)
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

        # 使用已训练模型获取 loss 曲线（通过 predict_proba 计算 log_loss）
        from sklearn.metrics import log_loss
        proba_train = self.model.predict_proba(X_train)
        loss_train = log_loss(y_train, proba_train)

        losses_val = None
        if X_val is not None and y_val is not None:
            proba_val = self.model.predict_proba(X_val)
            losses_val = log_loss(y_val, proba_val)

        history['loss_curve'] = {
            'iterations': [1],
            'train_loss': [loss_train],
            'val_loss': [losses_val] if losses_val is not None else None,
            'type': 'log_loss',
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
            return np.abs(self.model.coef_).mean(axis=0)
        return None
