"""KNN 分类器"""
import numpy as np
from typing import Dict, Optional
from sklearn.neighbors import KNeighborsClassifier
import time

from .base import BaseClassifier


class KNNClassifier(BaseClassifier):
    """K-Nearest Neighbors — 惰性学习算法"""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(name='KNN', params=params or {})

    def build(self, **kwargs):
        p = {**self.params, **kwargs}
        p.setdefault('n_neighbors', 5)
        p.setdefault('weights', 'distance')
        p.setdefault('metric', 'minkowski')
        p.setdefault('p', 2)

        self.model = KNeighborsClassifier(**p)
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

        # KNN 没有 loss 曲线，但记录不同 K 值下的验证准确率作为近似
        if X_val is not None and y_val is not None:
            k_values = [1, 3, 5, 7, 9, 11, 15, 20]
            val_accs = []
            for k in k_values:
                knn_temp = KNeighborsClassifier(
                    n_neighbors=k,
                    weights=self.params.get('weights', 'distance'),
                    metric=self.params.get('metric', 'minkowski'),
                    p=self.params.get('p', 2),
                )
                knn_temp.fit(X_train, y_train)
                val_accs.append(knn_temp.score(X_val, y_val))
            history['loss_curve'] = {
                'k_values': k_values,
                'val_accuracy': val_accs,
                'type': 'val_accuracy_vs_k',
            }
        else:
            history['loss_curve'] = {'type': 'na_lazy_learner'}

        self.is_trained = True
        self.training_history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
