"""SVM 分类器"""
import numpy as np
from typing import Dict, Optional
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time

from .base import BaseClassifier


class SVMClassifier(BaseClassifier):
    """SVM (RBF/Linear/Poly) — 核方法分类器"""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(name='SVM', params=params or {})

    def build(self, **kwargs):
        p = {**self.params, **kwargs}
        p.setdefault('C', 10.0)
        p.setdefault('kernel', 'rbf')
        p.setdefault('gamma', 'scale')
        p.setdefault('probability', True)
        p.setdefault('class_weight', 'balanced')
        p.setdefault('max_iter', 15000)
        p.setdefault('random_state', 42)

        self.model = SVC(**p)
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

        # SVM 没有原生 loss 曲线，记录不同 C 值下的验证准确率
        if X_val is not None and y_val is not None:
            c_values = [0.01, 0.1, 1, 10, 100]
            val_accs = []
            train_accs = []
            kernel = self.params.get('kernel', 'rbf')
            for c in c_values:
                svm_temp = SVC(C=c, kernel=kernel, gamma=self.params.get('gamma', 'scale'),
                    probability=True, random_state=42, max_iter=5000)
                svm_temp.fit(X_train, y_train)
                val_accs.append(accuracy_score(y_val, svm_temp.predict(X_val)))
                train_accs.append(accuracy_score(y_train, svm_temp.predict(X_train)))
            history['loss_curve'] = {
                'C_values': c_values,
                'val_accuracy': val_accs,
                'train_accuracy': train_accs,
                'type': 'accuracy_vs_C',
            }
        else:
            history['loss_curve'] = {'type': 'na_svm_no_validation'}

        self.is_trained = True
        self.training_history = history
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
