"""模型训练器 — 训练、交叉验证、学习曲线"""
import numpy as np
from typing import Dict, List, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
import logging

from models.base import BaseClassifier

logger = logging.getLogger(__name__)


class Trainer:
    """模型训练编排器"""

    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def train(self, model: BaseClassifier,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> BaseClassifier:
        """训练模型并返回"""
        logger.info(f"  训练 {model.name}...")
        model.fit(X_train, y_train, X_val, y_val)
        train_acc = model.predict(X_train)
        logger.info(f"    训练准确率: {np.mean(train_acc == y_train):.4f}")
        if X_val is not None and y_val is not None:
            val_acc = np.mean(model.predict(X_val) == y_val)
            logger.info(f"    验证准确率: {val_acc:.4f}")
        logger.info(f"    训练时间: {model.train_time:.2f}s")
        return model

    def cross_validate(self, model: BaseClassifier,
                       X: np.ndarray, y: np.ndarray) -> Dict:
        """K 折交叉验证"""
        logger.info(f"  {model.name} {self.cv_folds}-fold CV...")
        model.build()
        scores = cross_val_score(
            model.model, X, y, cv=self.cv, scoring='accuracy', n_jobs=-1
        )
        cv_results = {
            'fold_scores': scores.tolist(),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
        }
        logger.info(f"    CV Accuracy: {cv_results['mean']:.4f} ± {cv_results['std']:.4f}")
        return cv_results

    def learning_curve_data(self, model: BaseClassifier,
                            X_train: np.ndarray, y_train: np.ndarray,
                            train_sizes: List[float] = None) -> Dict:
        """生成学习曲线数据"""
        if train_sizes is None:
            train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        logger.info(f"  生成 {model.name} 学习曲线...")
        model.build()
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model.model, X_train, y_train,
            train_sizes=train_sizes, cv=self.cv,
            scoring='accuracy', n_jobs=-1, random_state=self.random_state,
        )
        return {
            'train_sizes': [float(ts) for ts in train_sizes_abs],
            'train_sizes_ratio': train_sizes,
            'train_mean': [float(s) for s in train_scores.mean(axis=1)],
            'train_std': [float(s) for s in train_scores.std(axis=1)],
            'val_mean': [float(s) for s in val_scores.mean(axis=1)],
            'val_std': [float(s) for s in val_scores.std(axis=1)],
        }
