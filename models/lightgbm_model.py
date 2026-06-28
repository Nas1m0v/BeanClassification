"""LightGBM 分类器 — ★ 课堂未讲算法

LightGBM (Light Gradient Boosting Machine) 是微软于2017年发布的高性能梯度提升框架。
与 XGBoost 的核心区别:
1. **Leaf-wise 生长策略**: 按叶子分裂（选择增益最大的叶子），而非 XGBoost 的按层生长
   - 优点: 收敛更快，同等迭代次数下误差更低
   - 风险: 可能过拟合（通过 min_child_samples / num_leaves 控制）
2. **Gradient-based One-Side Sampling (GOSS)**: 保留大梯度样本，对小梯度样本随机采样
3. **Exclusive Feature Bundling (EFB)**: 将互斥的稀疏特征捆绑，减少特征维度
4. **直方图算法**: 将连续特征离散化为固定数量的 bin，显著降低内存和计算开销

参考: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017.
"""
import numpy as np
from typing import Dict, Optional
import lightgbm as lgb
import time

from .base import BaseClassifier


class LightGBMClassifier(BaseClassifier):
    """LightGBM — Leaf-wise 梯度提升树（课堂未讲）"""

    def __init__(self, params: Optional[Dict] = None):
        super().__init__(name='LightGBM ★', params=params or {})

    def build(self, **kwargs):
        p = {**self.params, **kwargs}
        p.setdefault('n_estimators', 300)
        p.setdefault('num_leaves', 63)
        p.setdefault('learning_rate', 0.05)
        p.setdefault('min_child_samples', 20)
        p.setdefault('subsample', 0.8)
        p.setdefault('colsample_bytree', 0.8)
        p.setdefault('reg_alpha', 0.1)
        p.setdefault('reg_lambda', 0.1)
        p.setdefault('objective', 'multiclass')
        p.setdefault('metric', 'multi_logloss')
        p.setdefault('random_state', 42)
        p.setdefault('n_jobs', -1)
        p.setdefault('verbosity', -1)
        p.setdefault('class_weight', 'balanced')
        p.pop('class_weight', None)  # LightGBM 不用 class_weight

        # 使用 class weight 参数
        if 'is_unbalance' not in p:
            p.setdefault('is_unbalance', True)

        self.model = lgb.LGBMClassifier(**p)
        self.params = p

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict:
        if self.model is None:
            self.build()

        history = {'loss_curve': None, 'train_time': 0.0}

        # LightGBM 原生支持 eval_set 记录每轮 loss
        n_estimators = self.params.get('n_estimators', 300)
        callbacks = []

        # 使用 early stopping 防止过拟合
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))

        t0 = time.time()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)] if X_val is not None else [(X_train, y_train)],
            eval_names=['train', 'val'] if X_val is not None else ['train'],
            callbacks=callbacks if callbacks else None,
        )
        history['train_time'] = time.time() - t0
        self.train_time = history['train_time']

        # 提取 loss 曲线
        evals_result = self.model.evals_result_
        if 'train' in evals_result and 'multi_logloss' in evals_result['train']:
            history['loss_curve'] = {
                'iterations': list(range(1, len(evals_result['train']['multi_logloss']) + 1)),
                'train_loss': evals_result['train']['multi_logloss'],
                'val_loss': evals_result['val']['multi_logloss'] if 'val' in evals_result else None,
                'type': 'multi_logloss',
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
