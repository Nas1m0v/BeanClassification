"""模型基类"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, List
import time


class BaseClassifier(ABC):
    """所有分类器的抽象基类"""

    def __init__(self, name: str, params: Optional[Dict] = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_trained = False
        self.training_history: Dict = {}
        self.train_time: float = 0.0
        self.feature_names: List[str] = []

    @abstractmethod
    def build(self, **kwargs):
        """构建底层模型"""
        ...

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> Dict:
        """训练模型，返回 training_history"""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        ...

    def get_params(self) -> Dict:
        """返回当前超参数"""
        return self.params

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """返回特征重要性（如果模型支持）"""
        return None

    def get_loss_curve(self) -> Optional[Dict]:
        """返回损失曲线数据"""
        return self.training_history.get('loss_curve', None)
