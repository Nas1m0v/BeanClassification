"""预处理模块 - 特征缩放、数据分割"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """特征缩放与数据集分割"""

    def __init__(self, scaling: str = 'standard', random_state: int = 42):
        """
        Args:
            scaling: 'standard', 'minmax', 'robust', 或 'none'
            random_state: 随机种子
        """
        self.scaling = scaling
        self.random_state = random_state
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.feature_names: list = []

        if scaling == 'standard':
            self.scaler = StandardScaler()
        elif scaling == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling == 'robust':
            self.scaler = RobustScaler()
        # 'none' → scaler stays None

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """拟合并转换特征"""
        self.feature_names = list(X.columns)
        X_numeric = X.select_dtypes(include=[np.number])
        self.feature_names = list(X_numeric.columns)

        if self.scaler is not None:
            return self.scaler.fit_transform(X_numeric)
        return X_numeric.values

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """使用已拟合的缩放器转换特征"""
        X_numeric = X.select_dtypes(include=[np.number])
        if self.scaler is not None:
            return self.scaler.transform(X_numeric)
        return X_numeric.values

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """反向转换"""
        if self.scaler is not None:
            return self.scaler.inverse_transform(X)
        return X

    def encode_labels(self, y) -> np.ndarray:
        """将字符串标签编码为整数（0 ~ n_classes-1）"""
        if hasattr(y, 'values'):
            y = y.values
        if y.dtype == object or isinstance(y[0], str):
            return self.label_encoder.fit_transform(y)
        return np.asarray(y, dtype=int)

    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """将整数标签解码回字符串"""
        return self.label_encoder.inverse_transform(y_encoded.astype(int)) if hasattr(y_encoded, 'astype') else self.label_encoder.inverse_transform(y_encoded)

    def split_data(self, X: np.ndarray, y: np.ndarray,
                   X_test_orig: np.ndarray = None, y_test_orig: np.ndarray = None,
                   X_val_orig: np.ndarray = None, y_val_orig: np.ndarray = None,
                   test_size: float = 0.2) -> Dict[str, np.ndarray]:
        """
        分割/整理数据集。
        如果提供了 test/val 原始数据，直接使用；否则从训练集中分割

        Returns:
            {'X_train': ..., 'y_train': ..., 'X_val': ..., 'y_val': ..., 'X_test': ..., 'y_test': ...}
        """
        result = {}

        if X_val_orig is not None and y_val_orig is not None:
            result['X_train'] = X
            result['y_train'] = y
            result['X_val'] = X_val_orig
            result['y_val'] = y_val_orig
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, stratify=y, random_state=self.random_state
            )
            result['X_train'] = X_train
            result['y_train'] = y_train
            result['X_val'] = X_val
            result['y_val'] = y_val

        if X_test_orig is not None and y_test_orig is not None:
            result['X_test'] = X_test_orig
            result['y_test'] = y_test_orig
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                result['X_train'], result['y_train'],
                test_size=test_size, stratify=result['y_train'],
                random_state=self.random_state
            )
            result['X_train'] = X_train
            result['y_train'] = y_train
            result['X_test'] = X_test
            result['y_test'] = y_test

        # 打印分割统计
        for key in ['X_train', 'X_val', 'X_test']:
            if key in result:
                logger.info(f"  {key}: {result[key].shape}")

        return result
