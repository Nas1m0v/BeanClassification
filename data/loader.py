"""数据加载模块 - 统一读取三个CSV文件"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


class DataLoader:
    """加载 Dry Bean Dataset 的三个分裂文件，提供统一访问接口"""

    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 包含三个 CSV 文件的目录路径
        """
        self.data_dir = data_dir

    def _read_csv(self, filename: str) -> pd.DataFrame:
        """读取 CSV 文件，自动处理 BOM 和编码问题"""
        import os
        path = os.path.join(self.data_dir, filename)
        df = pd.read_csv(path, encoding='utf-8-sig')
        # 清理列名中的空格和 BOM
        df.columns = df.columns.str.strip().str.replace('﻿', '')
        return df

    def load_train(self) -> pd.DataFrame:
        """加载训练集"""
        return self._read_csv('Dry_Bean_Dataset_Dirty_train.csv')

    def load_test(self) -> pd.DataFrame:
        """加载测试集"""
        return self._read_csv('Dry_Bean_Dataset_Dirty_test.csv')

    def load_val(self) -> pd.DataFrame:
        """加载验证集"""
        return self._read_csv('Dry_Bean_Dataset_Dirty_val.csv')

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """返回 (train_df, test_df, val_df)"""
        return self.load_train(), self.load_test(), self.load_val()

    def get_summary(self, df: pd.DataFrame) -> Dict:
        """返回数据集摘要统计"""
        summary = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,
            'n_classes': df['Class'].nunique() if 'Class' in df.columns else 0,
            'class_distribution': df['Class'].value_counts().to_dict() if 'Class' in df.columns else {},
            'missing_counts': df.isin(['', '?', None]).sum().to_dict(),
            'dtype_info': {col: str(dtype) for col, dtype in df.dtypes.items()},
        }
        # 检查 NaN
        nan_counts = df.isna().sum()
        for col, cnt in nan_counts.items():
            if cnt > 0:
                summary['missing_counts'][col] = summary['missing_counts'].get(col, 0) + cnt
        return summary
