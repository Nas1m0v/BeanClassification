"""数据清洗模块 - 处理标签噪声、缺失值、后缀污染、异常值"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.impute import KNNImputer
import logging

logger = logging.getLogger(__name__)

# 目标类别（7 种标准豆类）
CANONICAL_CLASSES = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

# 标签映射表：将噪声标签映射到标准标签
LABEL_MAP = {
    # 尾部空格
    'DERMASON ': 'DERMASON',
    'SIRA ': 'SIRA',
    'SEKER ': 'SEKER',
    'HOROZ ': 'HOROZ',
    'CALI ': 'CALI',
    'BARBUNYA ': 'BARBUNYA',
    'BOMBAY ': 'BOMBAY',
    # 小写变体
    'dermason': 'DERMASON',
    'sira': 'SIRA',
    'seker': 'SEKER',
    'horoz': 'HOROZ',
    'cali': 'CALI',
    'barbunya': 'BARBUNYA',
    'bombay': 'BOMBAY',
    # Leet-speak 替换 (3→E, 0→O)
    'D3RMAS0N': 'DERMASON',
    'S3K3R': 'SEKER',
    'H0R0Z': 'HOROZ',
    'B0MBAY': 'BOMBAY',
}


class DataCleaner:
    """清洗脏数据：标签、缺失值、后缀、异常值"""

    def __init__(self, impute_strategy: str = 'knn', random_state: int = 42):
        """
        Args:
            impute_strategy: 'knn', 'median', 或 'iterative'
            random_state: 随机种子
        """
        self.impute_strategy = impute_strategy
        self.random_state = random_state
        self.cleaning_stats: Dict = {}
        self.knn_imputer: Optional[KNNImputer] = None

    def clean_labels(self, y: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        清洗标签噪声：
        1. 去掉首尾空格
        2. 通过映射表修正小写和 leet-speak 变体
        3. 验证清洗后的标签都在7个标准类别中

        Returns:
            (cleaned_series, stats_dict)
        """
        stats = {
            'original_unique': y.nunique(),
            'label_changes': {},
            'invalid_removed': 0,
        }

        cleaned = y.copy()

        # Step 1: 去除首尾空格
        cleaned = cleaned.str.strip()

        # Step 2: 应用映射表
        for dirty, clean in LABEL_MAP.items():
            mask = cleaned == dirty
            if mask.any():
                stats['label_changes'][dirty] = {'count': mask.sum(), 'mapped_to': clean}
                cleaned[mask] = clean

        # Step 3: 验证
        valid_mask = cleaned.isin(CANONICAL_CLASSES)
        invalid_count = (~valid_mask).sum()

        if invalid_count > 0:
            invalid_labels = cleaned[~valid_mask].unique()
            logger.warning(f"发现 {invalid_count} 条无效标签: {invalid_labels}")
            stats['invalid_labels'] = list(invalid_labels)
            stats['invalid_removed'] = invalid_count

        stats['cleaned_unique'] = cleaned[valid_mask].nunique()
        stats['final_class_distribution'] = cleaned[valid_mask].value_counts().to_dict()

        self.cleaning_stats['labels'] = stats
        return cleaned, stats, valid_mask

    def clean_numeric_suffixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理数值后缀污染：
        - Compactness 列中 " cm" 后缀 → 去除后缀并转为 float
        """
        cleaned = df.copy()
        col = 'Compactness'

        if col in cleaned.columns:
            # 检测非空且含 "cm" 的记录数
            mask = cleaned[col].astype(str).str.contains('cm', na=False)
            count = mask.sum()
            logger.info(f"  Compactness 列: {count} 条含 'cm' 后缀")

            # 去除后缀并转 float
            cleaned[col] = cleaned[col].astype(str).str.replace(' cm', '', regex=False).str.strip()
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

            self.cleaning_stats['suffix_cleaned'] = {
                'column': col,
                'affected_rows': count,
            }

        return cleaned

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值：
        1. 将空字符串和 "?" 替换为 NaN
        2. 使用 KNN Imputer 填充数值列中的 NaN
        """
        cleaned = df.copy()

        stats = {'before': {}, 'after': {}, 'method': self.impute_strategy}

        # Step 1: 将所有列尝试转为数值（空字符串和 "?" → NaN）
        # 先获取所有列（排除标签列可能在此 DataFrame 中）
        all_cols = cleaned.columns.tolist()
        for col in all_cols:
            if cleaned[col].dtype == object:
                # 将 "?" 等非标准缺失标记替换为 NaN
                cleaned[col] = cleaned[col].replace('?', np.nan)
                cleaned[col] = cleaned[col].replace('', np.nan)
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

        # 过滤出成功转为数值的列
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()

        # 记录缺失值数量
        for col in numeric_cols:
            nan_count = cleaned[col].isna().sum()
            if nan_count > 0:
                stats['before'][col] = int(nan_count)
                logger.info(f"  {col}: {nan_count} 个缺失值")

        # Step 2: KNN 填充
        if self.impute_strategy == 'knn':
            self.knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
            imputed_values = self.knn_imputer.fit_transform(cleaned[numeric_cols])
            cleaned[numeric_cols] = pd.DataFrame(imputed_values, columns=numeric_cols, index=cleaned.index)
        elif self.impute_strategy == 'median':
            for col in numeric_cols:
                cleaned[col].fillna(cleaned[col].median(), inplace=True)
        else:
            for col in numeric_cols:
                cleaned[col].fillna(cleaned[col].median(), inplace=True)

        # 验证填充后无缺失
        for col in numeric_cols:
            remaining = cleaned[col].isna().sum()
            stats['after'][col] = int(remaining)

        self.cleaning_stats['missing_values'] = stats
        return cleaned

    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr',
                        columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        异常值处理：使用 IQR 方法，按类别进行 Winsorization
        对于每个类别×特征组合，将超出 IQR×1.5 边界的值截断到边界
        """
        cleaned = df.copy()

        if columns is None:
            columns = cleaned.select_dtypes(include=[np.number]).columns.tolist()

        stats = {'method': method, 'columns_processed': [], 'per_class_stats': {}}

        if 'Class' in cleaned.columns:
            target_col = 'Class'
        elif 'Class_orig' in cleaned.columns:
            target_col = 'Class_orig'
        else:
            # 无类别列，全局处理
            target_col = None

        for col in columns:
            if col == target_col:
                continue

            if target_col and target_col in cleaned.columns:
                for cls in cleaned[target_col].unique():
                    mask = cleaned[target_col] == cls
                    series = cleaned.loc[mask, col]

                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR

                    outliers = (cleaned.loc[mask, col] < lower) | (cleaned.loc[mask, col] > upper)
                    n_outliers = outliers.sum()
                    if n_outliers > 0:
                        cleaned.loc[mask, col] = cleaned.loc[mask, col].clip(lower, upper)
                        if cls not in stats['per_class_stats']:
                            stats['per_class_stats'][cls] = {}
                        stats['per_class_stats'][cls][col] = int(n_outliers)
            else:
                series = cleaned[col].dropna()
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                cleaned[col] = cleaned[col].clip(lower, upper)

            stats['columns_processed'].append(col)

        self.cleaning_stats['outliers'] = stats
        return cleaned

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        执行完整清洗流水线

        Returns:
            (X_cleaned, y_cleaned)
        """
        logger.info("=" * 50)
        logger.info("开始数据清洗流程")
        logger.info("=" * 50)

        # 1. 清洗标签
        logger.info("[1/4] 清洗标签噪声...")
        y = df['Class'].copy()
        y_cleaned, label_stats, valid_mask = self.clean_labels(y)
        logger.info(f"  标签种类: {label_stats['original_unique']} → {label_stats['cleaned_unique']}")

        # 2. 去除数值后缀
        logger.info("[2/4] 清理数值后缀...")
        df_cleaned = self.clean_numeric_suffixes(df.drop(columns=['Class']))

        # 3. 处理缺失值
        logger.info("[3/4] 处理缺失值...")
        df_cleaned = self.handle_missing_values(df_cleaned)
        total_missing_before = sum(self.cleaning_stats.get('missing_values', {}).get('before', {}).values())
        logger.info(f"  共填充 {total_missing_before} 个缺失值")

        # 4. 处理异常值
        logger.info("[4/4] 处理异常值...")
        df_cleaned = self.handle_outliers(df_cleaned)

        # 应用标签有效性过滤
        X_cleaned = df_cleaned[valid_mask.values]
        y_cleaned = y_cleaned[valid_mask.values]
        X_cleaned = X_cleaned.reset_index(drop=True)
        y_cleaned = y_cleaned.reset_index(drop=True)

        logger.info(f"清洗完成: X={X_cleaned.shape}, y={len(y_cleaned)}")
        logger.info("=" * 50)

        return X_cleaned, y_cleaned
