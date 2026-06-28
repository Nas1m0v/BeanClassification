"""特征工程模块 - 交互特征、比率特征、多项式特征"""
import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """从原始形状测量特征中生成新特征"""

    def __init__(self, methods: Optional[List[str]] = None, k_best: int = 30):
        """
        Args:
            methods: 要应用的特征工程方法列表
                - 'ratios': 比率特征
                - 'interactions': Top-8 特征的交互项 (degree-2)
                - 'poly2': 所有特征的 2 次多项式（仅交互项）
            k_best: 特征选择保留的特征数
        """
        self.methods = methods or ['ratios', 'interactions']
        self.k_best = k_best
        self.selector = None
        self.selected_features: List[str] = []
        self.original_features: List[str] = []

    def _create_ratio_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """创建比率特征"""
        df = X.copy()
        new_features = {}

        # 面积比率
        if 'Area' in df.columns and 'Perimeter' in df.columns:
            new_features['AreaToPerimeter'] = df['Area'] / (df['Perimeter'] + 1e-8)
        if 'Area' in df.columns and 'ConvexArea' in df.columns:
            new_features['ConvexityRatio'] = df['Area'] / (df['ConvexArea'] + 1e-8)

        # 轴比率（可能与 AspectRation 冗余，但保留验证）
        if 'MajorAxisLength' in df.columns and 'MinorAxisLength' in df.columns:
            new_features['MajorToMinor'] = df['MajorAxisLength'] / (df['MinorAxisLength'] + 1e-8)
            new_features['Elongation'] = (df['MajorAxisLength'] - df['MinorAxisLength']) / \
                                         (df['MajorAxisLength'] + df['MinorAxisLength'] + 1e-8)

        # 紧凑度相关
        if 'EquivDiameter' in df.columns and 'MajorAxisLength' in df.columns:
            new_features['FormFactor'] = df['EquivDiameter'] / (df['MajorAxisLength'] + 1e-8)

        # 面积与轴的关系
        if 'Area' in df.columns and 'MajorAxisLength' in df.columns:
            new_features['AreaToMajorAxis'] = df['Area'] / (df['MajorAxisLength'] + 1e-8)

        # 将新特征添加到 DataFrame
        for name, values in new_features.items():
            df[name] = values

        logger.info(f"  创建了 {len(new_features)} 个比率特征: {list(new_features.keys())}")
        return df

    def _create_interaction_features(self, X: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
        """
        创建 Top-N 特征的交互项（degree-2, interaction-only）
        先通过 ANOVA F-test 选出 Top-N 特征，再创建交互项
        """
        from sklearn.preprocessing import PolynomialFeatures

        df = X.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 限制 Top-N
        top_cols = numeric_cols[:top_n] if len(numeric_cols) > top_n else numeric_cols

        if len(top_cols) < 2:
            return df

        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_values = poly.fit_transform(df[top_cols])
        poly_names = poly.get_feature_names_out(top_cols)

        # 只保留交互项 (去掉原始的一次项和平方项)
        added = 0
        for i, name in enumerate(poly_names):
            if ' ' in name:  # 交互项
                col_name = name.replace(' ', '_x_')
                df[col_name] = poly_values[:, i]
                added += 1

        logger.info(f"  创建了 {added} 个交互特征 (基于 Top-{top_n} 特征)")
        return df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        生成特征并进行特征选择

        Args:
            X: 原始特征 DataFrame
            y: 目标标签（用于特征选择）

        Returns:
            增强后的特征 DataFrame
        """
        self.original_features = list(X.columns)
        df = X.copy()

        # 应用各种特征工程方法
        if 'ratios' in self.methods:
            logger.info("[特征工程] 创建比率特征...")
            df = self._create_ratio_features(df)

        if 'interactions' in self.methods:
            logger.info("[特征工程] 创建交互特征...")
            df = self._create_interaction_features(df)

        logger.info(f"  特征数: {len(self.original_features)} → {df.shape[1]}")

        # 特征选择：基于 ANOVA F-test
        if len(df.columns) > self.k_best:
            logger.info(f"[特征选择] 从 {df.shape[1]} 个特征中选择 Top-{self.k_best}")
            self.selector = SelectKBest(score_func=f_classif, k=min(self.k_best, df.shape[1]))
            self.selector.fit(df.values, y)
            scores = self.selector.scores_
            feature_scores = sorted(zip(df.columns, scores), key=lambda x: x[1], reverse=True)

            # 记录 Top-10
            logger.info("  Top-10 特征 (ANOVA F-score):")
            for rank, (feat, score) in enumerate(feature_scores[:10], 1):
                logger.info(f"    {rank}. {feat}: {score:.2f}")

            self.selected_features = list(df.columns[self.selector.get_support()])
            df_selected = df[self.selected_features]
            logger.info(f"  特征选择后: {df_selected.shape[1]} 个特征")
            return df_selected

        self.selected_features = list(df.columns)
        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """对测试/验证数据应用相同的特征工程（必须已调用 fit_transform）"""
        df = X.copy()

        # 重新创建比率特征
        if 'ratios' in self.methods:
            df = self._create_ratio_features(df)

        if 'interactions' in self.methods:
            df = self._create_interaction_features(df)

        # 应用特征选择
        if self.selector is not None and len(self.selected_features) > 0:
            # 确保所有选择的特征都存在
            available = [f for f in self.selected_features if f in df.columns]
            df = df[available]

        return df
