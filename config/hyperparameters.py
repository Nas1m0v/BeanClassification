"""各算法的超参数搜索空间"""

# Logistic Regression
LR_PARAM_GRID = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga'],
    'max_iter': [1000, 3000, 5000],
    'class_weight': ['balanced'],
}

# KNN
KNN_PARAM_GRID = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2],
}

# SVM
SVM_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly', 'linear'],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'class_weight': ['balanced'],
    'probability': [True],
    'max_iter': [5000],
}

# Random Forest
RF_PARAM_GRID = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'bootstrap': [True],
}

# XGBoost
XGB_PARAM_GRID = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.3],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [1, 1.5, 2.0],
}

# LightGBM (★ 课堂未讲算法)
LGB_PARAM_GRID = {
    'n_estimators': [100, 200, 300, 500],
    'num_leaves': [31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0],
    'class_weight': ['balanced'],
}

# 参数网格映射
PARAM_GRIDS = {
    'lr': LR_PARAM_GRID,
    'knn': KNN_PARAM_GRID,
    'svm': SVM_PARAM_GRID,
    'rf': RF_PARAM_GRID,
    'xgboost': XGB_PARAM_GRID,
    'lightgbm': LGB_PARAM_GRID,
}

# 调参策略：grid 或 random
TUNE_METHOD = {
    'lr': 'grid',
    'knn': 'grid',
    'svm': 'grid',
    'rf': 'random',       # 搜索空间大
    'xgboost': 'random',  # 搜索空间大
    'lightgbm': 'random', # 搜索空间大
}

# RandomizedSearchCV 迭代次数
N_ITER_RANDOM = 50
