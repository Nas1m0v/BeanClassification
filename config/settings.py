"""全局配置常量"""
import os

# 数据路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'DryBeanDataset')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')

# 数据集文件名
TRAIN_FILE = 'Dry_Bean_Dataset_Dirty_train.csv'
TEST_FILE = 'Dry_Bean_Dataset_Dirty_test.csv'
VAL_FILE = 'Dry_Bean_Dataset_Dirty_val.csv'

# 7 个标准类别
CLASS_NAMES = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

# 标签纠错映射（leet-speak + 小写 + 空格）
LABEL_CLEAN_MAP = {}

# 随机种子
RANDOM_SEED = 42

# 交叉验证折数
CV_FOLDS = 5

# 特征列名（16个数值特征）
FEATURE_COLS = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter',
    'Extent', 'Solidity', 'roundness', 'Compactness',
    'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]

# 目标列
TARGET_COL = 'Class'

# 含缺失值的列
MISSING_COLS = ['Perimeter', 'Solidity']

# 含 "?" 的列
UNKNOWN_COLS = ['Solidity']

# 含 " cm" 后缀的列
SUFFIX_COLS = {'Compactness': ' cm'}

# 算法注册表
MODEL_REGISTRY = {
    'lr': 'LogisticRegressionClassifier',
    'knn': 'KNNClassifier',
    'svm': 'SVMClassifier',
    'rf': 'RandomForestClassifier',
    'xgboost': 'XGBoostClassifier',
    'lightgbm': 'LightGBMClassifier',
}

# 可视化设置
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
