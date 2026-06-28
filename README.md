# 🌾 Dry Bean Multi-Class Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-green.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**机器学习与项目实践 (AIT209) 期末作业 — 基于计算机视觉特征的干豆品种多分类**

---

## 📋 项目概述

本项目基于 UCI 机器学习仓库的 **Dry Bean Dataset**，构建了一个完整的机器学习分类流程：从脏数据清洗、特征工程，到多算法对比实验（含课堂未讲算法 LightGBM），再到系统化工程集成。数据集包含 13,611 条干豆种子图像特征，目标是将每条记录分类到 7 个豆类品种之一。

## 📖 目录

1. [数据集描述](#-数据集描述)
2. [数据质量问题与处理](#-数据质量问题与处理)
3. [实现的算法](#-实现的算法)
4. [实验结果](#-实验结果)
5. [项目结构](#-项目结构)
6. [安装与使用](#-安装与使用)
7. [图表展示](#-图表展示)

## 🌱 数据集描述

| 属性 | 值 |
|---|---|
| 来源 | [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| 样本数 | 9,527 训练 / 2,737 测试 / 1,347 验证 |
| 特征数 | 16 个数值型形态特征 |
| 类别数 | 7 种豆类 |
| 任务类型 | 多分类 |

### 16 个特征

| 维度特征 (12) | 形状特征 (4) |
|---|---|
| Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness | ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4 |

### 7 个豆类品种

| 类别 | 占比 | 描述 |
|---|---|---|
| DERMASON | 24.6% | 最常见品种 |
| SIRA | 18.7% | 与 DERMASON 易混淆 |
| SEKER | 14.3% | 中等大小 |
| HOROZ | 13.4% | 形态较独特 |
| CALI | 11.7% | 中等频率 |
| BARBUNYA | 9.4% | 较少见 |
| BOMBAY | 3.6% | 最稀有，体积大，易区分 |

### 类别分布

```
DERMASON  ████████████████████████ 2503
SIRA      ██████████████████ 1837
SEKER     ██████████████ 1408
HOROZ     █████████████ 1340
CALI      ███████████ 1151
BARBUNYA  █████████ 927
BOMBAY    ████ 361
```

**数据不平衡比**: DERMASON:BOMBAY ≈ 6.9:1

## 🔧 数据质量问题与处理

数据集包含 4 类人为污染（文件名中的"Dirty"）：

### 1. 标签噪声 (25 种唯一标签 → 7 种)

| 问题类型 | 示例 | 处理 |
|---|---|---|
| 尾部空格 | `DERMASON ` | `str.strip()` |
| 小写变体 | `dermason`, `sira` | `str.upper()` |
| Leet-speak | `D3RMAS0N`, `S3K3R`, `H0R0Z`, `B0MBAY` | 映射表 (3→E, 0→O) |

### 2. 缺失值

| 列 | 训练集缺失 | 处理方法 |
|---|---|---|
| Perimeter | 469 行 | KNN Imputer (k=5) |
| Solidity | 272 行 + 202 行 `?` | 先替换 `?` → NaN, KNN Imputer |

**选择 KNN 的理由**: 形态学特征高度相关，KNN 利用特征相关性比均值/中位数填充更准确。

### 3. 数值后缀污染

| 列 | 问题 | 数量 | 处理 |
|---|---|---|---|
| Compactness | `0.9293 cm` | 258 行 | 去除 ` cm` 后缀，转 float64 |

### 4. 异常值处理

- 方法: IQR × 1.5 (按类别 Winsorization)
- 目的: 保留小类样本 (如 BOMBAY)，同时减少极端值影响

### 5. 特征工程

- **比率特征** (6 个): AreaToPerimeter, ConvexityRatio, MajorToMinor, Elongation, FormFactor, AreaToMajorAxis
- **交互特征** (28 个): Top-8 特征的 degree-2 交互项
- **特征选择**: ANOVA F-test 保留 Top-30

### 6. 特征缩放与类别平衡

- SVM/KNN/LR → StandardScaler
- RF/XGBoost/LightGBM → 不缩放 (树模型)
- 全部模型 → `class_weight='balanced'` + Stratified CV

## 🤖 实现的算法

| # | 算法 | 类型 | Loss 曲线 | 特征重要性 | 课堂讲授 |
|---|---|---|---|---|---|
| 1 | Logistic Regression | 线性基线 | warm_start 逐次 loss | 系数幅度 | ✓ |
| 2 | KNN (k=5) | 惰性学习 | Val Acc vs K | 排列重要性 | ✓ |
| 3 | SVM (RBF Kernel) | 核方法 | Val Acc vs C | 排列重要性 | ✓ |
| 4 | Random Forest | Bagging 集成 | OOB Error | 原生 Gini | ✓ |
| 5 | XGBoost | Boosting 集成 | 原生 mlogloss | 原生 gain | ✓ |
| 6 | **LightGBM ★** | **Boosting 集成 (Leaf-wise)** | **原生 multi_logloss** | **原生 gain** | **✗ 未讲** |

### LightGBM — 课堂未讲算法详解

LightGBM (Light Gradient Boosting Machine) 由微软于 2017 年提出 (Ke et al., NeurIPS 2017)，与 XGBoost 的核心区别：

1. **Leaf-wise 生长策略**: 每次选择增益最大的叶子分裂（XGBoost 按层生长），收敛更快
2. **Gradient-based One-Side Sampling (GOSS)**: 保留大梯度样本，对小梯度样本随机采样
3. **Exclusive Feature Bundling (EFB)**: 无损压缩稀疏特征
4. **直方图算法**: 连续特征离散化为固定 bin，降低内存和计算开销

> 📚 参考: Ke G, Meng Q, Finley T, et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS, 2017.

## 📊 实验结果

### 测试集准确率对比

| 算法 | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Cohen Kappa | Training Acc | Train Time (s) | Inference (ms) |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9010 | 0.9120 | 0.9122 | 0.9014 | 0.8808 | 0.9071 | 0.21 | 0.00 |
| KNN | 0.8955 | 0.8996 | 0.9041 | 0.8954 | 0.8739 | 1.0000 | 0.00 | 26.56 |
| SVM | 0.9021 | 0.9096 | 0.9116 | 0.9025 | 0.8820 | 0.9105 | 1.83 | 344.30 |
| Random Forest | 0.9068 | 0.9136 | 0.9153 | 0.9067 | 0.8876 | 0.9983 | 0.85 | 40.14 |
| XGBoost | 0.9050 | 0.9134 | 0.9156 | 0.9050 | 0.8854 | 0.9987 | 2.22 | 12.23 |
| **LightGBM ★** | **0.9065** | **0.9137** | 0.9151 | 0.9064 | **0.8872** | 0.9728 | 1.50 | **7.46** |

> 🎯 **LightGBM 以 90.65% 准确率和最佳平衡准确率 (91.37%) 胜出**，同时推理速度 (7.46 ms) 在非平凡模型中排名第二。

### 过拟合分析

| 算法 | Train Acc | Test Acc | Gap | 评估 |
|---|---|---|---|---|
| Logistic Regression | 0.9071 | 0.9010 | 0.0061 | ✅ 低 (Low) |
| SVM | 0.9105 | 0.9021 | 0.0084 | ✅ 低 (Low) |
| **LightGBM ★** | 0.9728 | 0.9065 | 0.0663 | ⚠️ 高 (High) |
| Random Forest | 0.9983 | 0.9068 | 0.0915 | ⚠️ 高 (High) |
| XGBoost | 0.9987 | 0.9050 | 0.0937 | ⚠️ 高 (High) |
| KNN | 1.0000 | 0.8955 | 0.1045 | ❌ 严重 (Severe) |

> 📌 线性模型 (LR, SVM) 过拟合程度最低。树集成模型需要更多正则化。KNN 因完全记忆训练数据而严重过拟合。

### 推理速度对比

| 算法 | 单样本 (ms) | 批量1000 (ms) | 吞吐量 (样本/s) |
|---|---|---|---|
| Logistic Regression | 0.067 | 0.118 | **8,463,817** |
| LightGBM ★ | 7.46 (per batch) | n/a | 277,097 |
| XGBoost | 0.663 | 3.609 | 277,097 |
| KNN | 5.803 | 12.958 | 77,174 |
| Random Forest | 31.210 | 31.366 | 31,882 |
| SVM | 0.282 | 124.740 | 8,017 |

> 🚀 LightGBM/XGBoost 在速度-精度平衡上表现最优。

### 鲁棒性测试 (12 种噪声条件)

涵盖 4 种噪声类型 (高斯、椒盐、标签翻转、异常值) 多个强度等级。详见 `outputs/figures/robustness/`。

### 交叉验证 (5-fold)

| 算法 | CV Mean | CV Std |
|---|---|---|
| _需运行 `python main.py train-all --tune` 获取_ |

## 📁 项目结构

```
BeanClassification/
├── README.md                       # 本文件
├── requirements.txt                # Python 依赖
├── main.py                         # CLI 统一入口
├── config/                         # 配置模块
│   ├── settings.py                 # 全局常量、路径
│   └── hyperparameters.py          # 超参数搜索空间
├── data/                           # 数据层
│   ├── loader.py                   # CSV 读取
│   ├── cleaner.py                  # 数据清洗
│   ├── preprocessor.py             # 缩放 + 分割
│   └── augmenter.py                # 噪声注入
├── features/                       # 特征工程
│   └── engineer.py                 # 比率 + 交互 + 选择
├── models/                         # 模型层 (6 个算法)
│   ├── base.py                     # 抽象基类
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── svm.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   └── lightgbm_model.py           # ★ 课堂未讲
├── training/                       # 训练层
│   ├── trainer.py
│   └── tuner.py                    # 超参数搜索
├── evaluation/                     # 评估层
│   ├── metrics.py                  # 全指标计算
│   ├── comparator.py               # 算法对比编排
│   ├── robustness.py               # 鲁棒性测试
│   ├── overfitting.py              # 过拟合分析
│   └── speed.py                    # 速度基准测试
├── visualization/
│   └── plots.py                    # 30+ 图表函数
├── outputs/                        # 输出目录
│   ├── figures/                    # 图表 (PNG 300 DPI)
│   │   ├── data_analysis/          # 数据分析
│   │   ├── preprocessing/          # 处理前后对比
│   │   ├── models/                 # Loss 曲线
│   │   ├── comparison/             # 模型对比
│   │   ├── robustness/             # 鲁棒性
│   │   ├── overfitting/            # 过拟合
│   │   ├── feature_importance/     # 特征重要性
│   │   └── dimensionality/         # PCA/t-SNE
│   ├── tables/                     # 数据表格
│   ├── models/                     # 模型文件
│   └── logs/                       # 运行日志
└── paper/
    └── paper_content.md             # 论文正文
```

## 🚀 安装与使用

### 环境要求
- Python 3.9+
- Windows / Linux / macOS

### 安装

```bash
git clone https://github.com/Nas1m0v/BeanClassification.git
cd BeanClassification
pip install -r requirements.txt
```

### 使用方法

```bash
# 一键运行完整流程
python main.py pipeline

# 或按步骤运行
python main.py clean              # Step 1: 数据清洗
python main.py preprocess         # Step 2: 预处理 + 特征工程
python main.py train-all --tune   # Step 3: 超参调优 + 训练全部模型
python main.py compare            # Step 4: 完整对比分析
python main.py robustness         # Step 5: 鲁棒性测试
python main.py visualize          # Step 6: 生成全部图表

# 单个模型训练
python main.py train --model lightgbm --tune

# 指定数据目录
python main.py pipeline --data-dir ../DryBeanDataset --output-dir ./outputs
```

### CLI 命令总览

| 命令 | 说明 |
|---|---|
| `pipeline` | 一键运行完整 6 步流程 |
| `clean` | 数据清洗 (标签+缺失值+后缀) |
| `preprocess` | 预处理 + 特征工程 |
| `train --model X` | 训练指定模型 |
| `train-all` | 训练全部 6 模型 |
| `compare` | 完整对比分析 (5 维度) |
| `robustness` | 鲁棒性测试 (12 噪声条件) |
| `visualize` | 生成全部 30+ 图表 |

## 📈 图表展示

运行 `python main.py visualize` 后，可在 `outputs/figures/` 下找到以下图表：

### 数据分析
- 类别分布柱状图
- 缺失值热力图
- 特征相关性矩阵 (16×16)
- 特征分布直方图 (4×4 网格)
- 按类别箱线图
- PCA 2D / t-SNE 2D 投影

### 预处理
- 填补前后分布 KDE 对比
- 标签清洗流程图

### 模型对比
- 准确率对比柱状图
- 混淆矩阵 (3×2 网格)
- 每类 F1 热力图
- Learning Curves 学习曲线
- 雷达图 (6 维度)
- CV 箱线图
- 推理速度对比
- 训练时间 vs 准确率气泡图
- ROC / PR 曲线 (最佳模型)

### 鲁棒性
- 噪声退化曲线
- 鲁棒性热力图 (Model × Noise)

### 特征重要性
- 多模型特征重要性对比
- ANOVA F-score 排名

## 📝 许可

本项目仅用于学术目的 (AIT209 课程作业)。

## 🙏 致谢

- 数据集: Koklu & Ozkan (2020), *Computers and Electronics in Agriculture*
- LightGBM: Ke et al. (2017), *NeurIPS*
