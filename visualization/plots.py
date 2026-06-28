"""可视化模块 - 全部图表生成函数

包含约30个绘图函数，覆盖：数据分析、预处理、模型对比、
鲁棒性、过拟合、特征重要性、降维可视化等。
所有函数支持中文，保存 PNG (300 DPI)。
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)

# --- 中文/字体配置 ---
import matplotlib.font_manager as fm

# 强制重建字体缓存
try:
    fm._load_fontmanager(try_read_cache=False)
except Exception:
    pass

# 直接通过字体路径设置，绕过名称查找问题
CHINESE_FONT = None
FONT_CANDIDATES = ['SimHei', 'Microsoft YaHei', 'Noto Sans SC', 'STXihei',
                   'FangSong', 'KaiTi', 'WenQuanYi Micro Hei']

for font_name in FONT_CANDIDATES:
    try:
        font_path = fm.findfont(font_name, fallback_to_default=False)
        if font_path:
            # 注册字体
            fm.fontManager.addfont(font_path)
            prop = fm.FontProperties(fname=font_path)
            if prop.get_name() in [f.name for f in fm.fontManager.ttflist]:
                CHINESE_FONT = prop.get_name()
                break
    except Exception:
        continue

if CHINESE_FONT:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [CHINESE_FONT, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    logger.info(f"中文字体已配置: {CHINESE_FONT}")
else:
    logger.warning("未找到中文字体，图表中文可能显示为方框")

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.max_open_warning'] = 100

sns.set_style("whitegrid")

# seaborn 会重置字体设置，需要在此之后重新设置
if CHINESE_FONT:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [CHINESE_FONT, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 全局配色
PALETTE = sns.color_palette("Set2", 10)
MODEL_COLORS = {
    'Logistic Regression': PALETTE[0],
    'KNN': PALETTE[1],
    'SVM': PALETTE[2],
    'Random Forest': PALETTE[3],
    'XGBoost': PALETTE[4],
    'LightGBM ★': PALETTE[5],
    'LightGBM': PALETTE[5],
}

OUTPUT_DIR = ''  # 外部设置

def _save(fig, name: str, subdir: str = ''):
    """保存图表到 outputs/figures/"""
    path = os.path.join(OUTPUT_DIR, 'figures', subdir) if OUTPUT_DIR else ''
    if path:
        os.makedirs(path, exist_ok=True)
        fpath = os.path.join(path, name)
    else:
        fpath = name
    fig.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  [保存] {fpath}")


# ===================== 数据分析 (6 个图表) =====================

def plot_class_distribution(y: pd.Series, title: str = '类别分布',
                            save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """各类别样本数量柱状图"""
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(counts))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{val}\n({val/len(y)*100:.1f}%)', ha='center', fontsize=9)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('类别')
    ax.set_ylabel('样本数量')
    ax.tick_params(axis='x', rotation=30)
    if save_path:
        _save(fig, save_path, 'data_analysis')
    return fig, ax


def plot_missing_values_heatmap(df: pd.DataFrame, title: str = '缺失值分布热力图',
                                save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """缺失值位置热力图"""
    # 采样(避免太大)
    sample = df.sample(min(200, len(df)), random_state=42)
    missing = sample.isna() | sample.isin(['', '?'])
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(missing, cbar=True, cmap='Reds', ax=ax,
                xticklabels=True, yticklabels=False)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('特征列')
    if save_path:
        _save(fig, save_path, 'data_analysis')
    return fig, ax


def plot_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str],
                             title: str = '特征相关性矩阵',
                             save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Pearson 相关矩阵热力图"""
    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={'size': 7})
    ax.set_title(title, fontsize=14, fontweight='bold')
    if save_path:
        _save(fig, save_path, 'data_analysis')
    return fig, ax


def plot_feature_distributions(df: pd.DataFrame, feature_cols: List[str],
                               save_path: Optional[str] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """特征分布直方图 (4x4 grid)"""
    n = len(feature_cols)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=40, color=PALETTE[0], edgecolor='white', alpha=0.8)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle('特征分布直方图', fontsize=14, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save(fig, save_path, 'data_analysis')
    return fig, axes


def plot_boxplots_by_class(df: pd.DataFrame, feature_cols: List[str], target_col: str,
                           top_n: int = 8, save_path: Optional[str] = None):
    """特征按类别分组箱线图"""
    top_features = feature_cols[:top_n]
    n_cols = 4
    n_rows = (top_n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(top_features):
        ax = axes[i]
        df_temp = df[[col, target_col]].dropna()
        sns.boxplot(data=df_temp, x=target_col, y=col, ax=ax, hue=target_col,
                    palette='Set2', linewidth=0.8, legend=False)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=30)
    for i in range(top_n, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle('Top 特征按类别分布', fontsize=14, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save(fig, save_path, 'data_analysis')
    return fig, axes


def plot_pca_2d(X: np.ndarray, y: np.ndarray, class_names: List[str],
                title: str = 'PCA 2D 投影', save_path: Optional[str] = None):
    """PCA 降维 2D 散点图"""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(class_names):
        mask = y == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, s=8, alpha=0.7)
    ax.set_title(f'{title}\n(解释方差: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.legend(markerscale=3, fontsize=9)
    if save_path:
        _save(fig, save_path, 'dimensionality')
    return fig, ax


def plot_tsne_2d(X: np.ndarray, y: np.ndarray, class_names: List[str],
                 title: str = 't-SNE 2D 投影', save_path: Optional[str] = None):
    """t-SNE 降维 2D 散点图"""
    from sklearn.manifold import TSNE
    # 对大数据集采样
    n = min(X.shape[0], 5000)
    indices = np.random.RandomState(42).choice(X.shape[0], size=n, replace=False)
    X_sub, y_sub = X[indices], y[indices]
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=1)
    X_2d = tsne.fit_transform(X_sub)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(class_names):
        mask = y_sub == i
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=name, s=10, alpha=0.7)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(markerscale=3, fontsize=9)
    if save_path:
        _save(fig, save_path, 'dimensionality')
    return fig, ax


# ===================== 数据处理 (3 个图表) =====================

def plot_distribution_comparison(before: pd.Series, after: pd.Series, col_name: str,
                                 save_path: Optional[str] = None):
    """填补前后分布对比 KDE 图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(before.dropna(), label='填补前', color='red', linewidth=2, ax=ax)
    sns.kdeplot(after.dropna(), label='填补后', color='blue', linewidth=2, ax=ax)
    ax.set_title(f'{col_name} 填补前后分布对比', fontsize=14, fontweight='bold')
    ax.set_xlabel(col_name)
    ax.legend(fontsize=11)
    if save_path:
        _save(fig, save_path, 'preprocessing')
    return fig, ax


def plot_label_cleaning_summary(label_stats: Dict,
                                save_path: Optional[str] = None):
    """标签清洗前后对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Before: 展示噪声标签分布（简化版）
    changes = label_stats.get('label_changes', {})
    if changes:
        items = [(dirty, info['count']) for dirty, info in changes.items()]
        labels, counts = zip(*sorted(items, key=lambda x: x[1], reverse=True))
        ax1.barh(range(len(labels)), counts, color='salmon')
        ax1.set_yticks(range(len(labels)))
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_title('噪声标签检测', fontsize=13, fontweight='bold')
        ax1.set_xlabel('数量')

    # After
    final_dist = label_stats.get('final_class_distribution', {})
    if final_dist:
        names, vals = zip(*sorted(final_dist.items(), key=lambda x: x[1]))
        ax2.barh(range(len(names)), vals, color=sns.color_palette('viridis', len(names)))
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names)
        ax2.set_title('清洗后类别分布', fontsize=13, fontweight='bold')
        ax2.set_xlabel('数量')

    fig.suptitle('标签清洗结果', fontsize=15, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save(fig, save_path, 'preprocessing')
    return fig, (ax1, ax2)


def plot_feature_importance_bar(importance: np.ndarray, feature_names: List[str],
                                title: str = '特征重要性 (ANOVA F-score)',
                                top_n: int = 20, save_path: Optional[str] = None):
    """特征重要性水平柱状图"""
    if len(feature_names) > top_n:
        indices = np.argsort(importance)[-top_n:]
        imp = importance[indices]
        names = [feature_names[i] for i in indices]
    else:
        imp = importance
        names = feature_names
    sorted_idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_idx)))
    ax.barh([names[i] for i in sorted_idx], imp[sorted_idx], color=colors)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('F-score')
    if save_path:
        _save(fig, save_path, 'feature_importance')
    return fig, ax


# ===================== 模型对比 (10 个图表) =====================

def plot_accuracy_comparison(results: Dict, save_path: Optional[str] = None):
    """模型准确率对比柱状图"""
    names = []
    accs = []
    bal_accs = []
    f1s = []
    for name, r in results.items():
        names.append(name)
        m = r['metrics']
        accs.append(m['accuracy'])
        bal_accs.append(m['balanced_accuracy'])
        f1s.append(m['f1_weighted'])

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, accs, w, label='Accuracy', color=PALETTE[0], edgecolor='white')
    ax.bar(x, bal_accs, w, label='Balanced Acc', color=PALETTE[1], edgecolor='white')
    ax.bar(x + w, f1s, w, label='Weighted F1', color=PALETTE[2], edgecolor='white')
    # 标注数值
    for i in range(len(names)):
        ax.text(i - w, accs[i] + 0.003, f'{accs[i]:.3f}', ha='center', fontsize=8, rotation=90)
        ax.text(i, bal_accs[i] + 0.003, f'{bal_accs[i]:.3f}', ha='center', fontsize=8, rotation=90)
        ax.text(i + w, f1s[i] + 0.003, f'{f1s[i]:.3f}', ha='center', fontsize=8, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('模型测试集准确率对比', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0.75, 1.0)
    if save_path:
        _save(fig, save_path, 'comparison')
    return fig, ax


def plot_loss_curves(training_history: Dict, model_name: str,
                     save_path: Optional[str] = None):
    """Loss 曲线图"""
    lc = training_history.get('loss_curve')
    if lc is None or 'iterations' not in lc:
        return None, None
    fig, ax = plt.subplots(figsize=(10, 5))
    its = lc['iterations'][-50:] if len(lc['iterations']) > 100 else lc['iterations']
    train_l = lc['train_loss'][-50:] if len(lc.get('train_loss', [])) > 100 else lc.get('train_loss', [])
    ax.plot(its, train_l, label='Training Loss', color='blue', alpha=0.8)
    if lc.get('val_loss'):
        val_l = lc['val_loss'][-50:] if len(lc['val_loss']) > 100 else lc['val_loss']
        ax.plot(its, val_l, label='Validation Loss', color='red', alpha=0.8)
        ax.fill_between(its, train_l, val_l, alpha=0.1, color='gray')
    ax.set_title(f'{model_name} — Loss 曲线', fontsize=13, fontweight='bold')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.legend()
    if save_path:
        _save(fig, save_path, 'models')
    return fig, ax


def plot_confusion_matrices(results: Dict, class_names: List[str],
                            save_path: Optional[str] = None):
    """混淆矩阵网格 (按模型)"""
    n_models = len(results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for i, (name, r) in enumerate(results.items()):
        ax = axes[i]
        cm = np.array(r['metrics']['confusion_matrix'])
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=False, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax,
                    cbar_kws={'label': 'Recall'})
        # 添加文本标注
        for r_idx in range(cm_norm.shape[0]):
            for c_idx in range(cm_norm.shape[1]):
                val = cm_norm[r_idx, c_idx]
                text_color = 'white' if val > 0.6 else 'black'
                if val > 0.05:
                    ax.text(c_idx + 0.5, r_idx + 0.5, f'{val:.2f}',
                           ha='center', va='center', fontsize=7, color=text_color)
        ax.set_title(f'{name}\nAcc={r["metrics"]["accuracy"]:.3f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle('混淆矩阵对比 (行归一化)', fontsize=15, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save(fig, save_path, 'comparison')
    return fig, axes


def plot_per_class_f1_heatmap(results: Dict, class_names: List[str],
                              save_path: Optional[str] = None):
    """模型×类别 F1 热力图"""
    model_names = list(results.keys())
    data = np.zeros((len(model_names), len(class_names)))
    for i, mname in enumerate(model_names):
        pc = results[mname]['metrics']['per_class']
        for j, cls in enumerate(class_names):
            data[i, j] = pc.get(cls, {}).get('f1', 0)
    fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.2), max(6, len(model_names) * 0.8)))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=model_names, ax=ax,
                vmin=data.min() - 0.05, vmax=1.0)
    ax.set_title('每类 F1 Score 对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('类别')
    ax.set_ylabel('模型')
    if save_path:
        _save(fig, save_path, 'comparison')
    return fig, ax


def plot_inference_speed(speed_results: Dict, save_path: Optional[str] = None):
    """推理速度对比柱状图"""
    names = list(speed_results.keys())
    vals = [speed_results[n]['single_sample_ms'] for n in names]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [MODEL_COLORS.get(n, PALETTE[i % len(PALETTE)]) for i, n in enumerate(names)]
    bars = ax.bar(names, vals, color=colors, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.3f} ms', ha='center', fontsize=9)
    ax.set_ylabel('单样本推理时间 (ms)')
    ax.set_title('模型推理速度对比', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=20)
    if save_path:
        _save(fig, save_path, 'comparison')
    return fig, ax


def plot_overfitting_gap(overfitting_results: Dict, save_path: Optional[str] = None):
    """过拟合分析柱状图"""
    names = list(overfitting_results.keys())
    train_accs = [overfitting_results[n]['train_accuracy'] for n in names]
    test_accs = [overfitting_results[n]['test_accuracy'] for n in names]
    gaps = [overfitting_results[n]['gap'] for n in names]
    x = np.arange(len(names))
    w = 0.3
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, train_accs, w, label='Train Acc', color='steelblue', edgecolor='white')
    ax.bar(x + w/2, test_accs, w, label='Test Acc', color='coral', edgecolor='white')
    # 标记 gap
    for i in range(len(names)):
        mid = (train_accs[i] + test_accs[i]) / 2
        ax.annotate(f'Gap: {gaps[i]:.4f}', (i, mid),
                   ha='center', fontsize=8, bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('过拟合分析 (训练集 vs 测试集精度差)', fontsize=14, fontweight='bold')
    ax.legend()
    if save_path:
        _save(fig, save_path, 'overfitting')
    return fig, ax


def plot_learning_curves(learning_curves: Dict[str, Dict],
                         save_path: Optional[str] = None):
    """学习曲线网格"""
    n = len(learning_curves)
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    model_names = list(learning_curves.keys())
    for i, name in enumerate(model_names):
        ax = axes[i]
        lc = learning_curves[name]
        train_sizes = lc.get('train_sizes_ratio', lc.get('train_sizes', []))
        ax.plot(train_sizes, lc['train_mean'], 'o-', label='Train', color='blue')
        ax.plot(train_sizes, lc['val_mean'], 'o-', label='Val', color='red')
        if 'train_std' in lc:
            ax.fill_between(train_sizes,
                           np.array(lc['train_mean']) - np.array(lc['train_std']),
                           np.array(lc['train_mean']) + np.array(lc['train_std']),
                           alpha=0.15, color='blue')
            ax.fill_between(train_sizes,
                           np.array(lc['val_mean']) - np.array(lc['val_std']),
                           np.array(lc['val_mean']) + np.array(lc['val_std']),
                           alpha=0.15, color='red')
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Training Size Ratio')
        ax.set_ylabel('Accuracy')
        ax.legend(fontsize=8)
        ax.set_ylim(0.7, 1.02)
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle('学习曲线对比', fontsize=14, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save(fig, save_path, 'comparison')
    return fig, axes


def plot_cv_boxplot(cv_results: Dict[str, Dict], save_path: Optional[str] = None):
    """交叉验证箱线图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    labels = []
    for name, cvr in cv_results.items():
        if 'fold_scores' in cvr:
            data.append(cvr['fold_scores'])
            labels.append(f"{name}\n(μ={cvr['mean']:.4f})")
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, i in zip(bp['boxes'], range(len(data))):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.6)
    ax.set_ylabel('Accuracy')
    ax.set_title('交叉验证分数分布 (5-fold)', fontsize=14, fontweight='bold')
    if save_path:
        _save(fig, save_path, 'comparison')
    return fig, ax


def plot_robustness_degradation(degradation: Dict, save_path: Optional[str] = None):
    """鲁棒性退化曲线 - 噪声类型对比"""
    # degradation: {model_name: {noise_label: accuracy_drop}}
    fig, ax = plt.subplots(figsize=(14, 6))
    for model_name, noises in degradation.items():
        labels = list(noises.keys())
        # 简化标签
        short_labels = [l.replace('高斯噪声 σ=', 'σ=').replace('椒盐噪声 ', '').replace('标签翻转 ', '').replace('异常值噪声 ', '') for l in labels]
        drops = list(noises.values())
        x = range(len(labels))
        color = MODEL_COLORS.get(model_name, PALETTE[0])
        ax.plot(x, drops, 'o-', label=model_name, color=color, markersize=5, linewidth=1.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy Drop')
    ax.set_title('噪声鲁棒性对比 (干净准确率 - 噪声准确率)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    if save_path:
        _save(fig, save_path, 'robustness')
    return fig, ax


def plot_robustness_heatmap(robustness_results: Dict, save_path: Optional[str] = None):
    """鲁棒性热力图: model × noise_condition"""
    model_names = sorted(robustness_results.keys())
    # 获取所有噪声条件
    all_conditions = []
    for mn in model_names:
        for cond in robustness_results[mn].keys():
            if cond != '干净数据' and cond not in all_conditions:
                all_conditions.append(cond)

    data = np.zeros((len(model_names), len(all_conditions)))
    for i, mn in enumerate(model_names):
        for j, cond in enumerate(all_conditions):
            data[i, j] = robustness_results[mn].get(cond, 0)

    fig, ax = plt.subplots(figsize=(max(14, len(all_conditions) * 1.2), max(6, len(model_names) * 0.8)))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=all_conditions, yticklabels=model_names, ax=ax,
                vmin=0.6, vmax=1.0)
    ax.set_title('鲁棒性热力图 (噪声条件下的准确率)', fontsize=14, fontweight='bold')
    ax.set_xlabel('噪声条件')
    ax.set_ylabel('模型')
    ax.tick_params(axis='x', rotation=45)
    # 手动设置 x 轴标签水平对齐
    for label in ax.get_xticklabels():
        label.set_ha('right')
    if save_path:
        _save(fig, save_path, 'robustness')
    return fig, ax


def plot_feature_importance_comparison(feature_importances: Dict[str, np.ndarray],
                                       feature_names: List[str], top_n: int = 15,
                                       save_path: Optional[str] = None):
    """多模型特征重要性对比水平柱状图"""
    n_models = len(feature_importances)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 7))
    if n_models == 1:
        axes = [axes]
    # 确保 feature_names 是列表
    if not isinstance(feature_names, list):
        feature_names = [f'F_{j}' for j in range(len(list(feature_importances.values())[0]))]

    for i, (mname, imp) in enumerate(feature_importances.items()):
        ax = axes[i]
        if len(feature_names) > top_n:
            indices = np.argsort(imp)[-top_n:]
            imp_sub = imp[indices]
            names_sub = [feature_names[j] for j in indices]
        else:
            imp_sub = imp
            names_sub = feature_names
        sorted_idx = np.argsort(imp_sub)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
        ax.barh([names_sub[k] for k in sorted_idx], imp_sub[sorted_idx], color=colors)
        ax.set_title(mname, fontsize=12, fontweight='bold')
        ax.set_xlabel('Importance')
    fig.suptitle('特征重要性对比 (Top-15)', fontsize=15, fontweight='bold')
    fig.tight_layout()
    if save_path:
        _save(fig, save_path, 'feature_importance')
    return fig, axes


def plot_radar_chart(results: Dict, save_path: Optional[str] = None):
    """雷达图: 6维度模型综合对比"""
    from math import pi
    dimensions = ['Accuracy', 'Balanced Acc', 'F1 (Wtd)', 'Speed', 'Anti-Overfit', 'Cohen Kappa']
    n_dim = len(dimensions)
    angles = [n / float(n_dim) * 2 * pi for n in range(n_dim)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for name, r in results.items():
        m = r['metrics']
        # 速度归一化: 越快越好
        inf_time = r.get('inference_time', 0.01) or 0.01
        speed_score = min(1.0, 0.01 / max(inf_time, 1e-6))
        # 过拟合: 越小越好
        train_acc = r.get('train_accuracy', 0)
        test_acc = m['accuracy']
        anti_overfit = max(0, 1 - (train_acc - test_acc) * 10)

        values = [
            m['accuracy'],
            m['balanced_accuracy'],
            m['f1_weighted'],
            speed_score,
            anti_overfit,
            m['cohen_kappa'],
        ]
        values += values[:1]
        colors = MODEL_COLORS.get(name, PALETTE[0])
        ax.plot(angles, values, 'o-', label=name, color=colors, linewidth=1.5, markersize=4)
        ax.fill(angles, values, alpha=0.05, color=colors)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title('模型综合对比雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    if save_path:
        _save(fig, save_path, 'comparison')
    return fig, ax


def plot_training_time_vs_accuracy(results: Dict, save_path: Optional[str] = None):
    """训练时间 vs 准确率气泡图"""
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, r in results.items():
        train_time = r.get('train_time', 1)
        accuracy = r['metrics']['accuracy']
        size = 200 + train_time * 20
        color = MODEL_COLORS.get(name, PALETTE[0])
        ax.scatter(train_time, accuracy, s=size, c=[color], alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.annotate(name, (train_time, accuracy), textcoords="offset points",
                   xytext=(8, 5), fontsize=9)
    ax.set_xlabel('Training Time (s)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('训练时间 vs 测试准确率', fontsize=14, fontweight='bold')
    if save_path:
        _save(fig, save_path, 'comparison')
    return fig, ax


def plot_statistical_test_matrix(p_values: np.ndarray, model_names: List[str],
                                 save_path: Optional[str] = None):
    """统计检验 p-value 热力图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.eye(len(model_names), dtype=bool)
    sns.heatmap(p_values, annot=True, fmt='.3f', cmap='coolwarm_r',
                xticklabels=model_names, yticklabels=model_names,
                mask=mask, ax=ax, center=0.05, vmin=0, vmax=0.2)
    ax.set_title('Friedman Test Pairwise p-values', fontsize=13, fontweight='bold')
    if save_path:
        _save(fig, save_path, 'comparison')
    return fig, ax


def plot_hyperparameter_sensitivity(tuning_results: Dict, model_key: str,
                                    param_name: str, save_path: Optional[str] = None):
    """超参数敏感性曲线"""
    # 简化版：从 cv_results_ 中提取
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f'{model_key} — {param_name} 敏感性', fontsize=13, fontweight='bold')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Balanced Accuracy')
    if save_path:
        _save(fig, save_path, 'models')
    return fig, ax


def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, class_names: List[str],
                    title: str = 'ROC Curves (OvR)', save_path: Optional[str] = None):
    """一对多 ROC 曲线"""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n_classes))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})', linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    if save_path:
        _save(fig, save_path, 'comparison')
    return fig, ax


def plot_precision_recall_curves(y_true: np.ndarray, y_proba: np.ndarray,
                                 class_names: List[str],
                                 title: str = 'Precision-Recall Curves',
                                 save_path: Optional[str] = None):
    """逐类 Precision-Recall 曲线"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from sklearn.preprocessing import label_binarize
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n_classes))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, name in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        ax.plot(rec, prec, label=f'{name} (AP={ap:.3f})', linewidth=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8)
    if save_path:
        _save(fig, save_path, 'comparison')
    return fig, ax


# ===================== 辅助工具 =====================

def set_output_dir(path: str):
    """设置图表输出目录"""
    global OUTPUT_DIR
    OUTPUT_DIR = path
    os.makedirs(os.path.join(path, 'figures'), exist_ok=True)
    for subdir in ['data_analysis', 'preprocessing', 'models', 'comparison',
                   'robustness', 'overfitting', 'feature_importance', 'dimensionality']:
        os.makedirs(os.path.join(path, 'figures', subdir), exist_ok=True)
