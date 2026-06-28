"""BeanClassification CLI — Dry Bean 多分类项目统一入口

Usage:
    python main.py pipeline      一键运行完整流程
    python main.py clean         数据清洗
    python main.py preprocess    预处理 + 特征工程
    python main.py train --model lightgbm --tune   训练单个模型
    python main.py train-all     训练全部模型
    python main.py compare       完整对比分析
    python main.py robustness    鲁棒性测试
    python main.py visualize     生成全部图表
"""
import argparse
import sys
import os
import logging
import time
from datetime import datetime

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import (
    DATA_DIR, OUTPUT_DIR, CLASS_NAMES, FEATURE_COLS,
    RANDOM_SEED, CV_FOLDS, MODEL_REGISTRY,
)


def setup_logging(output_dir: str, verbose: bool = False):
    """配置日志"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    console_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'pipeline.log'), encoding='utf-8'),
        ],
    )
    # 设置控制台 handler 的级别
    logging.getLogger().handlers[0].setLevel(console_level)
    # 减少第三方库的日志
    for lib in ['matplotlib', 'PIL', 'numexpr']:
        logging.getLogger(lib).setLevel(logging.WARNING)


def cmd_clean(args):
    """数据清洗命令"""
    from data.loader import DataLoader
    from data.cleaner import DataCleaner

    logger = logging.getLogger(__name__)
    data_dir = args.data_dir or DATA_DIR
    logger.info(f"读取数据: {data_dir}")

    loader = DataLoader(data_dir)
    train_df, test_df, val_df = loader.load_all()

    cleaner = DataCleaner(impute_strategy=args.impute_strategy)

    # 清洗各数据集
    logger.info("清洗训练集...")
    X_train, y_train = cleaner.clean(train_df)
    logger.info("清洗测试集...")
    X_test, y_test = cleaner.clean(test_df)
    logger.info("清洗验证集...")
    X_val, y_val = cleaner.clean(val_df)

    # 保存
    out_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    import joblib
    joblib.dump((X_train, y_train, X_test, y_test, X_val, y_val),
                os.path.join(out_dir, 'cleaned_data.pkl'))
    logger.info(f"清洗后数据已保存到 {out_dir}/cleaned_data.pkl")

    # 打印统计
    logger.info(f"  Train: X={X_train.shape}, y={len(y_train)}")
    logger.info(f"  Test:  X={X_test.shape}, y={len(y_test)}")
    logger.info(f"  Val:   X={X_val.shape}, y={len(y_val)}")
    logger.info(f"  各类别分布:{dict(zip(*np.unique(y_train, return_counts=True)))}")


def cmd_preprocess(args):
    """预处理 + 特征工程"""
    import joblib
    from data.preprocessor import Preprocessor
    from features.engineer import FeatureEngineer
    import numpy as np

    logger = logging.getLogger(__name__)

    # 加载清洗后的数据
    data_path = os.path.join(args.output_dir or OUTPUT_DIR, 'cleaned_data.pkl')
    if not os.path.exists(data_path):
        logger.error("请先运行 'clean' 命令")
        return
    X_train, y_train, X_test, y_test, X_val, y_val = joblib.load(data_path)
    logger.info(f"加载清洗后数据: Train={X_train.shape}")

    # 标签编码（字符串 → 整数）
    preprocessor = Preprocessor(scaling=args.scaling or 'standard', random_state=RANDOM_SEED)
    logger.info("标签编码...")
    y_train_enc = preprocessor.encode_labels(y_train)
    y_test_enc = preprocessor.encode_labels(y_test)
    y_val_enc = preprocessor.encode_labels(y_val)

    # 特征工程
    eng = FeatureEngineer(methods=['ratios', 'interactions'])
    logger.info("特征工程 (fit_transform)...")
    X_train_eng = eng.fit_transform(X_train, y_train_enc)
    X_test_eng = eng.transform(X_test)
    X_val_eng = eng.transform(X_val)
    logger.info(f"  特征数: {X_train.shape[1]} → {X_train_eng.shape[1]}")

    # 缩放
    logger.info(f"特征缩放 ({args.scaling or 'standard'})...")
    X_train_scaled = preprocessor.fit_transform(X_train_eng)
    X_test_scaled = preprocessor.transform(X_test_eng)
    X_val_scaled = preprocessor.transform(X_val_eng)

    # 分割数据集
    data = preprocessor.split_data(
        X_train_scaled, y_train_enc,
        X_test_scaled, y_test_enc,
        X_val_scaled, y_val_enc,
    )

    # 保存
    out_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(data, os.path.join(out_dir, 'processed_data.pkl'))
    joblib.dump(preprocessor, os.path.join(out_dir, 'models', 'scaler.pkl'))
    joblib.dump(eng, os.path.join(out_dir, 'models', 'feature_engineer.pkl'))
    logger.info("预处理完成，数据已保存")


def _get_model_builders(best_params: dict = None):
    """返回模型构建器字典"""
    best_params = best_params or {}

    def build_lr():
        from models.logistic_regression import LogisticRegressionClassifier
        m = LogisticRegressionClassifier(params=best_params.get('lr'))
        m.build()
        return m

    def build_knn():
        from models.knn import KNNClassifier
        m = KNNClassifier(params=best_params.get('knn'))
        m.build()
        return m

    def build_svm():
        from models.svm import SVMClassifier
        m = SVMClassifier(params=best_params.get('svm'))
        m.build()
        return m

    def build_rf():
        from models.random_forest import RandomForestClassifier
        m = RandomForestClassifier(params=best_params.get('rf'))
        m.build()
        return m

    def build_xgb():
        from models.xgboost_model import XGBoostClassifier
        m = XGBoostClassifier(params=best_params.get('xgboost'))
        m.build()
        return m

    def build_lgb():
        from models.lightgbm_model import LightGBMClassifier
        m = LightGBMClassifier(params=best_params.get('lightgbm'))
        m.build()
        return m

    return {
        'Logistic Regression': build_lr,
        'KNN': build_knn,
        'SVM': build_svm,
        'Random Forest': build_rf,
        'XGBoost': build_xgb,
        'LightGBM ★': build_lgb,
    }


def cmd_train(args):
    """训练单个模型"""
    import joblib
    import numpy as np

    logger = logging.getLogger(__name__)

    data_path = os.path.join(args.output_dir or OUTPUT_DIR, 'processed_data.pkl')
    if not os.path.exists(data_path):
        logger.error("请先运行 'preprocess' 命令")
        return
    data = joblib.load(data_path)

    model_key = args.model
    builders = _get_model_builders()
    model_names = list(builders.keys())

    if model_key not in MODEL_REGISTRY:
        logger.error(f"未知模型: {model_key}. 可选: {list(MODEL_REGISTRY.keys())}")
        return

    # 找到对应的 builder
    model_idx = list(MODEL_REGISTRY.keys()).index(model_key)
    model_name = model_names[model_idx]
    build_fn = builders[model_name]

    # 超参数调优
    best_params = {}
    if args.tune:
        from training.tuner import HyperparameterTuner
        from config.hyperparameters import PARAM_GRIDS
        tuner = HyperparameterTuner()
        param_grid = PARAM_GRIDS.get(model_key, {})
        cls_map = {
            'lr': LogisticRegressionClassifier,
            'knn': KNNClassifier,
            'svm': SVMClassifier,
            'rf': RandomForestClassifier,
            'xgboost': XGBoostClassifier,
            'lightgbm': LightGBMClassifier,
        }
        from models.logistic_regression import LogisticRegressionClassifier
        from models.knn import KNNClassifier
        from models.svm import SVMClassifier
        from models.random_forest import RandomForestClassifier
        from models.xgboost_model import XGBoostClassifier
        from models.lightgbm_model import LightGBMClassifier
        cls_map = {
            'lr': LogisticRegressionClassifier,
            'knn': KNNClassifier,
            'svm': SVMClassifier,
            'rf': RandomForestClassifier,
            'xgboost': XGBoostClassifier,
            'lightgbm': LightGBMClassifier,
        }
        result = tuner.tune(cls_map[model_key], param_grid,
                           data['X_train'], data['y_train'], model_key)
        best_params[model_key] = result['best_params']
        joblib.dump(best_params, os.path.join(args.output_dir or OUTPUT_DIR, 'best_params.pkl'))

    # 训练
    from evaluation.comparator import AlgorithmComparator
    comparator = AlgorithmComparator(class_names=CLASS_NAMES)
    builders_with_params = _get_model_builders(best_params)
    build_fn = builders_with_params[model_name]
    result = comparator.run_single(
        build_fn(),
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test'],
        data.get('X_val'), data.get('y_val'),
    )

    # 保存模型
    out_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)
    joblib.dump(result, os.path.join(out_dir, 'models', f'{model_key}_result.pkl'))
    logger.info(f"模型已保存: {out_dir}/models/{model_key}_result.pkl")


def cmd_train_all(args):
    """训练全部模型"""
    import joblib

    logger = logging.getLogger(__name__)

    data_path = os.path.join(args.output_dir or OUTPUT_DIR, 'processed_data.pkl')
    if not os.path.exists(data_path):
        logger.error("请先运行 'preprocess' 命令")
        return
    data = joblib.load(data_path)

    best_params = {}
    if args.tune:
        from training.tuner import HyperparameterTuner
        from models.logistic_regression import LogisticRegressionClassifier
        from models.knn import KNNClassifier
        from models.svm import SVMClassifier
        from models.random_forest import RandomForestClassifier
        from models.xgboost_model import XGBoostClassifier
        from models.lightgbm_model import LightGBMClassifier

        cls_map = {
            'lr': LogisticRegressionClassifier,
            'knn': KNNClassifier,
            'svm': SVMClassifier,
            'rf': RandomForestClassifier,
            'xgboost': XGBoostClassifier,
            'lightgbm': LightGBMClassifier,
        }
        tuner = HyperparameterTuner()
        tuner.tune_all(cls_map, data['X_train'], data['y_train'])
        for key, r in tuner.results.items():
            best_params[key] = r.get('best_params', {})

        out_dir = args.output_dir or OUTPUT_DIR
        joblib.dump(best_params, os.path.join(out_dir, 'best_params.pkl'))

    # 训练所有模型
    builders = _get_model_builders(best_params)
    from evaluation.comparator import AlgorithmComparator
    comparator = AlgorithmComparator(class_names=CLASS_NAMES)
    all_results = comparator.run_all(builders, data)

    # 保存
    out_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(all_results, os.path.join(out_dir, 'all_results.pkl'))

    # 打印对比表
    table = comparator.create_comparison_table()
    logger.info("\n" + table.to_string(index=False))


def cmd_compare(args):
    """完整对比分析"""
    import joblib
    import pandas as pd

    logger = logging.getLogger(__name__)

    data_path = os.path.join(args.output_dir or OUTPUT_DIR, 'processed_data.pkl')
    results_path = os.path.join(args.output_dir or OUTPUT_DIR, 'all_results.pkl')

    if not os.path.exists(results_path):
        logger.info("未找到已有结果，开始训练...")
        cmd_train_all(args)

    data = joblib.load(data_path)
    all_results = joblib.load(results_path)

    from evaluation.comparator import AlgorithmComparator
    from evaluation.overfitting import OverfittingAnalyzer
    from evaluation.speed import SpeedBenchmark

    comparator = AlgorithmComparator(class_names=CLASS_NAMES)
    comparator.results = all_results

    # 创建对比表
    comp_table = comparator.create_comparison_table()
    logger.info("\n===== 算法对比表 =====\n" + comp_table.to_string(index=False))

    # 过拟合分析
    overfit_analyzer = OverfittingAnalyzer()
    overfit_results = overfit_analyzer.analyze_all(all_results)

    # 速度测试 - 使用已训练的模型
    models_dict = {}
    for name, r in all_results.items():
        trained_model = r.get('_model')
        if trained_model is not None:
            models_dict[name] = trained_model

    speed_bench = SpeedBenchmark()
    speed_results = speed_bench.benchmark_all(models_dict, data['X_test'])

    # 保存全部
    out_dir = args.output_dir or OUTPUT_DIR
    output = {
        'comparison_table': comp_table,
        'overfitting': overfit_results,
        'speed': speed_results,
    }
    joblib.dump(output, os.path.join(out_dir, 'comparison_results.pkl'))

    # 保存表格
    tables_dir = os.path.join(out_dir, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    comp_table.to_csv(os.path.join(tables_dir, 'accuracy_comparison.csv'), index=False, encoding='utf-8-sig')

    # 统计检验
    stat_result = comparator.statistical_test()
    logger.info(f"\nFriedman Test: statistic={stat_result.get('friedman', {}).get('statistic', 'N/A'):.4f}, "
                f"p={stat_result.get('friedman', {}).get('p_value', 'N/A'):.6f}")


def cmd_robustness(args):
    """鲁棒性测试"""
    import joblib

    logger = logging.getLogger(__name__)

    data_path = os.path.join(args.output_dir or OUTPUT_DIR, 'processed_data.pkl')
    best_params_path = os.path.join(args.output_dir or OUTPUT_DIR, 'best_params.pkl')

    if not os.path.exists(data_path):
        logger.error("请先运行 'preprocess' 命令")
        return

    data = joblib.load(data_path)
    best_params = {}
    if os.path.exists(best_params_path):
        best_params = joblib.load(best_params_path)

    from evaluation.robustness import RobustnessTester
    tester = RobustnessTester(random_state=RANDOM_SEED)
    builders = _get_model_builders(best_params)
    results = tester.test_all_noises(builders, data)

    out_dir = args.output_dir or OUTPUT_DIR
    joblib.dump(results, os.path.join(out_dir, 'robustness_results.pkl'))

    # 打印汇总
    degradation = tester.compute_degradation()
    logger.info("\n===== 鲁棒性汇总 (Accuracy Drop) =====")
    for model_name, noises in sorted(degradation.items()):
        avg_drop = np.mean([v for v in noises.values() if v is not None])
        logger.info(f"  {model_name}: 平均下降 {avg_drop:.4f}")


def cmd_visualize(args):
    """生成全部图表"""
    import joblib

    logger = logging.getLogger(__name__)

    out_dir = args.output_dir or OUTPUT_DIR
    results_path = os.path.join(out_dir, 'all_results.pkl')
    data_path = os.path.join(out_dir, 'processed_data.pkl')
    cleaned_path = os.path.join(out_dir, 'cleaned_data.pkl')
    robustness_path = os.path.join(out_dir, 'robustness_results.pkl')

    from visualization import plots
    plots.set_output_dir(out_dir)

    # 加载数据
    has_results = os.path.exists(results_path)
    has_robustness = os.path.exists(robustness_path)

    if has_results:
        all_results = joblib.load(results_path)
        data = joblib.load(data_path)

        logger.info("生成模型对比图表...")
        plots.plot_accuracy_comparison(all_results, 'accuracy_comparison.png')
        plots.plot_confusion_matrices(all_results, CLASS_NAMES, 'confusion_matrices.png')
        plots.plot_per_class_f1_heatmap(all_results, CLASS_NAMES, 'per_class_f1.png')
        plots.plot_radar_chart(all_results, 'radar_chart.png')
        plots.plot_training_time_vs_accuracy(all_results, 'train_time_vs_acc.png')

        # 学习曲线
        from training.trainer import Trainer
        trainer = Trainer(cv_folds=CV_FOLDS)
        learning_curves = {}
        for name, r in all_results.items():
            if r.get('train_accuracy', 0) > 0:
                builders = _get_model_builders()
                if name in builders:
                    m = builders[name]()
                    m.build()
                    try:
                        lc = trainer.learning_curve_data(m, data['X_train'], data['y_train'])
                        learning_curves[name] = lc
                    except Exception as e:
                        logger.warning(f"  学习曲线生成失败 ({name}): {e}")
        if learning_curves:
            plots.plot_learning_curves(learning_curves, 'learning_curves.png')

        # Loss 曲线（对支持的三模型）
        for name in ['XGBoost', 'LightGBM ★']:
            if name in all_results:
                th = all_results[name].get('training_history', {})
                if th.get('loss_curve'):
                    plots.plot_loss_curves(th, name, f'loss_curve_{name.replace(" ★", "").lower()}.png')

        # 特征重要性对比
        fi_dict = {}
        for name in ['Random Forest', 'XGBoost', 'LightGBM ★']:
            if name in all_results:
                fi = all_results[name].get('feature_importance')
                if fi is not None:
                    # 使用全部特征名（原始+工程化后）
                    fi_dict[name] = fi
        if fi_dict:
            # 尝试获取特征名
            eng = joblib.load(os.path.join(out_dir, 'models', 'feature_engineer.pkl'))
            feature_names = eng.selected_features if hasattr(eng, 'selected_features') and eng.selected_features else \
                           [f'Feature_{i}' for i in range(len(list(fi_dict.values())[0]))]
            if len(feature_names) < len(list(fi_dict.values())[0]):
                feature_names = [f'F_{i}' for i in range(len(list(fi_dict.values())[0]))]
            plots.plot_feature_importance_comparison(fi_dict, feature_names, save_path='feature_importance_comparison.png')

        # 过拟合
        from evaluation.overfitting import OverfittingAnalyzer
        oa = OverfittingAnalyzer()
        of_results = oa.analyze_all(all_results)
        plots.plot_overfitting_gap(of_results, 'overfitting_gap.png')

        # 速度
        from evaluation.speed import SpeedBenchmark
        models_dict = {}
        for name, r in all_results.items():
            trained_model = r.get('_model')
            if trained_model is not None:
                models_dict[name] = trained_model
        sb = SpeedBenchmark()
        speed_res = sb.benchmark_all(models_dict, data['X_test'])
        if speed_res:
            plots.plot_inference_speed(speed_res, 'inference_speed.png')

        # 最佳模型的 ROC 和 PR 曲线
        best_name = max(all_results.keys(), key=lambda n: all_results[n]['metrics']['accuracy'])
        best_result = all_results[best_name]
        if best_result.get('probabilities') is not None:
            plots.plot_roc_curves(data['y_test'], best_result['probabilities'],
                                  CLASS_NAMES, f'ROC Curves ({best_name})', 'roc_curves.png')
            plots.plot_precision_recall_curves(data['y_test'], best_result['probabilities'],
                                              CLASS_NAMES, f'PR Curves ({best_name})', 'pr_curves.png')

        # PCA / t-SNE
        plots.plot_pca_2d(data['X_train'], data['y_train'], CLASS_NAMES, 'PCA 2D (Train)', 'pca_2d.png')

    # 鲁棒性
    if has_robustness:
        rob_results = joblib.load(robustness_path)
        logger.info("生成鲁棒性图表...")
        from evaluation.robustness import RobustnessTester
        tester = RobustnessTester()
        tester.results = rob_results
        degradation = tester.compute_degradation()
        plots.plot_robustness_degradation(degradation, 'robustness_degradation.png')
        plots.plot_robustness_heatmap(rob_results, 'robustness_heatmap.png')

    logger.info("所有图表生成完毕")


def cmd_pipeline(args):
    """运行完整 pipeline"""
    logger = logging.getLogger(__name__)
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("Dry Bean Classification — 全流程 ML Pipeline")
    logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

    out_dir = args.output_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    for d in ['figures', 'tables', 'models', 'logs']:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    # Step 1: Clean
    logger.info("\n" + "█" * 50)
    logger.info("Step 1/6: 数据清洗")
    logger.info("█" * 50)
    clean_args = argparse.Namespace(
        data_dir=args.data_dir,
        output_dir=out_dir,
        impute_strategy='knn',
    )
    cmd_clean(clean_args)

    # Step 2: Preprocess
    logger.info("\n" + "█" * 50)
    logger.info("Step 2/6: 预处理 + 特征工程")
    logger.info("█" * 50)
    prep_args = argparse.Namespace(
        output_dir=out_dir,
        scaling='standard',
    )
    cmd_preprocess(prep_args)

    # Step 3: Train All
    logger.info("\n" + "█" * 50)
    logger.info("Step 3/6: 训练全部模型")
    logger.info("█" * 50)
    train_args = argparse.Namespace(
        output_dir=out_dir,
        tune=not args.skip_tuning,
        cv_folds=CV_FOLDS,
        seed=RANDOM_SEED,
    )
    cmd_train_all(train_args)

    # Step 4: Compare
    logger.info("\n" + "█" * 50)
    logger.info("Step 4/6: 完整对比分析")
    logger.info("█" * 50)
    cmd_compare(train_args)

    # Step 5: Robustness
    logger.info("\n" + "█" * 50)
    logger.info("Step 5/6: 鲁棒性测试")
    logger.info("█" * 50)
    rob_args = argparse.Namespace(output_dir=out_dir)
    cmd_robustness(rob_args)

    # Step 6: Visualize
    logger.info("\n" + "█" * 50)
    logger.info("Step 6/6: 生成全部图表")
    logger.info("█" * 50)
    vis_args = argparse.Namespace(output_dir=out_dir)
    cmd_visualize(vis_args)

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"Pipeline 完成! 总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"输出目录: {out_dir}")
    logger.info(f"  - figures/ : 图表 (PNG, 300 DPI)")
    logger.info(f"  - tables/  : 数据表格 (CSV)")
    logger.info(f"  - models/  : 模型文件 (pkl)")
    logger.info(f"  - logs/    : 运行日志")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Dry Bean Classification — ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # clean
    p = subparsers.add_parser('clean', help='数据清洗')
    p.add_argument('--data-dir', default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--impute-strategy', default='knn', choices=['knn', 'median'])

    # preprocess
    p = subparsers.add_parser('preprocess', help='预处理 + 特征工程')
    p.add_argument('--output-dir', default=None)
    p.add_argument('--scaling', default='standard', choices=['standard', 'minmax', 'robust', 'none'])

    # train
    p = subparsers.add_parser('train', help='训练单个模型')
    p.add_argument('--model', required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument('--tune', action='store_true')
    p.add_argument('--output-dir', default=None)

    # train-all
    p = subparsers.add_parser('train-all', help='训练全部模型')
    p.add_argument('--tune', action='store_true')
    p.add_argument('--output-dir', default=None)

    # compare
    p = subparsers.add_parser('compare', help='完整对比分析')
    p.add_argument('--output-dir', default=None)

    # robustness
    p = subparsers.add_parser('robustness', help='鲁棒性测试')
    p.add_argument('--output-dir', default=None)

    # visualize
    p = subparsers.add_parser('visualize', help='生成全部图表')
    p.add_argument('--output-dir', default=None)

    # pipeline
    p = subparsers.add_parser('pipeline', help='一键运行完整流程')
    p.add_argument('--data-dir', default=None)
    p.add_argument('--output-dir', default=None)
    p.add_argument('--skip-tuning', action='store_true')
    p.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 设置日志
    out_dir = getattr(args, 'output_dir', None) or OUTPUT_DIR
    setup_logging(out_dir, verbose=getattr(args, 'verbose', False))
    logger = logging.getLogger(__name__)

    # 分发
    if args.command == 'clean':
        cmd_clean(args)
    elif args.command == 'preprocess':
        cmd_preprocess(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'train-all':
        cmd_train_all(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'robustness':
        cmd_robustness(args)
    elif args.command == 'visualize':
        cmd_visualize(args)
    elif args.command == 'pipeline':
        cmd_pipeline(args)


if __name__ == '__main__':
    import numpy as np
    main()
