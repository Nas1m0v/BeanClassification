"""噪声增强模块 - 用于鲁棒性测试"""
import numpy as np
from typing import Dict, List


class NoiseAugmenter:
    """向训练数据添加不同类型的噪声，用于测试模型鲁棒性"""

    def __init__(self, random_state: int = 42):
        self.rng = np.random.RandomState(random_state)

    def add_gaussian_noise(self, X: np.ndarray, std_ratio: float) -> np.ndarray:
        """
        添加零均值高斯噪声
        std = std_ratio * feature_std
        """
        X_noisy = X.copy()
        feature_stds = X.std(axis=0)
        for j in range(X.shape[1]):
            noise = self.rng.normal(0, std_ratio * feature_stds[j], size=X.shape[0])
            X_noisy[:, j] += noise
        return X_noisy

    def add_salt_pepper_noise(self, X: np.ndarray, ratio: float) -> np.ndarray:
        """随机将 ratio 比例的值替换为该特征的最小值或最大值"""
        X_noisy = X.copy()
        n_samples, n_features = X.shape
        n_corrupt = int(n_samples * n_features * ratio)

        for _ in range(n_corrupt):
            i = self.rng.randint(0, n_samples)
            j = self.rng.randint(0, n_features)
            if self.rng.rand() < 0.5:
                X_noisy[i, j] = X[:, j].min()
            else:
                X_noisy[i, j] = X[:, j].max()
        return X_noisy

    def add_label_noise(self, y: np.ndarray, flip_ratio: float,
                        n_classes: int = 7) -> np.ndarray:
        """随机翻转 flip_ratio 比例的标签到另一个随机类别"""
        y_noisy = y.copy()
        n_samples = len(y)
        n_flip = int(n_samples * flip_ratio)
        flip_indices = self.rng.choice(n_samples, size=n_flip, replace=False)

        for idx in flip_indices:
            current_label = int(y[idx])
            # 随机选择一个不同的标签
            other_labels = [c for c in range(n_classes) if c != current_label]
            y_noisy[idx] = self.rng.choice(other_labels)

        return y_noisy

    def add_outlier_noise(self, X: np.ndarray, ratio: float,
                          magnitude: float = 5.0) -> np.ndarray:
        """将 ratio 比例的值替换为 feature_mean ± magnitude * feature_std"""
        X_noisy = X.copy()
        n_samples, n_features = X.shape
        n_corrupt = int(n_samples * n_features * ratio)
        feature_means = X.mean(axis=0)
        feature_stds = X.std(axis=0)

        for _ in range(n_corrupt):
            i = self.rng.randint(0, n_samples)
            j = self.rng.randint(0, n_features)
            sign = 1 if self.rng.rand() < 0.5 else -1
            X_noisy[i, j] = feature_means[j] + sign * magnitude * feature_stds[j]
        return X_noisy

    def get_noise_configs(self) -> List[Dict]:
        """返回全部待测试的噪声配置"""
        configs = []
        # 高斯噪声
        for std in [0.01, 0.05, 0.10, 0.15, 0.20]:
            configs.append({
                'type': 'gaussian',
                'param': std * 100,
                'label': f'高斯噪声 σ={int(std*100)}%',
                'category': '特征噪声',
            })
        # 椒盐噪声
        for ratio in [0.01, 0.05, 0.10]:
            configs.append({
                'type': 'salt_pepper',
                'param': ratio * 100,
                'label': f'椒盐噪声 {int(ratio*100)}%',
                'category': '特征噪声',
            })
        # 标签噪声
        for ratio in [0.05, 0.10, 0.15, 0.20]:
            configs.append({
                'type': 'label',
                'param': ratio * 100,
                'label': f'标签翻转 {int(ratio*100)}%',
                'category': '标签噪声',
            })
        # 异常值噪声
        for ratio in [0.01, 0.05]:
            configs.append({
                'type': 'outlier',
                'param': ratio * 100,
                'label': f'异常值噪声 {int(ratio*100)}%',
                'category': '特征噪声',
            })
        return configs

    def apply_noise(self, X: np.ndarray, y: np.ndarray,
                    noise_config: Dict, n_classes: int = 7) -> tuple:
        """根据配置应用噪声"""
        noise_type = noise_config['type']
        param = noise_config['param']

        if noise_type == 'gaussian':
            return self.add_gaussian_noise(X, param / 100.0), y.copy()
        elif noise_type == 'salt_pepper':
            return self.add_salt_pepper_noise(X, param / 100.0), y.copy()
        elif noise_type == 'label':
            return X.copy(), self.add_label_noise(y, param / 100.0, n_classes)
        elif noise_type == 'outlier':
            return self.add_outlier_noise(X, param / 100.0), y.copy()
        else:
            return X.copy(), y.copy()
