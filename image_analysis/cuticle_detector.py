"""
毛鳞片完整度检测模块

使用传统图像处理方法（LBP + GLCM + SVM）检测毛鳞片完整度等级
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


class CuticleDetector:
    """毛鳞片完整度检测器"""

    def __init__(self, model_path: Optional[str] = None):
        """
        初始化检测器

        Args:
            model_path: 模型路径（如果有预训练模型）
        """
        self.scaler = StandardScaler()
        self.model = None
        self.feature_history = []

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            # 创建默认SVM模型
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale')

    def extract_lbp_features(self, gray_image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """
        提取LBP特征（标准 LBP8,1：8采样点→256维直方图）

        Args:
            gray_image: 灰度图像
            radius: LBP邻域半径
            n_points: 圆周采样点数（必须<=8，否则LBP码超出uint8范围）

        Returns:
            LBP特征向量（256维归一化直方图）
        """
        # n_points必须<=8：n个采样点的LBP码范围为0~2^n-1，用uint8存储
        n_points = min(int(n_points), 8)
        # 简化版LBP实现
        h, w = gray_image.shape
        lbp_image = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray_image[i, j]
                binary_val = 0

                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(round(i + radius * np.sin(angle)))
                    y = int(round(j + radius * np.cos(angle)))

                    if gray_image[x, y] >= center:
                        binary_val |= (1 << k)

                lbp_image[i - radius, j - radius] = binary_val

        # 计算LBP直方图
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        hist /= (hist.sum() + 1e-7)  # 归一化

        return hist

    def extract_glcm_features(self, gray_image: np.ndarray, distances: list = [1, 2, 3],
                             angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4]) -> np.ndarray:
        """
        提取GLCM纹理特征

        Args:
            gray_image: 灰度图像
            distances: 采样距离列表
            angles: 采样角度列表

        Returns:
            GLCM特征向量
        """
        # 如果图像太大，先缩小
        if gray_image.shape[0] > 256 or gray_image.shape[1] > 256:
            gray_image = cv2.resize(gray_image, (256, 256))

        # 量化到8级灰度
        gray_quantized = (gray_image / 32).astype(np.uint8)

        features = []

        for d in distances:
            for angle in angles:
                # 计算GLCM
                glcm = self._compute_glcm(gray_quantized, d, angle)

                # 提取特征
                contrast = self._compute_contrast(glcm)
                correlation = self._compute_correlation(glcm)
                energy = self._compute_energy(glcm)
                homogeneity = self._compute_homogeneity(glcm)

                features.extend([contrast, correlation, energy, homogeneity])

        return np.array(features)

    def _compute_glcm(self, image: np.ndarray, distance: int, angle: float) -> np.ndarray:
        """
        计算GLCM矩阵

        Args:
            image: 量化灰度图像
            distance: 像素距离
            angle: 角度（弧度）

        Returns:
            GLCM矩阵 (levels, levels)
        """
        levels = 8
        glcm = np.zeros((levels, levels), dtype=np.float64)

        h, w = image.shape
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        for i in range(h):
            for j in range(w):
                i2 = int(round(i + distance * sin_a))
                j2 = int(round(j + distance * cos_a))

                if 0 <= i2 < h and 0 <= j2 < w:
                    glcm[image[i, j], image[i2, j2]] += 1

        # 归一化
        glcm_sum = glcm.sum()
        if glcm_sum > 0:
            glcm /= glcm_sum

        return glcm

    def _compute_contrast(self, glcm: np.ndarray) -> float:
        """计算对比度"""
        levels = glcm.shape[0]
        contrast = 0.0
        for i in range(levels):
            for j in range(levels):
                contrast += (i - j) ** 2 * glcm[i, j]
        return contrast

    def _compute_correlation(self, glcm: np.ndarray) -> float:
        """计算相关性"""
        levels = glcm.shape[0]
        i_vals = np.arange(levels).reshape(-1, 1)
        j_vals = np.arange(levels).reshape(1, -1)

        mean_i = (i_vals * glcm).sum()
        mean_j = (j_vals * glcm).sum()
        std_i = np.sqrt(((i_vals - mean_i) ** 2 * glcm).sum())
        std_j = np.sqrt(((j_vals - mean_j) ** 2 * glcm).sum())

        if std_i > 0 and std_j > 0:
            correlation = ((i_vals - mean_i) * (j_vals - mean_j) * glcm).sum() / (std_i * std_j)
        else:
            correlation = 0.0

        return correlation

    def _compute_energy(self, glcm: np.ndarray) -> float:
        """计算能量"""
        return (glcm ** 2).sum()

    def _compute_homogeneity(self, glcm: np.ndarray) -> float:
        """计算均匀性"""
        levels = glcm.shape[0]
        homogeneity = 0.0
        for i in range(levels):
            for j in range(levels):
                if i + j > 0:
                    homogeneity += glcm[i, j] / (1 + abs(i - j))
        return homogeneity

    def extract_edge_features(self, gray_image: np.ndarray) -> np.ndarray:
        """
        提取边缘特征

        Args:
            gray_image: 灰度图像

        Returns:
            边缘特征向量
        """
        # Canny边缘检测
        edges = cv2.Canny(gray_image, 50, 150)

        # 边缘密度
        edge_density = edges.sum() / (255 * edges.size)

        # 边缘长度（边缘点数量）
        edge_length = (edges > 0).sum()

        # 计算边缘方向的直方图
        # 使用Sobel算子
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobelx**2 + sobely**2)
        angle = np.arctan2(sobely, sobelx)

        # 方向直方图（8个bin）
        angle_hist, _ = np.histogram(angle[edges > 0], bins=8, range=(-np.pi, np.pi))

        # 归一化
        angle_hist = angle_hist.astype(np.float64)
        angle_hist /= (angle_hist.sum() + 1e-7)

        return np.array([edge_density, edge_length] + list(angle_hist))

    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        提取所有特征

        Args:
            image: 输入图像 (BGR)

        Returns:
            特征向量
        """
        # 灰度化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # LBP特征
        lbp_features = self.extract_lbp_features(blurred)

        # GLCM特征
        glcm_features = self.extract_glcm_features(blurred)

        # 边缘特征
        edge_features = self.extract_edge_features(blurred)

        # 合并所有特征
        all_features = np.concatenate([lbp_features, glcm_features, edge_features])

        return all_features

    def classify(self, features: np.ndarray) -> int:
        """
        分类毛鳞片等级

        Args:
            features: 特征向量

        Returns:
            毛鳞片等级 (0-3)
        """
        if self.model is None:
            raise ValueError("模型未加载或训练")

        # 标准化
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # 预测
        prediction = self.model.predict(features_scaled)

        return int(prediction[0])

    def detect(self, image: np.ndarray) -> Tuple[int, dict]:
        """
        检测毛鳞片完整度

        Args:
            image: 输入图像

        Returns:
            (等级, 详细信息字典)
        """
        # 提取特征
        features = self.extract_all_features(image)

        # 分类
        grade = self.classify(features)

        # 计算各特征值用于分析
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (255 * edges.size)

        details = {
            'grade': grade,
            'edge_density': float(edge_density),
            'features': features.tolist()
        }

        return grade, details

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        训练模型

        Args:
            X_train: 训练特征矩阵 (n_samples, n_features)
            y_train: 训练标签 (n_samples,)
        """
        # 标准化
        X_scaled = self.scaler.fit_transform(X_train)

        # 训练SVM
        self.model.fit(X_scaled, y_train)

    def save_model(self, model_path: str):
        """
        保存模型

        Args:
            model_path: 保存路径
        """
        model_data = {
            'scaler': self.scaler,
            'model': self.model
        }
        joblib.dump(model_data, model_path)

    def load_model(self, model_path: str):
        """
        加载模型

        Args:
            model_path: 模型路径
        """
        model_data = joblib.load(model_path)
        self.scaler = model_data['scaler']
        self.model = model_data['model']


def create_cuticle_detector(model_path: Optional[str] = None) -> CuticleDetector:
    """
    创建毛鳞片检测器

    Args:
        model_path: 模型路径

    Returns:
        CuticleDetector实例
    """
    return CuticleDetector(model_path=model_path)
