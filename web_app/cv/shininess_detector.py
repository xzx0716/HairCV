"""
光泽度检测模块

使用传统图像处理方法（亮度直方图分析）检测光泽度等级
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class ShininessResult:
    """光泽度检测结果"""
    grade: int  # 光泽度等级 (0-2)
    brightness_mean: float  # 平均亮度
    highlight_peak_height: float  # 高光峰值高度
    highlight_peak_width: float  # 高光峰值宽度


class ShininessDetector:
    """光泽度检测器"""

    def __init__(self, brightness_threshold: int = 150,
                 highlight_ratio: float = 1.5):
        """
        初始化检测器

        Args:
            brightness_threshold: 高光亮度阈值
            highlight_ratio: 高光峰比率（相对于平均值的倍数）
        """
        self.brightness_threshold = brightness_threshold
        self.highlight_ratio = highlight_ratio

    def segment_hair_region(self, image: np.ndarray) -> np.ndarray:
        """
        分割头发区域

        Args:
            image: 输入图像 (BGR)

        Returns:
            头发区域掩码
        """
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 头发颜色范围（扩展以覆盖不同发色）
        # 黑色头发
        lower_black1 = np.array([0, 0, 0])
        upper_black1 = np.array([180, 255, 100])

        # 深棕色头发
        lower_brown = np.array([0, 30, 0])
        upper_brown = np.array([30, 255, 100])

        # 创建掩码
        mask1 = cv2.inRange(hsv, lower_black1, upper_black1)
        mask2 = cv2.inRange(hsv, lower_brown, upper_brown)

        # 合并
        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学操作优化
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

    def calculate_brightness(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        计算头发区域的亮度

        Args:
            image: 输入图像
            mask: 头发掩码

        Returns:
            亮度值数组
        """
        # 计算亮度: Y = 0.299R + 0.587G + 0.114B
        b, g, r = cv2.split(image)
        brightness = 0.299 * r + 0.587 * g + 0.114 * b

        # 应用掩码
        masked_brightness = cv2.bitwise_and(
            brightness.astype(np.uint8),
            brightness.astype(np.uint8),
            mask=mask
        )

        return masked_brightness

    def compute_brightness_histogram(self, brightness: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算亮度直方图

        Args:
            brightness: 亮度数组
            mask: 掩码

        Returns:
            (直方图, bin边界)
        """
        # 获取有效像素
        valid_pixels = brightness[mask > 0]

        if len(valid_pixels) == 0:
            return np.zeros(256), np.arange(257)

        # 计算直方图
        hist, bins = np.histogram(valid_pixels, bins=256, range=(0, 256))

        return hist, bins

    def detect_highlight_peak(self, hist: np.ndarray) -> Tuple[float, float]:
        """
        检测高光峰值

        Args:
            hist: 亮度直方图

        Returns:
            (峰值高度, 峰值宽度)
        """
        # 找到直方图的峰值
        max_idx = np.argmax(hist)
        peak_height = float(hist[max_idx])

        # 计算峰值宽度（半高宽）
        half_max = peak_height / 2

        # 找到左右半高位置
        left_idx = 0
        right_idx = len(hist) - 1

        for i in range(max_idx, -1, -1):
            if hist[i] < half_max:
                left_idx = i
                break

        for i in range(max_idx, len(hist)):
            if hist[i] < half_max:
                right_idx = i
                break

        peak_width = float(right_idx - left_idx)

        return peak_height, peak_width

    def classify_shininess(self, brightness_mean: float,
                          peak_height: float,
                          peak_width: float) -> int:
        """
        分类光泽度

        Args:
            brightness_mean: 平均亮度
            peak_height: 高光峰值高度
            peak_width: 高光峰值宽度

        Returns:
            光泽度等级 (0-2)
        """
        # 计算高光比例
        total_pixels = peak_height * peak_width if peak_width > 0 else 1
        highlight_ratio = peak_height / (total_pixels + 1e-7)

        # 分类逻辑
        if brightness_mean < 80 or peak_height < 100:
            # 平均亮度低，无明显高光 -> 暗淡
            return 0
        elif brightness_mean < 150 and peak_height < 500:
            # 平均亮度中等，高光较弱 -> 中等光泽
            return 1
        elif brightness_mean >= 150 and peak_height >= 500:
            # 平均亮度高，高光明显 -> 有光泽
            return 2
        elif brightness_mean >= 120 and peak_height >= 300:
            # 介于两者之间
            return 1
        else:
            return 0

    def detect(self, image: np.ndarray) -> Tuple[int, dict]:
        """
        检测光泽度

        Args:
            image: 输入图像

        Returns:
            (光泽度等级, 详细信息字典)
        """
        # 分割头发区域
        hair_mask = self.segment_hair_region(image)

        # 计算亮度
        brightness = self.calculate_brightness(image, hair_mask)

        # 计算平均亮度
        valid_pixels = brightness[hair_mask > 0]
        if len(valid_pixels) > 0:
            brightness_mean = float(np.mean(valid_pixels))
        else:
            brightness_mean = 0.0

        # 计算直方图
        hist, _ = self.compute_brightness_histogram(brightness, hair_mask)

        # 检测高光峰值
        peak_height, peak_width = self.detect_highlight_peak(hist)

        # 分类
        grade = self.classify_shininess(brightness_mean, peak_height, peak_width)

        details = {
            'grade': grade,
            'brightness_mean': brightness_mean,
            'highlight_peak_height': peak_height,
            'highlight_peak_width': peak_width,
            'grade_description': self._get_grade_description(grade)
        }

        return grade, details

    def _get_grade_description(self, grade: int) -> str:
        """
        获取等级描述

        Args:
            grade: 等级 (0-2)

        Returns:
            等级描述
        """
        descriptions = {
            0: "暗淡 - 无明显光泽",
            1: "中等光泽 - 有一定光泽感",
            2: "有光泽 - 光泽感强"
        }
        return descriptions.get(grade, "未知")

    def detect_with_cnn(self, image: np.ndarray, model) -> Tuple[int, dict]:
        """
        使用CNN模型检测光泽度（深度学习方法）

        Args:
            image: 输入图像
            model: CNN模型

        Returns:
            (光泽度等级, 详细信息字典)
        """
        # TODO: 实现CNN检测逻辑
        # 使用CNN做3分类
        pass


def create_shininess_detector(brightness_threshold: int = 150,
                              highlight_ratio: float = 1.5) -> ShininessDetector:
    """
    创建光泽度检测器

    Args:
        brightness_threshold: 高光亮度阈值
        highlight_ratio: 高光峰比率

    Returns:
        ShininessDetector实例
    """
    return ShininessDetector(
        brightness_threshold=brightness_threshold,
        highlight_ratio=highlight_ratio
    )
