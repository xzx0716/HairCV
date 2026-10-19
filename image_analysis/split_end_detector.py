"""
分叉率检测模块

使用传统图像处理方法（阈值分割 + 轮廓检测）检测分叉率
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SplitEndResult:
    """分叉率检测结果"""
    split_end_count: int  # 分叉点数量
    total_hairs: int  # 总发丝数
    split_end_rate: float  # 分叉率(%)
    split_end_locations: list  # 分叉点位置坐标


class SplitEndDetector:
    """分叉率检测器"""

    def __init__(self, threshold_method: str = 'otsu'):
        """
        初始化检测器

        Args:
            threshold_method: 阈值分割方法 ('otsu' 或 'fixed')
        """
        self.threshold_method = threshold_method

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理

        Args:
            image: 输入图像 (BGR)

        Returns:
            预处理后的灰度图
        """
        # 灰度化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 高斯模糊去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        return blurred

    def segment_hairs(self, blurred: np.ndarray) -> np.ndarray:
        """
        阈值分割获取头发二值图像

        Args:
            blurred: 预处理后的灰度图

        Returns:
            二值图像
        """
        if self.threshold_method == 'otsu':
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        return binary

    def morphological_operations(self, binary: np.ndarray) -> np.ndarray:
        """
        形态学操作去噪并细化

        Args:
            binary: 二值图像

        Returns:
            处理后的二值图像
        """
        # 创建核
        kernel = np.ones((3, 3), np.uint8)

        # 开运算去除小噪点
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

        # 闭运算填充孔洞
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        return closed

    def skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """
        骨架化头发轮廓

        Args:
            binary: 二值图像

        Returns:
            骨架图像
        """
        # 确保是uint8类型
        skeleton = np.zeros_like(binary)

        # 创建结构元素
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # 迭代骨架化
        while True:
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(binary, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            binary = cv2.erode(binary, element)

            if cv2.countNonZero(binary) == 0:
                break

        return skeleton

    def detect_branch_points(self, skeleton: np.ndarray) -> list:
        """
        检测骨架中的分叉点

        Args:
            skeleton: 骨架图像

        Returns:
            分叉点位置列表 [(x, y), ...]
        """
        # 分叉点检测核
        kernel1 = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=np.uint8)  # T型

        kernel2 = np.array([[1, 0, 1],
                           [0, 0, 0],
                           [1, 0, 1]], dtype=np.uint8)  # X型

        # 卷积找到分叉点
        conv1 = cv2.filter2D(skeleton, -1, kernel1)
        conv2 = cv2.filter2D(skeleton, -1, kernel2)

        # 归一化
        conv1 = conv1 / 255
        conv2 = conv2 / 255

        # 合并结果
        branch_points = ((conv1 >= 2.5) | (conv2 >= 3.5)) & (skeleton > 0)

        # 获取坐标
        locations = np.where(branch_points)
        points = [(int(y), int(x)) for x, y in zip(locations[1], locations[0])]

        return points

    def count_hairs(self, binary: np.ndarray) -> int:
        """
        统计总发丝数

        Args:
            binary: 二值图像

        Returns:
            发丝数量
        """
        # 找到所有轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤太小的轮廓（噪点）
        min_area = 100
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        return len(valid_contours)

    def detect(self, image: np.ndarray, total_hairs_hint: Optional[int] = None) -> Tuple[int, dict]:
        """
        检测分叉率

        Args:
            image: 输入图像
            total_hairs_hint: 总发丝数提示（可选）

        Returns:
            (分叉率, 详细信息字典)
        """
        # 预处理
        blurred = self.preprocess(image)

        # 分割
        binary = self.segment_hairs(blurred)

        # 形态学操作
        processed = self.morphological_operations(binary)

        # 骨架化
        skeleton = self.skeletonize(processed)

        # 检测分叉点
        split_end_locations = self.detect_branch_points(skeleton)
        split_end_count = len(split_end_locations)

        # 统计发丝数
        if total_hairs_hint is not None:
            total_hairs = total_hairs_hint
        else:
            total_hairs = self.count_hairs(processed)

        # 计算分叉率
        if total_hairs > 0:
            split_end_rate = (split_end_count / total_hairs) * 100
        else:
            split_end_rate = 0.0

        # 限制最大值
        split_end_rate = min(split_end_rate, 100.0)

        details = {
            'split_end_count': split_end_count,
            'total_hairs': total_hairs,
            'split_end_rate': float(split_end_rate),
            'split_end_locations': split_end_locations
        }

        return split_end_rate, details

    def detect_with_yolo(self, image: np.ndarray, model) -> Tuple[float, dict]:
        """
        使用YOLO模型检测分叉点（深度学习方法）

        Args:
            image: 输入图像
            model: YOLO模型

        Returns:
            (分叉率, 详细信息字典)
        """
        # TODO: 实现YOLO检测逻辑
        # results = model.predict(image)
        # split_end_count = len(results[0].boxes)
        pass


def create_split_end_detector(threshold_method: str = 'otsu') -> SplitEndDetector:
    """
    创建分叉率检测器

    Args:
        threshold_method: 阈值分割方法

    Returns:
        SplitEndDetector实例
    """
    return SplitEndDetector(threshold_method=threshold_method)
