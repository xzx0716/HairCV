"""
毛躁度检测模块

使用传统图像处理方法（边缘检测 + 飞丝计数）检测毛躁度等级
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FrizzinessResult:
    """毛躁度检测结果"""
    grade: int  # 毛躁度等级 (0-3)
    flyaway_count: int  # 飞丝数量
    frizziness_score: float  # 毛躁度得分


class FrizzinessDetector:
    """毛躁度检测器"""

    def __init__(self, flyaway_threshold: int = 10):
        """
        初始化检测器

        Args:
            flyaway_threshold: 飞丝判定阈值（像素）
        """
        self.flyaway_threshold = flyaway_threshold

    def segment_hair_region(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割头发主体区域

        Args:
            image: 输入图像 (BGR)

        Returns:
            (头发掩码, 原始灰度图)
        """
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 头发颜色范围（黑色头发）
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 100])

        # 创建掩码
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # 形态学操作优化
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return mask, gray

    def find_main_hair_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """
        找到头发主体轮廓

        Args:
            mask: 头发掩码

        Returns:
            主体轮廓点或None
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # 找到最大轮廓
        largest_contour = max(contours, key=cv2.contourArea)

        # 简化轮廓
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        return approx

    def detect_flyaways(self, gray: np.ndarray, hair_mask: np.ndarray,
                       main_contour: np.ndarray) -> int:
        """
        检测飞丝数量

        Args:
            gray: 灰度图
            hair_mask: 头发掩码
            main_contour: 头发主体轮廓

        Returns:
            飞丝数量
        """
        # Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 获取主体轮廓的凸包
        hull = cv2.convexHull(main_contour)

        # 创建主体区域掩码
        hull_mask = np.zeros_like(hair_mask)
        cv2.drawContours(hull_mask, [hull], -1, 255, -1)

        # 膨胀主体区域
        kernel = np.ones((15, 15), np.uint8)
        expanded_mask = cv2.dilate(hull_mask, kernel, iterations=3)

        # 飞丝 = 边缘点不在膨胀后的主体区域内
        flyaway_edges = edges & (~expanded_mask)

        # 统计飞丝边缘点数量
        flyaway_pixels = cv2.countNonZero(flyaway_edges)

        # 根据像素数量估算飞丝根数
        # 假设每根飞丝约有50-100个边缘像素
        estimated_flyaways = flyaway_pixels // 75

        return estimated_flyaways

    def classify_frizziness(self, flyaway_count: int) -> int:
        """
        根据飞丝数量分类毛躁度

        Args:
            flyaway_count: 飞丝数量

        Returns:
            毛躁度等级 (0-3)
        """
        if flyaway_count == 0:
            return 0
        elif flyaway_count < 10:
            return 1
        elif flyaway_count <= 30:
            return 2
        else:
            return 3

    def detect(self, image: np.ndarray) -> Tuple[int, dict]:
        """
        检测毛躁度

        Args:
            image: 输入图像

        Returns:
            (毛躁度等级, 详细信息字典)
        """
        # 分割头发区域
        hair_mask, gray = self.segment_hair_region(image)

        # 找主体轮廓
        main_contour = self.find_main_hair_contour(hair_mask)

        if main_contour is None:
            # 没有找到头发区域
            grade = 0
            flyaway_count = 0
        else:
            # 检测飞丝
            flyaway_count = self.detect_flyaways(gray, hair_mask, main_contour)

            # 分类
            grade = self.classify_frizziness(flyaway_count)

        # 计算毛躁度得分（用于连续值输出）
        frizziness_score = min(flyaway_count / 30.0 * 10, 10.0)

        details = {
            'grade': grade,
            'flyaway_count': flyaway_count,
            'frizziness_score': float(frizziness_score),
            'grade_description': self._get_grade_description(grade)
        }

        return grade, details

    def _get_grade_description(self, grade: int) -> str:
        """
        获取等级描述

        Args:
            grade: 等级 (0-3)

        Returns:
            等级描述
        """
        descriptions = {
            0: "无毛躁 - 无明显飞丝",
            1: "轻微毛躁 - 少量飞丝(<10根)",
            2: "中度毛躁 - 中量飞丝(10-30根)",
            3: "严重毛躁 - 大量飞丝(>30根)"
        }
        return descriptions.get(grade, "未知")

    def detect_with_unet(self, image: np.ndarray, model) -> Tuple[int, dict]:
        """
        使用U-Net模型检测毛躁度（深度学习方法）

        Args:
            image: 输入图像
            model: U-Net模型

        Returns:
            (毛躁度等级, 详细信息字典)
        """
        # TODO: 实现U-Net检测逻辑
        # 使用语义分割分离头发区域
        # 基于分割结果统计飞丝
        pass


def create_frizziness_detector(flyaway_threshold: int = 10) -> FrizzinessDetector:
    """
    创建毛躁度检测器

    Args:
        flyaway_threshold: 飞丝判定阈值

    Returns:
        FrizzinessDetector实例
    """
    return FrizzinessDetector(flyaway_threshold=flyaway_threshold)
