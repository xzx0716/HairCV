"""
发质图像数据预处理模块

提供图像裁剪、增强、归一化等功能
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import json


class ImagePreprocessor:
    """图像预处理类"""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        初始化预处理器

        Args:
            target_size: 目标图像尺寸 (width, height)
        """
        self.target_size = target_size

    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像

        Args:
            image_path: 图像路径

        Returns:
            图像数组 (BGR格式)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return img

    def crop_microscope_roi(self, image: np.ndarray) -> np.ndarray:
        """
        显微镜图像裁剪 - 裁剪出单根发丝区域

        Args:
            image: 输入图像

        Returns:
            裁剪后的图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Otsu阈值分割获取发丝区域
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 找到发丝区域的轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # 找到最大轮廓（发丝）
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 添加边界margin
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y:y+h, x:x+w]

    def crop_split_end_roi(self, image: np.ndarray) -> np.ndarray:
        """
        微距分叉率图像裁剪 - 裁剪出发梢1cm区域

        Args:
            image: 输入图像

        Returns:
            裁剪后的图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 阈值分割获取头发区域
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # 找到非零像素的位置（头发区域）
        coords = np.column_stack(np.where(thresh > 0))

        if len(coords) == 0:
            return image

        # 找到发梢区域（y坐标最小的部分，即图像上方）
        min_y = coords[:, 0].min()
        max_y = min(min_y + int(image.shape[0] * 0.3), image.shape[0])  # 取上方30%

        # 在发梢区域内找最密集的区域
        roi = image[:max_y, :]

        return roi

    def crop_macro_roi(self, image: np.ndarray, side: str = "front") -> np.ndarray:
        """
        宏观图像裁剪 - 裁剪出头发区域

        Args:
            image: 输入图像
            side: 拍摄角度 ("front", "left", "right")

        Returns:
            裁剪后的图像
        """
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 头发颜色范围（黑色头发）
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 100])

        # 创建掩码
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 找到头发区域轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # 合并所有轮廓
        all_points = np.vstack([c for c in contours])
        x, y, w, h = cv2.boundingRect(all_points)

        # 添加边界margin
        margin = 30
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        return image[y:y+h, x:x+w]

    def resize_image(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        调整图像大小

        Args:
            image: 输入图像
            size: 目标尺寸，如果为None则使用默认尺寸

        Returns:
            调整大小后的图像
        """
        if size is None:
            size = self.target_size
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    def normalize(self, image: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        图像归一化

        Args:
            image: 输入图像
            method: 归一化方法 ("minmax" 或 "zscore")

        Returns:
            归一化后的图像
        """
        img_float = image.astype(np.float32)

        if method == "minmax":
            # Min-Max归一化到[0, 1]
            img_min = img_float.min()
            img_max = img_float.max()
            if img_max - img_min > 0:
                return (img_float - img_min) / (img_max - img_min)
            return img_float
        elif method == "zscore":
            # Z-score标准化
            mean = img_float.mean()
            std = img_float.std()
            if std > 0:
                return (img_float - mean) / std
            return img_float
        else:
            raise ValueError(f"未知的归一化方法: {method}")

    def augment_image(self, image: np.ndarray,
                     horizontal_flip: bool = True,
                     rotation: bool = True,
                     brightness: bool = True,
                     contrast: bool = True) -> List[np.ndarray]:
        """
        图像数据增强

        Args:
            image: 输入图像
            horizontal_flip: 是否水平翻转
            rotation: 是否随机旋转
            brightness: 是否调整亮度
            contrast: 是否调整对比度

        Returns:
            增强后的图像列表
        """
        augmented = [image.copy()]

        # 水平翻转
        if horizontal_flip:
            flipped = cv2.flip(image, 1)
            augmented.append(flipped)

        # 旋转（±15度）
        if rotation:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            for angle in [-15, 15]:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h))
                augmented.append(rotated)

        # 亮度调整
        if brightness:
            for factor in [0.8, 1.2]:
                adjusted = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
                augmented.append(adjusted)

        # 对比度调整
        if contrast:
            for factor in [0.8, 1.2]:
                mean = image.mean()
                adjusted = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
                augmented.append(adjusted)

        return augmented

    def preprocess_for_inference(self, image_path: str, image_type: str) -> np.ndarray:
        """
        推理时的一站式预处理

        Args:
            image_path: 图像路径
            image_type: 图像类型 ("microscope", "split_end", "macro")

        Returns:
            预处理后的图像
        """
        # 加载图像
        img = self.load_image(image_path)

        # 根据类型裁剪ROI
        if image_type == "microscope":
            img = self.crop_microscope_roi(img)
        elif image_type == "split_end":
            img = self.crop_split_end_roi(img)
        elif image_type == "macro":
            img = self.crop_macro_roi(img)

        # 调整大小
        img = self.resize_image(img)

        # 归一化
        img = self.normalize(img)

        return img


class LabelMeParser:
    """LabelMe标注文件解析器"""

    @staticmethod
    def parse_json(json_path: str) -> dict:
        """
        解析LabelMe JSON文件

        Args:
            json_path: JSON文件路径

        Returns:
            标注数据字典
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    @staticmethod
    def extract_cuticle_grade(json_path: str) -> int:
        """
        从标注文件中提取毛鳞片等级

        Args:
            json_path: JSON文件路径

        Returns:
            毛鳞片等级 (0-3)
        """
        data = LabelMeParser.parse_json(json_path)

        # 获取第一个形状的标签作为等级
        if data.get('shapes'):
            label = data['shapes'][0].get('label', '0')
            try:
                return int(label)
            except ValueError:
                return 0
        return 0

    @staticmethod
    def extract_split_end_count(json_path: str) -> Tuple[int, int]:
        """
        从标注文件中提取分叉点数量和总发丝数

        Args:
            json_path: JSON文件路径

        Returns:
            (分叉点数, 总发丝数)
        """
        data = LabelMeParser.parse_json(json_path)

        split_end_count = 0
        total_hairs = 20  # 默认值

        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            if label == 'split_end':
                split_end_count += 1
            elif label == 'total_hairs':
                # 从备注中获取总发丝数
                pass

        return split_end_count, total_hairs

    @staticmethod
    def extract_frizziness_grade(json_path: str) -> int:
        """
        从标注文件中提取毛躁度等级

        Args:
            json_path: JSON文件路径

        Returns:
            毛躁度等级 (0-3)
        """
        data = LabelMeParser.parse_json(json_path)

        if data.get('shapes'):
            label = data['shapes'][0].get('label', '0')
            try:
                return int(label)
            except ValueError:
                return 0
        return 0

    @staticmethod
    def extract_shininess_grade(json_path: str) -> int:
        """
        从标注文件中提取光泽度等级

        Args:
            json_path: JSON文件路径

        Returns:
            光泽度等级 (0-2)
        """
        data = LabelMeParser.parse_json(json_path)

        if data.get('shapes'):
            label = data['shapes'][0].get('label', '0')
            try:
                return int(label)
            except ValueError:
                return 0
        return 0


def create_preprocessor(config_path: Optional[str] = None) -> ImagePreprocessor:
    """
    创建预处理器实例

    Args:
        config_path: 配置文件路径（可选）

    Returns:
        ImagePreprocessor实例
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        target_size = tuple(config.get('target_size', [224, 224]))
        return ImagePreprocessor(target_size=target_size)
    return ImagePreprocessor()
