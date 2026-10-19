"""
发质CV分析模块测试
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import ImagePreprocessor, LabelMeParser
from cuticle_detector import CuticleDetector
from split_end_detector import SplitEndDetector
from frizziness_detector import FrizzinessDetector
from shininess_detector import ShininessDetector
from scorer import ScoreMapper, ScoreResult


class TestPreprocessing:
    """预处理模块测试"""

    def test_image_preprocessor_init(self):
        """测试预处理器初始化"""
        preprocessor = ImagePreprocessor()
        assert preprocessor.target_size == (224, 224)

    def test_image_preprocessor_custom_size(self):
        """测试自定义尺寸"""
        preprocessor = ImagePreprocessor(target_size=(512, 512))
        assert preprocessor.target_size == (512, 512)

    def test_resize_image(self):
        """测试图像缩放"""
        preprocessor = ImagePreprocessor()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = preprocessor.resize_image(test_image, (224, 224))
        assert resized.shape[:2] == (224, 224)

    def test_normalize_minmax(self):
        """测试Min-Max归一化"""
        preprocessor = ImagePreprocessor()
        test_image = np.array([[100, 150], [200, 250]], dtype=np.uint8)
        normalized = preprocessor.normalize(test_image, method="minmax")
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_augment_image(self):
        """测试图像增强"""
        preprocessor = ImagePreprocessor()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        augmented = preprocessor.augment_image(test_image)
        assert len(augmented) > 1  # 至少返回原图+增强图


class TestCuticleDetector:
    """毛鳞片检测器测试"""

    def test_detector_init(self):
        """测试检测器初始化"""
        detector = CuticleDetector()
        assert detector.model is not None

    def test_extract_edge_features(self):
        """测试边缘特征提取"""
        detector = CuticleDetector()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        features = detector.extract_edge_features(cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY))
        assert features.shape[0] > 0


class TestSplitEndDetector:
    """分叉率检测器测试"""

    def test_detector_init(self):
        """测试检测器初始化"""
        detector = SplitEndDetector()
        assert detector.threshold_method == 'otsu'

    def test_preprocess(self):
        """测试预处理"""
        detector = SplitEndDetector()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed = detector.preprocess(test_image)
        assert len(processed.shape) == 2  # 灰度图


class TestFrizzinessDetector:
    """毛躁度检测器测试"""

    def test_detector_init(self):
        """测试检测器初始化"""
        detector = FrizzinessDetector()
        assert detector.flyaway_threshold == 10

    def test_classify_frizziness(self):
        """测试毛躁度分类"""
        detector = FrizzinessDetector()
        assert detector.classify_frizziness(0) == 0
        assert detector.classify_frizziness(5) == 1
        assert detector.classify_frizziness(20) == 2
        assert detector.classify_frizziness(50) == 3


class TestShininessDetector:
    """光泽度检测器测试"""

    def test_detector_init(self):
        """测试检测器初始化"""
        detector = ShininessDetector()
        assert detector.brightness_threshold == 150

    def test_classify_shininess(self):
        """测试光泽度分类"""
        detector = ShininessDetector()
        # 暗淡情况
        assert detector.classify_shininess(50, 50, 10) == 0
        # 有光泽情况
        assert detector.classify_shininess(180, 600, 50) == 2


class TestScorer:
    """评分映射器测试"""

    def test_scorer_init(self):
        """测试评分映射器初始化"""
        scorer = ScoreMapper()
        assert scorer.CUTICLE_GRADE_TO_SCORE[0] == 10

    def test_map_cuticle_grade(self):
        """测试毛鳞片评分映射"""
        scorer = ScoreMapper()
        result = scorer.map_cuticle_grade(0)
        assert result.score == 10
        assert result.indicator == "毛鳞片完整度"

    def test_map_split_end_rate(self):
        """测试分叉率评分映射"""
        scorer = ScoreMapper()
        result = scorer.map_split_end_rate(3.0)
        assert result.score == 10
        assert result.indicator == "分叉率"

    def test_map_frizziness_grade(self):
        """测试毛躁度评分映射"""
        scorer = ScoreMapper()
        result = scorer.map_frizziness_grade(2)
        assert result.score == 4

    def test_map_shininess_grade(self):
        """测试光泽度评分映射"""
        scorer = ScoreMapper()
        result = scorer.map_shininess_grade(2)
        assert result.score == 10
        assert result.indicator == "光泽度"

    def test_to_json(self):
        """测试JSON输出"""
        scorer = ScoreMapper()
        results = {
            "cuticle": scorer.map_cuticle_grade(1)
        }
        json_str = scorer.to_json(results)
        assert "毛鳞片完整度" in json_str
        assert "7.0" in json_str

    def test_to_ahp_format(self):
        """测试AHP格式输出"""
        scorer = ScoreMapper()
        results = {
            "cuticle": scorer.map_cuticle_grade(1),
            "split_end": scorer.map_split_end_rate(10.0)
        }
        ahp_output = scorer.to_ahp_format(results)
        assert "毛鳞片完整度" in ahp_output
        assert "分叉率" in ahp_output
        assert ahp_output["毛鳞片完整度"] == 7.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
